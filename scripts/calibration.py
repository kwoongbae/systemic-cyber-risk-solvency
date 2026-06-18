"""
calibration.py -- Stage 2: calibrate the contagion intensity (beta).

The contagion intensity ``beta`` is calibrated, separately for each sector, so
that the systemic loss produced by the SIR loss-propagation model matches the
benchmark systemic loss reported by Welburn and Strong (2022). Repeating the
calibration many times under the stochastic loss model yields an empirical
*distribution* of beta, saved to ``data/infection_rates_on_{sector}.npy``.

Aligns with manuscript Section 3.3 (contagion-intensity calibration) and
Section 5.1 / Figure 3.

==============================================================================
Modelling assumptions
==============================================================================
1. Network homogeneity. The N firms form a homogeneous, fully-mixed network in
   which every infected firm can infect any susceptible firm with equal
   intensity beta (standard mass-action SIR; Section 3.2).
2. Single initial infection. Each scenario starts from one initially infected
   firm carrying the direct cost phi0 from Welburn and Strong (2022)
   (finance: $56M, information: $308M).
3. Lognormal indirect costs. The indirect cost incurred by each newly affected
   firm is i.i.d. lognormal: phi_f ~ Lognormal(ln(phi0), 1) * exp(beta / gamma),
   consistent with the heavy-tailed nature of cyber losses (Eling & Loperfido,
   2017; Eling & Jung, 2018).
4. Heterogeneous recovery. Each firm's operational resilience is drawn from
   gamma ~ U(0.95 * gamma_bar, 1.05 * gamma_bar) around the baseline
   gamma_bar = 0.1.
5. Deterministic SIR dynamics. Compartment sizes evolve via the deterministic
   SIR ODE system over T days; recoveries per day drive loss recognition.
6. Moment-matching calibration. beta = argmin_beta ( M_hat(beta) - M )^2, where
   M is the Welburn-Strong benchmark systemic loss and M_hat is the simulated
   total loss. The stochastic loss model makes each fit noisy, so the procedure
   is repeated num_simulations times to obtain the beta distribution.
==============================================================================

Run:
    python calibration.py                       # all sectors, 10,000 draws
    python calibration.py --sectors finance --num-simulations 2000
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

N = 1000                  # network size (Table C1)
GAMMA_BAR = 0.1           # baseline operational resilience used for calibration
T = 10                    # horizon for the calibration loss (days)
DEFAULT_NUM_SIMULATIONS = 10_000

# Welburn and Strong (2022) benchmarks: direct cost phi0 and systemic loss M.
WELBURN_STRONG = {
    "finance":     {"initial_loss": 56,  "systemic_loss": 3725,  "beta_initial": 0.000606},
    "information": {"initial_loss": 308, "systemic_loss": 20351, "beta_initial": 0.000608},
}


def sir_model(y, t, beta, gamma):
    S, I, R = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]


def calculate_loss(N, beta, gamma, T, initial_losses):
    """Simulate the indirect losses propagated through the SIR network."""
    I0 = len(initial_losses)
    y0 = [N - I0, I0, 0]
    t = np.linspace(0, T, T)
    ret = odeint(sir_model, y0, t, args=(beta, gamma))
    recovered_ts = np.round(np.diff(ret[:, 2], prepend=0)).astype(int)

    ibnr = {}
    for day, recovered in enumerate(recovered_ts):
        ibnr[day + 2] = (np.random.lognormal(mean=np.log(np.mean(initial_losses)),
                                              sigma=1, size=recovered)
                         * np.exp(beta / gamma))
    return ibnr


def find_optimal_beta(target_loss, beta_initial, gamma_bar=GAMMA_BAR, initial_loss=None):
    """One calibration draw: beta minimising (simulated loss - target)^2."""
    def loss_function(beta):
        gamma = np.random.uniform(0.95 * gamma_bar, 1.05 * gamma_bar)
        ibnr = calculate_loss(N, beta[0], gamma, T, initial_loss)
        simulated = sum(arr.sum() for arr in ibnr.values())
        return (simulated - target_loss) ** 2

    result = minimize(loss_function, [beta_initial], bounds=[(0, 1)], method="L-BFGS-B")
    return result.x[0]


def calibrate(sector, num_simulations=DEFAULT_NUM_SIMULATIONS, seed=0, data_dir=DATA_DIR):
    """Calibrate the beta distribution for one sector and save it to .npy."""
    params = WELBURN_STRONG[sector]
    initial_loss = [params["initial_loss"]]
    target_loss = params["systemic_loss"] - sum(initial_loss)

    random.seed(seed)
    np.random.seed(seed)
    beta_values = np.empty(num_simulations)
    for i in range(num_simulations):
        beta_values[i] = find_optimal_beta(
            target_loss, beta_initial=params["beta_initial"], initial_loss=initial_loss)
        if (i + 1) % 1000 == 0:
            print(f"[calibration] {sector}: {i + 1}/{num_simulations}")

    out = os.path.join(data_dir, f"infection_rates_on_{sector}.npy")
    np.save(out, beta_values)
    print(f"[calibration] {sector:12s}: mean beta={beta_values.mean():.6f} "
          f"(n={num_simulations}) -> {os.path.basename(out)}")
    return beta_values


def run(sectors=("finance", "information"), num_simulations=DEFAULT_NUM_SIMULATIONS,
        seed=0, data_dir=DATA_DIR):
    return {s: calibrate(s, num_simulations=num_simulations, seed=seed, data_dir=data_dir)
            for s in sectors}


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Calibrate beta from Welburn & Strong (2022) (Section 3.3).")
    p.add_argument("--sectors", nargs="+", default=["finance", "information"],
                   choices=list(WELBURN_STRONG))
    p.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=DATA_DIR)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    run(sectors=tuple(args.sectors), num_simulations=args.num_simulations,
        seed=args.seed, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
