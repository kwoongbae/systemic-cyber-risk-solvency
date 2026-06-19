"""
ibnr.py -- Stage 3: simulate Incurred-But-Not-Reported (IBNR) losses.

For a given reporting horizon ``T`` the contagion is propagated through the
network for each related-incident cluster in the ransomware sample, using the
calibrated beta distribution. Every firm that recovers (i.e. recognises and
reports its loss) on a given day generates a lognormal indirect cost, producing
a per-claim IBNR table stacked across the nine operational-resilience levels
gamma = 0.1, ..., 0.9. The observed losses are retained with the tag
``gamma = -1``.

Inputs : data/processed/ransomware_{sector}.csv, output/calibration/beta_{sector}.npy
Output : output/ibnr/ibnr_{sector}_{T}d.csv

Aligns with manuscript Section 3.4 / 5.2 (IBNR estimation).

Run:
    python ibnr.py --sector finance --T 10
    python ibnr.py --sector information --T 5
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from preprocess import processed_dir, calibration_dir, ibnr_dir

N = 1000
GAMMA_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SIGMA = 1.0


def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]


def recovered_per_day(N, beta, gamma, initial_infected, days):
    """Newly recovered (reported) firms on each day 0..days."""
    y0 = [N - initial_infected, initial_infected, 0]
    t = np.linspace(0, days, days + 1)
    solution = odeint(sir_model, y0, t, args=(N, beta, gamma))
    return np.round(np.diff(solution[:, 2], prepend=0)).astype(int)


def simulate_losses(data, N, beta_distr, gamma, days, sigma=SIGMA, i0=None):
    """Simulate IBNR claims for one resilience level (mirrors notebook 03-1).

    ``i0`` overrides the number of initially infected firms per incident cluster:
    by default (``None``) the empirical cluster size is used, but passing a fixed
    value sweeps i0 (e.g. {1, 3, 5, 7, 10}) as in the §5.3 / Appendix B analysis.
    """
    grouped = data.groupby("RELATED_ID").agg(
        count=("CPI_adjusted_loss", "count"),
        avg_amount=("CPI_adjusted_loss", "mean"),
        loss_date=("ACCIDENT_DATE", "max"),
    ).reset_index()

    results = []
    for _, row in grouped.iterrows():
        initial_infected = row["count"] if i0 is None else i0   # i0 (empirical, Table 3, or override)
        avg_amount = row["avg_amount"]             # phi0 (empirical, Table 3)
        loss_date = pd.to_datetime(row["loss_date"])

        beta = np.round(np.random.choice(beta_distr), 5)
        gamma_i = np.random.uniform(0.95 * gamma, 1.05 * gamma)

        recovered_ts = recovered_per_day(N, beta, gamma_i, int(initial_infected), days)
        for day, new_recovered in enumerate(recovered_ts):
            if new_recovered <= 0:
                continue
            new_date = loss_date + pd.Timedelta(days=day)
            phi = (np.random.lognormal(mean=np.log(avg_amount), sigma=sigma, size=new_recovered)
                   * np.exp(beta / gamma_i))
            for value in phi:
                results.append([row["RELATED_ID"], new_date, value, gamma, 1])

    return pd.DataFrame(results, columns=["RELATED_ID", "ACCIDENT_DATE",
                                          "CPI_adjusted_loss", "gamma", "ibnr"])


def run(sector, T, gamma_values=GAMMA_VALUES, seed=0, N=N, sigma=SIGMA, i0=None):
    """Build and save the IBNR table for one sector and reporting horizon T.

    The random seed is re-applied for every (sector, T), so each IBNR table is
    fully reproducible and independent of the order in which tables are built.

    N (network size), sigma (lognormal volatility of indirect costs), i0
    (initially infected firms, empirical when None) and gamma_values (the
    resilience levels stacked into the table) all default to the manuscript
    settings but can be overridden to re-simulate the IBNR table.
    """
    incidents = os.path.join(processed_dir(), f"ransomware_{sector}.csv")
    beta_path = os.path.join(calibration_dir(), f"beta_{sector}.npy")

    data = pd.read_csv(incidents)
    data["gamma"] = -1
    data["ibnr"] = 0
    data = data[["RELATED_ID", "ACCIDENT_DATE", "CPI_adjusted_loss", "gamma", "ibnr"]].copy()

    np.random.seed(seed)
    random.seed(seed)
    beta_distr = np.load(beta_path)
    beta_distr.sort()

    simulated = pd.concat(
        [simulate_losses(data, N, beta_distr, g, T, sigma=sigma, i0=i0) for g in gamma_values],
        ignore_index=True,
    )
    final = pd.concat([data, simulated], axis=0, ignore_index=True)
    final.columns = ["related_id", "accident_date", "CPI_adjusted_loss", "gamma", "ibnr"]
    final["accident_date"] = pd.to_datetime(final["accident_date"])
    final["accident_year"] = final["accident_date"].dt.year

    out = os.path.join(ibnr_dir(), f"ibnr_{sector}_{T}d.csv")
    final.to_csv(out, index=False)
    print(f"[ibnr] {sector} T={T}: {len(simulated)} simulated claims -> {os.path.basename(out)}")
    return out


def run_many(sector, T_values, gamma_values=GAMMA_VALUES, seed=0, N=N, sigma=SIGMA, i0=None):
    """Build IBNR tables for several reporting horizons T."""
    return [run(sector, T, gamma_values=gamma_values, seed=seed, N=N, sigma=sigma, i0=i0)
            for T in T_values]


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Simulate IBNR losses (Section 3.4 / 5.2).")
    p.add_argument("--sector", required=True, choices=["finance", "information", "manufacturing"])
    p.add_argument("--T", type=int, nargs="+", required=True,
                   help="reporting period(s) in days, e.g. --T 2 5 10")
    p.add_argument("--N", type=int, default=N, help="network size (default: 1000)")
    p.add_argument("--sigma", type=float, default=SIGMA,
                   help="lognormal volatility of indirect costs (default: 1.0)")
    p.add_argument("--i0", type=int, default=None,
                   help="initially infected firms per cluster; default empirical count "
                        "(sweep e.g. --i0 5)")
    p.add_argument("--gamma-values", type=float, nargs="+", default=list(GAMMA_VALUES),
                   help="resilience levels to stack into the table (default: 0.1 ... 0.9)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    run_many(args.sector, args.T, gamma_values=tuple(args.gamma_values), seed=args.seed,
             N=args.N, sigma=args.sigma, i0=args.i0)


if __name__ == "__main__":
    main()
