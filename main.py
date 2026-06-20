"""
main.py -- run the full pipeline end-to-end and print the SCR sensitivity table.

Chains the four stages:

    extract.py  ->  calibration.py  ->  ibnr.py  ->  scr.py
    (Section 4)     (Section 5.1)       (3.4/5.2)    (3.5/5.3, Table 5)

Project layout:
    data/raw/            advisen.csv, sas.csv          (proprietary, user-supplied)
    data/processed/      ransomware_{sector}.csv
    output/calibration/  beta_{sector}.npy
    output/ibnr/         ibnr_{sector}_{T}d.csv
    output/scr/          scr_{sector}.csv              (gamma x T table, incl. No-IBNR)

Every stochastic stage is seeded (default seed=0), so the whole pipeline is
reproducible: the same inputs and seed always give the same SCR table.

A stage is skipped when its output already exists; pass --force to rebuild from
the raw data. T and gamma accept several values to fill the sensitivity table.

Run:
    python main.py                                           # finance, full table
    python main.py --sector information --T 2 5 10 --gamma 0.1 0.5 0.9
    python main.py --sector finance --force                  # rebuild every stage
"""

from __future__ import annotations

import argparse
import os
import sys

# the pipeline stage modules live in scripts/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import extract
import calibration
import ibnr
import scr

DEFAULT_T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_GAMMA = [0.1, 0.5, 0.9]


def run(sector="finance", T_values=DEFAULT_T, gamma_values=DEFAULT_GAMMA,
        num_simulations=calibration.DEFAULT_NUM_SIMULATIONS, seed=0, force=False,
        # calibration knobs (Stage 2)
        N=calibration.N, gamma_bar=calibration.GAMMA_BAR, calib_T=calibration.T,
        # IBNR knobs (Stage 3)
        sigma=ibnr.SIGMA, i0=None,
        # SCR / portfolio knobs (Stage 4)
        portfolio_size=scr.PORTFOLIO_SIZE, limit=scr.COVERAGE_LIMIT, loss_ratio=scr.LOSS_RATIO,
        var_multiplier=scr.VAR_MULTIPLIER, vol_factor=scr.VOL_FACTOR):
    T_values = list(T_values)

    # Overriding an upstream knob invalidates the cached artefacts of that stage,
    # so rebuild the affected stage(s) instead of silently reusing stale files.
    calib_overridden = (N != calibration.N or gamma_bar != calibration.GAMMA_BAR
                        or calib_T != calibration.T)
    ibnr_overridden = (N != ibnr.N or sigma != ibnr.SIGMA or i0 is not None)
    force_calib = force or calib_overridden
    force_ibnr = force or calib_overridden or ibnr_overridden

    # Stage 1 -- ransomware sample (data/processed/)
    ransomware = os.path.join(extract.processed_dir(), f"ransomware_{sector}.csv")
    if force or not os.path.exists(ransomware):
        print("== Stage 1: extract ==")
        extract.run(sectors=(sector,))
    else:
        print(f"== Stage 1: extract (skip, {os.path.basename(ransomware)} exists) ==")

    # Stage 2 -- beta calibration (output/calibration/)
    beta = os.path.join(extract.calibration_dir(), f"beta_{sector}.npy")
    if force_calib or not os.path.exists(beta):
        print("== Stage 2: calibration ==")
        calibration.calibrate(sector, num_simulations=num_simulations, seed=seed,
                              N=N, gamma_bar=gamma_bar, T=calib_T)
    else:
        print(f"== Stage 2: calibration (skip, {os.path.basename(beta)} exists) ==")

    # Stage 3 -- IBNR simulation per T (output/ibnr/)
    print("== Stage 3: ibnr ==")
    for T in T_values:
        path = scr.ibnr_path(sector, T)
        if force_ibnr or not os.path.exists(path):
            ibnr.run(sector, T, seed=seed, N=N, sigma=sigma, i0=i0)
        else:
            print(f"== Stage 3: ibnr (skip, {os.path.basename(path)} exists) ==")

    # Stage 4 -- SCR sensitivity table (output/scr/)
    print("== Stage 4: scr ==")
    table = scr.build_table(sector, T_values, gamma_values, seed=seed,
                            portfolio_size=portfolio_size, limit=limit, loss_ratio=loss_ratio,
                            var_multiplier=var_multiplier, vol_factor=vol_factor)
    print(f"\n=== SCR ($M)  |  sector={sector} ===")
    print(table.to_string())
    return table


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Run the systemic cyber-risk SCR pipeline end-to-end.")
    p.add_argument("--sector", default="finance", choices=["finance", "information", "manufacturing"])
    p.add_argument("--T", type=int, nargs="+", default=DEFAULT_T,
                   help="reporting period(s) in days, e.g. --T 2 5 10")
    p.add_argument("--gamma", type=float, nargs="+", default=DEFAULT_GAMMA,
                   help="operational resilience value(s), e.g. --gamma 0.1 0.5 0.9")
    p.add_argument("--num-simulations", type=int, default=calibration.DEFAULT_NUM_SIMULATIONS,
                   help="calibration draws for beta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="rebuild every stage even if outputs exist")

    # --- key inputs / calibration choices (default to the manuscript values) ---
    # Overriding any of these automatically rebuilds the affected stage(s).
    g = p.add_argument_group("key inputs / calibration choices")
    g.add_argument("--N", type=int, default=calibration.N, help="network size (default: 1000)")
    g.add_argument("--gamma-bar", type=float, default=calibration.GAMMA_BAR,
                   help="baseline operational resilience for beta calibration (default: 0.1)")
    g.add_argument("--calib-T", type=int, default=calibration.T,
                   help="calibration horizon in days (default: 10)")
    g.add_argument("--sigma", type=float, default=ibnr.SIGMA,
                   help="lognormal volatility of indirect costs (default: 1.0)")
    g.add_argument("--i0", type=int, default=None,
                   help="initially infected firms per cluster; default empirical count")
    g.add_argument("--portfolio-size", type=int, default=scr.PORTFOLIO_SIZE,
                   help="n: number of policies (default: 150)")
    g.add_argument("--limit", type=float, default=scr.COVERAGE_LIMIT,
                   help="l: coverage limit per policy, $M (default: 50)")
    g.add_argument("--loss-ratio", type=float, default=scr.LOSS_RATIO,
                   help="b: insurer baseline loss ratio (default: 10-yr US P&C avg)")
    g.add_argument("--var-multiplier", type=float, default=scr.VAR_MULTIPLIER,
                   help="99.5%% VaR factor (default: 3.0)")
    g.add_argument("--vol-factor", type=float, default=scr.VOL_FACTOR,
                   help="aggregate volatility factor s (default: 0.14)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    run(sector=args.sector, T_values=args.T, gamma_values=args.gamma,
        num_simulations=args.num_simulations, seed=args.seed, force=args.force,
        N=args.N, gamma_bar=args.gamma_bar, calib_T=args.calib_T,
        sigma=args.sigma, i0=args.i0,
        portfolio_size=args.portfolio_size, limit=args.limit, loss_ratio=args.loss_ratio,
        var_multiplier=args.var_multiplier, vol_factor=args.vol_factor)


if __name__ == "__main__":
    main()
