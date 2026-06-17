"""
main.py -- run the full pipeline end-to-end and print the SCR.

Chains the four stages for a single scenario:

    preprocess.py  ->  calibration.py  ->  ibnr.py  ->  scr.py
    (Section 4)        (Section 3.3)       (3.4/5.2)    (3.5/5.3, Table 5)

Each stage writes its output into ``data/``. By default a stage is skipped when
its output already exists, so re-runs are fast; pass --force to regenerate
everything from the two base files (data/advisen.csv, data/sas.csv).

Run:
    python main.py                                   # finance, T=10, gamma=0.1
    python main.py --sector information --T 5 --gamma 0.9
    python main.py --force                           # rebuild every stage
"""

from __future__ import annotations

import argparse
import os
import sys

# the pipeline stage modules live in src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import preprocess
import calibration
import ibnr
import scr

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _exists(name):
    return os.path.exists(os.path.join(DATA_DIR, name))


def run(sector="finance", T=10, gamma=0.1, num_simulations=calibration.DEFAULT_NUM_SIMULATIONS,
        seed=0, force=False):
    ibnr_csv = f"{sector}_ibnr_{T}days.csv"

    # Stages 1-3 (preprocess -> calibration -> ibnr) regenerate the IBNR table
    # from the base data. They run only when the IBNR table is missing or when
    # --force is given. When the published IBNR table already sits in data/
    # (the seed-free CSV used for the manuscript), we use it directly so the SCR
    # reproduces the paper's Table 5.
    if force or not _exists(ibnr_csv):
        # Stage 1 -- ransomware sample
        if force or not _exists(f"{sector}_ransomware.csv"):
            print("== Stage 1: preprocess ==")
            preprocess.run(sectors=(sector,), data_dir=DATA_DIR)
        else:
            print(f"== Stage 1: preprocess (skip, {sector}_ransomware.csv exists) ==")

        # Stage 2 -- beta calibration
        if force or not _exists(f"infection_rates_on_{sector}.npy"):
            print("== Stage 2: calibration ==")
            calibration.calibrate(sector, num_simulations=num_simulations, seed=seed, data_dir=DATA_DIR)
        else:
            print(f"== Stage 2: calibration (skip, infection_rates_on_{sector}.npy exists) ==")

        # Stage 3 -- IBNR simulation
        print("== Stage 3: ibnr ==")
        ibnr.run(sector, T, seed=seed, data_dir=DATA_DIR)
    else:
        print(f"== Stages 1-3: using existing IBNR table data/{ibnr_csv} ==")

    # Stage 4 -- SCR
    print("== Stage 4: scr ==")
    value = scr.compute_scr(sector, T, gamma, seed=seed, data_dir=DATA_DIR)
    print(f"\nSCR  |  sector={sector}  T={T}  gamma={gamma}  ->  ${value} million")
    return value


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Run the systemic cyber-risk SCR pipeline end-to-end.")
    p.add_argument("--sector", default="finance", choices=["finance", "information", "manufacturing"])
    p.add_argument("--T", type=int, default=10, help="reporting period in days")
    p.add_argument("--gamma", type=float, default=0.1, help="operational resilience")
    p.add_argument("--num-simulations", type=int, default=calibration.DEFAULT_NUM_SIMULATIONS,
                   help="calibration draws for beta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="rebuild every stage even if outputs exist")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    run(sector=args.sector, T=args.T, gamma=args.gamma,
        num_simulations=args.num_simulations, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
