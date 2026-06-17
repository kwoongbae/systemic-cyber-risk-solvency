"""
scr.py -- Stage 4: Solvency Capital Requirement via the Loss Distribution Approach.

From the IBNR table produced by ``ibnr.py`` this builds, for one scenario
(sector, reporting period T, operational resilience gamma):

    * the empirical monthly claim-*frequency* distribution, and
    * the per-claim *severity* distribution, capped at the coverage limit l,

then runs a seeded Loss-Distribution-Approach (LDA) Monte-Carlo over the
insurer's portfolio and returns the Solvency II premium-risk SCR

    SCR = 3 * 0.14 * E[aggregate loss] / b           ($ million).

Input  : data/{sector}_ibnr_{T}days.csv
Output : the SCR value (printed; returned when imported)

Aligns with manuscript Section 3.5 / 5.3 (Solvency II SCR; Table 5).

Run:
    python scr.py --sector finance --T 10 --gamma 0.1
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PORTFOLIO_SIZE = 150          # n  : number of policies (Table C1)
COVERAGE_LIMIT = 50.0         # l  : coverage limit per policy, $M (Table C1)
# b : insurer baseline loss ratio = 10-yr (2014-2023) US P&C average.
LOSS_RATIO = (0.762 + 0.764 + 0.725 + 0.702 + 0.710
              + 0.714 + 0.762 + 0.722 + 0.693 + 0.690) / 10
VAR_MULTIPLIER = 3.0          # 99.5% VaR factor (Eling & Schnell, 2020)
VOL_FACTOR = 0.14             # aggregate volatility factor s


def ibnr_path(sector, T, data_dir=DATA_DIR):
    return os.path.join(data_dir, f"{sector}_ibnr_{T}days.csv")


def compute_scr(sector, T, gamma, portfolio_size=PORTFOLIO_SIZE, limit=COVERAGE_LIMIT,
                loss_ratio=LOSS_RATIO, var_multiplier=VAR_MULTIPLIER, vol_factor=VOL_FACTOR,
                seed=0, data_dir=DATA_DIR):
    """Return the SCR (in $ million) for one scenario."""
    path = ibnr_path(sector, T, data_dir=data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"IBNR table not found: {path}\nRun ibnr.py first.")

    systemic = pd.read_csv(path)
    data = systemic[(systemic["gamma"] == -1) | (systemic["gamma"] == gamma)].copy()
    data["accident_date"] = pd.to_datetime(data["accident_date"])

    # ---- monthly frequency distribution ----
    all_months = pd.date_range(
        start=data["accident_date"].min(),
        end=(data["accident_date"].max() + pd.offsets.MonthEnd()),
        freq="ME",
    ).to_period("M")
    monthly = data.groupby(data["accident_date"].dt.to_period("M")).size().reset_index()
    monthly.columns = ["Month", "Frequency"]
    full_monthly = pd.DataFrame({"Month": all_months}).merge(monthly, on="Month", how="left").fillna(0)
    frequencies = full_monthly["Frequency"].astype(int)

    # ---- severity distribution, capped at the coverage limit l ----
    severities = np.where(data["CPI_adjusted_loss"] > limit, limit, data["CPI_adjusted_loss"])

    # ---- seeded LDA Monte-Carlo over the portfolio ----
    np.random.seed(seed)
    aggregate_losses = []
    for _ in range(portfolio_size):
        num_events = np.random.choice(frequencies, size=1)[0]
        sampled = np.random.choice(severities, size=num_events, replace=True)
        aggregate_losses.append(sampled.sum())

    scr = var_multiplier * vol_factor * np.mean(aggregate_losses) / loss_ratio
    return round(scr, 2)


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Compute the Solvency II SCR (Section 3.5 / 5.3).")
    p.add_argument("--sector", required=True, choices=["finance", "information", "manufacturing"])
    p.add_argument("--T", type=int, required=True, help="reporting period in days")
    p.add_argument("--gamma", type=float, required=True, help="operational resilience, e.g. 0.1/0.5/0.9")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=DATA_DIR)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    value = compute_scr(args.sector, args.T, args.gamma, seed=args.seed, data_dir=args.data_dir)
    print(f"[scr] sector={args.sector}  T={args.T}  gamma={args.gamma}  ->  SCR = ${value} million")
    return value


if __name__ == "__main__":
    main()
