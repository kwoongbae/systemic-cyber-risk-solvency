"""
scr.py -- Stage 4: Solvency Capital Requirement via the Loss Distribution Approach.

From the IBNR table produced by ``ibnr.py`` this builds, for one scenario
(sector, reporting period T, operational resilience gamma):

    * the empirical monthly claim-*frequency* distribution, and
    * the per-claim *severity* distribution, capped at the coverage limit l,

then runs a seeded Loss-Distribution-Approach (LDA) Monte-Carlo over the
insurer's portfolio and returns the Solvency II premium-risk SCR

    SCR = 3 * 0.14 * E[aggregate loss] / b           ($ million).

Input  : output/ibnr/ibnr_{sector}_{T}d.csv
Output : output/scr/scr_{sector}.csv (gamma x T sensitivity table, incl. No-IBNR row);
         individual SCR values are printed and returned when imported.

Aligns with manuscript Section 3.5 / 5.3 (Solvency II SCR; Table 5).

Run:
    python scr.py --sector finance --T 10 --gamma 0.1
    python scr.py --sector finance --T 2 5 10 --gamma 0.1 0.5 0.9   # sensitivity table
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from preprocess import ibnr_dir, scr_dir

PORTFOLIO_SIZE = 150          # n  : number of policies (Table C1)
COVERAGE_LIMIT = 50.0         # l  : coverage limit per policy, $M (Table C1)
# b : insurer baseline loss ratio = 10-yr (2014-2023) US P&C average.
LOSS_RATIO = (0.762 + 0.764 + 0.725 + 0.702 + 0.710
              + 0.714 + 0.762 + 0.722 + 0.693 + 0.690) / 10
VAR_MULTIPLIER = 3.0          # 99.5% VaR factor (Eling & Schnell, 2020)
VOL_FACTOR = 0.14             # aggregate volatility factor s


def ibnr_path(sector, T):
    return os.path.join(ibnr_dir(), f"ibnr_{sector}_{T}d.csv")


def _load_ibnr(sector, T):
    path = ibnr_path(sector, T)
    if not os.path.exists(path):
        raise FileNotFoundError(f"IBNR table not found: {path}\nRun ibnr.py first.")
    return pd.read_csv(path)


def _lda_scr(data, portfolio_size, limit, loss_ratio, var_multiplier, vol_factor, seed):
    """Seeded LDA Monte-Carlo over the portfolio for one filtered claims table."""
    data = data.copy()
    data["accident_date"] = pd.to_datetime(data["accident_date"])

    # ---- monthly frequency distribution (months with no claim count as zero) ----
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


def compute_scr(sector, T, gamma, portfolio_size=PORTFOLIO_SIZE, limit=COVERAGE_LIMIT,
                loss_ratio=LOSS_RATIO, var_multiplier=VAR_MULTIPLIER, vol_factor=VOL_FACTOR,
                seed=0):
    """Return the with-IBNR SCR ($M) for one (sector, T, gamma) scenario."""
    systemic = _load_ibnr(sector, T)
    data = systemic[(systemic["gamma"] == -1) | (systemic["gamma"] == gamma)]
    return _lda_scr(data, portfolio_size, limit, loss_ratio, var_multiplier, vol_factor, seed)


def compute_scr_no_ibnr(sector, T, portfolio_size=PORTFOLIO_SIZE, limit=COVERAGE_LIMIT,
                        loss_ratio=LOSS_RATIO, var_multiplier=VAR_MULTIPLIER, vol_factor=VOL_FACTOR,
                        seed=0):
    """Return the No-IBNR baseline SCR ($M): observed losses only (gamma == -1).

    Uses no beta and no SIR simulation -- only the text-mined observed claims --
    so with a fixed seed it is fully deterministic and reproducible. It does not
    depend on T (the gamma==-1 rows are identical across T tables).
    """
    systemic = _load_ibnr(sector, T)
    data = systemic[systemic["gamma"] == -1]
    return _lda_scr(data, portfolio_size, limit, loss_ratio, var_multiplier, vol_factor, seed)


def build_table(sector, T_values, gamma_values, seed=0, save=True,
                portfolio_size=PORTFOLIO_SIZE, limit=COVERAGE_LIMIT, loss_ratio=LOSS_RATIO,
                var_multiplier=VAR_MULTIPLIER, vol_factor=VOL_FACTOR):
    """Build the gamma x T SCR sensitivity table (plus a No-IBNR row) for one sector.

    Returns a DataFrame indexed by gamma (with a leading 'No-IBNR' row) and
    columns T; optionally writes it to output/scr/scr_{sector}.csv. The portfolio
    knobs (n, l, b) and SCR-formula factors default to the manuscript values but
    can be overridden to recompute the whole table under a different portfolio or
    Solvency II calibration.
    """
    T_values = list(T_values)
    gamma_values = list(gamma_values)
    kw = dict(portfolio_size=portfolio_size, limit=limit, loss_ratio=loss_ratio,
              var_multiplier=var_multiplier, vol_factor=vol_factor, seed=seed)
    rows = {}
    # No-IBNR baseline: constant across T (observed losses only).
    base = compute_scr_no_ibnr(sector, T_values[0], **kw)
    rows["No-IBNR"] = {T: base for T in T_values}
    for g in gamma_values:
        rows[g] = {T: compute_scr(sector, T, g, **kw) for T in T_values}

    table = pd.DataFrame(rows).T
    table.columns = T_values
    table.index.name = "gamma"
    if save:
        out = os.path.join(scr_dir(), f"scr_{sector}.csv")
        table.to_csv(out)
        print(f"[scr] {sector}: SCR sensitivity table ({len(gamma_values)} gamma x {len(T_values)} T) "
              f"-> {os.path.basename(out)}")
    return table


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Compute the Solvency II SCR (Section 3.5 / 5.3).")
    p.add_argument("--sector", required=True, choices=["finance", "information", "manufacturing"])
    p.add_argument("--T", type=int, nargs="+", required=True,
                   help="reporting period(s) in days, e.g. --T 2 5 10")
    p.add_argument("--gamma", type=float, nargs="+", required=True,
                   help="operational resilience value(s), e.g. --gamma 0.1 0.5 0.9")
    p.add_argument("--portfolio-size", type=int, default=PORTFOLIO_SIZE,
                   help="n: number of policies (default: 150)")
    p.add_argument("--limit", type=float, default=COVERAGE_LIMIT,
                   help="l: coverage limit per policy, $M (default: 50)")
    p.add_argument("--loss-ratio", type=float, default=LOSS_RATIO,
                   help="b: insurer baseline loss ratio (default: 10-yr US P&C avg)")
    p.add_argument("--var-multiplier", type=float, default=VAR_MULTIPLIER,
                   help="99.5%% VaR factor (default: 3.0)")
    p.add_argument("--vol-factor", type=float, default=VOL_FACTOR,
                   help="aggregate volatility factor s (default: 0.14)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    kw = dict(portfolio_size=args.portfolio_size, limit=args.limit, loss_ratio=args.loss_ratio,
              var_multiplier=args.var_multiplier, vol_factor=args.vol_factor, seed=args.seed)
    # Single cell: print the one value. Multiple T or gamma: build the table.
    if len(args.T) == 1 and len(args.gamma) == 1:
        value = compute_scr(args.sector, args.T[0], args.gamma[0], **kw)
        base = compute_scr_no_ibnr(args.sector, args.T[0], **kw)
        print(f"[scr] sector={args.sector}  T={args.T[0]}  gamma={args.gamma[0]}  "
              f"->  SCR = ${value} million   (No-IBNR baseline = ${base} million)")
        return value
    table = build_table(args.sector, args.T, args.gamma, **kw)
    print(f"\n=== SCR ($M)  |  sector={args.sector} ===")
    print(table.to_string())
    return table


if __name__ == "__main__":
    main()
