"""
ibnr.py -- Stage 3: simulate Incurred-But-Not-Reported (IBNR) losses.

For a given reporting horizon ``T`` the contagion is propagated through the
network for each related-incident cluster in the ransomware sample, using the
calibrated beta distribution. Every firm that recovers (i.e. recognises and
reports its loss) on a given day generates a lognormal indirect cost, producing
a per-claim IBNR table stacked across the nine operational-resilience levels
gamma = 0.1, ..., 0.9. The observed losses are retained with the tag
``gamma = -1``.

Inputs : data/{sector}_ransomware.csv, data/infection_rates_on_{sector}.npy
Output : data/{sector}_ibnr_{T}days.csv

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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

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


def simulate_losses(data, N, beta_distr, gamma, days, sigma=SIGMA):
    """Simulate IBNR claims for one resilience level (mirrors notebook 03-1)."""
    grouped = data.groupby("RELATED_ID").agg(
        count=("CPI_adjusted_loss", "count"),
        avg_amount=("CPI_adjusted_loss", "mean"),
        loss_date=("ACCIDENT_DATE", "max"),
    ).reset_index()

    results = []
    for _, row in grouped.iterrows():
        initial_infected = row["count"]            # i0 (empirical, Table 3)
        avg_amount = row["avg_amount"]             # phi0 (empirical, Table 3)
        loss_date = pd.to_datetime(row["loss_date"])

        beta = np.round(np.random.choice(beta_distr), 5)
        gamma_i = np.random.uniform(0.95 * gamma, 1.05 * gamma)

        recovered_ts = recovered_per_day(N, beta, gamma_i, initial_infected, days)
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


def run(sector, T, gamma_values=GAMMA_VALUES, seed=0, data_dir=DATA_DIR):
    """Build and save the IBNR table for one sector and reporting horizon T."""
    incidents = os.path.join(data_dir, f"{sector}_ransomware.csv")
    beta_path = os.path.join(data_dir, f"infection_rates_on_{sector}.npy")

    data = pd.read_csv(incidents)
    data["gamma"] = -1
    data["ibnr"] = 0
    data = data[["RELATED_ID", "ACCIDENT_DATE", "CPI_adjusted_loss", "gamma", "ibnr"]].copy()

    np.random.seed(seed)
    random.seed(seed)
    beta_distr = np.load(beta_path)
    beta_distr.sort()

    simulated = pd.concat(
        [simulate_losses(data, N, beta_distr, g, T) for g in gamma_values],
        ignore_index=True,
    )
    final = pd.concat([data, simulated], axis=0, ignore_index=True)
    final.columns = ["related_id", "accident_date", "CPI_adjusted_loss", "gamma", "ibnr"]
    final["accident_date"] = pd.to_datetime(final["accident_date"])
    final["accident_year"] = final["accident_date"].dt.year

    out = os.path.join(data_dir, f"{sector}_ibnr_{T}days.csv")
    final.to_csv(out, index=False)
    print(f"[ibnr] {sector} T={T}: {len(simulated)} simulated claims -> {os.path.basename(out)}")
    return out


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Simulate IBNR losses (Section 3.4 / 5.2).")
    p.add_argument("--sector", required=True, choices=["finance", "information", "manufacturing"])
    p.add_argument("--T", type=int, required=True, help="reporting period in days")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=DATA_DIR)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    run(args.sector, args.T, seed=args.seed, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
