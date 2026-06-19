# Systemic cyber risks and insurance regulatory capital

A four-stage, reproducible pipeline that quantifies how **systemic** cyber risk
propagates into an insurer's regulatory capital, and estimates the Solvency II
**Solvency Capital Requirement (SCR)** for a cyber-insurance portfolio under that
risk. From two proprietary loss datasets it (1) text-mines a sample of ransomware
incidents at large firms; (2) calibrates the contagion intensity β by
moment-matching the simulated systemic loss to the Welburn & Strong (2022)
benchmark; (3) propagates each incident through a homogeneous firm network with a
deterministic **SIR** epidemic model and draws lognormal indirect costs to build
a per-claim **Incurred-But-Not-Reported (IBNR)** table; and (4) runs a
Loss-Distribution-Approach (LDA) Monte-Carlo over the portfolio to return the
premium-risk SCR. The pipeline is fully seeded, so identical inputs and seed
reproduce the same SCR, and every input in the tables below can be overridden
from the command line to re-run the whole chain under a different assumption.

```bash
python main.py --sector finance --T 10 --gamma 0.1
# SCR  |  sector=finance  T=10  gamma=0.1  ->  $124.8 million
```

---

## Pipeline

`main.py` (at the repo root) orchestrates four stage scripts in `scripts/`. You
supply the two raw inputs in `data/raw/`; every generated artifact is written
under `data/processed/` and `output/`:

```
data/raw/advisen.csv  +  data/raw/sas.csv           (raw inputs, you provide)
        │  preprocess.py
data/processed/ransomware_{sector}.csv              ransomware incident sample
        │  calibration.py
output/calibration/beta_{sector}.npy                calibrated beta distribution
        │  ibnr.py
output/ibnr/ibnr_{sector}_{T}d.csv                  per-claim IBNR table
        │  scr.py
output/scr/scr_{sector}.csv  +  SCR ($ million)
```


| Script           | What it does                                                                                                                                                                                         | Section                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `preprocess.py`  | Merges Advisen + SAS and applies the Florackis et al. (2023) text-mining lexicon to isolate ransomware incidents at large firms (≥250 employees, from 2000), inflation-adjusting losses to 2024 USD. | §4 Data (Table 3)      |
| `calibration.py` | Calibrates the contagion intensity β so the simulated systemic loss matches the Welburn & Strong (2022) benchmark; repeated to obtain a β distribution.                                              | §3.4 §5.1 / Fig. 3     |
| `ibnr.py`        | Propagates the shock through the network with the SIR model and draws lognormal indirect costs to build the IBNR table across γ = 0.1…0.9.                                                           | §3.2–3.3 §5.2 / Fig. 5 |
| `scr.py`         | Builds the monthly frequency and capped severity distributions, runs an LDA Monte-Carlo, and returns the SCR.                                                                                        | §3.5 §5.3 / Table 5    |
| `main.py`        | Orchestrates the stages for one scenario and prints the SCR.                                                                                                                                         | Appendix C             |


---

## Inputs

Every input is split into **Key inputs** — deterministic structural and
portfolio parameters — and **Calibration choices** — the stochastic and
regulatory factors of the loss model. Each carries a CLI flag and *defaults to
the manuscript value*, so the headline results reproduce out of the box while
any input can be overridden to re-run the pipeline under a different assumption
(see [Varying the inputs](#varying-the-inputs)). The last column records whether
the manuscript varies the input in its sensitivity analysis, and if so how.

### Key inputs (deterministic)

| Symbol | Meaning                                  | Default                          | CLI flag           | Varied in sensitivity analysis?                                                                                  |
| ------ | ---------------------------------------- | -------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `T`    | reporting period (days)                  | swept                            | `--T`              | **Yes** — set to 2, 5, 10 days (Table 5).                                                                        |
| `i0`   | initially infected firms per cluster     | empirical count per event        | `--i0`             | **Yes** — swept over {1, 3, 5, 7, 10}, and grouped as 1–5 / 6–10 / >10 (§5.3, Appendix B); default is empirical. |
| `N`    | network size                             | 1,000                            | `--N`              | No — held at 1,000.                                                                                              |
| `n`    | portfolio size (policies)                | 150                              | `--portfolio-size` | No — held at 150.                                                                                                |
| `l`    | coverage limit per policy                | $50M                             | `--limit`          | No — held at $50M.                                                                                               |
| `b`    | insurer baseline loss ratio              | 0.7244 (10-yr US P&C avg)        | `--loss-ratio`     | No — held at the 10-yr average.                                                                                  |
| `ϕ0`   | direct cost to first-hit firm            | empirical (fin. $56M, info $308M)| from data / W&S    | No — taken from the data and the Welburn & Strong (2022) benchmark.                                              |

`i0` is the number of firms sharing a common risk driver in each event. The
headline Table 5 uses the empirical `i0`; raising it (more firms hit
simultaneously = common-cause failure) sharply amplifies systemic loss.

### Calibration choices (stochastic / regulatory)

| Symbol | Meaning                                                       | Default                          | CLI flag          | Varied in sensitivity analysis?                                              |
| ------ | ------------------------------------------------------------- | -------------------------------- | ----------------- | --------------------------------------------------------------------------- |
| `γ`    | firm operational resilience level, firm-level ~ `U(0.95γ, 1.05γ)` | swept                       | `--gamma`         | **Yes** — set to 0.1, 0.5, 0.9 (Table 5).                                   |
| `γ̄`   | baseline operational resilience used to calibrate `β`         | 0.1                              | `--gamma-bar`     | No — fixed at 0.1 for the β calibration (the swept resilience is `γ` above). |
| `β`    | contagion intensity                                           | calibrated to Welburn & Strong (2022) | re-derived¹  | No — re-calibrated, not set directly.                                       |
| `σ`    | lognormal volatility of indirect costs                        | 1.0                              | `--sigma`         | No — held at 1.0.                                                            |
| —      | calibration horizon (days) for `β`                            | 10                               | `--calib-T`       | No — held at 10 days.                                                       |
| `ϕf`   | indirect cost per affected firm                               | `Lognormal(ln ϕ0, σ) · exp(β/γ)` | derived           | No — derived from `ϕ0`, `σ`, `β`, `γ`.                                       |
| —      | 99.5% VaR multiplier                                          | 3.0                              | `--var-multiplier`| No — Solvency II calibration (Eling & Schnell, 2020).                       |
| `s`    | aggregate volatility factor                                   | 0.14                             | `--vol-factor`    | No — Solvency II premium-risk factor.                                       |

¹ `β` has no flag of its own — it is recalibrated automatically whenever a
calibration input it depends on (`--N`, `--gamma-bar`, `--calib-T`) changes.

---

## Usage

> **Before use.** The two base datasets, `advisen.csv` and `sas.csv`, are
> **not included** in this repository: they originate from the proprietary
> Advisen and SAS OpRisk databases and cannot be redistributed for security and
> licensing reasons. The repository ships **no data** — drop your own copies of
> `advisen.csv` and `sas.csv` into `data/raw/`, and the pipeline generates
> everything else (into `data/processed/` and `output/`) on the way to the SCR.

```bash
uv sync                   # or: pip install numpy pandas scipy
```

Run a scenario with `main.py`:

```bash
python main.py --sector finance     --T 10 --gamma 0.1   # -> $124.8M
python main.py --sector information --T 5  --gamma 0.9   # -> $6.30M
```

`T ∈ {1,…,10}`, `γ ∈ {0.1,…,0.9}`, `sector ∈ {finance, information}`.

On the first run `main.py` builds the whole chain
(`preprocess` → `calibration` → `ibnr` → `scr`) from `advisen.csv` + `sas.csv`.
Later runs reuse the IBNR table already in `output/ibnr/` and go straight to
the SCR step; pass `--force` to rebuild every stage from scratch.

### Varying the inputs

Beyond the swept sensitivity variables (`--T`, `--gamma`, `--i0`), **every** input
in the [Inputs](#inputs) tables is exposed on `main.py` as a flag and defaults to
its manuscript value, so you can re-run the full pipeline under any single change
(or a combination):

```bash
# larger portfolio, lower coverage limit (Key inputs n, l)
python main.py --sector finance --T 10 --gamma 0.1 --portfolio-size 300 --limit 25

# bigger network and a heavier indirect-cost tail (N, sigma)
python main.py --sector finance --T 10 --gamma 0.1 --N 2000 --sigma 1.5

# alternative loss ratio and Solvency II factors (b, VaR multiplier, s)
python main.py --sector finance --T 10 --gamma 0.1 \
    --loss-ratio 0.80 --var-multiplier 2.5 --vol-factor 0.12

# common-cause failure: fix i0 = 5 firms hit at once (overrides the empirical count)
python main.py --sector finance --T 10 --gamma 0.1 --i0 5
```

`main.py` knows which stage each flag belongs to and **rebuilds only the affected
stages**: the portfolio/SCR knobs (`--portfolio-size`, `--limit`, `--loss-ratio`,
`--var-multiplier`, `--vol-factor`) recompute just the SCR step; `--sigma` / `--i0`
re-simulate the IBNR table; and `--N`, `--gamma-bar` or `--calib-T` re-run the β
calibration first (since `β` depends on them) and then everything downstream. You
therefore do **not** need `--force` when overriding an input — stale artefacts of
the affected stages are rebuilt automatically. Run `python main.py --help` for the
full list. The same flags are available on the individual stage scripts below.

### Running each stage on its own

`main.py` calls the stage functions directly, but every stage script is also a
standalone CLI exposing the same input flags, so the chain can be reproduced one
step at a time. Each stage reads its predecessor's output and writes its own, so
run them in order (`preprocess` → `calibration` → `ibnr` → `scr`). When you
override an input by hand, remember to re-run every downstream stage too.

**Stage 1 — `preprocess.py`** (§4): build the ransomware incident sample from
`advisen.csv` + `sas.csv`.

```bash
python scripts/preprocess.py                                # all sectors
python scripts/preprocess.py --sectors finance information  # selected sectors
# reads data/raw/{advisen,sas}.csv
# -> data/processed/ransomware_{sector}.csv
```

**Stage 2 — `calibration.py`** (§3.4 / 5.1): calibrate the contagion intensity
β against the Welburn & Strong (2022) benchmark, repeatedly, to get its
distribution.

```bash
python scripts/calibration.py                                  # all sectors, 10,000 draws
python scripts/calibration.py --sectors finance --num-simulations 10000
python scripts/calibration.py --sectors finance --N 2000 --gamma-bar 0.2 --T 10
# --N INT          network size (default: 1000)
# --gamma-bar F    baseline operational resilience (default: 0.1)
# --T INT          calibration horizon in days (default: 10)
# --seed N         RNG seed (default: 0)
# -> output/calibration/beta_{sector}.npy
```

**Stage 3 — `ibnr.py`** (§3.4 / 5.2): propagate the shock through the SIR
network and draw lognormal indirect costs, for one sector and reporting horizon
`T`. The output stacks all nine resilience levels γ = 0.1…0.9.

```bash
python scripts/ibnr.py --sector finance --T 10     # required: --sector, --T
python scripts/ibnr.py --sector finance --T 10 --N 2000 --sigma 1.5 --i0 5
# --N INT            network size (default: 1000)
# --sigma F          lognormal volatility of indirect costs (default: 1.0)
# --i0 INT           initially infected firms per cluster (default: empirical count)
# --gamma-values ..  resilience levels to stack (default: 0.1 ... 0.9)
# --seed N           RNG seed (default: 0)
# -> output/ibnr/ibnr_{sector}_{T}d.csv
```

**Stage 4 — `scr.py`** (§3.5 / 5.3, Table 5): build the frequency/severity
distributions, run the LDA Monte-Carlo and return the SCR for one scenario.

```bash
python scripts/scr.py --sector finance --T 10 --gamma 0.1   # required: --sector, --T, --gamma
python scripts/scr.py --sector finance --T 10 --gamma 0.1 --portfolio-size 300 --limit 25
# --portfolio-size INT   n: number of policies (default: 150)
# --limit F              l: coverage limit per policy, $M (default: 50)
# --loss-ratio F         b: insurer baseline loss ratio (default: 10-yr US P&C avg)
# --var-multiplier F     99.5% VaR factor (default: 3.0)
# --vol-factor F         aggregate volatility factor s (default: 0.14)
# --seed N               RNG seed (default: 0)
# -> prints SCR ($M); writes output/scr/scr_{sector}.csv for multi-value tables
```

`sector ∈ {finance, information}`, `T ∈ {1,…,10}`, `γ ∈ {0.1,…,0.9}`.

---

## Data

You provide the two raw inputs in `data/raw/`; the pipeline writes every
generated artifact into `data/processed/` and `output/`:


| File                                          | Tier      | Contents                                         |
| --------------------------------------------- | --------- | ------------------------------------------------ |
| `data/raw/advisen.csv`                        | raw       | Advisen cyber-loss records (with accident dates) |
| `data/raw/sas.csv`                            | raw       | SAS OpRisk records (with settlement dates)       |
| `data/processed/ransomware_{sector}.csv`      | generated | text-mined ransomware incident sample            |
| `output/calibration/beta_{sector}.npy`        | generated | calibrated contagion-intensity (β) distribution  |
| `output/ibnr/ibnr_{sector}_{T}d.csv`          | generated | per-claim IBNR tables consumed by `scr.py`       |
| `output/scr/scr_{sector}.csv`                 | generated | γ × T SCR sensitivity table (incl. No-IBNR row)  |


Nothing under `data/` or `output/` is tracked by git. The two raw datasets,
`advisen.csv` and `sas.csv`, are **not uploaded** to this repository (see the
*Before use* note above); place your own copies in `data/raw/` and everything
under `data/processed/` and `output/` is rebuilt by the pipeline.