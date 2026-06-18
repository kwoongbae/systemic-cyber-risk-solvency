# Systemic cyber risks and insurance regulatory capital

A pipeline that estimates the Solvency II **Solvency Capital Requirement (SCR)**
for a cyber-insurance portfolio exposed to systemic cyber risk. Starting from two
base datasets, it extracts ransomware incidents, calibrates the contagion
intensity, simulates Incurred-But-Not-Reported (IBNR) losses through an SIR
epidemic model, and aggregates them into the SCR.

```bash
python main.py --sector finance --T 10 --gamma 0.1
# SCR  |  sector=finance  T=10  gamma=0.1  ->  $124.8 million
```

---

## Pipeline

`main.py` (at the repo root) orchestrates four stage scripts in `scripts/`, each
reading from / writing to `data/`:

```
data/advisen.csv  +  data/sas.csv          (base inputs)
        │  preprocess.py
data/{sector}_ransomware.csv               ransomware incident sample
        │  calibration.py
data/infection_rates_on_{sector}.npy       calibrated beta distribution
        │  ibnr.py
data/{sector}_ibnr_{T}days.csv             per-claim IBNR table
        │  scr.py
SCR  ($ million)
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

**Key inputs** (deterministic) and **calibration choices** (stochastic):


| Symbol | Meaning                                                  | Value                                 |
| ------ | -------------------------------------------------------- | ------------------------------------- |
| `N`    | network size                                             | 1,000                                 |
| `T`    | reporting period (days)                                  | *sensitivity variable*                |
| `n`    | portfolio size (policies)                                | 150                                   |
| `l`    | coverage limit per policy                                | $50M                                  |
| `b`    | insurer baseline loss ratio                              | 0.7244 (10-yr US P&C avg)             |
| `i0`   | initially infected firms                                 | *sensitivity variable*                |
| `ϕ0`   | direct cost to first-hit firm                            | finance $56M, information $308M       |
| `γ`    | operational resilience, firm-level ~ `U(0.95γ̄, 1.05γ̄)` | *sensitivity variable*                |
| `β`    | contagion intensity                                      | calibrated to Welburn & Strong (2022) |
| `ϕf`   | indirect cost per supply-chain firm                      | `Lognormal(ln(ϕ0·β/γ), 1)`            |


### Sensitivity analysis

Three inputs are varied (the rest held at the values above):


| Variable                      | Range                                                                                        | Where            |
| ----------------------------- | -------------------------------------------------------------------------------------------- | ---------------- |
| reporting period `T`          | 2, 5, 10 days                                                                                | Table 5          |
| operational resilience `γ̄`   | 0.1, 0.5, 0.9                                                                                | Table 5          |
| initially infected firms `i0` | empirical count per event; also swept over {1, 3, 5, 7, 10}, and grouped as 1–5 / 6–10 / >10 | Appendix B, §5.3 |


`i0` is the number of firms sharing a common risk driver in each event. The
headline Table 5 uses the empirical `i0`; varying it (more firms hit
simultaneously = common-cause failure) sharply amplifies systemic loss.

---

## Usage

```bash
uv sync                   # or: pip install numpy pandas scipy
```

Run a scenario with `main.py`:

```bash
python main.py --sector finance     --T 10 --gamma 0.1   # -> $124.8M
python main.py --sector information --T 5  --gamma 0.9   # -> $6.30M
```

`T ∈ {1,…,10}`, `γ ∈ {0.1,…,0.9}`, `sector ∈ {finance, information}`.

By default `main.py` reads the IBNR table already in `data/` and goes straight
to the SCR step. Passing `--force` instead rebuilds the whole chain
(`preprocess` → `calibration` → `ibnr`) from `advisen.csv` + `sas.csv`.

---

## Data

All inputs live in `data/`:


| File                             | Contents                                         |
| -------------------------------- | ------------------------------------------------ |
| `data/advisen.csv`               | Advisen cyber-loss records (with accident dates) |
| `data/sas.csv`                   | SAS OpRisk records (with settlement dates)       |
| `data/{sector}_ibnr_{T}days.csv` | per-claim IBNR tables consumed by `scr.py`       |


The two base datasets, `advisen.csv` and `sas.csv`, are **not uploaded** to this
repository: they originate from the proprietary Advisen and SAS OpRisk databases
and cannot be redistributed for security and licensing reasons. Place your own
copies in `data/` to run the `--force` pipeline from scratch.