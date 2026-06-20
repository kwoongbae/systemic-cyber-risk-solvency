"""
extract.py -- Stage 1: build the ransomware incident sample.

Merges the two base datasets (``data/advisen.csv`` and ``data/sas.csv``) and
applies the text-mining procedure of Florackis et al. (2023) -- a curated
keyword/phrase lexicon with inclusion and exclusion rules -- to isolate
ransomware-type cyber incidents at large enterprises (>= 250 employees) that
occurred from 2000 onward. Losses are inflation-adjusted to 2024 USD with the
US CPI and the cleaned sample is written, per sector, to
``data/generated/ransomware_{sector}.csv``.

Aligns with manuscript Section 4 (Data) and produces the empirical sample
summarised in Table 3.

Inputs : data/raw/advisen.csv, data/raw/sas.csv
Output : data/processed/ransomware_{sector}.csv

Run:
    python extract.py                 # all sectors
    python extract.py --sectors finance information
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Project layout (repo-root relative, so scripts run from any directory):
#   data/raw/        advisen.csv, sas.csv          (proprietary, user-supplied)
#   data/processed/  ransomware_{sector}.csv       (text-mined sample)
#   output/calibration/  beta_{sector}.npy
#   output/ibnr/         ibnr_{sector}_{T}d.csv
#   output/scr/          scr_{sector}.csv
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
CALIB_DIR = os.path.join(ROOT, "output", "calibration")
IBNR_DIR = os.path.join(ROOT, "output", "ibnr")
SCR_DIR = os.path.join(ROOT, "output", "scr")

# kept for backwards-compatible call sites
DATA_DIR = os.path.join(ROOT, "data")


def _ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


def raw_dir():
    return _ensure(RAW_DIR)


def processed_dir():
    return _ensure(PROCESSED_DIR)


def calibration_dir():
    return _ensure(CALIB_DIR)


def ibnr_dir():
    return _ensure(IBNR_DIR)


def scr_dir():
    return _ensure(SCR_DIR)

# US CPI annual average (BLS), used to inflate historical losses to 2024 USD.
CPI = {
    2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9,
    2005: 195.3, 2006: 201.6, 2007: 207.3, 2008: 215.303, 2009: 214.537,
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736,
    2015: 237.017, 2016: 240.007, 2017: 245.12, 2018: 251.107, 2019: 255.657,
    2020: 258.811, 2021: 270.97, 2022: 292.655, 2023: 304.702, 2024: 313.689,
}
CPI_BASE_YEAR = 2024

# Map each NAIC sector label to the scenario name used downstream.
SECTOR_LABELS = {
    "finance": ("Finance and Insurance", "Financial Services"),
    "information": ("Information",),
    "manufacturing": ("Manufacturing",),
}

LARGE_FIRM_EMPLOYEES = 250          # "large enterprise" threshold (European Commission, 2003)
MIN_ADVISEN_LOSS = 100_000          # discard negligible Advisen losses (USD)
START_DATE = "2000-01-01"


# ---------------------------------------------------------------------------
# Florackis et al. (2023)-style text-mining classifier
# ---------------------------------------------------------------------------
def evaluate_description(sentence) -> int:
    """Return 1 if the incident description denotes a cyber/ransomware event."""
    if not isinstance(sentence, str):
        return 0
    sentence = sentence.lower()
    cyber_kw = ["cyber", "networks", "systems", "products", "services", "datacenter", "infrastructure"]
    cyber_prefix = r"\bcyber[a-z]*\b"

    def has(words):
        return any(re.search(r"\b" + re.escape(w) + r"\b", sentence) for w in words)

    if "attack" in sentence:
        if has(cyber_kw) or re.search(cyber_prefix, sentence):
            if not has(["terror", "war", "contraband", "bombs"]):
                return 1
        return 0
    if "threat" in sentence:
        if has(cyber_kw) or re.search(cyber_prefix, sentence):
            if not has(["terror", "simulator", "disease", "legal action", "competitive",
                        "competitors", "substitute", "patent", "nuclear", "life", "threatem"]):
                return 1
        return 0
    if "breaches" in sentence:
        if not has(["fiduciary duty", "fiduciary duties", "covenant", "credit", "agreement",
                    "warranty", "warranties", "obligations", "regulations", "contract", "resolution"]):
            return 1
        return 0
    if has(["hacker", "hacking", "social", "engineering", "denial-of-service", "phishing",
            "cyberattack", "cyberattacks", "cyber risk", "cyber security", "cybersecurity",
            "cyber intrusions", "unauthorized access", "breach in security", "security breach"]):
        return 1
    return 0


def _build_combined(advisen_path, sas_path):
    """Read, classify, filter and merge the two source datasets."""
    sas = (pd.read_csv(sas_path, encoding="cp949", low_memory=False)
           .drop_duplicates(subset="Reference ID Code", keep="first"))
    advisen = (pd.read_csv(advisen_path, encoding="cp949", low_memory=False)
               .drop_duplicates(subset="MSCAD_ID", keep="first"))
    advisen = advisen[advisen["CASESTATUS"] == "Estimate"]

    sas_col = ["Reference ID Code", "Firm Name", "Industry Sector Name", "Description of Event",
               "Loss Amount ($M)", "Month & Year of Settlement", "Multiple Firms Impacted Code",
               "Revenue ($M)", "# of Employees"]
    advisen_col = ["MSCAD_ID", "COMPANY_NAME_x", "NAIC_SECTOR_DESC", "CASE_DESCRIPTION",
                   "TOTAL_AMOUNT", "ACCIDENT_DATE", "RELATED_ID", "REVENUES", "EMPLOYEES"]

    sas = sas[sas_col].dropna()
    advisen = advisen[advisen_col].dropna()

    # text-mining classification
    sas["Cybersecurity"] = sas["Description of Event"].apply(evaluate_description)
    sas = sas[sas["Cybersecurity"] == 1]
    advisen["Cybersecurity"] = advisen["CASE_DESCRIPTION"].apply(evaluate_description)
    advisen = advisen[advisen["Cybersecurity"] == 1]

    # SAS: from 2000, large firms; tag source
    sas = sas.copy()
    sas["Month & Year of Settlement"] = pd.to_datetime(sas["Month & Year of Settlement"])
    sas = sas[sas["Month & Year of Settlement"] >= pd.to_datetime(START_DATE)]
    sas = sas[sas["# of Employees"] >= LARGE_FIRM_EMPLOYEES]
    sas["sas"] = 1

    # Advisen: from 2000, non-negligible loss, large firms; convert to $M; tag source
    advisen = advisen.copy()
    advisen["ACCIDENT_DATE"] = pd.to_datetime(advisen["ACCIDENT_DATE"])
    advisen = advisen[advisen["ACCIDENT_DATE"] >= pd.to_datetime(START_DATE)]
    advisen = advisen[advisen["TOTAL_AMOUNT"] > MIN_ADVISEN_LOSS]
    advisen = advisen[advisen["EMPLOYEES"] >= LARGE_FIRM_EMPLOYEES]
    advisen["TOTAL_AMOUNT"] /= 1_000_000
    advisen["sas"] = 0

    # align SAS columns onto the Advisen schema and stack
    sas.columns = advisen.columns
    combined = pd.concat([sas, advisen], ignore_index=True)

    # keep only related-incident clusters whose accident dates all coincide
    same_date = (combined.groupby("RELATED_ID")["ACCIDENT_DATE"]
                 .nunique().eq(1))
    keep_ids = same_date[same_date].index
    combined = combined[combined["RELATED_ID"].isin(keep_ids)].copy()

    # inflation-adjust to 2024 USD
    combined["ACCIDENT_YEAR"] = pd.to_datetime(combined["ACCIDENT_DATE"]).dt.year
    coeff = combined["ACCIDENT_YEAR"].map(lambda y: CPI[CPI_BASE_YEAR] / CPI.get(int(y), CPI[CPI_BASE_YEAR]))
    combined["CPI_adjusted_loss"] = combined["TOTAL_AMOUNT"] * coeff
    return combined


def run(sectors=("finance", "information", "manufacturing"), data_dir=None):
    """Build and save the ransomware sample for each requested sector."""
    advisen_path = os.path.join(raw_dir(), "advisen.csv")
    sas_path = os.path.join(raw_dir(), "sas.csv")
    combined = _build_combined(advisen_path, sas_path)

    out_dir = processed_dir()
    outputs = {}
    for sector in sectors:
        labels = SECTOR_LABELS[sector]
        sub = combined[combined["NAIC_SECTOR_DESC"].isin(labels)].copy()
        out = os.path.join(out_dir, f"ransomware_{sector}.csv")
        sub.to_csv(out, index=False)
        outputs[sector] = (out, len(sub))
        print(f"[extract] {sector:12s}: {len(sub):4d} incidents -> {os.path.basename(out)}")
    return outputs


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Extract ransomware incidents at large firms (Section 4).")
    p.add_argument("--sectors", nargs="+", default=["finance", "information", "manufacturing"],
                   choices=list(SECTOR_LABELS))
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    run(sectors=tuple(args.sectors))


if __name__ == "__main__":
    main()
