# policy_stability_check.py
# -*- coding: utf-8 -*-
"""
Generate basic policy-stability diagnostics from a V3 case-level table.

Usage:
    python policy_stability_check.py ^
      --input raw_granular_fair_skills_java_sampled_v3/00_case_level.csv ^
      --out policy_stability.csv

This script does not require old MAIN. It summarizes each intervention condition.
Later, you can compare these values against old MAIN or across INT1/INT2/INT3.
"""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path


SCORE_COLS = [
    "composite_score_mean",
    "stage1_composite_score_mean",
    "stage2_composite_score_mean",
    "score_Competency_mean",
    "score_Potential_mean",
    "score_Fit_mean",
    "score_Value_mean",
    "score_Risk_mean",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df["valid_decision"] = df["final_decision"].isin(["HIRE", "REJECT"])
    df["hire"] = (df["final_decision"] == "HIRE").astype(float)

    group_cols = [c for c in ["model", "occupation", "condition"] if c in df.columns]
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        valid = sub[sub["valid_decision"]].copy()
        row["n_cases"] = len(sub)
        row["n_valid"] = len(valid)
        row["valid_rate"] = len(valid) / len(sub) if len(sub) else None
        row["overall_hire_rate"] = valid["hire"].mean() if len(valid) else None
        for c in SCORE_COLS:
            if c in valid.columns:
                row[f"{c}_mean"] = pd.to_numeric(valid[c], errors="coerce").mean()
                row[f"{c}_std"] = pd.to_numeric(valid[c], errors="coerce").std()
                row[f"{c}_n"] = pd.to_numeric(valid[c], errors="coerce").notna().sum()
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
