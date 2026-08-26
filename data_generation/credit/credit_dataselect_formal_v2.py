#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCOPED-Credit formal seed selection.

Input:
  Home Credit Default Risk application_train.csv

Output:
  credit seed JSONL with fixed financial merit fields and hidden sampling meta.

Formal design:
  - Product types are task slices, analogous to occupations in SCOPED-Hiring.
  - Default products: Cash loans, Revolving loans.
  - Default sample size: 200 seeds per product.
  - TARGET is used only for stratified seed selection / difficulty coverage.
    It is not prompt-visible and is not treated as a ground-truth approval label.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

DOCUMENT_COLS = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
BUREAU_REQ_COLS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]


def clean_str(x: Any, default: Optional[str] = "Not specified") -> Optional[str]:
    if pd.isna(x):
        return default
    s = str(x).strip()
    if not s or s.upper() in {"XNA", "NAN", "NONE"}:
        return default
    return s


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if pd.isna(x):
        return default
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return default
    return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    if pd.isna(x):
        return default
    try:
        return int(float(x))
    except Exception:
        return default


def yn_to_bool(x: Any) -> Optional[bool]:
    if x == "Y":
        return True
    if x == "N":
        return False
    return None


def days_birth_to_age(days_value: Any) -> Optional[float]:
    d = safe_float(days_value)
    if d is None or d >= 0:
        return None
    return round((-d) / 365.25, 1)


def days_employed_to_years(days_value: Any) -> Optional[float]:
    # Home Credit special marker for pensioner / not applicable.
    d = safe_float(days_value)
    if d is None or d == 365243 or d >= 0:
        return None
    return round((-d) / 365.25, 2)


def ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den <= 0:
        return None
    return round(num / den, 4)


def sum_existing(row: pd.Series, cols: Iterable[str]) -> int:
    total = 0
    for c in cols:
        if c in row.index:
            total += safe_int(row.get(c), 0) or 0
    return int(total)


def bureau_req_detail(row: pd.Series) -> Dict[str, int]:
    return {c: safe_int(row.get(c), 0) or 0 for c in BUREAU_REQ_COLS if c in row.index}


def external_score_mean(row: pd.Series) -> Optional[float]:
    vals = [safe_float(row.get(c)) for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(float(np.mean(vals)), 4)


def affordability_band(credit_income: Optional[float], annuity_income: Optional[float]) -> str:
    if credit_income is None or annuity_income is None:
        return "unknown"
    if credit_income <= 2.5 and annuity_income <= 0.18:
        return "low_pressure"
    if credit_income <= 5.0 and annuity_income <= 0.32:
        return "medium_pressure"
    return "high_pressure"


def observed_risk_band(row: pd.Series) -> str:
    # Hidden-only sampling band. Not prompt-visible, not approval truth.
    target = safe_int(row.get("TARGET"), 0) or 0
    income = safe_float(row.get("AMT_INCOME_TOTAL"))
    credit = safe_float(row.get("AMT_CREDIT"))
    annuity = safe_float(row.get("AMT_ANNUITY"))
    ci = ratio(credit, income)
    ai = ratio(annuity, income)
    ext_mean = external_score_mean(row)

    if target == 1:
        return "observed_repayment_difficulty"
    if (ci is not None and ci > 5.0) or (ai is not None and ai > 0.32):
        return "high_affordability_pressure_no_observed_default"
    if ext_mean is not None and ext_mean < 0.35:
        return "low_external_score_no_observed_default"
    return "no_observed_repayment_difficulty"


def validate_input_columns(df: pd.DataFrame) -> None:
    required = [
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL",
        "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS",
        "CNT_CHILDREN", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def filter_eligible(df: pd.DataFrame, product: str, min_income: float, max_credit_income: float) -> pd.DataFrame:
    d = df[df["NAME_CONTRACT_TYPE"] == product].copy()
    d = d[d["AMT_INCOME_TOTAL"].notna() & (d["AMT_INCOME_TOTAL"] >= min_income)]
    d = d[d["AMT_CREDIT"].notna() & (d["AMT_CREDIT"] > 0)]
    d = d[d["AMT_ANNUITY"].notna() & (d["AMT_ANNUITY"] > 0)]
    d = d[d["CNT_FAM_MEMBERS"].notna()]
    d = d[d["CNT_CHILDREN"].notna()]
    d = d[d["NAME_INCOME_TYPE"].notna()]

    d["credit_income_ratio_tmp"] = d["AMT_CREDIT"] / d["AMT_INCOME_TOTAL"]
    d["annuity_income_ratio_tmp"] = d["AMT_ANNUITY"] / d["AMT_INCOME_TOTAL"]
    d = d[d["credit_income_ratio_tmp"].between(0.15, max_credit_income)]
    d = d[d["annuity_income_ratio_tmp"].between(0.01, 0.65)]

    d = d[d["NAME_INCOME_TYPE"] != "Pensioner"]
    d = d[d["DAYS_EMPLOYED"].notna()]
    d = d[d["DAYS_EMPLOYED"] != 365243]
    return d


def stratified_sample_product(
    d: pd.DataFrame,
    n: int,
    target_positive_frac: float,
    random_state: int,
) -> pd.DataFrame:
    if len(d) < n:
        raise ValueError(f"Not enough eligible rows for requested sample: need {n}, have {len(d)}")

    d = d.copy()
    d["target_bin"] = d["TARGET"].fillna(0).astype(int).clip(0, 1)
    d["affordability_bin"] = pd.qcut(
        d["annuity_income_ratio_tmp"].rank(method="first"), q=3, labels=["low", "mid", "high"]
    )

    n_pos = min(int(round(n * target_positive_frac)), int((d["target_bin"] == 1).sum()))
    n_neg = n - n_pos

    parts = []
    for target_bin, need, offset in [(1, n_pos, 11), (0, n_neg, 29)]:
        sub = d[d["target_bin"] == target_bin]
        if need <= 0:
            continue
        per = max(1, need // 3)
        sampled = (
            sub.groupby("affordability_bin", group_keys=False, observed=False)
            .apply(lambda x: x.sample(n=min(len(x), per), random_state=random_state + offset))
        )
        if len(sampled) < need:
            rest = sub.drop(index=sampled.index, errors="ignore")
            sampled = pd.concat([
                sampled,
                rest.sample(n=min(need - len(sampled), len(rest)), random_state=random_state + offset + 1),
            ])
        if len(sampled) > need:
            sampled = sampled.sample(n=need, random_state=random_state + offset + 2)
        parts.append(sampled)

    out = pd.concat(parts, axis=0)
    if len(out) < n:
        rest = d.drop(index=out.index, errors="ignore")
        out = pd.concat([out, rest.sample(n=n - len(out), random_state=random_state + 99)])
    out = out.sample(n=n, random_state=random_state + 101).reset_index(drop=True)
    return out


def build_seed_record(row: pd.Series, product_index: int, seed_order_in_product: int) -> Dict[str, Any]:
    income = safe_float(row.get("AMT_INCOME_TOTAL"))
    credit = safe_float(row.get("AMT_CREDIT"))
    annuity = safe_float(row.get("AMT_ANNUITY"))
    goods = safe_float(row.get("AMT_GOODS_PRICE"))
    ci = ratio(credit, income)
    ai = ratio(annuity, income)
    br = bureau_req_detail(row)

    product_type = clean_str(row.get("NAME_CONTRACT_TYPE"))
    product_slug = str(product_type).replace(" ", "_").replace("/", "_").upper()
    source_id = safe_int(row.get("SK_ID_CURR"))

    return {
        "seed_id": f"CREDIT_{product_slug}_{source_id}",
        "source_dataset": "home_credit_default_risk",
        "source_row_id": source_id,
        "scenario": "CREDIT",
        "product_type": product_type,
        "product_index": product_index,
        "seed_order_in_product": seed_order_in_product,
        "financial_merit": {
            "income_total": income,
            "credit_amount": credit,
            "annuity_amount": annuity,
            "goods_price": goods,
            "credit_income_ratio": ci,
            "annuity_income_ratio": ai,
            "affordability_band": affordability_band(ci, ai),
            "income_type": clean_str(row.get("NAME_INCOME_TYPE")),
            "occupation_type": clean_str(row.get("OCCUPATION_TYPE")),
            "employment_years": days_employed_to_years(row.get("DAYS_EMPLOYED")),
            "household_size": safe_float(row.get("CNT_FAM_MEMBERS")),
            "children_count": safe_int(row.get("CNT_CHILDREN")),
            "housing_type": clean_str(row.get("NAME_HOUSING_TYPE")),
            "owns_car": yn_to_bool(row.get("FLAG_OWN_CAR")),
            "owns_realty": yn_to_bool(row.get("FLAG_OWN_REALTY")),
            "documents_provided_count": sum_existing(row, DOCUMENT_COLS),
            "recent_credit_inquiries_total": int(sum(br.values())),
            "recent_credit_inquiries_detail": br,
        },
        "hidden_sampling_meta": {
            "target_repayment_difficulty": safe_int(row.get("TARGET")),
            "observed_risk_band": observed_risk_band(row),
            "raw_gender": clean_str(row.get("CODE_GENDER"), default=None),
            "raw_age_years": days_birth_to_age(row.get("DAYS_BIRTH")),
            "raw_education": clean_str(row.get("NAME_EDUCATION_TYPE"), default=None),
            "raw_family_status": clean_str(row.get("NAME_FAMILY_STATUS"), default=None),
            "raw_region_population_relative": safe_float(row.get("REGION_POPULATION_RELATIVE")),
            "raw_region_rating": safe_int(row.get("REGION_RATING_CLIENT")),
            "raw_region_rating_city": safe_int(row.get("REGION_RATING_CLIENT_W_CITY")),
            "raw_organization_type": clean_str(row.get("ORGANIZATION_TYPE"), default=None),
            "ext_source_1": safe_float(row.get("EXT_SOURCE_1")),
            "ext_source_2": safe_float(row.get("EXT_SOURCE_2")),
            "ext_source_3": safe_float(row.get("EXT_SOURCE_3")),
            "ext_source_mean": external_score_mean(row),
        },
    }


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_summary(records: List[Dict[str, Any]], path: Path) -> None:
    rows = []
    for r in records:
        fm = r["financial_merit"]
        hm = r["hidden_sampling_meta"]
        rows.append({
            "seed_id": r["seed_id"],
            "product_type": r["product_type"],
            "source_row_id": r["source_row_id"],
            "seed_order_in_product": r["seed_order_in_product"],
            "target_repayment_difficulty": hm.get("target_repayment_difficulty"),
            "observed_risk_band": hm.get("observed_risk_band"),
            "income_total": fm.get("income_total"),
            "credit_amount": fm.get("credit_amount"),
            "annuity_amount": fm.get("annuity_amount"),
            "credit_income_ratio": fm.get("credit_income_ratio"),
            "annuity_income_ratio": fm.get("annuity_income_ratio"),
            "income_type": fm.get("income_type"),
            "occupation_type": fm.get("occupation_type"),
            "employment_years": fm.get("employment_years"),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to application_train.csv")
    ap.add_argument("--out", required=True, help="Output seed JSONL")
    ap.add_argument("--summary-out", default=None, help="Optional seed CSV summary")
    ap.add_argument("--products", default="Cash loans,Revolving loans")
    ap.add_argument("--seeds-per-product", type=int, default=200)
    ap.add_argument("--target-positive-frac", type=float, default=0.30)
    ap.add_argument("--min-income", type=float, default=30000.0)
    ap.add_argument("--max-credit-income", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    validate_input_columns(df)

    products = [p.strip() for p in args.products.split(",") if p.strip()]
    records: List[Dict[str, Any]] = []

    for product_index, product in enumerate(products, start=1):
        d = filter_eligible(df, product, args.min_income, args.max_credit_income)
        sampled = stratified_sample_product(
            d=d,
            n=args.seeds_per_product,
            target_positive_frac=args.target_positive_frac,
            random_state=args.seed + product_index * 1000,
        )
        for i, (_, row) in enumerate(sampled.iterrows(), start=1):
            records.append(build_seed_record(row, product_index, i))
        print(f"[OK] product={product}: eligible={len(d)}, selected={len(sampled)}")

    write_jsonl(records, Path(args.out))
    print(f"[OK] wrote {len(records)} seeds to {args.out}")
    if args.summary_out:
        write_summary(records, Path(args.summary_out))
        print(f"[OK] wrote seed summary to {args.summary_out}")


if __name__ == "__main__":
    main()
