# SCOPED-Credit data generation scripts

## 1. Select formal seeds

Run from the repository root. Supply a lawfully obtained source file and write
all generated records to an untracked local workspace (shown as
`./local_data/` below).

```bash
python data_generation/credit/credit_dataselect_formal_v2.py \
  --input /path/to/application_train.csv \
  --out ./local_data/credit/seeds/credit_seeds_formal.jsonl \
  --summary-out ./local_data/credit/seeds/credit_seeds_formal_summary.csv \
  --products "Cash loans,Revolving loans" \
  --seeds-per-product 200 \
  --target-positive-frac 0.30 \
  --seed 42
```

## 2. Generate formal variants

```bash
python data_generation/credit/credit_var_datagen_formal_v2.py \
  --seeds ./local_data/credit/seeds/credit_seeds_formal.jsonl \
  --out ./local_data/credit/variants/credit_variants_formal.jsonl \
  --summary-out ./local_data/credit/variants/credit_variants_formal_summary.csv \
  --bucket-dir ./local_data/credit/variants/buckets \
  --variants-per-seed 30 \
  --focus-n-per-product 20 \
  --seed 42
```

Expected scale: 2 products × 200 seeds × 30 variants = 12,000 cases.

The variant generator inherits the full SCOPED-Hiring library from `Data_Gen_Final.py`: 12-country identity maps, first/last-name pools, university/city tiers, country-specific SES/hobby carriers, age groups, gap types, and P1-P6 focus profiles. It adds Credit-specific employer-tier and application-channel carriers.
