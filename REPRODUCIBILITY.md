# Reproducibility guide

## Reproduce the released analyses

1. Download the companion `SCOPED-Hiring-Trajectories` dataset.
2. Use `analysis/mas_fairbench_raw_granular_analyzer_v5_credit.py` to analyze
   aggregate run products, and `analysis/policy_stability_check.py` for the
   policy-stability checks.
3. Compare the products in `data/paper_tables/` with the paper tables.

The data release contains aggregate, direction-aligned metrics rather than the
complete raw trajectory corpus. Thus it supports verification of the reported
aggregations and paper tables, not byte-for-byte regeneration of every LLM run.

## Re-run the MAS experiments

1. Obtain lawful access to the source data described in `DATA_ACCESS.md`.
2. Generate controlled variants with `data_generation/hiring/` or
   `data_generation/credit/`.
3. Configure a provider in `.env.credit` from `.env.example`.
4. Edit the desired conditions in `experiment_configs.py`.
5. Run `python run_suite.py`, then analyze the generated products.

LLM outputs can vary with provider model versions, decoding settings, and
service behavior. The paper's reported evidence is therefore preserved as
aggregate release data in addition to the executable pipeline.
