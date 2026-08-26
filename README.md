# SCOPED-Hiring

Official research code for **Beyond Outcome Gaps: Process-Aware Fairness
Diagnosis for LLM-based Multi-Agent Decision Systems** (EMNLP 2026 Main
Conference).

SCOPED-Hiring audits fairness throughout an LLM-based multi-agent hiring
trajectory. It complements final-outcome checks with controlled, direction-aware
evidence about private assessments, scrutiny requests, pathway transitions,
deliberation dynamics, and system design choices.

## What this repository contains

- The configurable two-stage MAS runner (`run_suite.py`) and experiment matrix.
- The six-lens SCOPED analysis pipeline and aggregate-result analyzer.
- Scenario, agent, and Fair Skills intervention implementations.
- Scripts for regenerating controlled hiring and credit variants from separately
  obtained source data.
- BRI validation code and policy-stability analysis code.

The released aggregate result tables live in the companion Hugging Face dataset:
**[SCOPED-Hiring-Trajectories](https://huggingface.co/datasets/warrenlvlmgo/SCOPED-Hiring-Trajectories)**. This repository deliberately does **not** include
API keys, source CV data, complete raw trajectories, or representative raw
trajectory examples.

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env.credit
```

Set the provider credentials and model endpoint in `.env.credit`, then run a
configured suite:

```bash
python run_suite.py
```

`experiment_configs.py` defines the run matrix. The default configuration is a
research reproduction configuration; API access, model availability, and costs
remain the responsibility of the user.

## Data and reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the relationship between the
runner, aggregated release tables, and paper results; see
[DATA_ACCESS.md](DATA_ACCESS.md) for source-data boundaries. The credit
underwriting extension is an appendix-only portability study: it demonstrates
that the audit design can be rebound to another simulated high-stakes setting;
it is not a deployment, legal, or generalization claim.

## Responsible use

This artifact is for research and reproducibility. It is not designed or
validated for real hiring, candidate screening, ranking, compensation,
promotion, or other consequential decisions. See
[RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## Citation

If you use this artifact, please cite the EMNLP 2026 paper. A machine-readable
record is provided in [CITATION.cff](CITATION.cff).

## License

The code is released under [Apache-2.0](LICENSE). Attribution notices are in
[NOTICE](NOTICE). This license applies to the code and documentation in this
repository; source datasets and the companion aggregate data release have their
own terms.
