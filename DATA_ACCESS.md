# Data access and provenance

The hiring pipeline derives controlled variants from anonymized CV seeds based
on the Djinni Recruitment Dataset. The original source files are not
redistributed here. `data_generation/hiring/dataselect_v3.py` expects a locally
obtained source parquet file; it is optional and is not required to inspect the
released aggregate results.

The credit scripts likewise expect a separately obtained, lawful source table
at the path configured by the user. No source credit data is included here.

The companion data release contains only aggregate metric products, paper-table
sources, and data dictionaries. It excludes raw CV text, full raw JSONL
trajectories, and representative raw trajectory examples. This boundary avoids
re-distributing source records or generated text whose provenance and privacy
status require a separate release review.

Private-assessment fields in SCOPED are elicited audit artifacts; they are not
recovered latent chain-of-thought. Investigation desires are virtual scrutiny
signals, not executed external tool calls.
