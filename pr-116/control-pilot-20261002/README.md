# Flow-control development pilot — complete evidence snapshot

[Result plots, full fields and summary](report/README.md) cover all 12 learned-policy runs and all 8 warm-start diagnostics. [Raw outcomes and protocols](results/), [configs](configs/), [Slurm job IDs](submissions.json), and [artifact/source manifest](manifest.json) preserve the completed campaign. No follow-up experiment results are included.

The cheap linear controller beats every learned-policy checkpoint in this pilot. Warm-start direct action optimization improves the long-horizon linear baseline by 92.55% across these eight tasks; this is per-instance optimization, not a learned-policy advantage. These are development validation results with unequal, untuned training budgets, not a held-out test.

Dataset hashes differ: [the dataset audit](audits/dataset-audit.json) records the exact arrays and numerical differences, so the report makes no exact-paired cross-method claim. [The policy audit](audits/policy-audit.json) retains training/validation action errors and checkpoint diagnostics. Known generating controls add no labeling cost beyond common goal generation; extra expert optimization is charged separately.

The report includes one fixed full-field example for each experiment type and all-seed/checkpoint result plots. Raw large fields and model checkpoints remain in the cluster archives listed in the manifest. Frozen flow-control source files are extracted from the exact source archive whose SHA256 is recorded in every config. Reporting scripts are stored separately at this directory root.
