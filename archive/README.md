# archive/

Modules that no longer have a caller and are kept for reference. Nothing here
is imported by `src/` or by the test suite, and nothing here runs: they sit
outside `src/`, so they are off `PYTHONPATH` and cannot import the modules they
were written against. Git history is intact, so `git mv` them back if the need
returns.

Each entry says why it was retired. A file here is a judgement about the
current tree, not a claim that the code was wrong.

| file | why it is not in `src/` |
|---|---|
| `batch_export.py` | Exported four scenarios with charts plus the Nash tests. No caller. It builds on `market.adaptive_bidding`, which moved to `strategies.adaptive_bidding`. |
| `export_analysis.py` | No caller, and it runs a full simulation **at import time** — importing it to inspect it writes CSVs. That is reason enough not to leave it one `import` away from anything. |
| `export_charts_only.py` | No caller; a charts-only variant of the above. |
| `export_curves.py` | No caller; curve CSV export, superseded by the run artifacts under `outputs/rl/`. |
| `plot_diagrams.py` | Rendered architecture diagrams for a design that no longer exists: 24 discrete actions, an independent PPO actor-critic, a 20-dimensional state. The market is now continuous reservation prices, the learners are MATD3/TD3, and the observation is 12 dimensions. Left in place it would have produced figures that misdescribe the system, which is worse than producing none. |
| `probe_return_multimodality.py` | A one-off probe asking whether the return distribution is multimodal enough to justify a distributional critic. The question was answered and the distributional arms were set aside. |
| `pseudo_realtime.py` | Broken rather than obsolete: it calls `solve_opf_gurobi` with nine positional arguments against an eight-parameter signature, and unpacks five return values. It would raise on first use. |

## Not archived, and why

These look like candidates and are not:

- **`src/outputs.py`** — imported by `src/run.py`. It is currently unimportable
  because `matplotlib` is installed without its `__init__.py` in this
  environment, which also takes `run.py` down with it. That is a broken
  dependency, not dead code; reinstalling `matplotlib` fixes it. `matplotlib`
  is not in `requirements.txt`, which is how the drift went unnoticed.
- **`pywraplp` imports** in the export scripts — a load-order preload for a
  Windows DLL conflict, not an unused import.
- **The side-effect imports in `strategies/__init__.py` and `ui/shell.py`** —
  they register strategies and pages; removing them breaks the registries.
- **`rl_diffusion.py`, `rl_masac.py`, `rl_quantile.py`** — the archived
  distributional and entropy-regularized arms. They carry the same
  deviation-penalty defect that was repaired in `rl_td3`/`rl_bidding` (the
  reference still anchors at the logit zero, which is the box midpoint rather
  than the truthful point). They are **not** orphaned: `train_rl.py` reaches
  them through `--algo mad3pg|qmatd3|masac`, `rl_td3.load_policy` imports
  `rl_masac.StochasticActor`, and three test modules exercise them. Moving them
  is a change to the CLI surface, not a file move.
