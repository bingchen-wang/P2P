# P2P
Modular system for reconstructing population-level preferences from a compact LLM proxy ensemble. No finetuning, no demographic data.

```
P2P/
  pyproject.toml            # package + deps + pytest config + console_script  p2p = p2p.cli:main
  README.md   REVAMP.md
  p2p/
    __init__.py  __main__.py        # `python -m p2p` / `p2p <verb>`
    cli.py                          # argparse dispatcher (verb registry)          [REBUILD]
    config.py                       # load + validate ONE central YAML -> Config    [REBUILD]
    registry.py                     # generic Registry: BACKENDS/MODELS/STAGES/COMMANDS/METRICS [REBUILD]
    metrics.py                      # PURE kernels + project_to_simplex             [REBUILD]
    commands/                       # SIM-PANEL verbs (user workflows) — compose stages [REBUILD]
      generate.py elicit.py pipeline.py regress.py simulate.py ablate.py panel.py
      evaluate.py cluster.py validate.py docs.py
    stages/                         # internal reusable steps (thin over ported code) [REBUILD]
      generate.py elicit.py split.py targets.py fit.py evaluate.py report.py
    backends/ agents/               # ported, unified under one registry            [PORT+rewire]
    survey/ responses/ endowments/  # ported leaf modules (survey/response converters, manager) [PORT]
    models/ generators/ simulations/  # ported mostly as-is                          [PORT]
  config/central/*.yaml             # unified single-file configs (migrated from old schemas)
  data/                             # minimal fixture data only (w42 + one WVS locale)
  outputs/fixtures/                 # minimal cached runs used by golden tests
  tests/  conftest.py  golden/  fixtures/
  docs/                             # Sphinx (User Guide + API)
```