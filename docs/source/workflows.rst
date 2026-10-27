Workflows
=========

Offline re-analysis
-------------------

Cached ``responses.csv`` and ``endowments.csv`` are sufficient for Stage 2.
Point ``data.basis`` at their directory and use ``p2p regress``.  This supports
split changes, model comparisons, re-anchoring to a reference distribution,
other metric policies and any other re-analysis, without generation or
elicitation and at no cost.  Write each re-analysis to a new directory; the
source run is only read.

Full pipeline
-------------

``p2p pipeline`` performs generation, prepares responses, fits the configured
model, evaluates it, and writes canonical reports.  Active generation may
already elicit the responses needed for Stage 2; the pipeline reuses them
instead of paying for a redundant pass.

``config/smoke/studio_demo_pipeline.yaml`` runs the whole path offline on a
fictional survey with the rule-based ``simulated`` backend.  For a small real
run on your own machine, use the same survey with a local Ollama model:

.. code-block:: console

   p2p pipeline \
     --config config/smoke/ollama_demo.yaml \
     --output outputs/ollama_demo \
     --verbose

This expects an Ollama daemon with ``gemma3:4b``.  It exercises survey-level
attribute learning, the initial generation modes, entropy tracking, adaptive
expansion, a question patch, elicitation and response reuse, constrained
fitting, reporting and usage accounting.

DeepSeek V4
-----------

Install the provider extra and keep the credential outside configuration:

.. code-block:: console

   python -m pip install -e ".[deepseek]"
   export DEEPSEEK_API_KEY="..."

Select ``type: deepseek`` inline or reference
``config/backends/deepseek.yaml`` from a manifest.  The maintained profile
uses ``deepseek-v4-flash`` with thinking explicitly disabled for all three
pipeline roles: forced-choice survey answers benefit from bounded, directly
parseable output rather than provider-default reasoning.

Thinking mode is opt-in with ``thinking: enabled`` and optional
``reasoning_effort: high|max``.  The adapter omits temperature in this mode,
maps DeepSeek's cache-hit field, and keeps the provider's billable completion
count intact while recording its reasoning-token subset separately.  Run small
pilots and ``p2p budget --pilot-run`` before a full paid run, so the forecast
learns the model's actual token use.

Paid-run recovery
-----------------

Active generation checkpoints personas before their survey calls and commits
each completed response chunk atomically.  If a request, parser, process, or a
later pipeline stage fails, rerun the exact command with the same output
directory.  P2P restores RNG, entropy-tracker, attribute-bank, endowment,
response, and usage state and sends only missing agents to the provider.

Resume is deliberately strict.  The resolved configuration digest must match.
If an output directory contains nonempty ``usage.csv`` history but no
compatible checkpoint, P2P stops before constructing a backend and instructs
the user to preserve that directory and choose a fresh output path.  Do not
delete the checkpoint or edit the configuration between attempts.

Attribute discovery asks the model for a Python-style list.  Unusable
formatting receives bounded retries; if they run out, no new attributes are
recorded and generation continues, rather than discarding an otherwise
completed paid generation stage.

Standalone evaluation
---------------------

``p2p evaluate`` accepts a schema-versioned prediction or fitted artifact.
The configured model identity, splits, target, metric list, and prediction
policies are assertions.  Mismatches fail instead of being silently relabeled.

Aggregation
-----------

Locked aggregation is the path for reported tables.  An :class:`p2p.config.AggregateSpec`
declares runs, labels, grouping, requested metrics, expected repeats/cells,
missingness policy, hierarchy, summary statistics, and export formats.

The validator cross-checks declared labels against each run record and rejects:

* duplicate run IDs or duplicate inputs;
* missing expected repeats/cells;
* mixed split, cleaning, or prediction policies;
* conflicting metric scope or QID coverage;
* unsupported hierarchy/missingness combinations.

Exports include input and validation ledgers, machine-readable summaries, and
configured Markdown/LaTeX presentation views.  Full-precision machine outputs
are not rounded to presentation precision.

Sweeps
------

``simulate``, ``ablate``, and ``panel`` share deterministic expansion:

* nested/dotted overrides are normalized;
* cases and repeats receive child snapshots and stable run directories;
* child commands dispatch through the command registry;
* an optional shared aggregation specification reduces child results.

The default simulation runner writes ``simulation.json`` as a declaration
record only.  ``simulation.executor: scientific`` selects the built-in
cached-basis executor.  It supports the endowment- and response-fraction,
proxy-only and train/validation fraction, fixed-ground-truth and entropy-mode
estimands, plus the separate target-mean entropy subsampler.  Each child
writes its fit and statistics; the sweep root writes all-child and
mean/sample-SD summaries.

Live diagnostics
----------------

``p2p inspect`` builds an artifact-first interactive view of a run.  Static
mode writes a self-contained HTML file.
Live mode rebuilds it on every request and auto-refreshes, so it can follow a
pipeline while ``responses.csv``, entropy trajectories, usage, metrics, and
the log appear.

The maintained drilldowns are:

* question entropy → categorical response counts;
* prediction scatter → human/predicted distribution;
* proxy coefficients and canonical run metrics;
* adaptive entropy trajectories and usage by module;
* optional singular-value spectrum and MCA agent map.

Partial runs, and runs written in the earlier result format, show the views
their files support (see :doc:`earlier_formats`); no serialized model is ever
loaded.

The Studio's run page (:doc:`studio`) presents the same evidence alongside the
rest of the workspace.

Pre-run budgeting and pilot calibration
----------------------------------------

``p2p budget`` is an offline planning workflow over usage artifacts.  It
extracts target size, survey dimensions, generation mode, adaptive settings,
batch size, output caps, module, and configured model.  Each module's forecast
needs at least three earlier runs that used it: per-module random forests use
exact-model history when there are three matching runs and otherwise a
labelled fallback pooled across models.  The reported central interval
combines tree dispersion with out-of-bag residuals when coverage permits.

The planner writes ``budget.json``, a module-level ``budget.csv``, the exact
run-level training ledger ``budget_history.csv``, and ``budget.html``.  A
budget ceiling is assessed against the interval high as well as the point
estimate; unpriced models never silently become zero-cost.

A pilot run changes calibration, not the scientific target.  The planner
compares observed input/output tokens per API call with earlier runs of the
same module and model, and scales the full-run forecast.  During API-backed
commands, ``usage.csv`` is append-only while calls complete and is rewritten
canonically at completion.  Live budget mode rereads that file on every
request, so the observed token/cost trajectory and calibration update without
waiting for Stage 2.

Clustering
----------

``p2p cluster`` reads ``eid`` and ``endow_text`` from an existing endowment
CSV, builds TF-IDF features, runs seeded K-means, canonicalizes cluster labels,
and writes deterministic assignments and metadata.  It is offline and does
not embed text through a paid backend.
