# P2P Experiment Studio

Local browser interface for configuring and running P2P experiments. It is part
of this repository and is not connected to a hosting service.

## Run locally

```bash
cd studio
npm install
npm run build
cd ..
p2p ui
```

Open `http://127.0.0.1:5173/` if the browser does not open automatically.
The local Python server reads manifests and artifacts from this repository and
starts experiments through the same command handlers as the CLI. It binds to
the loopback interface by default and has no publishing or upload feature.

On a first launch, Studio presents a short welcome demonstration and then
behaves as an everyday research workspace. The introduction is stored in the
browser and can be replayed from Help.

## First local result

Choose **Run offline demo** from the welcome screen or Home. Studio runs the
whole pipeline on the license-free fictional-amenities survey under
`examples/studio_demo/` in a few seconds, without credentials, network access,
cost or licensed data:

- **Stage 1.** P2P's active generation creates residents of a fictional town,
  asks them the survey, and adds agents where answers are least settled
  (including question-targeted patches).
- **Stage 2.** A constrained lasso is fitted.
- **Evaluation.** The fit is scored on the held-out questions.

Stage 1 uses the `simulated` backend (`config/backends/simulated_demo.yaml`):
deterministic rules, not a language model. Its personas and answers come from a
small declarative world, so the demo shows how the pipeline works, not how well
any model answers.

The run opens with the generated population, the fitted ensemble, the entropy
explorer and the held-out evidence. Its canonical record appears under
`outputs/studio/` beside every other run. The demo inputs, the simulated world
and the attribute bank are written by `examples/studio_demo/build_demo_files.py`;
the manifest is `config/smoke/studio_demo_pipeline.yaml`. Stage-2-only examples on
a fixed synthetic agent basis are `config/smoke/studio_demo_fit_lasso.yaml` and
`config/smoke/studio_demo_fit_simple_average.yaml`.

The Data page can inspect a repository-local CSV, survey-schema YAML, and
aggregate-target JSON before an experiment is created. Inspection checks the
field mapping, unique question IDs, option support, split counts, and target
coverage. Registering a valid dataset writes only a small descriptor with
relative paths and SHA256 digests under `config/datasets/`; source files are
not rewritten. When a CSV has no schema yet, the mapping wizard lets the user
assign question ID, question text, coded answer, and split columns, recheck the
mapping, and write an explicit repository-local schema as part of registration.

Experiment creation has a non-writing preflight step. Python resolves and
validates the draft, reports its config digest, output conflict, provider
boundary, cost status, and CLI equivalent before the manifest or output
directory is created. Available workflows, models, backends, metrics, and
platform process controls come from `/api/capabilities`, not a duplicate list
in the React application. Draft fields are retained in local browser storage,
and registered dataset descriptors can populate their pinned survey, schema,
and target paths without copying the underlying data.
The default builder shows the decisions needed for an ordinary run: the Stage-2
method (lasso, elastic net or simple average) and, for fitted methods, the
Stage-2 configuration (mass penalty on or off, with a reference respondent, an
intercept or neither). Advanced settings add penalty selection (cross-validation
grouped by question, or a holdout split), the fold assignment (hashed, scikit-learn,
or replay of a recorded `folds.json`), the penalty grids (default from P2P:
10^-5 to 10^5, 30 values), block weighting, the metrics, the solver, seeds and
active-generation tuning; they retain their values when collapsed. The simple
average writes only `method: simple_average`, since P2P refuses penalty settings
for it. Every option and default comes from `/api/capabilities`.

The Results library discovers both canonical `run.json` bundles and historical
experiment directories containing `summary*.md`. Historical Markdown, metrics,
token summaries, prediction plots, model diagnostics, entropy trajectories,
MCA plots, and generation figures are grouped into one experiment view. Newer
canonical runs also expose interactive prediction-calibration and question-
entropy explorers from their structured diagnostics.

The library reads run and report summaries through paginated APIs. Current and
historical records retain their source format, capability flags, provenance,
and compatibility warnings. Missing fields in an older record are shown as
unavailable rather than reconstructed silently.

Each current run's record opens with **Run health**: plain-language checks read
from the run's own files (solver status and fallbacks, a penalty selected at a
grid edge, a flat validation curve, options with no effect, incomplete coverage,
rescaled or invalid distributions, and answers that failed every retry), beside
the facts that define the run (configuration, penalty selection, selected penalty,
solver, Σβ, reference-respondent weight or intercept, train MSE, folds and their
digest, config digest). The **Pooled summaries** tab lists every `p2p aggregate`
output in the workspace and shows its recorded centre ± spread per group, with
the validation checks and source files.

Studio views and opened runs or reports have durable query-string URLs, so
refresh, browser back/forward navigation, and direct local links preserve the
research context. The run library can compare two to four current or historical
records side by side and export the comparison as CSV. Runs and report folders
can be pinned and given persistent research labels without changing the source
artifacts. Labels are searchable, pinned runs and reports can be sorted first,
and the annotation store remains local under `outputs/.studio/`. Detail views explain
metric direction and coverage, surface provenance and compatibility warnings,
link a result back to its originating Studio job when available, and provide
safe downloads for files inside the selected output record.

Job metadata is persisted under `outputs/.studio/jobs/`. If Studio restarts,
completed outputs remain inspectable and an unattached in-progress record is
shown as interrupted with an explicit checkpoint-or-new-output recovery state.
Studio never claims that it can safely resume a process it can no longer
control. The Jobs workspace consolidates live controls, recovered records,
process logs, and completed-result navigation.
Any recorded job can load its source manifest into the experiment builder as a
new editable copy; the original manifest and output remain unchanged.

Backend cards provide a secret-safe readiness check for the profile, Python
dependency, and credential boundary. The check never sends a paid provider
request; local runtime availability is still confirmed when an experiment
starts.

For frontend-only development, run `npm run dev` from `studio/`. Repository
data and run controls require the `p2p ui` server.

## Backends

Profiles under `config/backends/` are selectable in the builder: `dummy`
(offline, fixed text), `simulated_demo` (offline, rule-based respondents for the
demo survey; not a language model), `ollama` (local model, no key), `openai`, `gemini` and `deepseek`
(metered; keys come from the environment, never from Studio or a manifest). The
Backends page lists each adapter's settings as read from its own constructor and
checks a profile locally without sending a provider request.
