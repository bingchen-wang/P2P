<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/p2p-logo-dark.png">
    <img src="docs/assets/p2p-logo-light.png" alt="P2P logo" width="360">
  </picture>
</p>

<h1 align="center">P2P: Prompts to Proxies</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2509.11311"><img src="https://img.shields.io/badge/arXiv-2509.11311-b31b1b.svg" alt="arXiv 2509.11311"></a>
  <a href="https://bingchen-wang.github.io/p2p-website/"><img src="https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg" alt="NeurIPS 2026"></a>
  <a href="https://bingchen-wang.github.io/p2p-website/"><img src="https://img.shields.io/badge/project-page-2ea44f.svg" alt="Project page"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-3776ab.svg" alt="Python 3.12+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT licence"></a>
</p>

P2P reconstructs how a human population answers a survey by building a pool of
LLM proxy respondents and weighting them so their combined answers match the
population's published answer shares. It needs only aggregate survey results,
not respondent-level demographic data or model fine-tuning.

This repository is the software behind the NeurIPS 2026 paper
*Prompts to Proxies: Emulating Human Preferences via a Compact LLM Ensemble*.
It is meant as a reusable research tool: one YAML configuration system and one
command-line interface covering persona generation, survey elicitation,
constrained regression, evaluation, sweeps and aggregation.

## How it works

1. **Stage 1, build a proxy pool.** P2P prompts an LLM to write personas from
   structured attributes and asks each persona the survey. Entropy-guided
   adaptive sampling steers generation toward questions where the pool's
   answers are still too uniform, so the pool spans a wide range of opinions.
2. **Stage 2, weight the pool.** A constrained regression (lasso, elastic net,
   ridge, or a simple average) finds weights on the proxies so that their
   weighted answers match the population's answer shares on training questions.
   The fitted ensemble then predicts held-out questions. Stage 2 can also be
   re-anchored to a reference distribution, so the proxies fit each target's
   change from it, as when refitting a survey to a new wave.

Weights are not restricted to be positive: a negatively weighted proxy is one
whose answers run counter to the target population, and P2P keeps this
capability deliberately.

## Install

P2P needs **Python 3.12 or later**.

```bash
git clone https://github.com/bingchen-wang/P2P.git
cd P2P
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
p2p --version
```

Optional extras add provider backends and the documentation toolchain:

| Extra | Adds |
|---|---|
| `openai` | OpenAI backend |
| `gemini` | Google Gemini backend |
| `deepseek` | DeepSeek backend (OpenAI-compatible transport) |
| `docs` | Sphinx, to build the documentation |
| `dev` | pytest |

For example, `python -m pip install -e ".[dev,openai]"`. Ollama needs no extra.

**Exact environment.** `environment/` pins every package version used to
produce the recorded results (a conda lock for macOS on Apple Silicon, plus a
pip lock). See [environment/README.md](environment/README.md).

Check the installation with the offline test suite:

```bash
python -m pytest -q
```

Tests that need licensed survey data skip automatically, because that data is
not distributed (see [Data](#data)). Tests that call paid APIs are deselected
by default.

## Quickstart (offline, no API key)

Run the whole pipeline on a small fictional survey. Stage 1 uses P2P's
rule-based `simulated` backend instead of a language model, so this needs no
network, no credentials and costs nothing:

```bash
p2p pipeline \
  --config config/smoke/studio_demo_pipeline.yaml \
  --output outputs/demo_pipeline
```

The output directory holds the generated personas (`endowments.csv`), their
answers (`responses.csv`), the fitted metrics (`metrics.csv`,
`metrics.json`), provenance (`run.json`, `config.resolved.yaml`) and token
usage. See [Run artifacts](#run-artifacts).

The same demo runs from the browser: build the [Experiment Studio](#experiment-studio)
once, start it with `p2p ui`, and choose **Run the offline demo** on its home page.

To run Stage 1 with a real model on your own machine, install
[Ollama](https://ollama.com), pull `gemma3:4b`, and run the same fictional
survey with it:

```bash
p2p pipeline \
  --config config/smoke/ollama_demo.yaml \
  --output outputs/ollama_demo
```

## Bring your own survey

P2P does not ship survey data. To use your own, provide three files and a
manifest that points to them. Paths in a manifest are relative to the manifest
file.

**1. Survey CSV.** One row per question, with an identifier, the question
text, a mapping from answer codes to answer labels, and the split the question
belongs to (`train`, `valid` or `test`):

```csv
qid,question,option_mapping,split
AMENITY_01,Which fictional neighborhood feature should receive priority?,"{'A': 'A shared garden', 'B': 'A quiet reading room', 'C': 'A night market'}",train
```

**2. Schema YAML.** Names the columns in your CSV:

```yaml
question_id: qid
question_text: question
code_to_answer: option_mapping
split: split
```

**3. Target JSON.** The population's answer shares for each question, keyed by
question identifier and answer label. Each question's shares must form a
distribution; P2P refuses a target that does not, rather than rescaling it.

```json
{
  "AMENITY_01": {"A shared garden": 0.44, "A quiet reading room": 0.27, "A night market": 0.29}
}
```

P2P scores predictions against the answer categories exactly as your target
file supplies them. A target file can also hold several populations at once,
one set of shares per subgroup; a run then names the subgroup it fits
(`target: {path: targets.json, subgroup: "SEX::Female"}`).

**4. Manifest.** Copy `config/smoke/studio_demo_pipeline.yaml`, change the
`data` paths, choose a backend profile, and validate it before a paid run:

```bash
p2p validate --verb pipeline --config my_pipeline.yaml
```

To forecast tokens and cost before a paid run, `p2p budget` learns from the
usage recorded by earlier runs under `outputs/`. It needs at least three
earlier runs that used each model module, for example three small pilots:

```bash
p2p budget --config my_pipeline.yaml --output outputs/my_budget
```

Add `--pilot-run outputs/my_pilot` to calibrate the forecast on a small or
in-progress run of the same plan.

P2P does not embed current provider prices. For dollar estimates, copy
`config/pricing.example.yaml`, fill in current prices and pass
`--pricing your-pricing.yaml`.

`examples/studio_demo/` holds a complete set of these files, and
`examples/studio_demo/build_demo_files.py` shows how they were generated.

## Backends and API keys

| Backend | `type` | Key (environment variable) | Profile |
|---|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` | `config/backends/openai.yaml` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | `config/backends/gemini.yaml` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `config/backends/deepseek.yaml` |
| Ollama (local) | `ollama` | none | `config/backends/ollama.yaml` |
| Rule-based simulation | `simulated` | none | `config/backends/simulated_demo.yaml` |
| Fixed replies, for tests | `dummy` | none | `config/backends/dummy.yaml` |

A manifest selects a profile with:

```yaml
backend:
  profile: ../backends/openai.yaml
```

**Never put API keys in a manifest:** P2P keeps the resolved configuration with
every run. Set keys as environment variables instead. The ignore rules cover
`.env`, `.env.*` (except `.env.example`) and `.secrets/`.

Paid runs that use active generation (the default) are resumable. P2P
checkpoints the personas before their survey calls and after every completed
chunk of responses; rerunning the identical command with the same `--output`
restores finished work and sends only the missing requests. A changed
configuration is rejected rather than mixed into the old run.

## CLI

```text
p2p pipeline   Run generation, elicitation, regression and reporting
p2p generate   Generate personas (active, vanilla or plugin)
p2p elicit     Ask an existing persona set the survey
p2p regress    Fit Stage 2 from existing responses and personas
p2p evaluate   Score existing predictions or fitted outputs
p2p aggregate  Validate and summarise run directories
p2p budget     Forecast tokens and cost before an API-backed run
p2p simulate   Run simulation sweeps over cached responses
p2p ablate     Expand and run an ablation sweep
p2p panel      Expand and run a panel sweep
p2p cluster    Cluster persona text offline
p2p inspect    Build or serve interactive diagnostics for a run
p2p validate   Check a manifest for a given command
p2p ui         Launch the local Experiment Studio in a browser
```

Use `p2p <command> --help` for each command's options.

## Experiment Studio

`p2p ui` serves a local browser interface: the one place to set up experiments,
follow them and read their results. Build it once with Node.js 22.13 or later:

```bash
cd studio && npm ci && npm run build && cd ..
p2p ui
```

- **Home** shows the latest result and the next steps; a new workspace starts
  with the offline demo.
- **New experiment** walks through the workflow in steps (experiment, data,
  Stage 1, Stage 2, evaluation, review), writes a validated manifest under
  `config/experiments/studio/`, and runs the same command as the CLI.
- **Runs** shows live and recent runs at the top and every recorded run below,
  with search, sorting, labels and side-by-side comparison. A run opens on its
  fitted ensemble, predictions against the targets, entropy and weights.
- **Data**, **Budgets** and **Backends** inspect survey files, cost forecasts and
  model backends.

The Studio binds to `127.0.0.1`, reads and writes only this workspace, and has
no upload or publishing feature. It can also read runs written in the earlier
result format, read-only.

## Run artifacts

A completed pipeline run writes, among others:

```text
endowments.csv          generated personas
responses.csv           their survey answers
metrics.json/.csv       scores by split, with prediction policy and question coverage
diagnostics.json        fit diagnostics
run.json                source paths and content digests
config.resolved.yaml    the fully resolved configuration
usage.csv/.json         token usage and cost
pipeline.log
```

Active generation also writes its entropy trajectories and a checkpoint
directory, which is a recovery artifact; keep it with the run.

## Documentation

The full user guide and API reference are built with Sphinx:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W -b html docs/source docs/build/html
```

Start at [docs/source/index.rst](docs/source/index.rst): quickstart, concepts,
the Studio, CLI, configuration reference, workflows and artifacts.

## Data

The paper evaluates on the Pew Research Center's American Trends Panel (ATP),
OpinionQA (built from ATP) and the World Values Survey (WVS). Their licences do
not allow redistribution, so **this repository contains no survey data from
them**. Obtain them from the providers:

- ATP: <https://www.pewresearch.org/american-trends-panel-datasets/>
- OpinionQA: <https://github.com/tatsu-lab/opinions_qa>
- WVS: <https://www.worldvaluessurvey.org/>

Then convert them to the three files described in
[Bring your own survey](#bring-your-own-survey).

## Relation to the paper

P2P produced the analyses in *Prompts to Proxies: Emulating
Human Preferences via a Compact LLM Ensemble* (NeurIPS 2026). The paper, its
appendices and its supplemental material describe the data pipeline, data
sources, train/validation/test splits, model backends, prompts,
hyperparameters, baselines and evaluation metrics needed to reproduce the main
experimental results. The raw datasets are available from their original
providers (see [Data](#data)).

P2P will keep growing after the paper. **The version used for the paper is
release [`v0.1.0`, "Prompts to Proxies"](https://github.com/bingchen-wang/P2P/releases/tag/v0.1.0).**
For replication, install that release rather than the latest code:

```bash
git clone --branch v0.1.0 https://github.com/bingchen-wang/P2P.git
```

and use the exact environment in `environment/`. Results from API-based models
can still differ between runs, because providers' inference is not fully
deterministic.

## Citation

If you use the P2P software, please cite it (see [CITATION.cff](CITATION.cff)):

```bibtex
@software{wang2026p2p,
  author  = {Wang, Bingchen},
  title   = {{P2P}: Prompts to Proxies},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/bingchen-wang/P2P}
}
```

If you build on the method or the empirical results, please also cite the
paper:

```bibtex
@inproceedings{wang2026prompts,
  title     = {Prompts to Proxies: Emulating Human Preferences via a Compact {LLM} Ensemble},
  author    = {Wang, Bingchen and Khoo, Zi-Yu and Wang, Jingtan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Licence and contact

P2P is released under the [MIT licence](LICENSE). Questions and bug reports:
please open a [GitHub issue](https://github.com/bingchen-wang/P2P/issues).
