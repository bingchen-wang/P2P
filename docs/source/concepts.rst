Concepts
========

One configuration, composable stages
------------------------------------

Every workflow starts from one resolved :class:`p2p.config.Config`.  Commands
are user-facing compositions; stages are reusable implementation units:

.. code-block:: text

   central YAML
       │
       ├── generate ──> endowments
       ├── elicit  ───> responses
       ├── split   ───> question-level train/valid/test matrices
       ├── targets ───> aligned population marginals
       ├── fit     ───> constrained model
       ├── evaluate───> policy-tagged metrics
       └── report  ───> run.json + metrics.json/csv

The ``pipeline`` command composes that path end to end.  ``generate``,
``elicit``, ``regress`` and ``evaluate`` expose useful boundaries for cached or
independently supplied artifacts.

Basis and target
----------------

The *basis* is the proxy personas (endowments) and their survey responses.  The
*target* maps question IDs and answer labels to population proportions.  Stage 2
learns proxy weights whose weighted answers approximate the target marginals.
The design needs aggregate marginals, not respondent-level demographic data.

Weights are not restricted in sign.  A negatively weighted proxy is an
*anti-type*: its answers run counter to the target population, and P2P keeps
this capability deliberately.

Surveys are split by question: whole questions go to train, validation or test,
and cross-validation folds group a question's option rows together.
Generation and elicitation cover the configured survey, while Stage 2 selects
the configured splits, so changing a question split can reuse a cached basis.

Recorded contracts
------------------

Every choice that affects a number is recorded with the run:

* ``cleaning_policy`` names how incomplete proxy answers were handled
  (``column_drop_v2`` drops proxies that did not answer every question).
* MSE records its prediction policy (raw predictions).
* Distribution metrics record the transform applied to predictions and the
  questions they cover.
* Aggregation rejects incompatible policies, cleaning policies, duplicate run
  IDs and incomplete expected cells.
* A resolved configuration snapshot and its digest accompany each run.

Backends and usage
------------------

The OpenAI, Gemini, DeepSeek, Ollama, simulated and dummy adapters implement one
normalized response and token-usage contract.  Logical callers account for
usage once, and each run can write ``usage.csv`` and ``usage.json``.  Local
backends report zero monetary cost unless local pricing is configured.

Adaptive generation
-------------------

Active generation measures the categorical entropy of the pool's answers.
Entropy drives a softmax allocation across generation modes; questions whose
answers stay low-entropy can receive question-specific patches, and persistent
low-entropy questions can be mixed with the strongest base mode.  Attribute
banks are immutable inputs; generators work on a run-local copy.

The target-mean entropy subsampler is a separate, explicitly selected offline
estimand for cached-basis simulations.  It uses the same entropy measure but is
never called by active generation.

Runs and outputs
----------------

Every command writes to the directory given by ``--output``.  Use a new
directory for each run; the run's own records, not its folder name, say what it
is.  Paid active generation is the one case that resumes in place (see
:doc:`workflows`).
