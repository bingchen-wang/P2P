Configuration and schema
========================

Loading and resolution
----------------------

:meth:`p2p.config.Config.load` reads one YAML manifest, recursively resolves
``extends``, applies defaults, resolves known paths, expands structured grids,
and freezes the result.  Mapping overrides merge recursively; sequences and
scalars replace their inherited value.

Paths are resolved relative to the final declaring manifest.  A run writes
``config.resolved.yaml`` with the absolute resolved values and a stable SHA256
configuration digest.

Top-level sections
------------------

``run``
   Run identity and deterministic seed.  Common fields include ``name``,
   ``seed``, ``repeat``, ``wave``, and labels used by result records.

``data``
   Survey CSV/schema, cached basis, target marginals, predictions, or fitted
   artifacts.  The required members depend on the verb.

``generation``
   Strategy (``active``, ``vanilla``, or injected), population size,
   adaptive-generation controls, endowment-model settings, and optional
   attribute bank.

``backend``
   A shared backend profile or role-specific ``response``, ``endowment``, and
   ``attribute_learner`` mappings.

   ``type: deepseek`` uses the provider-specific V4 adapter.  It reads
   ``DEEPSEEK_API_KEY`` and defaults to the official API endpoint.  Its
   ``kwargs`` accept ``thinking: enabled|disabled``, optional
   ``reasoning_effort: high|max``, ``temperature``, ``max_tokens``, and an
   optional ``base_url`` override.  Reasoning effort is valid only when
   thinking is enabled; temperature is deliberately omitted from thinking
   requests.  ``config/backends/deepseek.yaml`` is a non-thinking profile for
   forced-choice survey answers.

   ``type: openai`` accepts ``transport: chat_completions|responses``
   (default ``chat_completions``; never switched silently), ``store``
   (default ``false``: both APIs may store responses by default, and P2P
   never retrieves them), ``temperature``, and ``max_tokens``.  ``max_tokens``
   is P2P's provider-neutral output limit: it is sent as
   ``max_completion_tokens`` to Chat Completions and ``max_output_tokens`` to
   Responses, and both bound visible plus reasoning tokens.  With Responses,
   the system prompt is sent as ``instructions`` and the question as
   ``input``.

   ``type: simulated`` answers P2P's Stage-1 prompts by deterministic rules
   from a declarative ``world`` (attributes whose values prefer answer options);
   it is not a language model. It lets the whole pipeline run offline for
   demonstrations and tests (``config/backends/simulated_demo.yaml``); usage is
   zero. Its ``kwargs`` are ``world``, ``seed`` and ``noise``.

   ``type: gemini`` uses the ``google-genai`` SDK and accepts
   ``temperature`` and ``max_tokens`` (sent as ``max_output_tokens``).
   ``model_name`` has no default: Gemini models are retired on a published
   schedule, so every configuration names its model.

   A refusal, a blocked prompt, or a reply that ends without any text raises
   ``NoAnswerTextError`` naming the provider's stated reason (refusal text,
   ``finish_reason``, ``block_reason``, or incomplete status).  Partial text
   from a reply cut off at the token limit is returned unchanged.

``model``
   Regression method and hyperparameters: constrained ``lasso`` or
   ``elasticnet`` (optionally block-weighted), or ``simple_average``. See
   "Stage-2 model options" below.

``split``
   Question labels used for train/validation and test plus
   ``use_proxy_only``.

``metrics`` and ``evaluation``
   Requested metric names (``mse``, ``distribution_mse``, ``one_minus_jsd``,
   ``emd``, ``mcpa``; all scored on the test split) and explicit
   prediction/invalid-distribution
   policies.

``aggregate``
   Locked run list, asserted labels, grouping, expected completeness,
   hierarchy, summary statistics, and export formats.

``simulation``, ``ablate``, and ``panel``
   Sweep declarations and dotted or nested overrides.

``cluster``
   Offline clustering algorithm and number of clusters.

Verb requirements
-----------------

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Verb
     - Required contract
   * - ``generate``
     - Survey, generation strategy, and required backend roles.
   * - ``pipeline``
     - Generation contract plus target, split, model, and metrics.
   * - ``regress``
     - Survey, cached basis, target, split, model, and metrics.
   * - ``evaluate``
     - Survey, target, split, metrics, and either predictions or fitted input.
   * - ``aggregate``
     - Parsed ``aggregate`` specification.
   * - ``simulate``
     - ``simulation`` mapping; backend/generation only when
       ``invokes_generation`` is true.
   * - ``ablate`` / ``panel``
     - Corresponding sweep mapping.
   * - ``cluster``
     - ``data.basis`` and ``cluster``; checked by the cluster command.

Minimal pipeline example
------------------------

.. code-block:: yaml

   run:
     name: example_pipeline
     seed: 101

   data:
     survey:
       source: ../../tests/fixtures/data/survey.csv
       schema: ../../tests/fixtures/data/survey_schema.yaml
     target: ../../tests/fixtures/commands/pipeline/target.json

   generation:
     mode: vanilla
     target_n: 2
     batch_size: 2

   backend:
     response:
       type: dummy
       model_name: dummy
       kwargs: {answer_text: Option A}
     endowment:
       type: dummy
       model_name: dummy
       kwargs:
         answer_text: >-
           [{"eid": "a", "endow_text": "Synthetic A"},
            {"eid": "b", "endow_text": "Synthetic B"}]

   model:
     method: lasso
     alpha: [0.01]
     validation: {strategy: cv, cv_folds: 2}

   split:
     train_val: [train, valid]
     test: test
     use_proxy_only: true

   metrics: [mse, one_minus_jsd, emd, mcpa]
   evaluation:
     mse_prediction_policy: raw
     distribution_prediction_policy: clip_and_normalize
     invalid_distribution: error

Structured grids
----------------

Literal lists and safe structured grids replace evaluated NumPy expressions:

.. code-block:: yaml

   model:
     alpha:
       logspace: [-4, 0, 10]
     l1_ratio:
       linspace: [0.1, 0.9, 5]

Arbitrary expressions are rejected.

Stage-2 model options
---------------------

``model.method`` is ``lasso``, ``elasticnet`` or ``simple_average``. For the two
fitted methods, the options below are read; an unknown option is refused with a
spelling suggestion.  Ridge is elastic net with ``l1_ratio: [0.0]``; because its
many small weights are all meaningful, give it a small ``zero_threshold`` (the
Studio writes ``1e-10``).

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Option
     - Default
     - Meaning
   * - ``alpha``
     - ``{logspace: [-5, 5, 30]}``
     - Penalty grid. When omitted, P2P's default grid is used and the run's
       ``diagnostics.json`` records ``default_grids``. A selection at either end of
       the grid is reported (``alpha_at_grid_edge``).
   * - ``l1_ratio``
     - ``{linspace: [0.1, 1, 10]}``
     - Elastic net only: the lasso share of the penalty (1 is pure lasso).
   * - ``validation.strategy``
     - ``cv``
     - ``cv``: K-fold cross-validation grouped by question. ``holdout``: each
       penalty is fitted on the other training splits and scored on
       ``validation.holdout_split`` (default ``valid``); the best fit is kept
       without refitting.
   * - ``validation.cv_folds``
     - ``5``
     - Number of folds (``cv`` only).
   * - ``validation.folds``
     - ``{method: hashed}``
     - Fold assignment (``cv`` only). ``hashed`` is seeded by ``run.seed`` and
       identical on every machine; ``sklearn`` uses scikit-learn's
       ``GroupKFold`` (machine-dependent); ``{method: replay, file: …}`` replays a
       recorded ``folds.json``. Every run writes its own ``folds.json``.
   * - ``group_penalty_strength``
     - ``100``
     - Strength of the soft per-question mass penalty, which pulls each
       question's predicted mass towards ``group_total`` (default 1). ``0`` switches
       it off.
   * - ``reference_respondent``
     - ``false``
     - Adds a basis column that answers every option of a *k*-option question
       with 1/*k*, weighted like an agent (``reference_respondent_weight`` in the
       diagnostics).
   * - ``intercept``
     - ``false``
     - Adds an unpenalised constant on every option row; the mass penalty sees
       the *k* × intercept it adds to a question.
   * - ``block_weighting``
     - ``false``
     - Weights each question equally in the fit, whatever its number of options.
   * - ``zero_threshold``
     - ``1e-5``
     - Coefficients smaller in magnitude are treated as zero.
   * - ``soft_group_sum``, ``sum_to_one``, ``nonneg``
     - ``true``, ``false``, ``false``
     - Hard forms of the constraints. Signed weights (anti-type agents) are
       intended, so ``nonneg`` is off by default.

The six combinations of the mass penalty (on or off) with the reference
respondent, the intercept or neither are the Stage-2 configurations Studio offers
by name. ``max_iter`` and ``post_selection_refit`` are accepted so older
manifests stay readable, but they have no effect and are recorded as
``ignored_model_options``.

The solver is a command-line choice, not a manifest option:
``p2p regress --solver OSQP`` (default ``CLARABEL``). If a solve fails, P2P falls
back in the order Clarabel → OSQP → SCS and records the solver it used.

Reference distribution
----------------------

``data.reference_distribution`` names a JSON file in the target format, for
example an earlier survey wave's marginals. It enters every prediction with
weight one:

.. code-block:: yaml

   data:
     target: targets_w6.json
     reference_distribution: targets_w7.json   # prediction = reference + agents

The agents then explain each target's *deviation* from the reference rather than
its level, which is the drift refit across survey waves. The file is checked like
the targets (every survey question present, each distribution non-negative and
summing to one), and each question's reference mass over its modelled option rows
must be one. The mass penalty keeps its meaning: each question's full prediction
is pulled towards ``group_total``, so when every agent answers every question the
agents' weights sum to zero. ``soft_group_sum: false`` imposes this exactly.

Predictions are full distributions, so every metric is computed as usual, and the
run records the file and its digest (``reference_distribution_digest`` in the
diagnostics). Because the reference's weight is fixed at one, it cannot be combined
with ``intercept``, ``reference_respondent``, ``sum_to_one``, ``nonneg``,
``method: simple_average`` or ``evaluation.distribution_route: answer_file``;
such a manifest is refused.

Evaluation policies
-------------------

Raw MSE always scores the raw predictions (``mse_prediction_policy: raw``).
The distributional metrics (rescaled MSE, 1 − JSD, EMD, MCPA) score each
question's prediction after ``distribution_prediction_policy`` has turned it into
a distribution:

``clip_and_normalize`` (default)
   Set negative values to zero, then rescale each question to sum to one.
``project_to_simplex``
   Replace each question's prediction by the closest probability distribution
   (Euclidean projection).
``signed_sum_normalize``
   Divide by the signed sum (first by 100 if that sum exceeds 1.5), keeping
   negative values, and score with SciPy's routines, which renormalise their
   inputs.  ``legacy_signed_normalize`` is its earlier spelling and is still
   accepted.
``validate``
   Score predictions unchanged; one that is not already a distribution is
   invalid.

``invalid_distribution`` decides what happens to a question whose prediction is
still not a distribution: ``error`` (default) stops the evaluation; ``allow``
leaves it out of the distributional metrics and records how many were left out.

``distribution_route`` chooses where the distributional metrics' predictions
come from: ``model`` (default, the model's own predictions, as MSE uses) or
``answer_file`` (the agents' answer file, re-aggregated by agent name with the
fitted weights;
``legacy_answer_file`` is its earlier spelling and is still accepted).

Every metric record names the policy and route it was computed under, and the
earlier spellings are recorded exactly as written, so existing manifests keep
their digests.

No-regression baseline
----------------------

``method: simple_average`` weights every agent of the cleaned proxy panel equally
instead of fitting weights. It uses the same loader, cleaning and evaluation as a
fitted model, so the comparison isolates the regression step:

.. code-block:: yaml

   model:
     method: simple_average

   evaluation:
     distribution_prediction_policy: validate
     invalid_distribution: error

It takes no other model options; a penalty grid, validation settings, the
reference respondent or an intercept are refused rather than ignored. The fit
also refuses a design in which any agent lacks exactly one answer on a question,
so every question is averaged over the same agents.

Grouped targets
---------------

Existing flat target files remain ``{qid: {label: probability}}``.  A grouped
target is an explicit schema:

.. code-block:: json

   {
     "schema_version": 1,
     "default_subgroup": "overall",
     "subgroups": {
       "overall": {"Q1": {"yes": 0.5, "no": 0.5}},
       "party.democrat": {"Q1": {"yes": 0.7, "no": 0.3}}
     }
   }

Select a named slice without changing the survey, basis, split, model, or
metric kernels:

.. code-block:: yaml

   data:
     target:
       path: targets_by_subgroup.json
       subgroup: party.democrat

With multiple groups, selection must be explicit or the file must declare
``default_subgroup``.  The loader validates the selected distributions
without repair and records both the complete source digest and selected-slice
digest.

Scientific simulations
----------------------

``simulation.executor: scientific`` enables the maintained offline executor.
It requires the normal cached-basis ``data``, ``split``, and ``model``
sections.  A typical proxy-fraction sweep is:

.. code-block:: yaml

   simulation:
     executor: scientific
     kind: proxy
     invokes_generation: false
     target_source: configured
     ground_truth:
       source: existing
     fraction: 1.0
     sweep:
       target: simulate
       repeats: 3
       overrides:
         - simulation: {fraction: 0.25}
         - simulation: {fraction: 0.5}
         - simulation: {fraction: 1.0}

Kinds are ``endowment``, ``response``, ``proxy``, ``trainvalid``,
``entropy_modes``, and ``target_entropy``.  Fixed simulated targets use
``target_source: ground_truth`` plus a ground-truth assignment with
``source: existing``, ``random``, or ``modes``.  ``entropy_modes`` requires
``included_mode_count`` and ``n_proxies``.  The new entropy-target estimand is
opt-in:

.. code-block:: yaml

   simulation:
     executor: scientific
     kind: target_entropy
     sample_size: 30
     target_mean_entropy: 0.6
     max_evaluations: 5000

It uses the same mean normalized categorical entropy that active generation
tracks, but is never called by active generation.

Attribute banks
---------------

An active-generation bank may be a packaged resource or manifest-relative
file:

.. code-block:: yaml

   generation:
     mode: active
     attribute_bank:
       source: builtin:default

The loader returns immutable data.  Generation deep-copies it into run-local
state before adding survey- or question-specific attributes.

Backend profiles and secrets
----------------------------

Reusable backend mappings can live under ``config/backends/`` and be
referenced by path.  Keep credentials out of YAML.  Gemini can read
``GEMINI_API_KEY``; the OpenAI client follows its standard environment
configuration.  Ollama uses a local ``base_url`` and needs no API key.

Budget pricing
--------------

Provider prices are intentionally separate from central scientific
configuration because they change independently.  ``p2p budget --pricing``
accepts:

.. code-block:: yaml

   models:
     provider-model-name:
       input_per_million_usd: 1.0
       output_per_million_usd: 4.0
       cached_input_per_million_usd: 0.25

All values are nonnegative USD per one million tokens.  Copy
``config/pricing.example.yaml`` and replace its illustrative zeros with the
provider's current published rates.  The resolved source is recorded beside
each forecast.  P2P does not treat effective rates from earlier runs as
current prices.

Earlier-format configurations
-----------------------------

:func:`p2p.config_conversion.convert_earlier_config` converts regression,
pipeline, ablation, panel and simulation YAML written in the earlier
configuration format into the central contract.  It is a one-off conversion
helper, not a promise that every earlier key is accepted.  Review the result
before a paid or large local run.  See :doc:`earlier_formats`.
