Command-line interface
======================

The installed ``p2p`` command and ``python -m p2p`` use the same parser.

.. code-block:: console

   p2p --version
   p2p --help
   p2p <command> --help

Common options
--------------

Most commands accept:

``--config PATH``
   Required central YAML manifest.  Referenced paths are resolved before
   dispatch.

``--output PATH``
   Required destination for newly written artifacts.  Use a new directory for
   each run.

``--verbose``
   Enable stage progress and informational logging.

``--solver NAME``
   CVXPY solver for workflows that fit Stage 2 (``pipeline``, ``regress``);
   defaults to ``CLARABEL``.  If a solve fails, P2P falls back to OSQP, then
   SCS, and records the solver it used.  Pass ``OSQP`` to reproduce fits made
   with it.

Commands
--------

``generate``
   Generate active, vanilla, or injected-plugin endowments.

   .. code-block:: console

      p2p generate \
        --config config/central/generate_active_fixture.yaml \
        --output outputs/generate_fixture

``elicit``
   Elicit a survey from an existing ``endowments.csv`` rather than generating
   a new basis.

   .. code-block:: console

      p2p elicit \
        --config config/central/elicit_fixture.yaml \
        --endowments outputs/source/endowments.csv \
        --output outputs/elicit_fixture

``pipeline``
   Run generation, elicitation, Stage 2, evaluation, and canonical reporting.

   .. code-block:: console

      p2p pipeline \
        --config config/central/pipeline_fixture.yaml \
        --output outputs/pipeline_fixture \
        --solver OSQP --verbose

``regress``
   Run Stage 2 from cached responses/endowments and target marginals.

   .. code-block:: console

      p2p regress \
        --config config/central/regress_fixture.yaml \
        --output outputs/regress_fixture \
        --solver OSQP

``evaluate``
   Evaluate a self-contained prediction JSON or fitted artifact without
   refitting.

   .. code-block:: console

      p2p evaluate \
        --config config/central/evaluate_fixture.yaml \
        --output outputs/evaluate_fixture

``aggregate``
   Validate and summarize canonical run directories.  Locked mode uses an
   aggregation manifest:

   .. code-block:: console

      p2p aggregate \
        --config config/central/aggregate_fixture.yaml \
        --output outputs/aggregate_fixture

   Quick mode accepts explicit runs and requires grouping fields and metrics
   appropriate to those run records:

   .. code-block:: console

      p2p aggregate \
        --runs outputs/study/run_1 outputs/study/run_2 \
        --group-by model split \
        --metrics test_mse_raw one_minus_jsd \
        --output outputs/study/_summary

``budget``
   Forecast calls, tokens, and cost before an API-backed ``generate`` or
   ``pipeline`` command.  Forecasting is offline and learns from the usage
   recorded by earlier runs: ``--history`` names their folders (default
   ``outputs/``), and each model module needs at least three earlier runs.

   .. code-block:: console

      p2p budget \
        --config config/smoke/ollama_demo.yaml \
        --for-command pipeline \
        --budget-usd 1 \
        --compare-model gemini-2.0-flash \
        --output outputs/budget/ollama_demo

   ``--pilot-run`` calibrates per-call tokens from a small or in-progress
   run.  ``--serve --refresh 2`` rereads its streaming ``usage.csv`` and
   updates the dashboard.  ``--pricing`` accepts a current price file; without
   it, local backends are zero-cost, models seen in earlier runs use their
   labelled effective rates, and unknown paid models remain unpriced.

``simulate``, ``ablate``, and ``panel``
   Expand deterministic overrides/repeats and dispatch child commands.
   Simulation manifests are declaration-only by default.  Setting
   ``simulation.executor: scientific`` runs the cached-basis executor and
   writes child plus cross-repeat summaries.

   .. code-block:: console

      p2p simulate \
        --config config/central/simulate_scientific_fixture.yaml \
        --output outputs/simulate_fixture

      p2p ablate \
        --config config/central/ablate_fixture.yaml \
        --output outputs/ablate_fixture \
        --solver OSQP

      p2p panel \
        --config config/central/panel_fixture.yaml \
        --output outputs/panel_fixture \
        --solver OSQP

``inspect``
   Build a self-contained interactive HTML report, or serve a refreshing
   local view while a run is still in progress.  This command does not use a
   central manifest.

   .. code-block:: console

      p2p inspect \
        --run outputs/pipeline_fixture \
        --advanced

      p2p inspect \
        --run outputs/long_pipeline \
        --serve --refresh 2

   ``--advanced`` adds singular-value and multiple-correspondence views.
   The server binds to ``127.0.0.1:8050`` by default.

   Runs written in the earlier result format are read directly when their
   ``regression_config.yaml`` and referenced CSV/YAML/JSON inputs remain
   available.  Write the page outside the source run so it stays read-only:

   .. code-block:: console

      p2p inspect \
        --run path/to/earlier_run \
        --output outputs/inspect/earlier_run.html \
        --advanced

   The reader reconstructs the selected weighted-response prediction from the
   exported weights and never unpickles a serialized model.  See
   :doc:`earlier_formats`.

``cluster``
   Cluster existing endowment text deterministically with TF-IDF + K-means.
   This command enforces its own ``data.basis`` and ``cluster`` contract.

   .. code-block:: console

      p2p cluster \
        --config config/central/cluster_fixture.yaml \
        --output outputs/cluster_fixture

``validate``
   Resolve a manifest and validate it against a verb-scoped contract without
   running the workflow.

   .. code-block:: console

      p2p validate \
        --config config/central/regress_fixture.yaml \
        --verb regress

   The central validator currently supports ``generate``, ``pipeline``,
   ``regress``, ``evaluate``, ``aggregate``, ``simulate``, ``ablate``, and
   ``panel``.  ``elicit`` and ``cluster`` check their manifests when they run.

``ui``
   Start the local Experiment Studio (see :doc:`studio`).  It serves
   ``http://127.0.0.1:5173`` by default; ``--port`` and ``--host`` change that
   and ``--no-open-browser`` skips opening a browser.

   .. code-block:: console

      p2p ui

Exit behavior
-------------

Successful commands return exit code zero.  Invalid configuration, missing
artifacts, registry failures, incompatible aggregation inputs, and backend
errors fail loudly rather than creating a record that claims success.
