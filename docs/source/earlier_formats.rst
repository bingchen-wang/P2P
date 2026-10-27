Earlier formats
===============

P2P reads results and configurations written in the earlier result format, so
work done with it stays readable.  Nothing is migrated or rewritten to be
viewed: the readers below are read-only.

Earlier-format runs
-------------------

An earlier-format run directory holds ``regression_config.yaml``, the survey,
target and response files it references, exported weights
(``updated_endowments.csv``), ``summary_aggregate.csv``,
``token_usage_summary.csv`` and archived figures.

``p2p inspect --run <directory>`` reads such a run and reconstructs the
selected model's question-level predictions from the survey, the target, the
responses and the exported weights.  It imports the summary and usage tables
and embeds the archived prediction and diagnostic figures.  It never imports old
Python modules or loads a ``joblib`` pickle, and a run missing some inputs
still shows every view its files support and names what is missing.  Write the
page outside the source run:

.. code-block:: console

   p2p inspect \
     --run path/to/earlier_run \
     --output outputs/inspect/earlier_run.html \
     --advanced

The Studio lists earlier-format runs found under ``outputs/paper/`` with an
*earlier format* badge, and result folders with Markdown summaries and figures
under *Result folders*.  Their run pages show the same views as current runs,
plus an *Original report files* tab.

The Python reader is :func:`p2p.earlier_runs.load_earlier_run`.

Earlier-format configurations
-----------------------------

:func:`p2p.config_conversion.convert_earlier_config` converts regression,
pipeline, ablation, panel and simulation YAML written in the earlier
configuration format into a central manifest.  It replaces evaluated NumPy
expressions with structured grids and separates persona-generation settings
(batching, attribute caps) from backend arguments.  Review the result before a
paid or large local run.

Cleaning policies
-----------------

Every run records how incomplete proxy answers were handled.  P2P fits with
``column_drop_v2``, which drops proxies that did not answer every question.
Earlier-format runs may carry ``row_drop_v1``, which instead dropped any
question row with an incomplete proxy answer.  Aggregation refuses to pool runs
with different cleaning policies.

Earlier option spellings
------------------------

Two option values have behaviour-named spellings; the earlier ones are still
accepted and are recorded exactly as written, so existing manifests run
unchanged and keep their configuration digests:

.. list-table::
   :header-rows: 1
   :widths: 36 32 32

   * - Option
     - Name
     - Earlier spelling (accepted)
   * - ``evaluation.distribution_prediction_policy``
     - ``signed_sum_normalize``
     - ``legacy_signed_normalize``
   * - ``evaluation.distribution_route``
     - ``answer_file``
     - ``legacy_answer_file``

``max_iter`` and ``post_selection_refit`` in a model block are also accepted,
have no effect, and are recorded as ``ignored_model_options``.

Target renormalisation
----------------------

``data.target.renormalize: true`` divides each target vector by its own sum
(first by 100 if the sum exceeds 1.5) before scoring, so a target whose shares
do not quite sum to one can still be scored; an all-zero target is recorded as
degenerate instead of raising.  The fit sees the target block as given and only
the scoring sees it rescaled.  It is off by default: without it, a target that
is not a distribution is refused.
