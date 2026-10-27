Artifacts and reproducibility
=============================

Run records
-----------

A completed pipeline normally writes:

.. code-block:: text

   active_generation.checkpoint/
   endowments.csv
   responses.csv
   attribute_bank.resolved.yaml
   aggregate_entropy_trajectory.csv
   persistent_mode_entropy.csv
   question_entropy_trajectories.csv
   mode_summary.csv
   usage.csv
   usage.json
   pipeline.log
   config.resolved.yaml
   run.json
   metrics.json
   metrics.csv
   diagnostics.json

``run.json`` records the run's identity, its source files and their digests,
the split and model labels, the cleaning policy and the evaluation policy.
``metrics.json`` and ``metrics.csv`` hold one record per metric, with its split,
unit, prediction policy and question coverage.  A regression also writes
``diagnostics.json`` with the coefficients and each question's target and
predicted distributions; ``p2p inspect`` reads it, without loading a pickled
model, and writes ``diagnostics.html`` on request.

With active generation, ``active_generation.checkpoint/`` is what lets a paid
run recover.  It holds the saved generation state plus readable partial
personas, responses and the learned attribute bank, and it is tied to the
digest of the resolved configuration.  Rerunning the same command with the same
output directory restores the completed response chunks and requests only the
missing personas.  If the digest differs, or the directory has paid usage
records but no matching checkpoint, P2P stops rather than mixing runs.

Simulation sweep artifacts
--------------------------

Each simulation in a ``p2p simulate`` sweep writes ``simulation.json`` and
``simulation.csv``.  The sweep's top folder writes ``simulation_summary.json``,
``simulation_summary.csv`` and a long-format ``simulation_aggregate.csv`` with
means, sample standard deviations and counts.  A value with an empty
denominator is recorded as null rather than infinity.

Aggregation artifacts
---------------------

Aggregation from a manifest (``p2p aggregate --config``) writes the resolved
specification, ledgers of the inputs and their validation, one row per run,
the summary rows and the requested table formats.  Every reported cell can
therefore be traced back to the run and metric records it came from.

Budget artifacts
----------------

``p2p budget`` writes:

.. code-block:: text

   budget.json
   budget.csv
   budget_history.csv
   budget.html

``budget.json`` holds the full forecast: the digests of the configuration
and the usage history, the features and uncertainty level used, a model card
for each module, where the prices came from, any pilot calibration, the totals
for each scenario and the budget status.  ``budget.csv`` is the same forecast
as a flat table by module and scenario.  ``budget_history.csv`` lists the
earlier runs the forecast learned from (usage only, no prompts).
``budget.html`` presents the same numbers as a self-contained page.

API-backed generation and elicitation append a row to ``usage.csv`` as each
call completes.  When the command finishes, P2P rewrites the file in its final
form and writes ``usage.json``.  ``reasoning_tokens`` (usage schema version 2)
is the part of ``output_tokens`` spent on reasoning; it is reported separately
and not counted twice in the cost.  A resumed command keeps the existing rows
and appends new ones.

Reproduction checklist
----------------------

#. Keep the source run unchanged; write every re-analysis to a new directory.
#. Validate the central manifest for the intended verb.
#. Pin the cleaning and prediction policies explicitly.
#. Keep the resolved configuration and its digest.
#. Keep each run's ``run.json``, ``metrics.json`` and ``metrics.csv``.
#. Build reported tables with ``p2p aggregate --config``.
#. Use the exact environment in ``environment/`` and the same solver
   (``--solver``) as the run being reproduced.
