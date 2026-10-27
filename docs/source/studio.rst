Experiment Studio
=================

The Studio is a local browser interface for P2P: the one place to set up
experiments, follow them while they run and read their results.  Everything it
does goes through the same commands and files as the CLI.

Start it
--------

Build the interface once (Node.js 22.13 or later), then start it from the
repository root:

.. code-block:: console

   cd studio && npm ci && npm run build && cd ..
   p2p ui

``p2p ui`` serves ``http://127.0.0.1:5173`` and opens it in your browser
(``--no-open-browser`` to skip that, ``--port`` to change the port).  The server
binds to the loopback interface, reads and writes only this workspace, and has
no upload or publishing feature.  Provider backends make calls only when you
run an experiment that uses them.

The Dock
--------

Six sections sit in the Dock at the top of the window (at the bottom on a phone):

Home
   The latest result and its fitted ensemble, recent runs, and the next steps.
   A new workspace starts with three: run the offline demo, add your data,
   build an experiment.

New experiment
   A builder that follows the workflow in steps (Experiment, Data, Stage 1,
   Stage 2, Evaluation, Review).  The step rail at the top marks each step
   complete as you fill it in and jumps to it.  Only the steps the chosen
   workflow needs appear: a fit of an existing basis has no Stage 1.

Runs
   Live and recent runs started from the Studio sit at the top; select one for
   its status, process controls, *Duplicate setup* and its process log.  Below,
   the library lists every run under ``outputs/`` with search, sorting by
   metric, labels, pinning and side-by-side comparison of up to four runs.

Budgets
   Cost forecasts written by ``p2p budget``: scenarios, uncertainty and the
   modules that drive cost.

Data
   Inspect a survey CSV, map its columns, check its splits and the target's
   coverage, and register it as a reusable dataset.

Backends
   Every model backend with its options, dependencies and credential status.
   *Check* validates a profile locally without calling the provider.

Building an experiment
----------------------

The builder writes a manifest to ``config/experiments/studio/<name>.yaml`` and
runs the same command as the CLI; the sidebar shows the equivalent command line.

* **Data**: survey CSV, schema YAML and target JSON.  For a grouped target file
  (one distribution per subgroup), name the subgroup; preflight lists the
  file's subgroups and refuses an unknown or missing one.
* **Stage 1**: generation strategy, proxy count, backend profile and, for
  active generation, the attribute bank and update steps.
* **Stage 2**: method (lasso, elastic net, ridge or a simple average) and
  configuration (mass penalty on or off, each with no uniform component, a
  reference respondent or an intercept).  An optional **reference
  distribution** re-anchors the fit: the proxies then explain each target's
  change from the reference.  Advanced settings cover penalty selection,
  folds, grids, exact question mass, the zero threshold, block weighting and
  the solver.
* **Evaluation**: metrics, the distribution policy and what happens to a
  prediction that is still not a distribution.
* **Review**: the generated YAML.  *Review preflight* validates the manifest
  exactly as the command will, reports its digest and any warnings, and only
  then allows *Create YAML & run*.

Reading a run
-------------

A run opens on **Ensemble ↔ Evidence**: every generated proxy drawn as a figure
coloured by template family, sized by the magnitude of its weight, striped when
the weight is negative, and dimmed when the fit did not retain it.  Selecting a
proxy shows its persona and coefficient; the evidence side shows each held-out
question's predicted and target distributions.

The other tabs are the **Run record** (metrics, health checks, provenance and
downloads), the **Prediction explorer** (predicted against target shares for
train, validation and test, with a question drill-down), the **Entropy
explorer** (how adaptive generation explored the modes) and **Proxy weights**.

Earlier-format runs
-------------------

The Studio also reads runs written in the earlier result format, read-only, and
shows them with an *earlier format* badge.  It finds them under
``outputs/paper/``; result folders with Markdown summaries and figures appear
under *Result folders*.  See :doc:`earlier_formats`.
