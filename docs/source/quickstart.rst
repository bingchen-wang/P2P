Quickstart
==========

Install
-------

P2P needs Python 3.12 or later.  From the repository root:

.. code-block:: console

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[dev]"
   p2p --version

Add ``docs`` to the extras to build this guide, and ``openai``, ``gemini`` or
``deepseek`` for provider backends.  ``environment/`` pins the exact package
versions used for the recorded results; see ``environment/README.md``.

Run the offline demo
--------------------

The demo runs the whole pipeline on a small fictional survey.  Stage 1 uses the
rule-based ``simulated`` backend instead of a language model, so it needs no
network, no credentials and no money:

.. code-block:: console

   p2p pipeline \
     --config config/smoke/studio_demo_pipeline.yaml \
     --output outputs/demo_pipeline

The output directory holds the generated personas (``endowments.csv``), their
answers (``responses.csv``), the fitted metrics, the resolved configuration,
usage records and a log; see :doc:`artifacts`.

To do the same in the browser, build the Studio once and choose **Run the
offline demo** on its home page (see :doc:`studio`):

.. code-block:: console

   cd studio && npm ci && npm run build && cd ..
   p2p ui

Run with a local model
----------------------

With `Ollama <https://ollama.com>`_ installed and ``gemma3:4b`` pulled, the
same fictional survey runs with a real model for every Stage-1 role, at no
cost:

.. code-block:: console

   p2p pipeline \
     --config config/smoke/ollama_demo.yaml \
     --output outputs/ollama_demo

Validate a manifest
-------------------

Check a manifest against a command's contract before running it:

.. code-block:: console

   p2p validate \
     --config config/central/pipeline_fixture.yaml \
     --verb pipeline

A valid manifest prints nothing and exits with status 0; an invalid one stops
with an error naming the problem.

Reuse a proxy basis
-------------------

Stage 2 runs on its own against saved responses and personas:

.. code-block:: console

   p2p regress \
     --config config/central/regress_fixture.yaml \
     --output outputs/regress_fixture

Point ``data.basis`` at an existing run directory (or at explicit response and
persona files), change only what you mean to test, and write to a new
directory.  Refitting costs nothing.

Budget an API-backed run
------------------------

``p2p budget`` forecasts calls, tokens and cost offline, learning from the
usage recorded by earlier runs under ``outputs/``.  It needs at least three
earlier runs that used each model module, for example three small pilots:

.. code-block:: console

   p2p budget \
     --config config/smoke/ollama_demo.yaml \
     --output outputs/budget/ollama_demo

It writes JSON and CSV records and a self-contained ``budget.html``.  Local
backends are priced at zero; pass ``--pricing`` with a price file for dollar
estimates.  ``--pilot-run`` calibrates the forecast on a small or in-progress
run of the same plan.

Inspect a result
----------------

.. code-block:: console

   p2p inspect \
     --run outputs/regress_fixture \
     --advanced

This writes a self-contained ``diagnostics.html`` inside the run.  During a
longer pipeline, ``--serve --refresh 2`` rebuilds the view as artifacts appear.
The Studio's run page shows the same evidence.

Verify the installation
-----------------------

.. code-block:: console

   python -m pytest -q

The default suite is offline and excludes tests that call paid APIs.  Tests that
need licensed survey data skip automatically, because that data is not
distributed.

Next, read :doc:`concepts` for the model behind the commands, or :doc:`cli` for
the full command surface.
