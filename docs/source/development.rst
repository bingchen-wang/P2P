Development
===========

Environment
-----------

Any Python 3.12+ environment works:

.. code-block:: console

   python -m pip install -e ".[dev,docs]"

``environment/`` pins the exact versions used for the recorded results (a conda
lock for macOS on Apple Silicon and a pip lock); see ``environment/README.md``.

Tests
-----

The default pytest configuration excludes the ``paid`` marker and is fully
offline:

.. code-block:: console

   python -m pytest

Useful focused gates:

.. code-block:: console

   python -m pytest tests/test_central_manifests.py -q
   python -m pytest tests/test_studio.py tests/test_studio_stage2.py -q

The golden-parity tests (``tests/test_golden_parity.py``) compare Stage 2 with
recorded results on licensed survey data; without that data they skip.

The local Ollama integration is opt-in:

.. code-block:: console

   P2P_RUN_OLLAMA=1 python -m pytest tests/test_ollama_integration.py -q

The live DeepSeek check is metered, requires ``DEEPSEEK_API_KEY``, and is
excluded from the default suite:

.. code-block:: console

   P2P_RUN_DEEPSEEK=1 \
     python -m pytest -m paid tests/test_deepseek_integration.py -q

Documentation
-------------

Build with warnings promoted to errors:

.. code-block:: console

   sphinx-build -W -b html docs/source docs/build/html

Generated HTML lives under the gitignored ``docs/build/html`` directory.

Studio
------

The Studio's source is under ``studio/`` (React, TypeScript, Vite).  Build it
with ``npm ci && npm run build``; ``p2p ui`` serves ``studio/dist``.  Type-check
with ``npx tsc --noEmit -p tsconfig.json``.  The server side is
``p2p/studio.py``; its capabilities endpoint reads the live registries, so the
builder offers exactly the methods, metrics and backends P2P supports.

Extension points
----------------

Registries
   Backends, models, stages, commands, and metrics use
   :class:`p2p.registry.Registry`.  Registration rejects duplicate exact names.
   Import the defining module from the appropriate package initializer so the
   registration runs before CLI dispatch.

Backends
   Implement the keyword-only ``chat(system=..., user=...)`` contract and
   return :class:`p2p.backends.base.BackendResponse` with normalized
   :class:`p2p.backends.base.TokenUsage`.  Usage is accounted by the logical
   caller, not inside multiple wrapper layers.

Commands and stages
   Put reusable implementation in ``p2p.stages`` and user-facing composition
   in ``p2p.commands``.  Commands receive resolved :class:`p2p.config.Config`
   objects and explicit destinations.

Metrics
   Keep kernels pure.  Add explicit validation/transformation policy in the
   evaluation layer and record it in canonical metric artifacts.

Configuration
   Add schema validation to the verb requirement matrix, include a fixture
   manifest, and test both valid and invalid cases.  Do not add evaluated
   expressions or behavior selected by script filename.

Result compatibility
   Any change to result records, cleaning, splits, prediction policies, or
   aggregation must update the corresponding validation and parity tests.

Repository layout
-----------------

.. code-block:: text

   p2p/            package and registries
   p2p/commands/   user-facing workflow composition
   p2p/stages/     reusable pipeline stages
   config/         central manifests, smoke runs and reusable profiles
   examples/       a fictional demo survey, basis and targets
   studio/         the Experiment Studio (browser interface)
   docs/source/    narrative guide and API reference
   docs/assets/    logos used by the README
   environment/    pinned environments for exact reproduction
   outputs/        run outputs (gitignored)
   tests/          unit, integration and parity tests
