:hide-toc:

.. raw:: html

   <div class="p2p-spots" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div>
   <p class="p2p-eyebrow">Prompts to Proxies &middot; v0.1.0</p>

P2P documentation
=================

.. raw:: html

   <div class="p2p-hero">
     <p class="p2p-lede">P2P reconstructs how a human population answers a survey by
     building a pool of LLM proxy respondents and weighting them so their combined
     answers match the population's published answer shares. It needs only aggregate
     survey results, not respondent-level demographic data or model fine-tuning.</p>
     <div class="p2p-actions">
       <a class="p2p-button p2p-button-primary" href="quickstart.html">Get started</a>
       <a class="p2p-button" href="studio.html">Experiment Studio</a>
       <a class="p2p-button" href="https://github.com/bingchen-wang/P2P">GitHub</a>
     </div>
   </div>

One YAML configuration system and one command-line interface cover persona
generation, survey elicitation, constrained regression, evaluation, sweeps,
clustering and aggregation.  The Experiment Studio (``p2p ui``) puts the same
workflow in a local browser interface.

The quickstart, the fixture workflows and the test suite run offline.  Provider
backends (OpenAI, Gemini, DeepSeek) are optional, and Ollama runs the real
pipeline on your own machine.

Where to start
--------------

.. raw:: html

   <div class="p2p-cards">
     <a class="p2p-card" href="quickstart.html"><span class="p2p-card-step">01</span><strong>Quickstart</strong><span class="p2p-card-copy">Install P2P and run the whole pipeline offline on a fictional survey.</span></a>
     <a class="p2p-card" href="concepts.html"><span class="p2p-card-step">02</span><strong>Concepts</strong><span class="p2p-card-copy">The two stages, the basis and target, and what every run records.</span></a>
     <a class="p2p-card" href="studio.html"><span class="p2p-card-step">03</span><strong>Experiment Studio</strong><span class="p2p-card-copy">Build, run and read experiments in a local browser interface.</span></a>
     <a class="p2p-card" href="cli.html"><span class="p2p-card-step">04</span><strong>Command line</strong><span class="p2p-card-copy">Every <code>p2p</code> command, its options and examples.</span></a>
     <a class="p2p-card" href="configuration.html"><span class="p2p-card-step">05</span><strong>Configuration</strong><span class="p2p-card-copy">The YAML manifest: data, Stage 1, Stage 2 and evaluation.</span></a>
     <a class="p2p-card" href="api/index.html"><span class="p2p-card-step">06</span><strong>API reference</strong><span class="p2p-card-copy">Modules, classes and functions, generated from the code.</span></a>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: User guide
   :hidden:

   quickstart
   concepts
   studio
   cli
   configuration
   workflows
   artifacts
   earlier_formats
   development

.. toctree::
   :maxdepth: 2
   :caption: API reference
   :hidden:

   api/index

The repository ``README.md`` gives the installation, the data sources, the
relation to the NeurIPS 2026 paper *Prompts to Proxies* and how to cite P2P.
