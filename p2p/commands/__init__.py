"""Registered user-facing command implementations."""

from p2p.commands import ablate
from p2p.commands import aggregate
from p2p.commands import budget
from p2p.commands import cluster
from p2p.commands import elicit
from p2p.commands import evaluate
from p2p.commands import generate
from p2p.commands import inspect
from p2p.commands import panel
from p2p.commands import pipeline
from p2p.commands import regress
from p2p.commands import simulate
from p2p.commands import validate
from p2p.registry import COMMANDS


COMMANDS.register(
    "generate",
    generate.run,
)
COMMANDS.register(
    "elicit",
    elicit.run,
)
COMMANDS.register(
    "regress",
    regress.run,
)
COMMANDS.register(
    "pipeline",
    pipeline.run,
)
COMMANDS.register(
    "evaluate",
    evaluate.run,
)
COMMANDS.register(
    "aggregate",
    aggregate.run,
)
COMMANDS.register(
    "budget",
    budget.run,
)
COMMANDS.register(
    "simulate",
    simulate.run,
)
COMMANDS.register(
    "ablate",
    ablate.run,
)
COMMANDS.register(
    "panel",
    panel.run,
)
COMMANDS.register(
    "cluster",
    cluster.run,
)
COMMANDS.register(
    "validate",
    validate.run,
)
COMMANDS.register(
    "inspect",
    inspect.run,
)


__all__ = [
    "ablate",
    "aggregate",
    "budget",
    "cluster",
    "elicit",
    "evaluate",
    "generate",
    "inspect",
    "panel",
    "pipeline",
    "regress",
    "simulate",
    "validate",
]
