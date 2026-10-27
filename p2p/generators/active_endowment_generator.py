from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from itertools import count, islice
import logging
import os
import random
import time
from typing import Any, Callable
import warnings

import numpy as np
from tqdm import tqdm
import yaml

from p2p.endowments.manager import ActiveEndowments
from p2p.generators.theme_variability_tracker import ThemeVariabilityTracker
from p2p.responses.converter import Responses


logger = logging.getLogger(__name__)

SURVEY_GLOBAL_MODE = "survey.global"
QUESTION_MODE_PREFIX = "question."

class ActiveEndowmentGenerator:
    """
    Generate endowments adaptively to increase response variability in simulated
    survey experiments.

    The generator owns a deep-copied, run-local attribute bank. Attributes
    discovered during a run use the reserved modes ``survey.global`` and
    ``question.<QID>``. Response simulation is delegated to an injected
    conductor factory, keeping backend and agent integration outside this
    module.

    Attributes:
        survey: Survey containing question definitions and metadata.
        endowment_model: Object that generates endowments from attribute lists.
        attribute_learner: Object that discovers survey and question attributes.
        agent_config: Agent settings consumed by the injected conductor.
        attribute_bank: Run-local copy of the injected attribute bank.
        conductor_factory: Optional callable that constructs survey conductors.
        tracker: Variability tracker used for entropy-guided sampling.
        target_n: Total number of endowments to generate.
        initial_n: Number sampled per initial mode.
        num_update_steps: Maximum number of adaptive expansion steps.
        parallel: Whether sampling and simulation may use worker threads.
        max_workers: Maximum number of worker threads.
        question_patch: Question-targeting allocation settings.
        seed: Random seed used by the generator.
        response_log: Collected agent-level response records.
        mode_history: Modes and attributes used during sampling.
        token_tracker: Optional injected usage tracker.
    """ 
    def __init__(
        self,
        survey,
        endowment_model,
        attribute_learner,
        agent_config: Any,
        attribute_bank: dict,
        conductor_factory: Callable[..., Any] | None = None,
        variability_tracker=None,
        target_n=100,
        initial_n=10,
        num_update_steps=4,
        seed=None,
        parallel=False,
        max_workers=10,
        verbose=True,
        question_patch: dict | None = None,
        logger=None,
        token_tracker=None,
        enable_survey_attributes: bool = True,
        checkpoint_callback: Callable[[dict[str, Any]], Any] | None = None,
        resume_state: dict[str, Any] | None = None,
    ):
        """
        Initialize an active endowment generator.

        Args:
            survey: Survey containing the questions to simulate.
            endowment_model: Object exposing
                ``generate_endowments(attributes, n, mode)``.
            attribute_learner: Object exposing survey-level and question-level
                attribute generation methods.
            agent_config: Configuration object exposing ``agent_type``,
                ``model_name``, ``formality``, and ``agent_kwargs``. It may
                optionally expose ``delay``.
            attribute_bank: Attribute-bank mapping containing ``templates`` and
                ``attributes``. It is required and deep-copied into run-local
                state.
            conductor_factory: Optional callable used to construct a conductor
                from the survey, an endowment manager, agent settings, and
                runtime options. It is required only when response simulation
                is invoked.
            variability_tracker: Optional tracker for entropy-based allocation.
                When omitted, a tracker is created from the survey's training
                questions.
            target_n: Total number of endowments to generate.
            initial_n: Number of endowments sampled per initial mode.
            num_update_steps: Number of adaptive expansion steps.
            seed: Seed for reproducible generator sampling.
            parallel: Whether to use parallel sampling and simulation.
            max_workers: Maximum number of worker threads.
            verbose: Whether to emit progress logging.
            question_patch: Question-targeting settings such as allocation
                fraction, top-k count, and ineffective-mode threshold.
            logger: Optional logger. The module logger is used by default.
            token_tracker: Optional injected usage tracker, forwarded to the
                conductor and used for usage reporting.
            enable_survey_attributes: Whether to discover and register the
                reserved ``survey.global`` attributes during initialization.

        Raises:
            ValueError: If ``attribute_bank`` is not provided.
        """
        if attribute_bank is None:
            raise ValueError("attribute_bank must be provided") 
               
        self.survey = survey
        self.attribute_learner = attribute_learner
        self.endowment_model = endowment_model
        self.agent_config = agent_config
        self.tracker = variability_tracker or ThemeVariabilityTracker(
            questions=self.survey.get_questions_by_split("train"),
            smoothing= 0
        )

        self.target_n = target_n
        self.initial_n = initial_n
        self.num_update_steps = num_update_steps
        self.seed = seed
        self.parallel = parallel
        self.verbose = verbose
        self.max_workers = max_workers

        self.endowment_manager = ActiveEndowments.from_endowment_list([])

        self.question_patch = question_patch or {
            "fraction": 0.3,
            "top_k": 3,
            "min_repeats": 2  # threshold to trigger reallocation of ineffective question modes
        }

        self.mode_history = []
        self.response_log = []
        self.attribute_bank = deepcopy(attribute_bank)
        self.conductor_factory = conductor_factory
        self.random_state = random.Random(seed)
        self.numpy_random = np.random.default_rng(seed)
        self.logger = logger or logging.getLogger(__name__)
        self.token_tracker = token_tracker

        self.qid_low_entropy_counts = Counter()
        self.enable_survey_attributes = enable_survey_attributes
        self.checkpoint_callback = checkpoint_callback
        self.initial_complete = False
        self.updates_done = 0
        self.inflight_kind: str | None = None
        self.inflight_eids: list[str] = []
        self.committed_batch_sizes: list[int] = []

        if resume_state is not None:
            self._restore_checkpoint_state(resume_state)
        else:
            # Learn survey attributes in the generator's run-local bank.
            if self.enable_survey_attributes:
                self.add_survey_attributes()
            else:
                self._log(
                    "[Attribute Bank Expansion] Skipping survey-level "
                    "attributes (enable_survey_attributes=False)"
                )
            self._checkpoint()

    @staticmethod
    def _json_value(value):
        if isinstance(value, dict):
            return {
                str(key): ActiveEndowmentGenerator._json_value(child)
                for key, child in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [
                ActiveEndowmentGenerator._json_value(child)
                for child in value
            ]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    @staticmethod
    def _tuple_tree(value):
        if isinstance(value, list):
            return tuple(
                ActiveEndowmentGenerator._tuple_tree(child)
                for child in value
            )
        return value

    def checkpoint_state(self) -> dict[str, Any]:
        """Return JSON-safe state sufficient for an exact paid-run resume."""

        endowment_random_state = getattr(
            self.endowment_model,
            "random_state",
            None,
        )
        return self._json_value(
            {
                "schema_version": 1,
                "target_n": self.target_n,
                "initial_complete": self.initial_complete,
                "updates_done": self.updates_done,
                "inflight_kind": self.inflight_kind,
                "inflight_eids": list(self.inflight_eids),
                "committed_batch_sizes": list(
                    self.committed_batch_sizes
                ),
                "endowments": deepcopy(
                    self.endowment_manager.get_endowments()
                ),
                "response_log": deepcopy(self.response_log),
                "attribute_bank": deepcopy(self.attribute_bank),
                "mode_history": deepcopy(self.mode_history),
                "qid_low_entropy_counts": dict(
                    self.qid_low_entropy_counts
                ),
                "numpy_random_state": deepcopy(
                    self.numpy_random.bit_generator.state
                ),
                "generator_random_state": self.random_state.getstate(),
                "endowment_random_state": (
                    endowment_random_state.getstate()
                    if endowment_random_state is not None
                    else None
                ),
                "generation_complete": (
                    not self.inflight_eids
                    and len(self.endowment_manager) >= self.target_n
                ),
            }
        )

    def _checkpoint(self) -> None:
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(
                self.checkpoint_state()
            )

    def _restore_checkpoint_state(
        self,
        state: dict[str, Any],
    ) -> None:
        if not isinstance(state, dict):
            raise TypeError("resume_state must be a mapping")
        if state.get("schema_version") != 1:
            raise ValueError(
                "unsupported active-generator resume schema"
            )
        if state.get("target_n") != self.target_n:
            raise ValueError(
                "active-generator checkpoint target_n mismatch"
            )

        raw_endowments = state.get("endowments")
        raw_responses = state.get("response_log")
        raw_bank = state.get("attribute_bank")
        raw_sizes = state.get("committed_batch_sizes")
        if not isinstance(raw_endowments, list):
            raise ValueError(
                "checkpoint endowments must be a list"
            )
        if not isinstance(raw_responses, list):
            raise ValueError(
                "checkpoint response_log must be a list"
            )
        if not isinstance(raw_bank, dict):
            raise ValueError(
                "checkpoint attribute_bank must be a mapping"
            )
        if (
            not isinstance(raw_sizes, list)
            or any(
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                for size in raw_sizes
            )
        ):
            raise ValueError(
                "checkpoint committed_batch_sizes must contain "
                "nonnegative integers"
            )

        endowments = deepcopy(raw_endowments)
        for endowment in endowments:
            endowment["mode"] = tuple(
                endowment.get("mode", ())
            )
        responses = deepcopy(raw_responses)
        for response in responses:
            response["mode"] = tuple(
                response.get("mode", ())
            )
        mode_history = deepcopy(
            state.get("mode_history", [])
        )
        for entry in mode_history:
            entry["mode"] = tuple(
                entry.get("mode", ())
            )

        self.endowment_manager = (
            ActiveEndowments.from_endowment_list(
                endowments
            )
        )
        self.response_log = responses
        self.attribute_bank = deepcopy(raw_bank)
        self.mode_history = mode_history
        self.qid_low_entropy_counts = Counter(
            state.get("qid_low_entropy_counts", {})
        )
        self.initial_complete = bool(
            state.get("initial_complete", False)
        )
        self.updates_done = state.get("updates_done", 0)
        if (
            isinstance(self.updates_done, bool)
            or not isinstance(self.updates_done, int)
            or self.updates_done < 0
        ):
            raise ValueError(
                "checkpoint updates_done must be nonnegative"
            )
        self.inflight_kind = state.get("inflight_kind")
        if self.inflight_kind not in {None, "initial", "adaptive"}:
            raise ValueError(
                "checkpoint inflight_kind is invalid"
            )
        self.inflight_eids = list(
            state.get("inflight_eids", [])
        )
        self.committed_batch_sizes = list(raw_sizes)

        committed_count = sum(self.committed_batch_sizes)
        if committed_count > len(self.response_log):
            raise ValueError(
                "checkpoint commits more responses than it stores"
            )
        if not self.inflight_eids and (
            committed_count != len(self.response_log)
        ):
            raise ValueError(
                "checkpoint has uncommitted responses without an "
                "in-flight batch"
            )

        # Replaying committed batches reconstructs the entropy/adaptive state
        # without serializing implementation-specific tracker internals.
        cursor = 0
        for size in self.committed_batch_sizes:
            batch = self.response_log[cursor : cursor + size]
            filtered = self.tracker.filter_records(
                batch,
                survey=self.survey,
            )
            self.tracker.update_from_records(filtered)
            cursor += size

        numpy_state = state.get("numpy_random_state")
        if not isinstance(numpy_state, dict):
            raise ValueError(
                "checkpoint numpy_random_state must be a mapping"
            )
        self.numpy_random.bit_generator.state = numpy_state
        self.random_state.setstate(
            self._tuple_tree(
                state.get("generator_random_state")
            )
        )
        endowment_state = state.get(
            "endowment_random_state"
        )
        if endowment_state is not None and hasattr(
            self.endowment_model,
            "random_state",
        ):
            self.endowment_model.random_state.setstate(
                self._tuple_tree(endowment_state)
            )

        identifiers = []
        for endowment in endowments:
            eid = str(endowment.get("eid", ""))
            if eid.startswith("active_"):
                try:
                    identifiers.append(
                        int(eid.removeprefix("active_"))
                    )
                except ValueError:
                    pass
        if hasattr(self.endowment_model, "_ids"):
            self.endowment_model._ids = count(
                max(identifiers, default=0) + 1
            )

        self._log(
            "[Checkpoint] Resuming active generation with "
            f"{committed_count} committed and "
            f"{len(self.response_log) - committed_count} "
            "in-flight completed agents"
        )

    def add_survey_attributes(self):
        """
        Register attributes related to the overall survey under the reserved
        ``survey.global`` mode.
        """
        attributes = self.attribute_learner.generate_survey_attributes()

        survey_modes = self.attribute_bank["templates"].setdefault("survey", [])
        if SURVEY_GLOBAL_MODE not in survey_modes:
            survey_modes.append(SURVEY_GLOBAL_MODE)

        self.attribute_bank["attributes"][SURVEY_GLOBAL_MODE] = attributes

        if self.verbose:
            print(
                "[Attribute Bank Expansion] "
                f"Added {len(attributes)} survey specific attributes:\n {attributes}"
            )
    
    def add_question_attributes(self, qid, max_attributes=None):
        """
        Register attributes for a question under its reserved
        ``question.<QID>`` mode.
        """
        attributes = self.attribute_learner.generate_question_attributes(
            qid, max_attributes
        )
        question_mode = f"{QUESTION_MODE_PREFIX}{qid}"

        question_modes = self.attribute_bank["templates"].setdefault("question", [])
        if question_mode not in question_modes:
            question_modes.append(question_mode)

        if question_mode not in self.attribute_bank["attributes"]:
            self.attribute_bank["attributes"][question_mode] = attributes
            if self.verbose:
                print(
                    "[Attribute Bank Expansion] "
                    f"Added {len(attributes)} question specific attributes:\n "
                    f"{attributes}"
                )
        elif self.verbose:
            print(
                f"[Attribute Bank] Attributes for QID {qid!r} already exist, "
                "skipping"
            )

    def _log(self, msg, level: str = "info"):
        """
        Utility method for conditional logging with level control.

        Args:
            msg (str): The message to log.
            level (str): Logging level as string ("debug", "info", "warning", "error", "critical").
        """
        if getattr(self, 'verbose', False) and self.logger:
            log_fn = getattr(self.logger, level.lower(), self.logger.info)
            log_fn(msg)

    def generate(self):
        """
        Orchestrates the full active generation loop. Starts with initial sampling, then 
        repeatedly expands the pool using entropy-weighted sampling until `target_n` 
        endowments are generated.

        Returns:
            An ActiveEndowments manager with all generated endowments.
        """
        if getattr(self, "inflight_eids", []):
            self._log(
                "[Checkpoint] Completing interrupted "
                f"{self.inflight_kind} simulation batch"
            )
            self._resume_inflight_batch()

        if not getattr(self, "initial_complete", False):
            self._log("Starting initial sampling...")
            initial_size = len(self.endowment_manager)
            self._initial_sampling()
            if len(self.endowment_manager) <= initial_size:
                raise RuntimeError(
                    "active generation made no progress during "
                    "initial sampling"
                )

        while len(self.endowment_manager.endowments) < self.target_n:
            remaining = self.target_n - len(self.endowment_manager)
            steps_left = max(
                1,
                self.num_update_steps - self.updates_done,
            )
            expansion_batch_size = min(remaining, int(np.ceil(remaining / steps_left)))

            self._log(f"Expanding pool: current size = {len(self.endowment_manager)}, target = {self.target_n}")
            size_before = len(self.endowment_manager)
            self._expand_pool(expansion_batch_size)
            if len(self.endowment_manager) <= size_before:
                raise RuntimeError(
                    "active generation made no progress while expanding "
                    f"the pool at update step {self.updates_done + 1}"
                )

        self._log(f"Generation complete. Final pool size: {len(self.endowment_manager)}")
        self._checkpoint()

        mode_counts = Counter(tuple(e['mode']) for e in self.endowment_manager.get_endowments())
        self._log(f"Unique modes sampled: {len(mode_counts)}")
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            self._log(f"  Mode {mode}: {count} endowments")

        if self.verbose and self.token_tracker:
            self.report_token_usage()

        return self.endowment_manager

    def _initial_sampling(self):
        """
        Sample the preset core, thematic, and theoretical modes, always adding
        the reserved ``survey.global`` mode. Simulate the resulting endowments
        to initialize entropy estimates.
        """
        n = self.initial_n
        templates = self.attribute_bank["templates"]
        mode_n_list = []

        found_presets = False

        # Core modes
        if templates.get("core"):
            self._log("Queuing core modes:")
            for core_mode in templates["core"]:
                self._log(f"  Queued: core mode `{core_mode}` with n={n}")
                mode_n_list.append(((core_mode,), n))
            found_presets = True

        # Thematic modes
        if templates.get("thematic"):
            self._log("Queuing thematic modes:")
            for theme in templates["thematic"]:
                mode = ("core", theme)
                self._log(f"  Queued: thematic mode `{mode}` with n={n}")
                mode_n_list.append((mode, n))
            found_presets = True

        # Theoretical modes
        if templates.get("theoretical"):
            self._log("Queuing theoretical modes:")
            for theory in templates["theoretical"]:
                mode = ("core", theory)
                self._log(f"  Queued: theoretical mode `{mode}` with n={n}")
                mode_n_list.append((mode, n))
            found_presets = True

        if not found_presets:
            self._log("No preset core/thematic/theoretical modes found. Skipping structured sampling.")

        # Survey-specific mode is always included
        self._log(f"Queuing survey-specific mode:  n={n}")
        mode_n_list.append(((SURVEY_GLOBAL_MODE,), n))

        self._log(f"Sampling {len(mode_n_list)} modes in parallel...")
        new_endowments = []
        new_endowments.extend(
            batch for batch in self._parallel_sample_modes(mode_n_list) if batch
        )

        self._log("Simulating survey responses...")
        self._simulate_and_commit_batch(
            new_endowments,
            kind="initial",
        )

    def _expand_pool(self, batch_total=10, question_patch = None):
        """
        Expand the pool using entropy-weighted base modes and targeted
        ``question.<QID>`` modes.

        Args:
            batch_total: Total number of new agents to allocate.
            question_patch: Optional override for question-targeting settings.
        """
        if not question_patch:
            question_patch = self.question_patch
        question_quota = int(batch_total * question_patch["fraction"])
        base_quota = batch_total - question_quota

        mode_n_list = []

        # === BASE MODES: thematic/core/theoretical ===
        weights = self.tracker.get_softmax_weights(temperature=0.3)
        modes = list(weights.keys())
        probs = np.array([weights[m] for m in modes])
        draws = self.numpy_random.multinomial(base_quota, probs)
        allocation = Counter({modes[i]: draws[i] for i in range(len(modes)) if draws[i] > 0})

        for mode, n in allocation.items():
            self._log(f"Queuing base mode: {mode} with n={n}")
            mode_n_list.append((mode, n))

        
        # === QUESTION-SPECIFIC MODES: even allocation across top-k low-entropy questions ===
        if question_quota > 0:
            top_k = question_patch["top_k"]
            min_repeats = question_patch["min_repeats"]
            low_entropy_qids = self.tracker.get_low_entropy_questions(top_k=top_k)

            for qid in low_entropy_qids:
                self.qid_low_entropy_counts[qid] += 1

            per_q = question_quota // top_k
            leftovers = question_quota % top_k

            top_mode = max(weights.items(), key=lambda x: x[1])[0]

            for i, qid in enumerate(low_entropy_qids):
                n = per_q + (1 if i < leftovers else 0)
                if n <= 0:
                    continue

                question_mode = f"{QUESTION_MODE_PREFIX}{qid}"

                if question_mode not in self.attribute_bank["attributes"]:
                    self.add_question_attributes(qid)
                else:
                    self._log(
                        f"[Attribute Bank] Attributes for QID {qid!r} "
                        "already exist, skipping"
                    )

                # Reallocate if ineffective
                if self.qid_low_entropy_counts[qid] >= min_repeats:
                    mixed_mode = tuple(set(top_mode + (question_mode,)))
                    self._log(f"Reallocating question mode '{qid}' to mixed mode: {mixed_mode} with n={n}")
                    mode_n_list.append((mixed_mode, n))
                else:
                    q_mode = (question_mode,)
                    self._log(f"Queuing question-specific mode: {q_mode} with n={n}")
                    mode_n_list.append((q_mode, n))

        self._log(f"Sampling {len(mode_n_list)} modes in parallel...")
        new_endowments = [
            batch for batch in self._parallel_sample_modes(mode_n_list)
            if batch
        ]

        self._log(f"Simulating survey for {len(new_endowments)} sampled mode(s)...")
        self._simulate_and_commit_batch(
            new_endowments,
            kind="adaptive",
        )

    def _simulate_and_commit_batch(
        self,
        endowment_batches,
        *,
        kind: str,
        resuming: bool = False,
    ) -> None:
        if kind not in {"initial", "adaptive"}:
            raise ValueError(
                "simulation batch kind must be initial or adaptive"
            )
        flat_endowments = [
            endowment
            for batch in endowment_batches
            for endowment in batch
        ]
        if not flat_endowments:
            raise RuntimeError(
                f"{kind} sampling produced no endowments"
            )

        batch_eids = [
            str(endowment["eid"])
            for endowment in flat_endowments
        ]
        if not resuming:
            if self.inflight_eids:
                raise RuntimeError(
                    "cannot start a new batch while another batch "
                    "is in flight"
                )
            self.inflight_kind = kind
            self.inflight_eids = batch_eids
            # Commit generated personas before response elicitation starts.
            self._checkpoint()
        elif (
            self.inflight_kind != kind
            or self.inflight_eids != batch_eids
        ):
            raise RuntimeError(
                "resumed in-flight batch does not match checkpoint"
            )

        agent_records = self._simulate_survey_batch(
            [flat_endowments]
        )
        record_eids = [
            str(record["agent_id"])
            for record in agent_records
        ]
        if (
            len(record_eids) != len(batch_eids)
            or len(record_eids) != len(set(record_eids))
            or set(record_eids) != set(batch_eids)
        ):
            raise RuntimeError(
                "simulation batch did not return exactly one record "
                "per in-flight agent"
            )

        filtered_records = self.tracker.filter_records(
            agent_records,
            survey=self.survey,
        )
        self.tracker.update_from_records(filtered_records)
        self.committed_batch_sizes.append(
            len(agent_records)
        )
        if kind == "initial":
            self.initial_complete = True
        else:
            self.updates_done += 1
        self.inflight_kind = None
        self.inflight_eids = []
        self._checkpoint()

    def _resume_inflight_batch(self) -> None:
        if (
            self.inflight_kind is None
            or not self.inflight_eids
        ):
            raise RuntimeError(
                "there is no in-flight batch to resume"
            )
        endowments = []
        for eid in self.inflight_eids:
            endowment = (
                self.endowment_manager.get_endowment_by_eid(
                    eid
                )
            )
            if endowment is None:
                raise RuntimeError(
                    "checkpoint in-flight agent is missing its "
                    f"endowment: {eid}"
                )
            endowments.append(endowment)
        self._simulate_and_commit_batch(
            [endowments],
            kind=self.inflight_kind,
            resuming=True,
        )

    def _sample_mode(self, mode_tuple, n):
        """
        Generate endowments from the union of attributes for a mode tuple.

        Args:
            mode_tuple: Attribute-bank keys, such as ``("core", "economics")``,
                ``("survey.global",)``, or ``("question.Q20",)``.
            n: Number of endowments to generate.

        Returns:
            Generated endowments, or ``None`` when the selected modes contain
            no attributes.
        """
        if isinstance(mode_tuple, str):
            mode_tuple = (mode_tuple,)

        attributes = sorted(set(attr for mode in mode_tuple for attr in self.attribute_bank["attributes"].get(mode, [])))

        if not attributes:
            warnings.warn(f"No attributes found for mode {mode_tuple}")
            return

        # self._log(f"  Generating {n} endowments for mode {mode_tuple} with {len(attributes)} attributes")
        new_endowments = self.endowment_model.generate_endowments(attributes, n=n, mode=mode_tuple)
        self.endowment_manager.add_batch(new_endowments)
        self.mode_history.append({
            "mode": mode_tuple,
            "attributes": attributes,
            "n": n
        })
        return new_endowments
    
    def _parallel_sample_modes(self, mode_n_list):
        """
        Parallel wrapper around _sample_mode for a list of (mode_tuple, n) pairs.
        Returns a list of endowment batches, one per mode.
        """
        if not self.parallel or self.max_workers <= 1:
            return [self._sample_mode(mode, n) for mode, n in mode_n_list]

        results = [None] * len(mode_n_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._sample_mode, mode, n): i
                for i, (mode, n) in enumerate(mode_n_list)
            }

            iterator = as_completed(future_to_idx)
            if self.verbose:
                iterator = tqdm(iterator, total=len(future_to_idx), desc="Sampling modes")

            for future in iterator:
                i = future_to_idx[future]
                results[i] = future.result()

        return results

    def _simulate_survey_batch(self, endowment_batches) -> list[dict]:
        """
        Simulate batches using conductors created by ``conductor_factory``.

        Args:
            endowment_batches: Endowment lists grouped by sampled mode.

        Returns:
            Flat agent-level response records produced by the conductors.

        Raises:
            RuntimeError: If response simulation is requested without an
                injected conductor factory.
        """
        if self.conductor_factory is None:
            raise RuntimeError(
                "conductor_factory is required to simulate survey responses"
            )
        
        self._log(f"Simulating survey responses with dynamic chunking (parallel={self.parallel})")

        # Flatten all endowments into a single list. A checkpoint may already
        # contain completed chunks from an interrupted attempt, so only send
        # the missing agents back to the paid backend.
        flat_endowments = [e for batch in endowment_batches for e in batch]
        batch_eids = [str(e["eid"]) for e in flat_endowments]
        if len(batch_eids) != len(set(batch_eids)):
            raise RuntimeError("simulation batch contains duplicate endowment ids")

        batch_eid_set = set(batch_eids)
        existing_by_eid = {}
        for record in self.response_log:
            eid = str(record.get("agent_id", ""))
            if eid not in batch_eid_set:
                continue
            if eid in existing_by_eid:
                raise RuntimeError(
                    f"checkpoint contains duplicate response records for {eid}"
                )
            existing_by_eid[eid] = record

        pending_endowments = [
            endowment
            for endowment in flat_endowments
            if str(endowment["eid"]) not in existing_by_eid
        ]
        total = len(pending_endowments)
        if not total:
            self._log(
                f"Reusing all {len(batch_eids)} checkpointed agent responses"
            )
            return [existing_by_eid[eid] for eid in batch_eids]

        # Decide chunk size based on max_workers
        if self.parallel:
            factor = 2  # Number of chunks per worker
            chunk_size = max(1, total // (self.max_workers * factor))
        else:
            chunk_size = total  # process all in one chunk

        self._log(f"Chunking {total} endowments into chunks of size {chunk_size}")

        # Helper to chunk the list
        def chunked_iter(iterable, n):
            it = iter(iterable)
            while True:
                chunk = list(islice(it, n))
                if not chunk:
                    break
                yield chunk

        chunks = list(chunked_iter(pending_endowments, chunk_size))

        def run_conductor(endowments):
            if hasattr(self.agent_config, "delay") and self.agent_config.delay:
                time.sleep(self.agent_config.delay)

            manager = ActiveEndowments.from_endowment_list(endowments)
            conductor = self.conductor_factory(
                survey=self.survey,
                endowments=manager,
                agent_type=self.agent_config.agent_type,
                model_name=self.agent_config.model_name,
                formality=self.agent_config.formality,
                verbose=self.verbose,
                token_tracker=self.token_tracker
            )
            conductor.run(save=False, parallel_worker=self.parallel, **self.agent_config.agent_kwargs)
            return conductor.to_agent_records()

        errors = []

        def commit_records(records):
            record_eids = [str(record.get("agent_id", "")) for record in records]
            if len(record_eids) != len(set(record_eids)):
                raise RuntimeError("simulation chunk returned duplicate agent ids")
            foreign = set(record_eids) - batch_eid_set
            if foreign:
                raise RuntimeError(
                    "simulation chunk returned agent ids outside the active batch: "
                    + ", ".join(sorted(foreign))
                )
            overlap = set(record_eids) & set(existing_by_eid)
            if overlap:
                raise RuntimeError(
                    "simulation chunk returned already-checkpointed agent ids: "
                    + ", ".join(sorted(overlap))
                )
            for record in records:
                existing_by_eid[str(record["agent_id"])] = record
            self.response_log.extend(records)
            # Persist each completed chunk before another paid request can fail.
            self._checkpoint()

        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    # micro-delay to prevent synchronized burst submission
                    time.sleep(0.015 * i)   # 10 ms per worker; adjust as needed (0.005–0.02)
                    futures.append(executor.submit(run_conductor, chunk))
                for f in tqdm(as_completed(futures), total=len(futures), desc="Simulating agents", disable= not self.verbose):
                    try:
                        commit_records(f.result())
                    except Exception as exc:
                        errors.append(exc)
        else:
            for chunk in chunks:
                try:
                    commit_records(run_conductor(chunk))
                except Exception as exc:
                    errors.append(exc)
                    break

        if errors:
            raise RuntimeError(
                "Survey simulation failed after completed chunks were "
                "checkpointed. Rerun the same command with the same output "
                "directory to resume only the missing agents."
            ) from errors[0]

        missing = batch_eid_set - set(existing_by_eid)
        if missing:
            raise RuntimeError(
                "simulation batch finished without response records for: "
                + ", ".join(sorted(missing))
            )

        results = [existing_by_eid[eid] for eid in batch_eids]
        self._log(
            f"Collected {len(results)} agent response records from "
            f"{len(chunks)} new chunks; reused "
            f"{len(results) - len(pending_endowments)} checkpointed records"
        )
        return results

    def export_mode_summary(self, path: str):
        """
        Saves a summary of endowments generated per mode to CSV.

        Args:
            path (str): Path to save the summary file.
        """
        from collections import defaultdict
        import pandas as pd

        # Collect rows with mode, n, and attributes
        rows = []
        for entry in self.mode_history:
            mode = entry["mode"]
            n = entry["n"]
            attributes = entry["attributes"]
            rows.append({
                "mode": "+".join(mode),
                "n": n,
                "attributes": ", ".join(attributes)
            })

        df = pd.DataFrame(rows)
        df = df.groupby("mode").agg({
            "n": "sum",
            "attributes": lambda x: "; ".join(set(x))  # de-duplicate
        }).reset_index()
        df = df.sort_values("n", ascending=False)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        if self.verbose:
            self._log(f"[Mode Summary Export] Saved to: {path}")

    def export_responses_to_csv(self, path: str, format="code"):
        """
        Converts and saves all collected agent responses to a CSV file.

        Args:
            path: Path to save the file.
            format: Output format to use in Responses class (e.g., "code" or "text").
        """
        qids = [q["id"] for q in self.survey.questions]

        flat_records = []
        for agent_record in self.response_log:
            eid = agent_record["agent_id"]
            for qid, answer in zip(qids, agent_record["responses"]):
                flat_records.append({
                    "eid": eid,
                    "qid": qid,
                    "answer": answer if answer is not None else ""
                })

        responses = Responses(flat_records, self.survey, output_format=format)
        responses.save(path)
    
    def export_attribute_bank(self, filepath: str = "attribute_bank.yaml"):
        """
        Saves the current attribute bank to a YAML file.

        Args:
            filepath: Target path for saving the attribute bank.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        with open(filepath, "w") as f:
            yaml.dump(self.attribute_bank, f, sort_keys=False, allow_unicode=True)

        if self.verbose:
            self._log(f"[Mode Summary Export] Saved to: {filepath}")

    def get_responses(self, output_format="answer"):
        """
        Converts the response log to a `Responses` object (for analysis or saving).

        Args:
            output_format (str): Either "code" or "text" for the response content.

        Returns:
            Responses: Structured Responses object.
        """
        qids = [q["id"] for q in self.survey.questions]

        flat_records = []
        for agent_record in self.response_log:
            eid = agent_record["agent_id"]
            for qid, answer in zip(qids, agent_record["responses"]):
                flat_records.append({
                    "eid": eid,
                    "qid": qid,
                    "answer": answer if answer is not None else ""
                })

        return Responses(flat_records, self.survey, output_format=output_format, clean=True)

    def report_token_usage(self):
        if self.token_tracker:
            print("\n--- Token Usage Summary ---")
            print(self.token_tracker.report())

    @property
    def responses(self):
        return self.get_responses()
    
    @property
    def responses_code(self):
        return self.get_responses(output_format="code")

    def __len__(self):
        """
        Returns:
            The number of generated endowments.
        """
        return len(self.endowment_manager.endowments)   
