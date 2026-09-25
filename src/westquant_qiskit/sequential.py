from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from westquant_core import (
    EquivalenceKind,
    RepGraph,
    Representation,
    RepresentationKind,
    TransformationRecord,
)

from .adapter import QiskitAdapter, circuit_metrics, numeric_metrics
from .compiler import QiskitCompiler
from .config import PipelineConfig
from .search import DEFAULT_BASELINE_CONFIG, CandidateResult, _framework_version, _target_context
from .verification import QiskitEquivalenceVerifier, VerificationReport


class CompilerLike(Protocol):
    def compile(self, circuit: Any, config: PipelineConfig, **kwargs: Any) -> Any: ...


class VerifierLike(Protocol):
    def verify(self, original: Any, candidate: Any) -> VerificationReport: ...


STAGE_ORDER = ("layout", "routing", "translation", "optimization")


@dataclass(frozen=True)
class SequentialSearchSpace:
    """Ordered action space for policy-style representation search.

    At each step the selected prefix is completed with explicit defaults and
    compiled end-to-end.  This gives every action a comparable rollout reward
    without depending on Qiskit's private intermediate-pass internals.
    """

    layout_methods: tuple[str, ...] = ("trivial", "dense", "sabre")
    routing_methods: tuple[str, ...] = ("basic", "lookahead", "sabre")
    translation_methods: tuple[str, ...] = ("translator", "synthesis")
    optimization_levels: tuple[int, ...] = (0, 1, 2, 3)
    seed_transpiler: int = 0
    approximation_degree: float = 1.0
    stage_order: tuple[str, ...] = STAGE_ORDER

    def __post_init__(self) -> None:
        allowed = set(STAGE_ORDER)
        if set(self.stage_order) != allowed or len(self.stage_order) != len(allowed):
            raise ValueError(f"stage_order must contain each of {STAGE_ORDER} exactly once")
        if not all((self.layout_methods, self.routing_methods, self.translation_methods, self.optimization_levels)):
            raise ValueError("all sequential action dimensions must be non-empty")

    def choices(self, stage: str) -> tuple[Any, ...]:
        if stage == "layout":
            return self.layout_methods
        if stage == "routing":
            return self.routing_methods
        if stage == "translation":
            return self.translation_methods
        if stage == "optimization":
            return self.optimization_levels
        raise KeyError(stage)

    def config_for(self, prefix: dict[str, Any]) -> PipelineConfig:
        return PipelineConfig(
            optimization_level=int(prefix.get("optimization", 2)),
            layout_method=str(prefix.get("layout", "default")),
            routing_method=str(prefix.get("routing", "default")),
            translation_method=str(prefix.get("translation", "default")),
            seed_transpiler=self.seed_transpiler,
            approximation_degree=self.approximation_degree,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_methods": list(self.layout_methods),
            "routing_methods": list(self.routing_methods),
            "translation_methods": list(self.translation_methods),
            "optimization_levels": list(self.optimization_levels),
            "seed_transpiler": self.seed_transpiler,
            "approximation_degree": self.approximation_degree,
            "stage_order": list(self.stage_order),
        }

    @property
    def search_space_id(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "qiskit-seq-space:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class SequentialState:
    state_id: str
    parent_state_id: str | None
    step_index: int
    stage: str | None
    action: Any | None
    prefix: dict[str, Any]
    candidate: CandidateResult | None
    kept_in_beam: bool = False
    terminal: bool = False

    @property
    def metrics(self) -> dict[str, Any]:
        return self.candidate.metrics if self.candidate else {}

    @property
    def selectable(self) -> bool:
        return bool(self.candidate and self.candidate.selectable)


@dataclass
class SequentialSearchResult:
    challenge_id: str
    graph: RepGraph
    input_metrics: dict[str, Any]
    baseline: CandidateResult
    states: list[SequentialState]
    final_beam: list[SequentialState]
    beam_width: int

    @property
    def best_state(self) -> SequentialState | None:
        selectable = [s for s in self.final_beam if s.selectable]
        if not selectable:
            return None
        return min(selectable, key=_state_selection_key)

    def training_records(self) -> list[dict[str, Any]]:
        by_id = {s.state_id: s for s in self.states}
        records: list[dict[str, Any]] = []
        for state in self.states:
            if state.parent_state_id is None or state.candidate is None:
                continue
            parent = by_id.get(state.parent_state_id)
            before = parent.metrics if parent else self.baseline.metrics
            after = state.candidate.metrics
            action_payload = {
                "stage": state.stage,
                "name": str(state.action),
                "parameters": {},
            }
            records.append({
                "schema_version": "wqt-policy-v0.1",
                "challenge_id": self.challenge_id,
                "framework": "qiskit",
                "search_space_id": self.graph.metadata.get("search_space_id"),
                "context": {"target_context": self.graph.metadata.get("target_context", {})},
                "target_context": self.graph.metadata.get("target_context", {}),
                "state_id": state.parent_state_id,
                "next_state_id": state.state_id,
                "step_index": state.step_index,
                "stage": state.stage,
                "action": action_payload,
                "action_name": state.action,
                "prefix_before": parent.prefix if parent else {},
                "prefix_after": state.prefix,
                "pipeline_rollout": state.candidate.config.to_dict(),
                "compile_success": state.candidate.compile_success,
                "success": state.candidate.compile_success,
                "selectable": state.candidate.selectable,
                "outcome_class": _outcome_class(state.candidate),
                "verification": state.candidate.verification.to_dict() if state.candidate.verification else None,
                "metrics_before": before,
                "metrics_after": after,
                "reward_vector": CandidateResult._deltas(after, before),
                "delta_vs_default_baseline": CandidateResult._deltas(after, self.baseline.metrics),
                "compile_seconds": state.candidate.compile_seconds,
                "kept_in_beam": state.kept_in_beam,
                "terminal": state.terminal,
                "error": state.candidate.error,
            })
        return records

    def summary(self) -> dict[str, Any]:
        best = self.best_state
        return {
            "challenge_id": self.challenge_id,
            "search": "sequential_beam",
            "beam_width": self.beam_width,
            "n_action_evaluations": len([s for s in self.states if s.parent_state_id is not None]),
            "n_compile_success": sum(bool(s.candidate and s.candidate.compile_success) for s in self.states),
            "n_final_states": len(self.final_beam),
            "default_baseline_metrics": self.baseline.metrics,
            "best_state_id": best.state_id if best else None,
            "best_prefix": best.prefix if best else None,
            "best_metrics": best.metrics if best else None,
            "best_delta_vs_default_baseline": (
                CandidateResult._deltas(best.metrics, self.baseline.metrics) if best else None
            ),
        }


def _outcome_class(candidate: CandidateResult) -> str:
    """Conservative label that never conflates compile success with proof of equivalence."""
    if not candidate.compile_success:
        return "compile_failed"
    if candidate.verification is None:
        return "compiled_unknown"
    if candidate.verification.equivalence == EquivalenceKind.INVALID:
        return "verification_invalid"
    if candidate.verification.verified and candidate.verification.equivalence == EquivalenceKind.EXACT:
        return "verified_exact"
    return "compiled_unknown"


def _state_selection_key(state: SequentialState) -> tuple[Any, ...]:
    if state.candidate is None or not state.candidate.selectable:
        return (math.inf, math.inf, math.inf, math.inf, math.inf)
    verified_penalty = 0 if state.candidate.verification and state.candidate.verification.verified else 1
    return (verified_penalty, *state.candidate.objective_tuple())


def _state_id(challenge_id: str, prefix: dict[str, Any]) -> str:
    blob = json.dumps(prefix, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return f"{challenge_id}:seq:{digest}"


class SequentialSearchEngine:
    """Beam search over ordered Qiskit representation decisions.

    Each partial state is evaluated by an end-to-end rollout where unchosen
    later stages use Qiskit's explicit ``default`` methods.  The approach is
    intentionally public-API-only and produces policy-ready state/action/reward
    traces while remaining deterministic for fixed targets and seeds.
    """

    def __init__(
        self,
        *,
        search_space: SequentialSearchSpace | None = None,
        beam_width: int = 3,
        compiler: CompilerLike | None = None,
        verifier: VerifierLike | None = None,
        adapter: Any | None = None,
        baseline_config: PipelineConfig = DEFAULT_BASELINE_CONFIG,
    ) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        self.search_space = search_space or SequentialSearchSpace()
        self.beam_width = beam_width
        self.compiler = compiler or QiskitCompiler()
        self.verifier = verifier or QiskitEquivalenceVerifier()
        self.adapter = adapter or QiskitAdapter()
        self.baseline_config = baseline_config

    def _compile(
        self,
        circuit: Any,
        config: PipelineConfig,
        *,
        default_pipeline: bool,
        backend: Any,
        target: Any,
        coupling_map: Any,
        basis_gates: list[str] | None,
        dt: float | None,
    ) -> tuple[Any | None, CandidateResult]:
        started = time.perf_counter()
        try:
            kwargs = dict(
                backend=backend,
                target=target,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                dt=dt,
            )
            if default_pipeline and hasattr(self.compiler, "compile_default"):
                compiled = self.compiler.compile_default(circuit, config, **kwargs)
            else:
                compiled = self.compiler.compile(circuit, config, **kwargs)
            elapsed = time.perf_counter() - started
            verification = self.verifier.verify(circuit, compiled)
            return compiled, CandidateResult(
                candidate_id=config.candidate_id,
                config=config,
                compile_success=True,
                role="sequential_rollout",
                metrics=circuit_metrics(compiled),
                compile_seconds=elapsed,
                verification=verification,
            )
        except Exception as exc:
            return None, CandidateResult(
                candidate_id=config.candidate_id,
                config=config,
                compile_success=False,
                role="sequential_rollout",
                compile_seconds=time.perf_counter() - started,
                error=repr(exc),
            )

    def _state_representation(
        self,
        *,
        state: SequentialState,
        compiled: Any | None,
        challenge_id: str,
    ) -> Representation:
        compiled_payload: dict[str, Any] | None = None
        if compiled is not None:
            imported = self.adapter.import_native(
                compiled,
                representation_id=f"{state.state_id}:rollout-circuit",
                semantic_root=challenge_id,
                metadata={"role": "sequential_rollout_circuit"},
            )
            compiled_payload = imported.payload
        candidate = state.candidate
        return Representation(
            id=state.state_id,
            kind=RepresentationKind.SEARCH_STATE,
            semantic_root=challenge_id,
            framework="qiskit",
            payload={
                "step_index": state.step_index,
                "stage": state.stage,
                "action": state.action,
                "prefix": state.prefix,
                "pipeline_rollout": candidate.config.to_dict() if candidate else None,
                "compile_success": candidate.compile_success if candidate else True,
                "metrics": candidate.metrics if candidate else {},
                "verification": candidate.verification.to_dict() if candidate and candidate.verification else None,
                "error": candidate.error if candidate else None,
                "compiled_circuit": compiled_payload,
            },
            metadata={"role": "search_state"},
        )

    def search(
        self,
        circuit: Any,
        *,
        challenge_id: str,
        backend: Any = None,
        target: Any = None,
        coupling_map: Any = None,
        basis_gates: list[str] | None = None,
        dt: float | None = None,
        challenge_metadata: dict[str, Any] | None = None,
    ) -> SequentialSearchResult:
        input_metrics = circuit_metrics(circuit)
        root_id = f"{challenge_id}:root"
        root = self.adapter.import_native(
            circuit,
            representation_id=root_id,
            semantic_root=challenge_id,
            metadata={"role": "logical_input"},
        )
        target_context = _target_context(
            backend=backend,
            target=target,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            dt=dt,
        )
        graph = RepGraph(
            graph_id=f"repgraph:{challenge_id}:sequential",
            metadata={
                "framework": "qiskit",
                "framework_version": _framework_version(),
                "search": "sequential_beam",
                "beam_width": self.beam_width,
                "search_space_id": self.search_space.search_space_id,
                "search_space": self.search_space.to_dict(),
                "target_context": target_context,
                "challenge_metadata": dict(challenge_metadata or {}),
            },
        )
        graph.add_representation(root)

        baseline_circuit, baseline = self._compile(
            circuit,
            self.baseline_config,
            default_pipeline=True,
            backend=backend,
            target=target,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            dt=dt,
        )
        baseline_id = f"{challenge_id}:default-baseline"
        if baseline_circuit is not None:
            baseline_node = self.adapter.import_native(
                baseline_circuit,
                representation_id=baseline_id,
                semantic_root=challenge_id,
                metadata={"role": "default_baseline"},
            )
        else:
            baseline_node = Representation(
                id=baseline_id,
                kind=RepresentationKind.EVALUATION,
                semantic_root=challenge_id,
                framework="qiskit",
                payload={"status": "compile_failed", "error": baseline.error},
                metadata={"role": "default_baseline"},
            )
        graph.add_representation(baseline_node)
        graph.add_transformation(TransformationRecord(
            id=f"edge:{challenge_id}:default-baseline",
            input_id=root_id,
            output_id=baseline_id,
            transform_id="qiskit.transpile.default",
            transform_version="0.2",
            equivalence=(baseline.verification.equivalence if baseline.verification else EquivalenceKind.UNKNOWN),
            parameters=self.baseline_config.to_dict(),
            verification=(baseline.verification.to_dict() if baseline.verification else {}),
            outcome={"compile_success": baseline.compile_success},
            metrics_before=numeric_metrics(input_metrics),
            metrics_after=numeric_metrics(baseline.metrics),
            framework="qiskit",
            cost={"compile_seconds": baseline.compile_seconds},
        ))

        seq_root_id = f"{challenge_id}:seq:root"
        seq_root = SequentialState(
            state_id=seq_root_id,
            parent_state_id=None,
            step_index=-1,
            stage=None,
            action=None,
            prefix={},
            candidate=baseline,
            kept_in_beam=True,
        )
        graph.add_representation(Representation(
            id=seq_root_id,
            kind=RepresentationKind.SEARCH_STATE,
            semantic_root=challenge_id,
            framework="qiskit",
            payload={
                "step_index": -1,
                "prefix": {},
                "rollout_role": "default_baseline",
                "metrics": baseline.metrics,
            },
            metadata={"role": "search_root"},
        ))
        graph.add_transformation(TransformationRecord(
            id=f"edge:{challenge_id}:search-init",
            input_id=root_id,
            output_id=seq_root_id,
            transform_id="westquant.search.initialize",
            transform_version="0.2",
            equivalence=EquivalenceKind.UNKNOWN,
            parameters={"beam_width": self.beam_width},
            outcome={"initialized": True},
            metrics_before=numeric_metrics(input_metrics),
            metrics_after=numeric_metrics(baseline.metrics),
            framework="qiskit",
        ))

        states: list[SequentialState] = [seq_root]
        beam: list[SequentialState] = [seq_root]

        for step_index, stage in enumerate(self.search_space.stage_order):
            expanded: list[SequentialState] = []
            compiled_by_state: dict[str, Any | None] = {}
            for parent in beam:
                for action in self.search_space.choices(stage):
                    prefix = {**parent.prefix, stage: action}
                    config = self.search_space.config_for(prefix)
                    compiled, candidate = self._compile(
                        circuit,
                        config,
                        default_pipeline=False,
                        backend=backend,
                        target=target,
                        coupling_map=coupling_map,
                        basis_gates=basis_gates,
                        dt=dt,
                    )
                    state = SequentialState(
                        state_id=_state_id(challenge_id, prefix),
                        parent_state_id=parent.state_id,
                        step_index=step_index,
                        stage=stage,
                        action=action,
                        prefix=prefix,
                        candidate=candidate,
                        terminal=(step_index == len(self.search_space.stage_order) - 1),
                    )
                    expanded.append(state)
                    compiled_by_state[state.state_id] = compiled

            selectable = sorted((s for s in expanded if s.selectable), key=_state_selection_key)
            beam = selectable[: self.beam_width]
            beam_ids = {s.state_id for s in beam}
            for state in expanded:
                state.kept_in_beam = state.state_id in beam_ids
                graph.add_representation(self._state_representation(
                    state=state,
                    compiled=compiled_by_state[state.state_id],
                    challenge_id=challenge_id,
                ))
                parent_state = next((s for s in states if s.state_id == state.parent_state_id), None)
                before_metrics = parent_state.metrics if parent_state else baseline.metrics
                candidate = state.candidate
                graph.add_transformation(TransformationRecord(
                    id=f"edge:{state.state_id}",
                    input_id=state.parent_state_id or seq_root_id,
                    output_id=state.state_id,
                    transform_id=f"qiskit.search.choose.{stage}",
                    transform_version="0.2",
                    equivalence=(candidate.verification.equivalence if candidate and candidate.verification else EquivalenceKind.UNKNOWN),
                    parameters={
                        "stage": stage,
                        "action": state.action,
                        "prefix": state.prefix,
                        "pipeline_rollout": candidate.config.to_dict() if candidate else None,
                    },
                    verification=(candidate.verification.to_dict() if candidate and candidate.verification else {}),
                    outcome={
                        "compile_success": candidate.compile_success if candidate else False,
                        "kept_in_beam": state.kept_in_beam,
                        "terminal": state.terminal,
                        "error": candidate.error if candidate else None,
                    },
                    metrics_before=numeric_metrics(before_metrics),
                    metrics_after=numeric_metrics(candidate.metrics if candidate else {}),
                    framework="qiskit",
                    cost={"compile_seconds": candidate.compile_seconds if candidate else None},
                ))
            states.extend(expanded)
            if not beam:
                break

        graph.metadata["n_action_evaluations"] = len(states) - 1
        graph.metadata["n_compile_success"] = sum(bool(s.candidate and s.candidate.compile_success) for s in states[1:])
        graph.metadata["final_beam_size"] = len(beam)

        return SequentialSearchResult(
            challenge_id=challenge_id,
            graph=graph,
            input_metrics=input_metrics,
            baseline=baseline,
            states=states,
            final_beam=beam,
            beam_width=self.beam_width,
        )
