from __future__ import annotations

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
from .config import PipelineConfig, SearchSpace
from .verification import QiskitEquivalenceVerifier, VerificationReport


class CompilerLike(Protocol):
    def compile(self, circuit: Any, config: PipelineConfig, **kwargs: Any) -> Any: ...


class VerifierLike(Protocol):
    def verify(self, original: Any, candidate: Any) -> VerificationReport: ...


DEFAULT_BASELINE_CONFIG = PipelineConfig(
    optimization_level=2,
    layout_method="default",
    routing_method="default",
    translation_method="default",
    seed_transpiler=0,
    approximation_degree=1.0,
)


@dataclass
class CandidateResult:
    candidate_id: str
    config: PipelineConfig
    compile_success: bool
    role: str = "search_candidate"
    metrics: dict[str, Any] = field(default_factory=dict)
    compile_seconds: float | None = None
    verification: VerificationReport | None = None
    error: str | None = None
    representation_id: str | None = None
    pareto: bool = False
    rank: int | None = None

    @property
    def invalid(self) -> bool:
        return bool(
            self.verification is not None
            and self.verification.equivalence == EquivalenceKind.INVALID
        )

    @property
    def selectable(self) -> bool:
        return self.compile_success and not self.invalid

    def objective_tuple(self) -> tuple[float, ...]:
        # Compile time is recorded as cost, not used to choose a winner. This
        # keeps selection reproducible across machines while preserving runtime
        # data for later resource-aware policies.
        if not self.selectable:
            return (math.inf,) * 4
        return (
            float(self.metrics.get("two_qubit_gates", math.inf)),
            float(self.metrics.get("swap_gates", math.inf)),
            float(self.metrics.get("depth", math.inf)),
            float(self.metrics.get("size", math.inf)),
        )

    @staticmethod
    def _deltas(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {}
        for key in ("depth", "size", "two_qubit_gates", "swap_gates", "duration"):
            a, b = after.get(key), before.get(key)
            out[key] = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        return out

    def to_record(
        self,
        *,
        challenge_id: str,
        input_metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
        budget_index: int,
        search_space_id: str,
        target_context: dict[str, Any],
    ) -> dict[str, Any]:
        after = self.metrics
        return {
            "schema_version": "0.1",
            "challenge_id": challenge_id,
            "framework": "qiskit",
            "search_space_id": search_space_id,
            "target_context": target_context,
            "candidate_id": self.candidate_id,
            "role": self.role,
            "budget_index": budget_index,
            "representation_action": self.config.to_dict(),
            "compile_success": self.compile_success,
            "verification": self.verification.to_dict() if self.verification else None,
            "input_metrics": input_metrics,
            "default_baseline_metrics": baseline_metrics,
            "metrics_after": after,
            "delta_vs_input": self._deltas(after, input_metrics),
            "delta_vs_default_baseline": self._deltas(after, baseline_metrics),
            "compile_seconds": self.compile_seconds,
            "pareto": self.pareto,
            "rank": self.rank,
            "error": self.error,
        }


@dataclass
class SearchResult:
    challenge_id: str
    graph: RepGraph
    root_representation: Representation
    input_metrics: dict[str, Any]
    baseline: CandidateResult
    candidates: list[CandidateResult]

    @property
    def baseline_metrics(self) -> dict[str, Any]:
        return self.baseline.metrics

    @property
    def successful(self) -> list[CandidateResult]:
        return [c for c in self.candidates if c.selectable]

    @property
    def verified(self) -> list[CandidateResult]:
        return [c for c in self.successful if c.verification and c.verification.verified]

    @property
    def pareto_front(self) -> list[CandidateResult]:
        return [c for c in self.candidates if c.pareto]

    @property
    def best(self) -> CandidateResult | None:
        pool = self.verified or self.successful
        return min(pool, key=lambda c: c.objective_tuple()) if pool else None

    def training_records(self, *, include_baseline: bool = True) -> list[dict[str, Any]]:
        search_space_id = str(self.graph.metadata.get("search_space_id", "unknown"))
        target_context = dict(self.graph.metadata.get("target_context", {}))
        records = []
        if include_baseline:
            records.append(self.baseline.to_record(
                challenge_id=self.challenge_id,
                input_metrics=self.input_metrics,
                baseline_metrics=self.baseline_metrics,
                budget_index=-1,
                search_space_id=search_space_id,
                target_context=target_context,
            ))
        records.extend(
            c.to_record(
                challenge_id=self.challenge_id,
                input_metrics=self.input_metrics,
                baseline_metrics=self.baseline_metrics,
                budget_index=i,
                search_space_id=search_space_id,
                target_context=target_context,
            )
            for i, c in enumerate(self.candidates)
        )
        return records

    def summary(self) -> dict[str, Any]:
        best = self.best
        improvement = None
        if best is not None and self.baseline.compile_success:
            improvement = CandidateResult._deltas(best.metrics, self.baseline.metrics)
        return {
            "challenge_id": self.challenge_id,
            "n_candidates": len(self.candidates),
            "n_compile_success": sum(c.compile_success for c in self.candidates),
            "n_verified": sum(bool(c.verification and c.verification.verified) for c in self.candidates),
            "n_invalid": sum(c.invalid for c in self.candidates),
            "n_pareto": len(self.pareto_front),
            "input_metrics": self.input_metrics,
            "default_baseline": {
                "compile_success": self.baseline.compile_success,
                "config": self.baseline.config.to_dict(),
                "metrics": self.baseline.metrics,
                "verification": self.baseline.verification.to_dict() if self.baseline.verification else None,
                "error": self.baseline.error,
            },
            "best_candidate_id": best.candidate_id if best else None,
            "best_config": best.config.to_dict() if best else None,
            "best_metrics": best.metrics if best else None,
            "best_delta_vs_default_baseline": improvement,
        }


def _dominates(a: CandidateResult, b: CandidateResult) -> bool:
    av, bv = a.objective_tuple(), b.objective_tuple()
    if any(math.isinf(v) for v in av):
        return False
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))


def _mark_pareto_and_rank(candidates: list[CandidateResult]) -> None:
    selectable = [c for c in candidates if c.selectable]
    for c in selectable:
        c.pareto = not any(_dominates(other, c) for other in selectable if other is not c)
    ordered = sorted(selectable, key=lambda c: c.objective_tuple())
    for rank, c in enumerate(ordered, start=1):
        c.rank = rank


def _target_context(
    *,
    backend: Any = None,
    target: Any = None,
    coupling_map: Any = None,
    basis_gates: list[str] | None = None,
    dt: float | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {"basis_gates": list(basis_gates or []), "dt": dt}
    if backend is not None:
        name = getattr(backend, "name", None)
        context["backend"] = name() if callable(name) else (str(name) if name is not None else type(backend).__name__)
    if target is not None:
        context["target_type"] = type(target).__name__
        try:
            context["target_operations"] = sorted(str(x) for x in target.operation_names)
        except Exception:
            pass
    if coupling_map is not None:
        try:
            context["coupling_edges"] = [list(edge) for edge in coupling_map.get_edges()]
            context["num_physical_qubits"] = int(coupling_map.size())
        except Exception:
            context["coupling_map_type"] = type(coupling_map).__name__
    return context


def _framework_version() -> str:
    try:
        import qiskit
        return str(qiskit.__version__)
    except Exception:
        return "unavailable"


class DeterministicSearchEngine:
    """Budgeted, reproducible search over Qiskit compilation representations."""

    def __init__(
        self,
        *,
        search_space: SearchSpace | None = None,
        compiler: CompilerLike | None = None,
        verifier: VerifierLike | None = None,
        adapter: Any | None = None,
        baseline_config: PipelineConfig = DEFAULT_BASELINE_CONFIG,
    ) -> None:
        self.search_space = search_space or SearchSpace()
        self.compiler = compiler or QiskitCompiler()
        self.verifier = verifier or QiskitEquivalenceVerifier()
        self.adapter = adapter or QiskitAdapter()
        self.baseline_config = baseline_config

    def _compile_one(
        self,
        circuit: Any,
        config: PipelineConfig,
        *,
        role: str,
        use_default_pipeline: bool,
        backend: Any,
        target: Any,
        coupling_map: Any,
        basis_gates: list[str] | None,
        dt: float | None,
    ) -> tuple[Any | None, CandidateResult]:
        started = time.perf_counter()
        try:
            if use_default_pipeline and hasattr(self.compiler, "compile_default"):
                compiled = self.compiler.compile_default(
                    circuit,
                    config,
                    backend=backend,
                    target=target,
                    coupling_map=coupling_map,
                    basis_gates=basis_gates,
                    dt=dt,
                )
            else:
                compiled = self.compiler.compile(
                    circuit,
                    config,
                    backend=backend,
                    target=target,
                    coupling_map=coupling_map,
                    basis_gates=basis_gates,
                    dt=dt,
                )
            elapsed = time.perf_counter() - started
            metrics = circuit_metrics(compiled)
            verification = self.verifier.verify(circuit, compiled)
            return compiled, CandidateResult(
                candidate_id=config.candidate_id,
                config=config,
                compile_success=True,
                role=role,
                metrics=metrics,
                compile_seconds=elapsed,
                verification=verification,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return None, CandidateResult(
                candidate_id=config.candidate_id,
                config=config,
                compile_success=False,
                role=role,
                compile_seconds=elapsed,
                error=repr(exc),
            )

    @staticmethod
    def _add_attempt_to_graph(
        graph: RepGraph,
        *,
        challenge_id: str,
        root_id: str,
        circuit: Any | None,
        result: CandidateResult,
        adapter: Any,
        input_metrics: dict[str, Any],
    ) -> None:
        suffix = "default-baseline" if result.role == "default_baseline" else result.candidate_id
        if circuit is not None:
            representation_id = f"{challenge_id}:{suffix}"
            node = adapter.import_native(
                circuit,
                representation_id=representation_id,
                semantic_root=challenge_id,
                metadata={
                    "role": result.role,
                    "candidate_id": result.candidate_id,
                    "pipeline": result.config.to_dict(),
                },
            )
            result.representation_id = representation_id
            equivalence = result.verification.equivalence if result.verification else EquivalenceKind.UNKNOWN
            verification = result.verification.to_dict() if result.verification else {}
            metrics_after = numeric_metrics(result.metrics)
            outcome = {"compile_success": True, "role": result.role}
        else:
            representation_id = f"{challenge_id}:{suffix}:compile-failure"
            node = Representation(
                id=representation_id,
                kind=RepresentationKind.EVALUATION,
                semantic_root=challenge_id,
                framework="qiskit",
                payload={
                    "status": "compile_failed",
                    "error": result.error,
                    "pipeline": result.config.to_dict(),
                },
                metadata={"candidate_id": result.candidate_id, "role": result.role},
            )
            result.representation_id = representation_id
            equivalence = EquivalenceKind.UNKNOWN
            verification = {
                "equivalence": EquivalenceKind.UNKNOWN.value,
                "verified": False,
                "method": "not_run",
                "reason": "compilation failed before equivalence verification",
            }
            metrics_after = {}
            outcome = {"compile_success": False, "error": result.error, "role": result.role}

        graph.add_representation(node)
        graph.add_transformation(TransformationRecord(
            id=f"edge:{challenge_id}:{suffix}",
            input_id=root_id,
            output_id=representation_id,
            transform_id="qiskit.transpile.default" if result.role == "default_baseline" else "qiskit.transpile.pipeline",
            transform_version="0.1",
            equivalence=equivalence,
            parameters=result.config.to_dict(),
            verification=verification,
            outcome=outcome,
            metrics_before=numeric_metrics(input_metrics),
            metrics_after=metrics_after,
            framework="qiskit",
            cost={"compile_seconds": result.compile_seconds},
        ))

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
        max_candidates: int | None = None,
        challenge_metadata: dict[str, Any] | None = None,
    ) -> SearchResult:
        input_metrics = circuit_metrics(circuit)
        root_id = f"{challenge_id}:root"
        root = self.adapter.import_native(
            circuit,
            representation_id=root_id,
            semantic_root=challenge_id,
            metadata={"role": "logical_input"},
        )
        target_context = _target_context(
            backend=backend, target=target, coupling_map=coupling_map,
            basis_gates=basis_gates, dt=dt,
        )
        graph = RepGraph(
            graph_id=f"repgraph:{challenge_id}",
            metadata={
                "framework": "qiskit",
                "framework_version": _framework_version(),
                "search": "deterministic_grid",
                "search_space_size": len(self.search_space),
                "search_space_id": self.search_space.search_space_id,
                "search_space": self.search_space.to_dict(),
                "target_context": target_context,
                "challenge_metadata": dict(challenge_metadata or {}),
            },
        )
        graph.add_representation(root)

        baseline_circuit, baseline = self._compile_one(
            circuit,
            self.baseline_config,
            role="default_baseline",
            use_default_pipeline=True,
            backend=backend,
            target=target,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            dt=dt,
        )
        self._add_attempt_to_graph(
            graph,
            challenge_id=challenge_id,
            root_id=root_id,
            circuit=baseline_circuit,
            result=baseline,
            adapter=self.adapter,
            input_metrics=input_metrics,
        )

        configs: Iterable[PipelineConfig] = self.search_space.iter_configs()
        if max_candidates is not None:
            if max_candidates < 1:
                raise ValueError("max_candidates must be >= 1")
            configs = list(configs)[:max_candidates]

        results: list[CandidateResult] = []
        for config in configs:
            compiled, result = self._compile_one(
                circuit,
                config,
                role="search_candidate",
                use_default_pipeline=False,
                backend=backend,
                target=target,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                dt=dt,
            )
            self._add_attempt_to_graph(
                graph,
                challenge_id=challenge_id,
                root_id=root_id,
                circuit=compiled,
                result=result,
                adapter=self.adapter,
                input_metrics=input_metrics,
            )
            results.append(result)

        _mark_pareto_and_rank(results)
        graph.metadata["n_candidates"] = len(results)
        graph.metadata["n_compile_success"] = sum(c.compile_success for c in results)
        graph.metadata["n_invalid"] = sum(c.invalid for c in results)
        graph.metadata["default_baseline_compile_success"] = baseline.compile_success
        return SearchResult(
            challenge_id=challenge_id,
            graph=graph,
            root_representation=root,
            input_metrics=input_metrics,
            baseline=baseline,
            candidates=results,
        )
