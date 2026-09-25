from __future__ import annotations

from dataclasses import dataclass

from westquant_core import EquivalenceKind, Representation, RepresentationKind
from westquant_qiskit.config import SearchSpace
from westquant_qiskit.search import DeterministicSearchEngine
from westquant_qiskit.verification import VerificationReport


class Inst:
    def __init__(self, n):
        self.qubits = [object() for _ in range(n)]


class FakeCircuit:
    num_qubits = 4
    num_clbits = 0
    duration = None

    def __init__(self, depth=10, twoq=5, swaps=0):
        self._depth = depth
        self._twoq = twoq
        self._swaps = swaps
        self.data = [Inst(2) for _ in range(twoq)] + [Inst(1) for _ in range(max(0, depth-twoq))]

    def count_ops(self):
        return {"cx": self._twoq, "swap": self._swaps}

    def depth(self):
        return self._depth

    def size(self):
        return len(self.data)


class FakeAdapter:
    def import_native(self, circuit, *, representation_id, semantic_root=None, metadata=None, **kwargs):
        return Representation(
            id=representation_id,
            kind=RepresentationKind.CIRCUIT,
            semantic_root=semantic_root,
            framework="qiskit",
            payload={"fake": True},
            metadata=metadata or {},
        )


class FakeCompiler:
    def compile(self, circuit, config, **kwargs):
        # Higher optimization level reduces depth and two-qubit count in this toy oracle.
        return FakeCircuit(
            depth=max(2, 10-config.optimization_level),
            twoq=max(1, 5-config.optimization_level),
            swaps=0 if config.routing_method == "sabre" else 1,
        )


class FakeVerifier:
    def verify(self, original, candidate):
        return VerificationReport(
            equivalence=EquivalenceKind.EXACT,
            verified=True,
            method="fake",
        )


def test_search_space_is_deterministic():
    space = SearchSpace.smoke()
    ids1 = [c.candidate_id for c in space.iter_configs()]
    ids2 = [c.candidate_id for c in space.iter_configs()]
    assert len(space) == 8
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)


def test_search_builds_repgraph_and_selects_best():
    engine = DeterministicSearchEngine(
        search_space=SearchSpace.smoke(),
        compiler=FakeCompiler(),
        verifier=FakeVerifier(),
        adapter=FakeAdapter(),
    )
    result = engine.search(FakeCircuit(), challenge_id="fake-1")
    assert len(result.candidates) == 8
    assert len(result.graph.nodes) == 10
    assert len(result.graph.edges) == 9
    assert result.graph.validate() == []
    assert result.best is not None
    assert result.best.config.optimization_level == 2
    assert result.best.config.routing_method == "sabre"
    assert result.baseline.role == "default_baseline"
    assert result.baseline.compile_success
    assert len(result.training_records()) == 9
    assert any(c.pareto for c in result.candidates)


def test_failed_candidate_becomes_training_record_not_node():
    class SometimesFail(FakeCompiler):
        def compile(self, circuit, config, **kwargs):
            if config.layout_method == "trivial":
                raise RuntimeError("routing impossible")
            return super().compile(circuit, config, **kwargs)

    engine = DeterministicSearchEngine(
        search_space=SearchSpace.smoke(),
        compiler=SometimesFail(),
        verifier=FakeVerifier(),
        adapter=FakeAdapter(),
    )
    result = engine.search(FakeCircuit(), challenge_id="fake-fail")
    failed = [c for c in result.candidates if not c.compile_success]
    assert failed
    assert len(result.training_records()) == 9
    assert len(result.graph.nodes) == 10
    assert len(result.graph.edges) == 9
    assert any(node.kind == RepresentationKind.EVALUATION for node in result.graph.nodes.values())
