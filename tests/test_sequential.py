from __future__ import annotations

from westquant_core import EquivalenceKind, Representation, RepresentationKind
from westquant_qiskit.sequential import SequentialSearchEngine, SequentialSearchSpace
from westquant_qiskit.verification import VerificationReport


class Inst:
    def __init__(self, n):
        self.qubits = [object() for _ in range(n)]


class FakeCircuit:
    num_qubits = 4
    num_clbits = 0
    duration = None

    def __init__(self, depth=12, twoq=6, swaps=1):
        self._depth = depth
        self._twoq = twoq
        self._swaps = swaps
        self.data = [Inst(2) for _ in range(twoq)] + [Inst(1) for _ in range(max(0, depth - twoq))]

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
            payload={"fake": True, "depth": circuit.depth()},
            metadata=metadata or {},
        )


class FakeCompiler:
    def compile_default(self, circuit, config, **kwargs):
        return FakeCircuit(depth=10, twoq=5, swaps=1)

    def compile(self, circuit, config, **kwargs):
        layout_bonus = 2 if config.layout_method == "sabre" else 0
        routing_bonus = 1 if config.routing_method == "sabre" else 0
        trans_bonus = 1 if config.translation_method == "synthesis" else 0
        opt_bonus = config.optimization_level
        score = layout_bonus + routing_bonus + trans_bonus + opt_bonus
        return FakeCircuit(
            depth=max(2, 12 - score),
            twoq=max(1, 6 - score // 2),
            swaps=0 if config.routing_method == "sabre" else 1,
        )


class FakeVerifier:
    def verify(self, original, candidate):
        return VerificationReport(
            equivalence=EquivalenceKind.EXACT,
            verified=True,
            method="fake",
        )


def tiny_space():
    return SequentialSearchSpace(
        layout_methods=("trivial", "sabre"),
        routing_methods=("basic", "sabre"),
        translation_methods=("translator", "synthesis"),
        optimization_levels=(0, 3),
    )


def test_sequential_beam_produces_policy_records():
    engine = SequentialSearchEngine(
        search_space=tiny_space(),
        beam_width=2,
        compiler=FakeCompiler(),
        verifier=FakeVerifier(),
        adapter=FakeAdapter(),
    )
    result = engine.search(FakeCircuit(), challenge_id="seq-1")

    # 2 + 4 + 4 + 4 evaluations for beam width 2 across four binary stages.
    assert result.summary()["n_action_evaluations"] == 14
    assert len(result.training_records()) == 14
    assert result.best_state is not None
    assert result.best_state.prefix == {
        "layout": "sabre",
        "routing": "sabre",
        "translation": "synthesis",
        "optimization": 3,
    }
    assert result.graph.validate() == []
    assert any(n.kind == RepresentationKind.SEARCH_STATE for n in result.graph.nodes.values())
    assert all("reward_vector" in r for r in result.training_records())


def test_sequential_search_is_deterministic_with_fake_oracle():
    kwargs = dict(
        search_space=tiny_space(),
        beam_width=2,
        compiler=FakeCompiler(),
        verifier=FakeVerifier(),
        adapter=FakeAdapter(),
    )
    r1 = SequentialSearchEngine(**kwargs).search(FakeCircuit(), challenge_id="same")
    r2 = SequentialSearchEngine(**kwargs).search(FakeCircuit(), challenge_id="same")
    assert [r["prefix_after"] for r in r1.training_records()] == [r["prefix_after"] for r in r2.training_records()]
    assert r1.best_state.prefix == r2.best_state.prefix
