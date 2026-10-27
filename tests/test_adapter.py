from westquant_qiskit.adapter import circuit_metrics


class Inst:
    def __init__(self, n):
        self.qubits = [object() for _ in range(n)]


class FakeCircuit:
    num_qubits = 3
    num_clbits = 1
    data = [Inst(1), Inst(2), Inst(2)]
    def count_ops(self): return {"h": 1, "cx": 2}
    def depth(self): return 3
    def size(self): return 3


def test_metrics():
    m = circuit_metrics(FakeCircuit())
    assert m["two_qubit_gates"] == 2
    assert m["depth"] == 3


def test_noise_aware_metrics_backend_v2_target():
    from qiskit import QuantumCircuit, transpile
    from qiskit.providers.fake_provider import GenericBackendV2
    from westquant_qiskit.adapter import noise_aware_metrics

    backend = GenericBackendV2(num_qubits=3, seed=1)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    tqc = transpile(qc, backend=backend, seed_transpiler=1)
    metrics = noise_aware_metrics(tqc, backend=backend)
    assert 0 < metrics["estimated_fidelity"] < 1
    assert metrics["total_duration"] > 0
