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
