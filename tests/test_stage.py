import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap, PassManagerConfig, TranspilerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins

from westquant_qiskit.stage import WestQuantSearchPass, _count_2q_gates, _dag_depth

COUPLING = CouplingMap.from_line(5)
BASIS = ["cx", "rz", "sx", "x"]
STAGES = {"layout": "layout_method", "routing": "routing_method", "optimization": "optimization_method"}


def circuit(measure=False):
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 3)
    qc.ry(0.4, 1)
    qc.cx(1, 3)
    qc.cx(0, 2)
    qc.cx(1, 2)  # interaction 4-cycle: cannot be embedded in a line, routing is required
    qc.rz(0.3, 2)
    qc.cx(2, 0)
    qc.cx(3, 1)
    if measure:
        qc.measure_all()
    return qc


def transpile(qc, level=2, **methods):
    pm = generate_preset_pass_manager(optimization_level=level, coupling_map=COUPLING, basis_gates=BASIS, seed_transpiler=7, **methods)
    return pm, pm.run(qc)


def assert_on_coupling(tqc):
    edges = set(COUPLING.get_edges())
    for item in tqc.data:
        if len(item.qubits) == 2:
            pair = tuple(tqc.find_bit(q).index for q in item.qubits)
            assert pair in edges or pair[::-1] in edges


def assert_same_distribution(qc, tqc):
    ideal = Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict()
    physical = tqc.layout.final_index_layout()
    routed = Statevector(tqc.remove_final_measurements(inplace=False)).probabilities_dict(qargs=physical)
    for key in set(ideal) | set(routed):
        assert abs(ideal.get(key, 0) - routed.get(key, 0)) < 1e-8


def assert_selected_is_best(selection):
    ok = [c for c in selection["candidates"] if c["ok"]]
    best = min(ok, key=lambda c: (c["two_qubit"], c["depth"]))
    assert selection["selected"] == best["candidate"]


@pytest.mark.parametrize("stage", list(STAGES))
def test_plugins_are_discoverable(stage):
    assert "westquant" in list_stage_plugins(stage)


@pytest.mark.parametrize("stage", list(STAGES))
def test_stage_runs_and_preserves_unitary(stage):
    qc = circuit()
    pm, tqc = transpile(qc, **{STAGES[stage]: "westquant"})
    selection = pm.property_set["westquant_selection"][stage]
    assert [c["candidate"] for c in selection["candidates"]] == WestQuantSearchPass(stage=stage)._get_candidates()
    assert all(c["ok"] or c["error"] for c in selection["candidates"]), selection
    assert_selected_is_best(selection)
    assert_on_coupling(tqc)
    ancillas = tqc.num_qubits - qc.num_qubits
    assert Operator.from_circuit(tqc).equiv(Operator(qc).expand(Operator(np.eye(2**ancillas))))


def test_all_stages_with_measurements_preserve_distribution():
    qc = circuit(measure=True)
    pm, tqc = transpile(qc, layout_method="westquant", routing_method="westquant", optimization_method="westquant")
    assert set(pm.property_set["westquant_selection"]) == set(STAGES)
    assert tqc.count_ops()["measure"] == 4
    assert_on_coupling(tqc)
    assert_same_distribution(qc, tqc)


def test_routing_pass_returns_selected_candidate_dag():
    physical = QuantumCircuit(5)  # already on physical qubits, but not routed
    physical.cx(0, 4)
    physical.cx(1, 3)
    physical.cx(0, 2)
    physical.cx(4, 1)
    dag = circuit_to_dag(physical)
    search = WestQuantSearchPass(stage="routing", config=PassManagerConfig(coupling_map=COUPLING, seed_transpiler=7))
    out = search(physical)
    assert_on_coupling(out)
    record = search.property_set["westquant_selection"]["routing"]
    chosen = next(c for c in record["candidates"] if c["candidate"] == record["selected"])
    out_dag = circuit_to_dag(out)
    assert (_count_2q_gates(out_dag), _dag_depth(out_dag)) == (chosen["two_qubit"], chosen["depth"])
    assert search.property_set["final_layout"] is not None
    assert _count_2q_gates(out_dag) > _count_2q_gates(dag)  # SWAPs were inserted: the pass acted


@pytest.mark.parametrize("level", [1, 2, 3])
def test_optimization_never_worse_than_default(level):
    qc = circuit()
    _, default = transpile(qc, level=level)
    _, westquant = transpile(qc, level=level, optimization_method="westquant")
    def cost(c):
        dag = circuit_to_dag(c)
        return _count_2q_gates(dag), _dag_depth(dag)
    assert cost(westquant) <= cost(default)


def test_all_candidates_failing_raises():
    search = WestQuantSearchPass(stage="routing", config=PassManagerConfig(coupling_map=None))
    with pytest.raises(TranspilerError, match="All WestQuant routing candidates failed"):
        search(circuit())


def test_optimization_level_zero_is_respected():
    assert WestQuantSearchPass(stage="layout", optimization_level=0).optimization_level == 0
