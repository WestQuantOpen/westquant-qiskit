from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    circuit: Any
    coupling_map: Any
    basis_gates: tuple[str, ...] = ("rz", "sx", "x", "cx")
    metadata: dict[str, Any] | None = None


def _line_edges(n: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(n - 1):
        edges.extend([(i, i + 1), (i + 1, i)])
    return edges


def _ring_edges(n: int) -> list[tuple[int, int]]:
    edges = _line_edges(n)
    if n > 2:
        edges.extend([(0, n - 1), (n - 1, 0)])
    return edges


def coupling_map(topology: str, n: int) -> Any:
    from qiskit.transpiler import CouplingMap

    if topology == "line":
        edges = _line_edges(n)
    elif topology == "ring":
        edges = _ring_edges(n)
    else:
        raise ValueError(f"unsupported topology: {topology}")
    return CouplingMap(couplinglist=edges)


def ghz_circuit(n: int) -> Any:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def qft_like_circuit(n: int) -> Any:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name=f"qft_like_{n}")
    for target in range(n):
        qc.h(target)
        for control in range(target + 1, n):
            # Controlled-phase angles are deliberately simple and deterministic.
            angle = 3.141592653589793 / (2 ** (control - target))
            qc.cp(angle, control, target)
    for i in range(n // 2):
        qc.swap(i, n - i - 1)
    return qc


def random_entangling_circuit(n: int, depth: int, seed: int) -> Any:
    import random
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    qc = QuantumCircuit(n, name=f"random_{n}_{depth}_{seed}")
    one_qubit = ("h", "x", "sx", "rz")
    for layer in range(depth):
        for q in range(n):
            gate = one_qubit[rng.randrange(len(one_qubit))]
            if gate == "rz":
                qc.rz(rng.uniform(-3.141592653589793, 3.141592653589793), q)
            else:
                getattr(qc, gate)(q)
        order = list(range(n))
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            qc.cx(a, b)
    return qc


def generate_suite(mode: str = "smoke") -> Iterator[Challenge]:
    if mode == "smoke":
        specs = [
            ("ghz4-line", ghz_circuit(4), "line", 4),
            ("qft4-line", qft_like_circuit(4), "line", 4),
            ("random4d4-ring", random_entangling_circuit(4, 4, 42), "ring", 4),
        ]
    elif mode == "train":
        specs = []
        for n in (4, 5, 6, 7, 8):
            specs.append((f"ghz{n}-line", ghz_circuit(n), "line", n))
            specs.append((f"qft{n}-line", qft_like_circuit(n), "line", n))
            for depth in (4, 8, 16):
                for seed in (0, 1, 2):
                    specs.append((
                        f"random{n}d{depth}s{seed}-ring",
                        random_entangling_circuit(n, depth, seed),
                        "ring",
                        n,
                    ))
    else:
        raise ValueError("mode must be 'smoke' or 'train'")

    for cid, circuit, topology, n in specs:
        yield Challenge(
            challenge_id=cid,
            circuit=circuit,
            coupling_map=coupling_map(topology, n),
            metadata={"family": cid.split(str(n))[0], "topology": topology, "n_qubits": n},
        )
