from __future__ import annotations

from typing import Any
from westquant_core import Representation, RepresentationKind


def _bit_index(circuit: Any, bit: Any) -> int | str:
    try:
        return int(circuit.find_bit(bit).index)
    except Exception:
        try:
            return int(list(circuit.qubits).index(bit))
        except Exception:
            return str(bit)


def _safe_param(value: Any) -> float | int | str:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def circuit_instructions(circuit: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for position, instruction in enumerate(circuit.data):
        op = getattr(instruction, "operation", instruction)
        qubits = getattr(instruction, "qubits", ())
        clbits = getattr(instruction, "clbits", ())
        out.append({
            "position": position,
            "name": str(getattr(op, "name", type(op).__name__)),
            "qubits": [_bit_index(circuit, q) for q in qubits],
            "clbits": [str(c) for c in clbits],
            "params": [_safe_param(p) for p in getattr(op, "params", ())],
        })
    return out


def circuit_metrics(circuit: Any) -> dict[str, Any]:
    counts = {str(k): int(v) for k, v in circuit.count_ops().items()}
    two_qubit = 0
    for instruction in circuit.data:
        if len(getattr(instruction, "qubits", ())) == 2:
            two_qubit += 1
    duration = getattr(circuit, "duration", None)
    return {
        "n_qubits": int(circuit.num_qubits),
        "n_clbits": int(circuit.num_clbits),
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size()),
        "two_qubit_gates": int(two_qubit),
        "swap_gates": int(counts.get("swap", 0)),
        "duration": float(duration) if isinstance(duration, (int, float)) else None,
        "operations": counts,
    }


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    keys = ("n_qubits", "n_clbits", "depth", "size", "two_qubit_gates", "swap_gates", "duration")
    return {key: metrics.get(key) for key in keys}


class QiskitAdapter:
    framework = "qiskit"

    def import_native(
        self,
        circuit: Any,
        *,
        representation_id: str = "qiskit:circuit",
        semantic_root: str | None = None,
        include_instructions: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Representation:
        payload: dict[str, Any] = {"metrics": circuit_metrics(circuit)}
        if include_instructions:
            payload["instructions"] = circuit_instructions(circuit)
        global_phase = getattr(circuit, "global_phase", None)
        if global_phase is not None:
            payload["global_phase"] = _safe_param(global_phase)
        return Representation(
            id=representation_id,
            kind=RepresentationKind.CIRCUIT,
            semantic_root=semantic_root,
            framework=self.framework,
            payload=payload,
            metadata={"native_type": type(circuit).__name__, **(metadata or {})},
        )
