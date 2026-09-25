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


def noise_aware_metrics(
    circuit: Any,
    backend: Any = None,
    target: Any = None,
) -> dict[str, Any]:
    """Compute noise/fidelity-aware metrics for a circuit.

    Returns a dict with ``estimated_fidelity``, ``total_duration`` and
    ``noise_aware_depth``.  When ``backend``/``target`` is ``None`` or does not
    expose per-gate error/duration data, the corresponding fields are ``None``.
    """
    result: dict[str, Any] = {
        "estimated_fidelity": None,
        "total_duration": None,
        "noise_aware_depth": None,
    }

    # Collect gate names in execution order.
    ops: list[str] = []
    for instruction in getattr(circuit, "data", []):
        op = getattr(instruction, "operation", instruction)
        ops.append(str(getattr(op, "name", type(op).__name__)).lower())

    if not ops:
        return result

    # Try to obtain a properties/props source from backend or target.
    props = None
    if backend is not None:
        props = getattr(backend, "properties", None)
        if callable(props):
            try:
                props = props()
            except Exception:
                props = None
    if props is None and target is not None:
        props = getattr(target, "props", None)

    if props is None:
        return result

    # Build per-gate error rate and duration lookup tables.
    def _gate_error(props: Any, name: str, qubits: tuple) -> float | None:
        getter = getattr(props, "gate_error", None)
        if callable(getter):
            try:
                return float(getter(name, qubits))
            except Exception:
                return None
        return None

    def _gate_length(props: Any, name: str, qubits: tuple) -> float | None:
        for attr in ("gate_length", "gate_duration"):
            getter = getattr(props, attr, None)
            if callable(getter):
                try:
                    return float(getter(name, qubits))
                except Exception:
                    continue
        return None

    fidelity = 1.0
    total_duration = 0.0
    weighted_depth = 0.0
    have_fidelity = False
    have_duration = False

    for instruction in getattr(circuit, "data", []):
        op = getattr(instruction, "operation", instruction)
        name = str(getattr(op, "name", type(op).__name__)).lower()
        qubits = tuple(
            int(getattr(circuit.find_bit(q), "index", 0))
            for q in getattr(instruction, "qubits", ())
        )
        err = _gate_error(props, name, qubits)
        length = _gate_length(props, name, qubits)
        if err is not None:
            fidelity *= (1.0 - err)
            have_fidelity = True
        if length is not None:
            total_duration += length
            have_duration = True
            # Weight this gate's contribution to depth by its error rate.
            weighted_depth += (1.0 - (err if err is not None else 0.0)) * length

    if have_fidelity:
        result["estimated_fidelity"] = fidelity
    if have_duration:
        result["total_duration"] = total_duration
        result["noise_aware_depth"] = weighted_depth
    return result


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
