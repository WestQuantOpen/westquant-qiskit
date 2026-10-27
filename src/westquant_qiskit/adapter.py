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


def _build_v1_lookup(props: Any) -> dict[str, dict[tuple, dict[str, float | None]]]:
    """Build a (gate_name, qubits) -> {error, duration} lookup from a
    BackendV1 ``BackendProperties`` object.

    Returns an empty dict if ``props`` does not expose the V1 ``gate_error`` /
    ``gate_length`` callable interface.
    """
    lookup: dict[str, dict[tuple, dict[str, float | None]]] = {}
    if not (callable(getattr(props, "gate_error", None))
            or callable(getattr(props, "gate_length", None))):
        return lookup

    # BackendProperties exposes gate names + qubits via .gates list.
    gates = getattr(props, "gates", None)
    if not gates:
        return lookup

    for gate in gates:
        name = str(getattr(gate, "name", "")).lower()
        qubits = tuple(getattr(gate, "qubits", ()))
        if not name or not qubits:
            continue
        err = None
        dur = None
        gate_error_fn = getattr(props, "gate_error", None)
        gate_length_fn = getattr(props, "gate_length", None)
        if callable(gate_error_fn):
            try:
                err = float(gate_error_fn(name, list(qubits)))
            except Exception:
                err = None
        if callable(gate_length_fn):
            try:
                dur = float(gate_length_fn(name, list(qubits)))
            except Exception:
                dur = None
        if err is None and dur is None:
            continue
        lookup.setdefault(name, {})[qubits] = {"error": err, "duration": dur}
    return lookup


def _build_v2_lookup(target: Any) -> dict[str, dict[tuple, dict[str, float | None]]]:
    """Build a (gate_name, qubits) -> {error, duration} lookup from a
    BackendV2 ``Target`` object.

    Iterates ``target.instructions`` (list of ``(Instruction, qargs)`` tuples)
    and reads ``target.instruction_properties(index).error`` / ``.duration``.
    Returns an empty dict if ``target`` does not expose the V2 interface.
    """
    lookup: dict[str, dict[tuple, dict[str, float | None]]] = {}
    instructions = getattr(target, "instructions", None)
    instruction_properties = getattr(target, "instruction_properties", None)
    if not instructions or not callable(instruction_properties):
        return lookup

    for index, entry in enumerate(instructions):
        try:
            instr, qargs = entry
        except (ValueError, TypeError):
            continue
        name = str(getattr(instr, "name", "")).lower()
        if not name:
            continue
        qubits = tuple(int(q) for q in qargs) if qargs else ()
        try:
            ip = instruction_properties(index)
        except Exception:
            continue
        if ip is None:
            continue
        err = getattr(ip, "error", None)
        dur = getattr(ip, "duration", None)
        err = float(err) if isinstance(err, (int, float)) else None
        dur = float(dur) if isinstance(dur, (int, float)) else None
        if err is None and dur is None:
            continue
        lookup.setdefault(name, {})[qubits] = {"error": err, "duration": dur}
    return lookup


def noise_aware_metrics(
    circuit: Any,
    backend: Any = None,
    target: Any = None,
) -> dict[str, Any]:
    """Compute noise/fidelity-aware metrics for a circuit.

    Supports both the legacy BackendV1 API (``backend.properties()`` returning
    a ``BackendProperties`` object with ``gate_error``/``gate_length``) and
    the current BackendV2 API (``backend.target`` returning a ``Target``
    object accessed via ``instruction_properties(index).error``/``.duration``).

    Returns a dict with ``estimated_fidelity``, ``total_duration`` and
    ``noise_aware_depth``.  When no per-gate error/duration data is available,
    the corresponding fields are ``None``.
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

    # ------------------------------------------------------------------
    # Build a unified lookup: gate_name -> {qubit_tuple -> {error, duration}}
    # ------------------------------------------------------------------
    lookup: dict[str, dict[tuple, dict[str, float | None]]] = {}

    # 1. Try BackendV1: backend.properties() -> BackendProperties
    if backend is not None:
        props = getattr(backend, "properties", None)
        if callable(props):
            try:
                props = props()
            except Exception:
                props = None
        else:
            props = None
        if props is not None:
            lookup = _build_v1_lookup(props)

    # 2. Try BackendV2: backend.target -> Target (or explicit target arg)
    if not lookup:
        v2_target = None
        if target is not None:
            v2_target = target
        elif backend is not None:
            v2_target = getattr(backend, "target", None)
        if v2_target is not None:
            lookup = _build_v2_lookup(v2_target)

    if not lookup:
        return result

    # ------------------------------------------------------------------
    # Compute fidelity, duration, and noise-aware depth from the lookup.
    # ------------------------------------------------------------------
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

        # Look up this gate: try exact qubit match first, then any-qubit fallback.
        gate_data = None
        name_lookup = lookup.get(name)
        if name_lookup:
            gate_data = name_lookup.get(qubits) or next(iter(name_lookup.values()), None)

        if gate_data is None:
            continue

        err = gate_data.get("error")
        length = gate_data.get("duration")

        if err is not None:
            fidelity *= (1.0 - err)
            have_fidelity = True
        if length is not None:
            total_duration += length
            have_duration = True
            # Weight this gate's contribution to depth by its fidelity.
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
