from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from westquant_core import EquivalenceKind


@dataclass(frozen=True)
class VerificationReport:
    equivalence: EquivalenceKind
    verified: bool
    method: str
    reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalence": self.equivalence.value,
            "verified": self.verified,
            "method": self.method,
            "reason": self.reason,
            "diagnostics": self.diagnostics,
        }


class QiskitEquivalenceVerifier:
    """Conservative circuit-equivalence verification.

    Exact unitary verification is attempted only when both circuits are small,
    unitary, and have the same width.  Qiskit's Operator.from_circuit() accounts
    for a transpiled circuit's stored layout.  Larger/non-unitary cases are
    explicitly UNKNOWN rather than silently trusted.
    """

    NON_UNITARY_NAMES = {"measure", "reset", "initialize", "delay"}

    def __init__(self, *, max_operator_qubits: int = 8, bind_parameters: bool = True) -> None:
        self.max_operator_qubits = max_operator_qubits
        self.bind_parameters = bind_parameters

    @staticmethod
    def _operation_name(item: Any) -> str:
        op = getattr(item, "operation", item)
        return str(getattr(op, "name", type(op).__name__)).lower()

    @staticmethod
    def _has_parameters(circuit: Any) -> bool:
        """Return True if the circuit contains any Parameter objects."""
        try:
            from qiskit.circuit import Parameter
        except Exception:
            return False
        for instruction in list(getattr(circuit, "data", [])):
            op = getattr(instruction, "operation", instruction)
            for param in getattr(op, "params", ()):
                if isinstance(param, Parameter):
                    return True
        return False

    @staticmethod
    def _bind_parameters(circuit: Any) -> Any:
        """Bind every Parameter in the circuit to 0.5 (deterministic default)."""
        params = getattr(circuit, "parameters", None)
        if not params:
            return circuit
        binding = {param: 0.5 for param in params}
        return circuit.assign_parameters(binding)

    def verify(self, original: Any, candidate: Any) -> VerificationReport:
        # Handle parameterized circuits: Operator() cannot compute the unitary
        # of a circuit that still has free parameters.
        original_has_params = self._has_parameters(original)
        candidate_has_params = self._has_parameters(candidate)
        if original_has_params or candidate_has_params:
            if not self.bind_parameters:
                return VerificationReport(
                    equivalence=EquivalenceKind.UNKNOWN,
                    verified=False,
                    method="operator",
                    reason="parameterized circuit — set bind_parameters=True to verify",
                )
            if original_has_params:
                original = self._bind_parameters(original)
            if candidate_has_params:
                candidate = self._bind_parameters(candidate)

        if int(original.num_qubits) != int(candidate.num_qubits):
            return VerificationReport(
                equivalence=EquivalenceKind.UNKNOWN,
                verified=False,
                method="operator",
                reason="circuit widths differ; exact operator comparison skipped",
                diagnostics={
                    "original_qubits": int(original.num_qubits),
                    "candidate_qubits": int(candidate.num_qubits),
                },
            )

        names = {self._operation_name(item) for item in list(original.data) + list(candidate.data)}
        bad = sorted(names & self.NON_UNITARY_NAMES)
        if bad:
            return VerificationReport(
                equivalence=EquivalenceKind.UNKNOWN,
                verified=False,
                method="operator",
                reason="non-unitary operations present",
                diagnostics={"non_unitary_operations": bad},
            )

        if int(original.num_qubits) > self.max_operator_qubits:
            return VerificationReport(
                equivalence=EquivalenceKind.UNKNOWN,
                verified=False,
                method="operator",
                reason="circuit exceeds exact operator verification limit",
                diagnostics={"max_operator_qubits": self.max_operator_qubits},
            )

        try:
            from qiskit.quantum_info import Operator

            original_op = Operator.from_circuit(original, ignore_set_layout=True)
            candidate_op = Operator.from_circuit(candidate)
            equivalent = bool(original_op.equiv(candidate_op))
        except Exception as exc:  # QiskitError and optional numerical failures
            return VerificationReport(
                equivalence=EquivalenceKind.UNKNOWN,
                verified=False,
                method="operator",
                reason="operator verification raised an exception",
                diagnostics={"exception": repr(exc)},
            )

        return VerificationReport(
            equivalence=EquivalenceKind.EXACT if equivalent else EquivalenceKind.INVALID,
            verified=True,
            method="qiskit.quantum_info.Operator.from_circuit().equiv",
            reason=None if equivalent else "operators are not equivalent up to global phase",
        )
