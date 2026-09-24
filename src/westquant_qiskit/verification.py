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

    def __init__(self, *, max_operator_qubits: int = 8) -> None:
        self.max_operator_qubits = max_operator_qubits

    @staticmethod
    def _operation_name(item: Any) -> str:
        op = getattr(item, "operation", item)
        return str(getattr(op, "name", type(op).__name__)).lower()

    def verify(self, original: Any, candidate: Any) -> VerificationReport:
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
