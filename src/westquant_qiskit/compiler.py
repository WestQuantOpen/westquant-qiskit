from __future__ import annotations

from typing import Any

from .config import PipelineConfig


class QiskitCompiler:
    """Thin wrapper around Qiskit's public preset-pass-manager API."""

    @staticmethod
    def _constraints(*, backend: Any = None, target: Any = None, coupling_map: Any = None, basis_gates: list[str] | None = None, dt: float | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if backend is not None:
            kwargs["backend"] = backend
        if target is not None:
            kwargs["target"] = target
        if coupling_map is not None:
            kwargs["coupling_map"] = coupling_map
        if basis_gates is not None:
            kwargs["basis_gates"] = basis_gates
        if dt is not None:
            kwargs["dt"] = dt
        return kwargs

    def compile(
        self,
        circuit: Any,
        config: PipelineConfig,
        *,
        backend: Any = None,
        target: Any = None,
        coupling_map: Any = None,
        basis_gates: list[str] | None = None,
        dt: float | None = None,
    ) -> Any:
        from qiskit.transpiler import generate_preset_pass_manager

        kwargs: dict[str, Any] = {
            "optimization_level": config.optimization_level,
            "layout_method": config.layout_method,
            "routing_method": config.routing_method,
            "translation_method": config.translation_method,
            "seed_transpiler": config.seed_transpiler,
            "approximation_degree": config.approximation_degree,
            "qubits_initially_zero": False,
            **self._constraints(
                backend=backend, target=target, coupling_map=coupling_map,
                basis_gates=basis_gates, dt=dt,
            ),
        }
        manager = generate_preset_pass_manager(**kwargs)
        return manager.run(circuit)

    def compile_default(
        self,
        circuit: Any,
        config: PipelineConfig,
        *,
        backend: Any = None,
        target: Any = None,
        coupling_map: Any = None,
        basis_gates: list[str] | None = None,
        dt: float | None = None,
    ) -> Any:
        """Frozen Qiskit baseline with no explicit stage-method overrides."""
        from qiskit.transpiler import generate_preset_pass_manager

        kwargs: dict[str, Any] = {
            "optimization_level": config.optimization_level,
            "seed_transpiler": config.seed_transpiler,
            "approximation_degree": config.approximation_degree,
            "qubits_initially_zero": False,
            **self._constraints(
                backend=backend, target=target, coupling_map=coupling_map,
                basis_gates=basis_gates, dt=dt,
            ),
        }
        manager = generate_preset_pass_manager(**kwargs)
        return manager.run(circuit)
