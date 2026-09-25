"""WestQuant Qiskit transpiler stage plugins.

IMPORTANT: These plugins currently DELEGATE to Qiskit's default stage
pass managers. They do not execute WestQuant representation search.

The actual WestQuant search runs through the CLI:
    westquant-qiskit-search --suite smoke --search sequential --beam-width 2

These plugins exist so that:
1. Qiskit discovers 'westquant' in layout/routing/optimization groups.
2. Future versions can wire real WestQuant optimization into the
   Qiskit transpiler pipeline without changing the entry-point contract.
3. Users can reference 'westquant' as a stage name in generate_preset_pass_manager.

Recursion guard: the baseline is always 'default', never 'westquant',
so there is no risk of infinite recursion.
"""
from __future__ import annotations

from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePlugin,
    PassManagerStagePluginManager,
)


class _DelegatingStage(PassManagerStagePlugin):
    """Delegates to Qiskit's default stage pass manager.

    This is NOT WestQuant optimization. It is a placeholder that
    ensures plugin discovery works. Real WestQuant search runs
    through the CLI, not through Qiskit's transpiler stage hooks.
    """

    stage: str
    baseline: str = "default"

    def pass_manager(self, pass_manager_config, optimization_level=None):
        manager = PassManagerStagePluginManager()
        return manager.get_passmanager_stage(
            self.stage,
            self.baseline,
            pass_manager_config,
            optimization_level=optimization_level,
        )


class WestQuantLayoutPlugin(_DelegatingStage):
    """Delegates layout to Qiskit default. Not WestQuant search."""

    stage = "layout"


class WestQuantRoutingPlugin(_DelegatingStage):
    """Delegates routing to Qiskit default. Not WestQuant search."""

    stage = "routing"


class WestQuantOptimizationPlugin(_DelegatingStage):
    """Delegates optimization to Qiskit default. Not WestQuant search."""

    stage = "optimization"
