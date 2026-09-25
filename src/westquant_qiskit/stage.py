from __future__ import annotations

from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePlugin,
    PassManagerStagePluginManager,
)


class _DelegatingStage(PassManagerStagePlugin):
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
    stage = "layout"


class WestQuantRoutingPlugin(_DelegatingStage):
    stage = "routing"


class WestQuantOptimizationPlugin(_DelegatingStage):
    stage = "optimization"
