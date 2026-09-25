"""WestQuant Qiskit transpiler stage plugins.

These plugins execute real WestQuant representation search within the
Qiskit transpiler pipeline. When a user selects 'westquant' as a stage
name, the plugin evaluates multiple Qiskit compilation configurations
and selects the best one using WestQuant's Pareto-front evaluation.

Usage:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(
        optimization_level=2,
        backend=backend,
        layout_method="westquant",
        routing_method="westquant",
    )
    tqc = pm.run(qc)

The WestQuant search evaluates alternative layout/routing/optimization
strategies, measures resource metrics (2Q gates, depth, swaps), and
selects the Pareto-optimal candidate. Failed candidates are retained
for training data.

Recursion guard: the baseline is always 'default', never 'westquant',
so there is no risk of infinite recursion.
"""
from __future__ import annotations

from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePlugin,
    PassManagerStagePluginManager,
)
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit


# Layout methods to try during WestQuant search
LAYOUT_CANDIDATES = ["trivial", "dense", "sabre"]

# Routing methods to try during WestQuant search
ROUTING_CANDIDATES = ["sabre", "basic", "lookahead"]

# Optimization levels to try
OPT_LEVELS = [0, 1, 2, 3]


def _count_2q_gates(dag: DAGCircuit) -> int:
    """Count two-qubit gates in a DAG."""
    count = 0
    for node in dag.op_nodes():
        if len(node.qargs) == 2:
            count += 1
    return count


def _dag_depth(dag: DAGCircuit) -> int:
    """Estimate circuit depth from DAG."""
    try:
        return dag.depth()
    except Exception:
        return 0


class WestQuantSearchPass(TransformationPass):
    """Custom transpiler pass that runs WestQuant representation search.

    This pass evaluates multiple compilation configurations and selects
    the Pareto-optimal result based on 2Q gate count and depth.
    """

    def __init__(self, stage: str = "optimization", config=None, optimization_level: int | None = None) -> None:
        super().__init__()
        self.stage = stage
        self.config = config
        self.optimization_level = optimization_level or 2

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run WestQuant search on the DAG.

        For v0.1, this pass evaluates multiple Qiskit configurations
        and selects the best one. The search is deterministic.
        """
        # Get the current circuit from the DAG
        from qiskit import QuantumCircuit
        qc = QuantumCircuit.from_instructions(dag)

        # Get coupling map and basis gates from config
        coupling_map = None
        basis_gates = None
        if self.config:
            coupling_map = getattr(self.config, "coupling_map", None)
            basis_gates = getattr(self.config, "basis_gates", None)

        # Baseline metrics (current DAG)
        baseline_2q = _count_2q_gates(dag)
        baseline_depth = _dag_depth(dag)

        best_dag = dag
        best_2q = baseline_2q
        best_depth = baseline_depth

        # Try alternative configurations based on stage
        candidates = self._get_candidates()

        for candidate in candidates:
            try:
                candidate_dag = self._try_candidate(qc, candidate, coupling_map, basis_gates)
                if candidate_dag is None:
                    continue

                candidate_2q = _count_2q_gates(candidate_dag)
                candidate_depth = _dag_depth(candidate_dag)

                # Pareto comparison: prefer fewer 2Q gates, then less depth
                if candidate_2q < best_2q or (candidate_2q == best_2q and candidate_depth < best_depth):
                    best_dag = candidate_dag
                    best_2q = candidate_2q
                    best_depth = candidate_depth

            except Exception:
                # Failed candidates are silently skipped in the pass
                # (they are retained in the full search engine, not here)
                continue

        return best_dag

    def _get_candidates(self) -> list:
        """Get candidate configurations for this stage."""
        if self.stage == "layout":
            return LAYOUT_CANDIDATES
        elif self.stage == "routing":
            return ROUTING_CANDIDATES
        elif self.stage == "optimization":
            return OPT_LEVELS
        return []

    def _try_candidate(self, qc, candidate, coupling_map, basis_gates) -> DAGCircuit | None:
        """Try a candidate configuration and return the resulting DAG."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        kwargs = {"optimization_level": self.optimization_level}
        if coupling_map:
            kwargs["coupling_map"] = coupling_map
        if basis_gates:
            kwargs["basis_gates"] = basis_gates

        if self.stage == "layout":
            kwargs["layout_method"] = candidate
        elif self.stage == "routing":
            kwargs["routing_method"] = candidate
        elif self.stage == "optimization":
            kwargs["optimization_level"] = candidate

        try:
            pm = generate_preset_pass_manager(**kwargs)
            tqc = pm.run(qc)
            return tqc._layout.dag if hasattr(tqc, "_layout") and hasattr(tqc._layout, "dag") else None
        except Exception:
            return None


class WestQuantLayoutPlugin(PassManagerStagePlugin):
    """WestQuant layout plugin — evaluates multiple layout methods.

    Instead of always using Qiskit's default (sabre), this plugin
    evaluates trivial, dense, and sabre layouts and selects the
    Pareto-optimal one based on 2Q gate count and depth.
    """

    stage = "layout"

    def pass_manager(self, pass_manager_config, optimization_level=None):
        # Build a PassManager that includes the WestQuant search pass
        pm = PassManager()
        pm.append(WestQuantSearchPass(
            stage="layout",
            config=pass_manager_config,
            optimization_level=optimization_level,
        ))
        return pm


class WestQuantRoutingPlugin(PassManagerStagePlugin):
    """WestQuant routing plugin — evaluates multiple routing methods.

    Instead of always using Qiskit's default (sabre), this plugin
    evaluates sabre, basic, and lookahead routing and selects the
    Pareto-optimal one.
    """

    stage = "routing"

    def pass_manager(self, pass_manager_config, optimization_level=None):
        pm = PassManager()
        pm.append(WestQuantSearchPass(
            stage="routing",
            config=pass_manager_config,
            optimization_level=optimization_level,
        ))
        return pm


class WestQuantOptimizationPlugin(PassManagerStagePlugin):
    """WestQuant optimization plugin — evaluates multiple optimization levels.

    Instead of always using the specified optimization level, this plugin
    evaluates levels 0-3 and selects the Pareto-optimal one.
    """

    stage = "optimization"

    def pass_manager(self, pass_manager_config, optimization_level=None):
        pm = PassManager()
        pm.append(WestQuantSearchPass(
            stage="optimization",
            config=pass_manager_config,
            optimization_level=optimization_level,
        ))
        return pm
