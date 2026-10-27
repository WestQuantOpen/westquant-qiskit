"""WestQuant Qiskit transpiler stage plugins.

These plugins execute a deterministic best-of search within the Qiskit
transpiler pipeline. When a user selects 'westquant' as a stage name, the
plugin evaluates several built-in Qiskit strategies for that stage and keeps
the candidate with the lowest two-qubit cost (SWAP counted as three 2Q gates),
breaking ties by depth.

Usage:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(
        optimization_level=2,
        backend=backend,
        layout_method="westquant",
        routing_method="westquant",
        optimization_method="westquant",
        seed_transpiler=7,
    )
    tqc = pm.run(qc)
    pm.property_set["westquant_selection"]  # per-stage candidate records

Stage semantics:
    layout        Scores trivial/dense/sabre initial layouts by their fully
                  transpiled cost, sets ``property_set["layout"]`` to the best
                  one and then applies Qiskit's standard embedding.
    routing       Runs SabreSwap, BasicSwap and LookaheadSwap on the physical
                  DAG and keeps the cheapest routed DAG and its final layout,
                  wrapped in Qiskit's standard routing stage.
    optimization  Runs Qiskit's default optimization stage for levels 0-3 and
                  keeps the cheapest result.

Every candidate result is recorded. Failed candidates are logged as warnings
and recorded with their error; if every candidate fails, a TranspilerError is
raised instead of silently returning the input. Candidates are always built
from Qiskit's built-in methods, never 'westquant', so there is no recursion.
Results are deterministic for a fixed ``seed_transpiler`` (default seed 0).
"""
from __future__ import annotations

import copy
import logging

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import Layout, PassManager, TranspilerError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap, SabreSwap, SetLayout
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePlugin,
    PassManagerStagePluginManager,
)

logger = logging.getLogger(__name__)

# Layout methods to try during WestQuant search
LAYOUT_CANDIDATES = ["trivial", "dense", "sabre"]

# Routing methods to try during WestQuant search
ROUTING_CANDIDATES = ["sabre", "basic", "lookahead"]

# Optimization levels to try
OPT_LEVELS = [0, 1, 2, 3]


def _count_2q_gates(dag: DAGCircuit) -> int:
    """Two-qubit cost of a DAG; a SWAP counts as three 2Q gates."""
    return sum(3 if node.name == "swap" else 1 for node in dag.op_nodes() if len(node.qargs) == 2)


def _dag_depth(dag: DAGCircuit) -> int:
    return dag.depth()


def _coupling(config):
    target = getattr(config, "target", None)
    return getattr(config, "coupling_map", None) or (target.build_coupling_map() if target is not None else None)


class WestQuantSearchPass(TransformationPass):
    """Evaluate the candidates of one transpiler stage and apply the best one."""

    def __init__(self, stage: str = "optimization", config=None, optimization_level: int | None = None) -> None:
        super().__init__()
        if stage not in ("layout", "routing", "optimization"):
            raise TranspilerError(f"Unknown WestQuant stage: {stage!r}")
        self.stage = stage
        self.config = config
        self.optimization_level = 2 if optimization_level is None else optimization_level
        seed = getattr(config, "seed_transpiler", None)
        self.seed = 0 if seed is None or seed < 0 else seed

    def _get_candidates(self) -> list:
        return {"layout": LAYOUT_CANDIDATES, "routing": ROUTING_CANDIDATES, "optimization": OPT_LEVELS}[self.stage]

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        records, best = [], None
        for candidate in self._get_candidates():
            try:
                result = self._try_candidate(dag, candidate)
            except Exception as exc:  # recorded and logged, never silently dropped
                logger.warning("WestQuant %s candidate %r failed: %s", self.stage, candidate, exc)
                records.append({"candidate": candidate, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                continue
            cost_dag, applied = result
            record = {"candidate": candidate, "ok": True, "two_qubit": _count_2q_gates(cost_dag), "depth": _dag_depth(cost_dag)}
            records.append(record)
            if best is None or (record["two_qubit"], record["depth"]) < (best[0]["two_qubit"], best[0]["depth"]):
                best = (record, applied)
        if best is None:
            raise TranspilerError(f"All WestQuant {self.stage} candidates failed: {records}")
        record, applied = best
        selection = dict(self.property_set["westquant_selection"] or {})
        selection[self.stage] = {"selected": record["candidate"], "candidates": records}
        self.property_set["westquant_selection"] = selection
        return self._apply(dag, applied)

    def _try_candidate(self, dag: DAGCircuit, candidate):
        """Return (DAG used for scoring, payload needed to apply the candidate)."""
        if self.stage == "layout":
            return self._layout_candidate(dag, candidate)
        if self.stage == "routing":
            return self._routing_candidate(dag, candidate)
        return self._optimization_candidate(dag, candidate)

    def _layout_candidate(self, dag, method):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        cfg = self.config
        routing = cfg.routing_method if cfg.routing_method not in (None, "westquant") else "sabre"
        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level,
            target=cfg.target,
            coupling_map=None if cfg.target is not None else cfg.coupling_map,
            basis_gates=None if cfg.target is not None else cfg.basis_gates,
            layout_method=method,
            routing_method=routing,
            seed_transpiler=self.seed,
        )
        tqc = pm.run(dag_to_circuit(dag))
        physical = tqc.layout.initial_index_layout(filter_ancillas=True)
        return circuit_to_dag(tqc), Layout({dag.qubits[i]: p for i, p in enumerate(physical)})

    def _routing_candidate(self, dag, method):
        coupling = _coupling(self.config)
        if coupling is None:
            raise TranspilerError("Routing requires a coupling map or target.")
        routing_pass = {
            "sabre": lambda: SabreSwap(coupling, heuristic="decay", seed=self.seed),
            "basic": lambda: BasicSwap(coupling),
            "lookahead": lambda: LookaheadSwap(coupling),
        }[method]()
        routed = routing_pass.run(copy.deepcopy(dag))
        return routed, (routed, routing_pass.property_set["final_layout"])

    def _optimization_candidate(self, dag, level):
        pm = PassManagerStagePluginManager().get_passmanager_stage("optimization", "default", self.config, optimization_level=level)
        out = circuit_to_dag(pm.run(dag_to_circuit(dag))) if pm is not None else copy.deepcopy(dag)
        return out, out

    def _apply(self, dag, applied):
        if self.stage == "layout":
            self.property_set["layout"] = applied
            return dag
        if self.stage == "routing":
            routed, final_layout = applied
            if final_layout is not None:
                current = self.property_set["final_layout"]
                self.property_set["final_layout"] = final_layout if current is None else current.compose(final_layout, dag.qubits)
            return routed
        return applied


class WestQuantLayoutPlugin(PassManagerStagePlugin):
    """WestQuant layout plugin: best of trivial, dense and sabre layouts."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        pm = PassManager([SetLayout(pass_manager_config.initial_layout)])
        coupling = pass_manager_config.target or pass_manager_config.coupling_map
        if coupling is None:
            return pm
        pm.append(ConditionalController(
            WestQuantSearchPass(stage="layout", config=pass_manager_config, optimization_level=optimization_level),
            condition=lambda property_set: not property_set["layout"],
        ))
        pm.append(common.generate_embed_passmanager(coupling).to_flow_controller())
        return pm


class WestQuantRoutingPlugin(PassManagerStagePlugin):
    """WestQuant routing plugin: best of SabreSwap, BasicSwap and LookaheadSwap."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        level = 2 if optimization_level is None else optimization_level
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            level, pass_manager_config.layout_method, pass_manager_config.initial_layout
        )
        return common.generate_routing_passmanager(
            WestQuantSearchPass(stage="routing", config=pass_manager_config, optimization_level=level),
            pass_manager_config.target,
            coupling_map=pass_manager_config.coupling_map,
            vf2_call_limit=None if level == 0 else vf2_call_limit,
            vf2_max_trials=None if level == 0 else vf2_max_trials,
            seed_transpiler=-1,
            check_trivial=level == 1,
            use_barrier_before_measurement=True,
        )


class WestQuantOptimizationPlugin(PassManagerStagePlugin):
    """WestQuant optimization plugin: best of Qiskit optimization levels 0-3."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        return PassManager([
            WestQuantSearchPass(stage="optimization", config=pass_manager_config, optimization_level=optimization_level)
        ])
