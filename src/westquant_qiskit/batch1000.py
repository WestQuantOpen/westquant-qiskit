from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .challenges import Challenge
from .io import save_sequential_result
from .sequential import SequentialSearchEngine, SequentialSearchSpace

GENERATOR_VERSION = "0.1"
DEFAULT_MASTER_SEED = 20260924
FAMILIES = ("ghz", "qft_like", "random_entangling", "hardware_efficient", "qaoa_ring")
TOPOLOGIES = ("line", "ring", "grid", "star", "ladder")
QUBIT_SIZES = (4, 5, 6, 8, 10, 12, 14, 16)
VARIANTS = tuple(range(5))
DEFAULT_BASIS_GATES = ("rz", "sx", "x", "cx")


@dataclass(frozen=True)
class ChallengeSpec:
    challenge_id: str
    ordinal: int
    family: str
    topology: str
    n_qubits: int
    variant: int
    circuit_seed: int
    parameters: dict[str, Any]
    basis_gates: tuple[str, ...] = DEFAULT_BASIS_GATES

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["basis_gates"] = list(self.basis_gates)
        return value


def _stable_seed(master_seed: int, *parts: Any) -> int:
    text = "|".join([str(master_seed), *(str(p) for p in parts)])
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _spec_parameters(family: str, n: int, variant: int) -> dict[str, Any]:
    if family == "ghz":
        return {"permuted_chain": True}
    if family == "qft_like":
        return {"phase_scale": (1.0, 0.75, 0.5, 1.25, 1.5)[variant]}
    if family == "random_entangling":
        return {"depth": (4, 6, 8, 12, 16)[variant]}
    if family == "hardware_efficient":
        return {"layers": (2, 3, 4, 6, 8)[variant]}
    if family == "qaoa_ring":
        return {"p": (1, 2, 3, 4, 5)[variant]}
    raise KeyError(family)


def generate_1000_specs(master_seed: int = DEFAULT_MASTER_SEED) -> list[ChallengeSpec]:
    """Return exactly 1000 stable, balanced challenge specifications.

    Design: 5 circuit families x 5 target topologies x 8 sizes x 5 variants.
    IDs and circuit seeds are deterministic for a fixed master seed.
    """
    specs: list[ChallengeSpec] = []
    ordinal = 0
    for family in FAMILIES:
        for topology in TOPOLOGIES:
            for n in QUBIT_SIZES:
                for variant in VARIANTS:
                    circuit_seed = _stable_seed(master_seed, family, topology, n, variant)
                    short = hashlib.sha256(
                        f"{master_seed}|{family}|{topology}|{n}|{variant}".encode()
                    ).hexdigest()[:8]
                    challenge_id = (
                        f"wq1000-{ordinal:04d}-{family}-{topology}-"
                        f"n{n:02d}-v{variant}-{short}"
                    )
                    specs.append(ChallengeSpec(
                        challenge_id=challenge_id,
                        ordinal=ordinal,
                        family=family,
                        topology=topology,
                        n_qubits=n,
                        variant=variant,
                        circuit_seed=circuit_seed,
                        parameters=_spec_parameters(family, n, variant),
                    ))
                    ordinal += 1
    if len(specs) != 1000:
        raise AssertionError(f"generator invariant violated: expected 1000, got {len(specs)}")
    if len({s.challenge_id for s in specs}) != 1000:
        raise AssertionError("challenge IDs are not unique")
    return specs


def _bidirectional(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for a, b in edges:
        if a == b:
            continue
        out.add((a, b))
        out.add((b, a))
    return sorted(out)


def topology_edges(topology: str, n: int) -> list[tuple[int, int]]:
    if n < 2:
        return []
    if topology == "line":
        return _bidirectional((i, i + 1) for i in range(n - 1))
    if topology == "ring":
        edges = [(i, i + 1) for i in range(n - 1)]
        if n > 2:
            edges.append((n - 1, 0))
        return _bidirectional(edges)
    if topology == "star":
        return _bidirectional((0, i) for i in range(1, n))
    if topology == "ladder":
        # Zig-zag ladder that remains connected for odd n.
        split = (n + 1) // 2
        top = list(range(split))
        bottom = list(range(split, n))
        edges: list[tuple[int, int]] = []
        edges.extend(zip(top, top[1:]))
        edges.extend(zip(bottom, bottom[1:]))
        edges.extend((top[i], bottom[i]) for i in range(min(len(top), len(bottom))))
        if len(top) > len(bottom) and bottom:
            edges.append((top[-1], bottom[-1]))
        return _bidirectional(edges)
    if topology == "grid":
        rows = max(2, int(math.floor(math.sqrt(n))))
        cols = int(math.ceil(n / rows))
        edges: list[tuple[int, int]] = []
        for idx in range(n):
            r, c = divmod(idx, cols)
            right = idx + 1
            down = idx + cols
            if c + 1 < cols and right < n:
                edges.append((idx, right))
            if r + 1 < rows and down < n:
                edges.append((idx, down))
        # The truncated last row can rarely leave an awkward tail; line edge is
        # a deterministic connectivity guard, harmless if already present.
        edges.extend((i, i + 1) for i in range(n - 1) if not edges)
        return _bidirectional(edges)
    raise ValueError(f"unsupported topology: {topology}")


def _coupling_map(topology: str, n: int) -> Any:
    from qiskit.transpiler import CouplingMap

    return CouplingMap(couplinglist=topology_edges(topology, n))


def _ghz_variant(n: int, seed: int) -> Any:
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    qc = QuantumCircuit(n, name=f"ghz_variant_{n}_{seed}")
    qc.h(order[0])
    for a, b in zip(order, order[1:]):
        qc.cx(a, b)
    return qc


def _qft_like_variant(n: int, phase_scale: float) -> Any:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name=f"qft_like_{n}_{phase_scale:g}")
    for target in range(n):
        qc.h(target)
        for control in range(target + 1, n):
            angle = phase_scale * math.pi / (2 ** (control - target))
            qc.cp(angle, control, target)
    for i in range(n // 2):
        qc.swap(i, n - i - 1)
    return qc


def _random_entangling(n: int, depth: int, seed: int) -> Any:
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    qc = QuantumCircuit(n, name=f"random_entangling_{n}_{depth}_{seed}")
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice(("h", "x", "sx", "rz", "rx", "ry"))
            if gate in {"rz", "rx", "ry"}:
                getattr(qc, gate)(rng.uniform(-math.pi, math.pi), q)
            else:
                getattr(qc, gate)(q)
        order = list(range(n))
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            qc.cx(a, b)
    return qc


def _hardware_efficient(n: int, layers: int, seed: int) -> Any:
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    qc = QuantumCircuit(n, name=f"hardware_efficient_{n}_{layers}_{seed}")
    for layer in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-math.pi, math.pi), q)
            qc.rz(rng.uniform(-math.pi, math.pi), q)
        offset = layer % 2
        for q in range(offset, n - 1, 2):
            qc.cx(q, q + 1)
        if n > 2:
            qc.cx(n - 1, 0)
    return qc


def _qaoa_ring(n: int, p: int, seed: int) -> Any:
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    qc = QuantumCircuit(n, name=f"qaoa_ring_{n}_p{p}_{seed}")
    for q in range(n):
        qc.h(q)
    for _ in range(p):
        gamma = rng.uniform(0.05, 1.25)
        beta = rng.uniform(0.05, 1.25)
        for q in range(n):
            qc.rzz(2.0 * gamma, q, (q + 1) % n)
        for q in range(n):
            qc.rx(2.0 * beta, q)
    return qc


def materialize_challenge(spec: ChallengeSpec) -> Challenge:
    if spec.family == "ghz":
        circuit = _ghz_variant(spec.n_qubits, spec.circuit_seed)
    elif spec.family == "qft_like":
        circuit = _qft_like_variant(spec.n_qubits, float(spec.parameters["phase_scale"]))
    elif spec.family == "random_entangling":
        circuit = _random_entangling(spec.n_qubits, int(spec.parameters["depth"]), spec.circuit_seed)
    elif spec.family == "hardware_efficient":
        circuit = _hardware_efficient(spec.n_qubits, int(spec.parameters["layers"]), spec.circuit_seed)
    elif spec.family == "qaoa_ring":
        circuit = _qaoa_ring(spec.n_qubits, int(spec.parameters["p"]), spec.circuit_seed)
    else:
        raise ValueError(f"unknown family: {spec.family}")

    metadata = {
        "dataset": "westquant-qiskit-sequential-1000",
        "generator_version": GENERATOR_VERSION,
        "ordinal": spec.ordinal,
        "family": spec.family,
        "topology": spec.topology,
        "n_qubits": spec.n_qubits,
        "variant": spec.variant,
        "circuit_seed": spec.circuit_seed,
        "parameters": spec.parameters,
    }
    return Challenge(
        challenge_id=spec.challenge_id,
        circuit=circuit,
        coupling_map=_coupling_map(spec.topology, spec.n_qubits),
        basis_gates=spec.basis_gates,
        metadata=metadata,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(output: Path, specs: list[ChallengeSpec], master_seed: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": "westquant-qiskit-sequential-1000",
        "schema_version": "0.1",
        "generator_version": GENERATOR_VERSION,
        "created_at": _utc_now(),
        "master_seed": master_seed,
        "n_challenges": len(specs),
        "factorial_design": {
            "families": list(FAMILIES),
            "topologies": list(TOPOLOGIES),
            "n_qubits": list(QUBIT_SIZES),
            "variants": list(VARIANTS),
            "formula": "5 families x 5 topologies x 8 sizes x 5 variants = 1000",
        },
        "specs": [s.to_dict() for s in specs],
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _public_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "repr": repr(exc),
    }


def run_spec(
    spec: ChallengeSpec,
    *,
    root: Path,
    beam_width: int,
    force: bool = False,
) -> dict[str, Any]:
    challenge_dir = root / "challenges" / spec.challenge_id
    run_path = challenge_dir / "run.json"
    if run_path.exists() and not force:
        try:
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                return existing
        except Exception:
            pass

    challenge_dir.mkdir(parents=True, exist_ok=True)
    started_wall = _utc_now()
    started = time.perf_counter()
    run_record: dict[str, Any] = {
        "challenge_id": spec.challenge_id,
        "status": "running",
        "started_at": started_wall,
        "spec": spec.to_dict(),
        "beam_width": beam_width,
    }
    run_path.write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        challenge = materialize_challenge(spec)
        engine = SequentialSearchEngine(
            search_space=SequentialSearchSpace(seed_transpiler=spec.circuit_seed),
            beam_width=beam_width,
        )
        result = engine.search(
            challenge.circuit,
            challenge_id=challenge.challenge_id,
            coupling_map=challenge.coupling_map,
            basis_gates=list(challenge.basis_gates),
            challenge_metadata=challenge.metadata,
        )
        save_sequential_result(result, challenge_dir)
        baseline_verification = (
            result.baseline.verification.to_dict() if result.baseline.verification else None
        )
        run_record.update({
            "status": "completed",
            "completed_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "baseline_compile_success": result.baseline.compile_success,
            "baseline_verification": baseline_verification,
            "baseline_error": result.baseline.error,
            "summary": result.summary(),
        })
    except Exception as exc:
        run_record.update({
            "status": "runner_error",
            "completed_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": _public_error(exc),
            "traceback": traceback.format_exc(),
        })

    run_path.write_text(json.dumps(run_record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return run_record


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _describe(values: Iterable[float]) -> dict[str, Any]:
    xs = sorted(float(x) for x in values)
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        k = (len(xs) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - k) + xs[hi] * (k - lo)

    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p10": pct(0.10),
        "p90": pct(0.90),
        "min": xs[0],
        "max": xs[-1],
    }


def _error_type(error: Any) -> str:
    if not error:
        return "none"
    text = str(error)
    match = re.match(r"(?:[\w.]+\.)?([A-Za-z_][A-Za-z0-9_]*)", text)
    return match.group(1) if match else "unknown"


def _relation(best: dict[str, Any] | None, baseline: dict[str, Any] | None) -> str:
    if not best or not baseline:
        return "not_comparable"
    keys = ("two_qubit_gates", "swap_gates", "depth", "size")
    pairs: list[tuple[float, float]] = []
    for key in keys:
        a, b = _num(best.get(key)), _num(baseline.get(key))
        if a is not None and b is not None:
            pairs.append((a, b))
    if not pairs:
        return "not_comparable"
    better = any(a < b for a, b in pairs)
    worse = any(a > b for a, b in pairs)
    no_worse = all(a <= b for a, b in pairs)
    no_better = all(a >= b for a, b in pairs)
    if no_worse and better:
        return "westquant_dominates_baseline"
    if no_better and worse:
        return "baseline_dominates_westquant"
    if not better and not worse:
        return "equal"
    return "tradeoff"


def aggregate_statistics(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs_by_id = {item["challenge_id"]: item for item in manifest["specs"]}

    challenge_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    stage_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    strata: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    merged_trajectory = root / "all_trajectories.jsonl"

    with merged_trajectory.open("w", encoding="utf-8") as merged:
        for challenge_id, spec in specs_by_id.items():
            cdir = root / "challenges" / challenge_id
            run_path = cdir / "run.json"
            if not run_path.exists():
                challenge_rows.append({
                    **spec,
                    "status": "not_run",
                    "elapsed_seconds": None,
                    "baseline_compile_success": None,
                    "search_has_best": False,
                    "relation_to_baseline": "not_comparable",
                })
                continue
            run = json.loads(run_path.read_text(encoding="utf-8"))
            summary = run.get("summary") or {}
            best = summary.get("best_metrics")
            baseline = summary.get("default_baseline_metrics") or {}
            baseline_ok = run.get("baseline_compile_success")
            has_best = best is not None
            relation = _relation(best, baseline) if baseline_ok and has_best else "not_comparable"
            if baseline_ok is False and has_best:
                relation = "baseline_failed_search_succeeded"
            elif baseline_ok is True and not has_best:
                relation = "baseline_succeeded_search_failed"
            elif baseline_ok is False and not has_best:
                relation = "both_failed"

            crow = {
                **spec,
                "status": run.get("status"),
                "elapsed_seconds": run.get("elapsed_seconds"),
                "baseline_compile_success": baseline_ok,
                "search_has_best": has_best,
                "relation_to_baseline": relation,
                "n_action_evaluations": summary.get("n_action_evaluations"),
                "n_compile_success": summary.get("n_compile_success"),
                "best_state_id": summary.get("best_state_id"),
                "best_prefix": summary.get("best_prefix"),
                "best_metrics": best,
                "default_baseline_metrics": baseline,
                "runner_error_type": (run.get("error") or {}).get("type"),
            }
            challenge_rows.append(crow)
            strata[(spec["family"], spec["topology"], int(spec["n_qubits"]))].append(crow)

            if run.get("status") == "runner_error":
                failure_counts[f"runner:{(run.get('error') or {}).get('type', 'unknown')}"] += 1

            trajectory_path = cdir / "trajectory.jsonl"
            if not trajectory_path.exists():
                continue
            for line in trajectory_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                enriched = {
                    **record,
                    "family": spec["family"],
                    "topology": spec["topology"],
                    "n_qubits": spec["n_qubits"],
                    "variant": spec["variant"],
                }
                action_rows.append(enriched)
                merged.write(json.dumps(enriched, sort_keys=True, default=str) + "\n")
                action_value = record.get("action")
                if isinstance(action_value, dict):
                    action_value = action_value.get("name", str(action_value))
                stage_action[(str(record.get("stage")), str(action_value))].append(enriched)
                if not record.get("compile_success"):
                    failure_counts[f"compile:{_error_type(record.get('error'))}"] += 1
                verification = record.get("verification") or {}
                if str(verification.get("equivalence", "unknown")).lower() == "invalid":
                    failure_counts["verification:INVALID"] += 1

    completed = [r for r in challenge_rows if r.get("status") == "completed"]
    runner_errors = [r for r in challenge_rows if r.get("status") == "runner_error"]
    not_run = [r for r in challenge_rows if r.get("status") == "not_run"]
    compile_success_rows = [r for r in action_rows if r.get("compile_success", r.get("success", False))]
    compile_failed_rows = [r for r in action_rows if not r.get("compile_success", r.get("success", False))]
    exact_rows = [r for r in action_rows if str((r.get("verification") or {}).get("equivalence", "unknown")).lower() == "exact"]
    invalid_rows = [r for r in action_rows if str((r.get("verification") or {}).get("equivalence", "unknown")).lower() == "invalid"]
    unknown_rows = [r for r in action_rows if not r.get("verification") or str((r.get("verification") or {}).get("equivalence", "unknown")).lower() == "unknown"]

    metric_names = ("two_qubit_gates", "swap_gates", "depth", "size", "duration")
    delta_stats: dict[str, Any] = {}
    for metric in metric_names:
        values = [
            float(r["delta_vs_default_baseline"][metric])
            for r in action_rows
            if isinstance((r.get("delta_vs_default_baseline") or {}).get(metric), (int, float))
        ]
        delta_stats[metric] = {
            **_describe(values),
            "improved_fraction": (sum(v < 0 for v in values) / len(values)) if values else None,
            "equal_fraction": (sum(v == 0 for v in values) / len(values)) if values else None,
            "worse_fraction": (sum(v > 0 for v in values) / len(values)) if values else None,
        }

    verification_counts = Counter(
        str((r.get("verification") or {}).get("equivalence", "unknown")).upper() for r in action_rows
    )
    def action_outcome(row: dict[str, Any]) -> str:
        explicit = row.get("outcome_class")
        if explicit:
            return str(explicit)
        if not row.get("compile_success", row.get("success", False)):
            return "compile_failed"
        eq = str((row.get("verification") or {}).get("equivalence", "unknown")).lower()
        verified = bool((row.get("verification") or {}).get("verified"))
        if eq == "invalid":
            return "verification_invalid"
        if eq == "exact" and verified:
            return "verified_exact"
        return "compiled_unknown"

    action_outcome_counts = Counter(action_outcome(r) for r in action_rows)
    relation_counts = Counter(r.get("relation_to_baseline", "unknown") for r in challenge_rows)

    by_stage_action: list[dict[str, Any]] = []
    for (stage, action), rows in sorted(stage_action.items()):
        deltas_2q = [
            float(r["delta_vs_default_baseline"]["two_qubit_gates"])
            for r in rows
            if isinstance((r.get("delta_vs_default_baseline") or {}).get("two_qubit_gates"), (int, float))
        ]
        by_stage_action.append({
            "stage": stage,
            "action": action,
            "attempts": len(rows),
            "compile_success": sum(bool(r.get("compile_success", r.get("success", False))) for r in rows),
            "compile_success_rate": sum(bool(r.get("compile_success", r.get("success", False))) for r in rows) / len(rows),
            "kept_in_beam": sum(bool(r.get("kept_in_beam")) for r in rows),
            "kept_rate": sum(bool(r.get("kept_in_beam")) for r in rows) / len(rows),
            "exact": sum(str((r.get("verification") or {}).get("equivalence", "unknown")).lower() == "exact" for r in rows),
            "invalid": sum(str((r.get("verification") or {}).get("equivalence", "unknown")).lower() == "invalid" for r in rows),
            "unknown": sum(str((r.get("verification") or {}).get("equivalence", "unknown")).lower() == "unknown" for r in rows),
            "delta_2q_vs_baseline": _describe(deltas_2q),
            "compile_seconds": _describe([
                float(r["compile_seconds"]) for r in rows if isinstance(r.get("compile_seconds"), (int, float))
            ]),
        })

    by_stratum: list[dict[str, Any]] = []
    for (family, topology, n), rows in sorted(strata.items()):
        completed_rows = [r for r in rows if r.get("status") == "completed"]
        by_stratum.append({
            "family": family,
            "topology": topology,
            "n_qubits": n,
            "n": len(rows),
            "completed": len(completed_rows),
            "runner_errors": sum(r.get("status") == "runner_error" for r in rows),
            "baseline_compile_success_rate": (
                sum(r.get("baseline_compile_success") is True for r in completed_rows) / len(completed_rows)
                if completed_rows else None
            ),
            "search_has_best_rate": (
                sum(bool(r.get("search_has_best")) for r in completed_rows) / len(completed_rows)
                if completed_rows else None
            ),
            "westquant_dominates_rate": (
                sum(r.get("relation_to_baseline") == "westquant_dominates_baseline" for r in completed_rows) / len(completed_rows)
                if completed_rows else None
            ),
            "elapsed_seconds": _describe([
                float(r["elapsed_seconds"]) for r in completed_rows if isinstance(r.get("elapsed_seconds"), (int, float))
            ]),
        })

    stats = {
        "dataset_id": manifest["dataset_id"],
        "generated_at": _utc_now(),
        "planned_challenges": len(challenge_rows),
        "completed_challenges": len(completed),
        "runner_error_challenges": len(runner_errors),
        "not_run_challenges": len(not_run),
        "challenge_completion_rate": len(completed) / len(challenge_rows) if challenge_rows else None,
        "action_evaluations": len(action_rows),
        "compile_success_actions": len(compile_success_rows),
        "compile_failed_actions": len(compile_failed_rows),
        "compile_success_rate": len(compile_success_rows) / len(action_rows) if action_rows else None,
        "verification_counts": dict(verification_counts),
        "action_outcomes": dict(action_outcome_counts),
        "exact_verification_rate": len(exact_rows) / len(action_rows) if action_rows else None,
        "invalid_verification_rate": len(invalid_rows) / len(action_rows) if action_rows else None,
        "unknown_verification_rate": len(unknown_rows) / len(action_rows) if action_rows else None,
        "challenge_outcomes": dict(relation_counts),
        "challenge_elapsed_seconds": _describe([
            float(r["elapsed_seconds"]) for r in completed if isinstance(r.get("elapsed_seconds"), (int, float))
        ]),
        "action_compile_seconds": _describe([
            float(r["compile_seconds"]) for r in action_rows if isinstance(r.get("compile_seconds"), (int, float))
        ]),
        "delta_vs_default_baseline": delta_stats,
        "failure_taxonomy": dict(failure_counts.most_common()),
        "by_stage_action": by_stage_action,
        "by_stratum": by_stratum,
    }

    (root / "statistics.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    _write_csv(root / "challenge_summary.csv", challenge_rows)
    _write_csv(root / "stage_action_summary.csv", by_stage_action)
    _write_csv(root / "strata_summary.csv", by_stratum)
    _write_failure_csv(root / "failure_summary.csv", failure_counts)
    _write_dataset_card(root / "DATASET_CARD.md", manifest, stats)
    return stats


def _flat(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flat(row.get(key)) for key in fields})


def _write_failure_csv(path: Path, counts: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["failure_class", "count"])
        writer.writeheader()
        for name, count in counts.most_common():
            writer.writerow({"failure_class": name, "count": count})


def _write_dataset_card(path: Path, manifest: dict[str, Any], stats: dict[str, Any]) -> None:
    outcomes = stats.get("challenge_outcomes", {})
    text = "# WestQuant Qiskit Sequential 1000 - Dataset Card\n\n"
    text += f"- Dataset ID: `{manifest['dataset_id']}`\n"
    text += f"- Generator version: `{manifest['generator_version']}`\n"
    text += f"- Master seed: `{manifest['master_seed']}`\n"
    text += f"- Planned challenges: {stats['planned_challenges']}\n"
    text += f"- Completed: {stats['completed_challenges']}\n"
    text += f"- Runner errors: {stats['runner_error_challenges']}\n"
    text += f"- Action evaluations: {stats['action_evaluations']}\n"
    text += f"- Compile success rate: {stats['compile_success_rate']}\n"
    text += f"- Exact verification rate: {stats['exact_verification_rate']}\n\n"
    text += "## Balanced design\n\n"
    text += "5 circuit families x 5 target topologies x 8 qubit sizes x 5 variants = 1000 challenges.\n\n"
    text += "## Challenge outcomes\n\n"
    for key, value in sorted(outcomes.items()):
        text += f"- `{key}`: {value}\n"
    text += "\n## Failure retention\n\n"
    text += (
        "Compilation failures, invalid equivalence checks, UNKNOWN verification outcomes, "
        "beam-pruned actions, and challenge-level runner errors are retained. They are data, "
        "not missing observations.\n"
    )
    text += "\n## Files\n\n"
    text += "- `manifest.json`: frozen challenge specifications\n"
    text += "- `all_trajectories.jsonl`: merged action-level WQT records\n"
    text += "- `statistics.json`: aggregate statistics\n"
    text += "- `challenge_summary.csv`: one row per challenge\n"
    text += "- `stage_action_summary.csv`: action-level aggregate statistics\n"
    text += "- `strata_summary.csv`: family/topology/size aggregates\n"
    text += "- `failure_summary.csv`: failure taxonomy\n"
    path.write_text(text, encoding="utf-8")


def _load_specs(path: Path) -> list[ChallengeSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ChallengeSpec(
        challenge_id=s["challenge_id"],
        ordinal=int(s["ordinal"]),
        family=s["family"],
        topology=s["topology"],
        n_qubits=int(s["n_qubits"]),
        variant=int(s["variant"]),
        circuit_seed=int(s["circuit_seed"]),
        parameters=dict(s["parameters"]),
        basis_gates=tuple(s.get("basis_gates", DEFAULT_BASIS_GATES)),
    ) for s in payload["specs"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and run the balanced WestQuant 1000 sequential Qiskit challenges"
    )
    parser.add_argument("--output", default="results/westquant-qiskit-sequential-1000")
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N specs; manifest still contains 1000")
    parser.add_argument("--start", type=int, default=0, help="Start ordinal for this invocation")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"

    if args.stats_only:
        stats = aggregate_statistics(root)
        print(json.dumps({
            "completed_challenges": stats["completed_challenges"],
            "action_evaluations": stats["action_evaluations"],
            "compile_success_rate": stats["compile_success_rate"],
        }, sort_keys=True))
        return 0

    if manifest_path.exists():
        specs = _load_specs(manifest_path)
    else:
        specs = generate_1000_specs(args.master_seed)
        write_manifest(root, specs, args.master_seed)

    if args.manifest_only:
        print(str(manifest_path))
        return 0

    selected = [s for s in specs if s.ordinal >= args.start]
    if args.limit is not None:
        selected = selected[: args.limit]

    counts = Counter()
    for index, spec in enumerate(selected, start=1):
        record = run_spec(spec, root=root, beam_width=args.beam_width, force=args.force)
        counts[record.get("status", "unknown")] += 1
        print(json.dumps({
            "progress": f"{index}/{len(selected)}",
            "challenge_id": spec.challenge_id,
            "status": record.get("status"),
            "elapsed_seconds": record.get("elapsed_seconds"),
        }, sort_keys=True))

    stats = aggregate_statistics(root)
    print(json.dumps({
        "run_status_counts": dict(counts),
        "completed_challenges": stats["completed_challenges"],
        "action_evaluations": stats["action_evaluations"],
        "compile_success_rate": stats["compile_success_rate"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
