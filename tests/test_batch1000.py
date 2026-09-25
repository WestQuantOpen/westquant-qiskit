from __future__ import annotations

import json
from pathlib import Path

from westquant_qiskit.batch1000 import (
    FAMILIES,
    QUBIT_SIZES,
    TOPOLOGIES,
    VARIANTS,
    aggregate_statistics,
    generate_1000_specs,
    topology_edges,
    write_manifest,
)


def test_generate_exactly_1000_balanced_specs():
    specs = generate_1000_specs(1234)
    assert len(specs) == 1000
    assert len({s.challenge_id for s in specs}) == 1000

    counts = {}
    for spec in specs:
        key = (spec.family, spec.topology, spec.n_qubits, spec.variant)
        counts[key] = counts.get(key, 0) + 1

    assert len(counts) == len(FAMILIES) * len(TOPOLOGIES) * len(QUBIT_SIZES) * len(VARIANTS)
    assert set(counts.values()) == {1}


def test_spec_generation_is_deterministic():
    a = generate_1000_specs(42)
    b = generate_1000_specs(42)
    c = generate_1000_specs(43)
    assert [x.challenge_id for x in a] == [x.challenge_id for x in b]
    assert [x.circuit_seed for x in a] == [x.circuit_seed for x in b]
    assert [x.challenge_id for x in a] != [x.challenge_id for x in c]


def test_topologies_are_bidirectional_and_cover_nodes():
    for topology in TOPOLOGIES:
        for n in (4, 5, 8, 16):
            edges = topology_edges(topology, n)
            assert edges
            edge_set = set(edges)
            for a, b in edges:
                assert (b, a) in edge_set
            touched = {q for edge in edges for q in edge}
            assert touched == set(range(n))


def test_aggregate_keeps_successes_and_failures(tmp_path: Path):
    specs = generate_1000_specs(7)
    root = tmp_path / "batch"
    write_manifest(root, specs, 7)

    first = specs[0]
    cdir = root / "challenges" / first.challenge_id
    cdir.mkdir(parents=True)
    run = {
        "challenge_id": first.challenge_id,
        "status": "completed",
        "elapsed_seconds": 1.25,
        "baseline_compile_success": True,
        "summary": {
            "n_action_evaluations": 2,
            "n_compile_success": 1,
            "best_state_id": "s1",
            "best_prefix": {"layout": "sabre"},
            "best_metrics": {"two_qubit_gates": 8, "swap_gates": 1, "depth": 12, "size": 20},
            "default_baseline_metrics": {"two_qubit_gates": 10, "swap_gates": 2, "depth": 15, "size": 24},
        },
    }
    (cdir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    records = [
        {
            "challenge_id": first.challenge_id,
            "stage": "layout",
            "action": "sabre",
            "compile_success": True,
            "verification": {"equivalence": "EXACT"},
            "delta_vs_default_baseline": {"two_qubit_gates": -2, "swap_gates": -1, "depth": -3, "size": -4, "duration": None},
            "compile_seconds": 0.2,
            "kept_in_beam": True,
        },
        {
            "challenge_id": first.challenge_id,
            "stage": "routing",
            "action": "basic",
            "compile_success": False,
            "verification": None,
            "delta_vs_default_baseline": {"two_qubit_gates": None, "swap_gates": None, "depth": None, "size": None, "duration": None},
            "compile_seconds": 0.1,
            "kept_in_beam": False,
            "error": "TranspilerError('bad route')",
        },
    ]
    with (cdir / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    stats = aggregate_statistics(root)
    assert stats["completed_challenges"] == 1
    assert stats["not_run_challenges"] == 999
    assert stats["action_evaluations"] == 2
    assert stats["compile_success_actions"] == 1
    assert stats["compile_failed_actions"] == 1
    assert stats["verification_counts"]["EXACT"] == 1
    assert stats["verification_counts"]["UNKNOWN"] == 1
    assert stats["challenge_outcomes"]["westquant_dominates_baseline"] == 1
    assert stats["failure_taxonomy"]["compile:TranspilerError"] == 1
    assert (root / "all_trajectories.jsonl").exists()
    assert (root / "statistics.json").exists()
    assert (root / "challenge_summary.csv").exists()
    assert (root / "stage_action_summary.csv").exists()
    assert (root / "strata_summary.csv").exists()
    assert (root / "failure_summary.csv").exists()
    assert (root / "DATASET_CARD.md").exists()
