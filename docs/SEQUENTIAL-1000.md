# WestQuant Qiskit Sequential 1000

A balanced, resumable generator for 1000 sequential Qiskit representation-search challenges.

## Design

The dataset is a full factorial design:

- 5 circuit families: `ghz`, `qft_like`, `random_entangling`, `hardware_efficient`, `qaoa_ring`
- 5 target topologies: `line`, `ring`, `grid`, `star`, `ladder`
- 8 qubit sizes: `4, 5, 6, 8, 10, 12, 14, 16`
- 5 variants per cell

Total: `5 x 5 x 8 x 5 = 1000` challenges.

With beam width 3, a fully surviving challenge evaluates up to 30 sequential actions plus one Qiskit default baseline, so the full run is approximately 31,000 transpilation calls. Failed beams can reduce that total.

## 1. Freeze the manifest

```bash
westquant-qiskit-1000 \
  --output results/wq-qiskit-sequential-1000 \
  --master-seed 20260924 \
  --manifest-only
```

Do not regenerate the manifest after the first production run starts.

## 2. Pilot 25 challenges

```bash
westquant-qiskit-1000 \
  --output results/wq-qiskit-sequential-1000 \
  --beam-width 3 \
  --limit 25
```

Inspect `statistics.json`, `DATASET_CARD.md`, and a sample of successful and failed `trajectory.jsonl` rows.

## 3. Run all 1000

```bash
westquant-qiskit-1000 \
  --output results/wq-qiskit-sequential-1000 \
  --beam-width 3
```

The runner is resumable. A repeated invocation skips challenges whose `run.json` status is `completed`. Use `--force` only when intentionally rerunning and version the new results separately.

## 4. Recompute statistics without rerunning

```bash
westquant-qiskit-1000 \
  --output results/wq-qiskit-sequential-1000 \
  --stats-only
```

## Failure retention

Every sequential action is kept, including:

- successful exact verification: `verified_exact`
- compiled but not exactly verified: `compiled_unknown`
- explicit equivalence failure: `verification_invalid`
- transpilation/compilation failure: `compile_failed`
- beam-pruned actions (`kept_in_beam=false`)
- challenge-level runner exceptions (`status=runner_error`)

UNKNOWN is not treated as proof of correctness and is not treated as a compile failure.

## Outputs

Top level:

- `manifest.json`: frozen 1000 challenge specs
- `all_trajectories.jsonl`: merged action-level WQT records
- `statistics.json`: global statistics
- `challenge_summary.csv`: one row per challenge
- `stage_action_summary.csv`: action and stage aggregates
- `strata_summary.csv`: family/topology/qubit-size aggregates
- `failure_summary.csv`: failure taxonomy
- `DATASET_CARD.md`: generated dataset card

Per challenge:

- `run.json`: challenge-level execution status, timing, baseline status, and runner errors
- `summary.json`: sequential search summary
- `repgraph.json`: complete representation/search graph
- `trajectory.jsonl`: state-action-next-state outcomes

## Statistical reporting principles

Always separate:

1. compile success from exact verification;
2. UNKNOWN verification from INVALID;
3. baseline failures from WestQuant failures;
4. beam pruning from failure;
5. challenge-level runner errors from candidate-level compile errors.

Report both successful and unsuccessful attempts. Do not delete failed rows, silently rerun them, or condition aggregate performance only on successful candidates without explicitly labeling the conditioning.
