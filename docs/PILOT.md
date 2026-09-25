# Pilot protocol: Qiskit deterministic representation search

## Goal

Validate the full pipeline from Qiskit circuit to RepGraph/WQT-ready traces,
then quantify how often deterministic representation search changes compilation
outcomes relative to a frozen Qiskit default baseline.

## Gate 0: environment

- Python 3.10-3.12
- Qiskit 2.5.x
- `westquant-core` 0.2 alpha
- `westquant-qiskit` 0.1 alpha

Record package versions and machine metadata with the run artifact.

## Gate 1: smoke

Run:

```bash
westquant-qiskit-search \
  --suite smoke \
  --output results/qiskit-smoke
```

Expected search volume:

- 3 challenge circuits
- 8 search candidates per challenge
- 1 frozen Qiskit default baseline per challenge
- 27 total training rows

Pass conditions:

- every challenge produces `summary.json`, `repgraph.json`, `training.jsonl`;
- RepGraph validates as a DAG;
- no silent equivalence claims;
- compilation failures are retained.

## Gate 2: training pilot

Run:

```bash
westquant-qiskit-search \
  --suite train \
  --seed-count 3 \
  --output results/qiskit-train
```

Current v0.1 volume:

- 55 structured/random circuits
- 216 explicit pipeline candidates per circuit
- 11,880 WestQuant candidate traces
- 55 frozen Qiskit default baseline traces
- **11,935 training rows total**

All circuits in the first pilot are at most 8 qubits so the exact-unitary
verification path can be attempted when transpiled width permits it.

## Gate 3: acceptance metrics

Report per challenge and globally:

- compilation success rate;
- exact-verification rate;
- invalid-candidate rate;
- Pareto-front size;
- two-qubit-gate delta versus Qiskit default;
- SWAP delta versus Qiskit default;
- depth delta versus Qiskit default;
- circuit-size delta versus Qiskit default;
- compile cost distribution;
- which representation actions dominate for each circuit/topology family.

Do **not** collapse the pilot to one headline score. The dataset is intended to
learn contextual policy, so heterogeneity across targets and circuit families is
part of the signal.

## Gate 4: WQT20-ready conversion

Only after the deterministic run is stable:

- add `budget_remaining` and prior-action history;
- add pairwise preferences among candidates;
- add value targets for Pareto survival and expected improvement;
- split by circuit family and topology to prevent near-duplicate leakage;
- hold out complete target topologies for OOD evaluation.
