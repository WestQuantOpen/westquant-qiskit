# westquant-qiskit

**Wave 1 deterministic representation search for Qiskit.**

`westquant-qiskit` searches the *compilation representation*, not only compiler
parameters.  A logical circuit is compiled through a reproducible grid of Qiskit
pipeline choices and every success, failure, metric delta, and verification
outcome is written to RepGraph/WQT-ready records.

## v0.1 search space

The first deterministic engine searches the Cartesian product of:

- optimization level: `0, 1, 2, 3`
- layout: `trivial, dense, sabre`
- routing: `basic, lookahead, sabre`
- translation: `translator, synthesis`
- fixed transpiler seeds
- approximation degree

The default training grid with three seeds contains **216 pipeline
representations per circuit**.  This is intentionally deterministic: the same
circuit, target, search-space version, and seeds reproduce the candidate set.

## Run a smoke suite

```bash
pip install -e ../westquant-core
pip install -e .
python -m westquant_qiskit.cli --suite smoke --output results/qiskit-smoke
```

A larger local data-generation suite:

```bash
python -m westquant_qiskit.cli \
  --suite train \
  --seed-count 3 \
  --output results/qiskit-train
```

Each challenge writes:

```text
summary.json
repgraph.json
training.jsonl
```

`training.jsonl` is the primary WQT20 precursor format.  Each row contains the
logical baseline metrics, one pipeline choice, compilation success/failure,
conservative equivalence verification, resulting metrics, metric deltas,
compile cost, Pareto membership, and rank.

## Verification policy

WestQuant does **not** silently call a candidate equivalent.  For small unitary
circuits of equal width, v0.1 uses
`qiskit.quantum_info.Operator.from_circuit(...).equiv(...)`, which accounts for
stored transpiler layouts.  Circuits that cannot be checked under that regime
are labeled `UNKNOWN`, not `EXACT`.

## Current objective vector

Candidates are Pareto-compared by minimizing:

1. two-qubit gate count
2. SWAP count
3. depth
4. circuit size
5. compile time

The lexicographic `best` is used only as a transparent deterministic default;
the full Pareto front and all losing candidates remain in the dataset.

## Qiskit integration

The package registers `westquant` as a Qiskit transpiler stage plugin for the
`layout`, `routing` and `optimization` stages:

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

pm = generate_preset_pass_manager(
    optimization_level=2, backend=backend, seed_transpiler=7,
    layout_method="westquant", routing_method="westquant", optimization_method="westquant",
)
tqc = pm.run(qc)
print(pm.property_set["westquant_selection"])
```

Each stage is a deterministic best-of selection among Qiskit's own built-in
strategies (layout: trivial/dense/sabre; routing: SabreSwap/BasicSwap/
LookaheadSwap; optimization: levels 0-3), scored by two-qubit cost (SWAP = 3)
and then depth. Every candidate, including failures and their error messages,
is recorded in `property_set["westquant_selection"]`; if every candidate fails a
`TranspilerError` is raised. The routing stage only runs when the circuit is
not already mapped to the device. Tests check unitary equivalence, ideal output
distributions with measurements, coupling-map compliance and that the selected
candidate is the one returned. This is a best-of wrapper over Qiskit, not the
full sequential policy search described below.

## Sequential policy search

Wave 1 now also includes a policy-style sequential search mode. Instead of
choosing a complete compiler pipeline in one Cartesian-grid action, WestQuant
chooses an ordered sequence:

```text
layout -> routing -> translation -> optimization
```

Each partial prefix is evaluated through an end-to-end rollout with unchosen
later stages set to Qiskit's explicit defaults. A beam retains the strongest
states under the deterministic objective vector. This intentionally uses only
public Qiskit APIs while producing WQT-ready state/action/reward trajectories.

```bash
westquant-qiskit-search \
  --suite smoke \
  --search sequential \
  --beam-width 3 \
  --output results/qiskit-sequential
```

Each challenge writes `trajectory.jsonl` in addition to `summary.json` and
`repgraph.json`. A row contains `state_id`, `stage`, `action`, prefix before and
after the action, rollout pipeline, verification, metrics, reward vector,
baseline delta, and whether the state survived the beam.

## Sequential 1000 dataset generator

Generate a frozen balanced manifest:

```bash
westquant-qiskit-1000 --output results/wq-qiskit-sequential-1000 --manifest-only
```

Pilot 25 challenges:

```bash
westquant-qiskit-1000 --output results/wq-qiskit-sequential-1000 --beam-width 3 --limit 25
```

Run/resume the full 1000:

```bash
westquant-qiskit-1000 --output results/wq-qiskit-sequential-1000 --beam-width 3
```

Regenerate statistics from raw saved outputs only:

```bash
westquant-qiskit-1000 --output results/wq-qiskit-sequential-1000 --stats-only
```

See `docs/SEQUENTIAL-1000.md` for the frozen design and failure-retention policy.
