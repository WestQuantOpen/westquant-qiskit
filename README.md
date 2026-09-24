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

The package also exposes alpha stage entry points for Qiskit's native
transpiler-plugin mechanism.  The plugin stages still delegate to Qiskit
baselines in this Wave 1 build.  The deterministic search engine is the first
real WestQuant search implementation and will be wired into stage plugins after
benchmarking and recursion guards are complete.
