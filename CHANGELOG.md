# Changelog

## [0.1.0a2] - 2026-09-24

### Added
- Deterministic representation search engine over Qiskit compilation pipelines.
- Search space: 4 optimization levels × 3 layout methods × 3 routing methods × 2 translation methods × 3 seeds = 216 candidates per circuit.
- Frozen Qiskit default baseline (optimization level 2, seed 0).
- Conservative exact-unitary equivalence verification via Operator.from_circuit().equiv().
- Compile failures stored as first-class RepGraph evaluation nodes.
- Pareto frontier over 2Q gates, SWAP count, depth, circuit size.
- RepGraph JSON + WQT-ready training.jsonl export.
- Challenge generators: GHZ, QFT-like, random entangling circuits over line/ring topologies.
- CLI for smoke and training suite generation.
- GitHub Actions CI pinned to Qiskit 2.5.x.
- Native Qiskit transpiler plugin entry points (layout, routing, optimization).
- Data contract and pilot protocol documentation.
