# Changelog

## 0.1.2

- Fix: the `westquant` layout, routing and optimization stage plugins crashed
  on Qiskit 2.x (`QuantumCircuit.from_instructions(dag)` with a `DAGCircuit`)
  and the candidate path read a private `tqc._layout.dag` attribute. The stages
  are reimplemented as best-of selections over Qiskit built-in strategies that
  set `layout` / `final_layout` correctly and use Qiskit's standard embed and
  routing stage wrappers.
- Candidate results, including failures, are recorded in
  `property_set["westquant_selection"]` and failures are logged; if every
  candidate fails a `TranspilerError` is raised instead of silently returning
  the input.
- `optimization_level=0` is no longer replaced by 2 inside the plugin.
- Brings the BackendV2 `Target` support in `noise_aware_metrics` that shipped in
  the 0.1.1 wheel into the source repository.
- Tests: plugin discovery, unitary equivalence, measurement distribution,
  coupling-map compliance, selected-candidate application and all-fail error.

## 0.1.1

Published to PyPI on 2026-09-26 from a working tree that was not committed;
its only source difference from 0.1.0a4 was BackendV2 support in
`adapter.noise_aware_metrics`.
