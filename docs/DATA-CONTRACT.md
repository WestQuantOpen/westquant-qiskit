# Qiskit deterministic-search data contract v0.1

Every challenge records the same three layers:

1. **logical input**: framework-native circuit before target compilation;
2. **default baseline**: Qiskit's preset pipeline with optimization level 2,
   fixed seed 0, and no explicit stage-method overrides;
3. **WestQuant candidates**: explicit layout × routing × translation ×
   optimization-level × seed configurations.

The target context is first-class data. Records include the basis gate set,
coupling edges or target operations when available, physical-qubit count, and
backend/target identity where exposed.

## `training.jsonl`

One row is emitted for the default baseline and one for every attempted
WestQuant candidate, including compilation failures.

Important fields:

```text
challenge_id
framework
search_space_id
target_context
candidate_id
role
budget_index
representation_action
compile_success
verification
input_metrics
default_baseline_metrics
metrics_after
delta_vs_input
delta_vs_default_baseline
compile_seconds
pareto
rank
error
```

This is designed to become a WQT20 transition record:

```text
(context, representation_action) -> outcome
```

where context contains the circuit structure, target constraints, baseline,
and remaining search budget.

## Equivalence policy

`EXACT` is used only when exact unitary verification succeeds under the v0.1
regime. Unsupported/too-large/non-unitary cases remain `UNKNOWN`. A compilation
failure is an explicit negative outcome, not a missing row.

## Selection objectives

Search candidates are compared on a Pareto front minimizing:

1. two-qubit gates;
2. SWAP gates;
3. depth;
4. total circuit size.

Compile time is recorded as cost but deliberately excluded from deterministic
winner selection because it is machine dependent.
