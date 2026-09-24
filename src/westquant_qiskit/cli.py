from __future__ import annotations

import argparse
import json
from pathlib import Path

from .challenges import generate_suite
from .config import SearchSpace
from .io import save_search_result
from .search import DeterministicSearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WestQuant deterministic Qiskit representation search")
    parser.add_argument("--suite", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--output", default="results/westquant-qiskit")
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    space = SearchSpace.smoke() if args.suite == "smoke" else SearchSpace.training(seed_count=args.seed_count)
    engine = DeterministicSearchEngine(search_space=space)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for challenge in generate_suite(args.suite):
        result = engine.search(
            challenge.circuit,
            challenge_id=challenge.challenge_id,
            coupling_map=challenge.coupling_map,
            basis_gates=list(challenge.basis_gates),
            max_candidates=args.max_candidates,
            challenge_metadata=challenge.metadata,
        )
        save_search_result(result, root / challenge.challenge_id)
        summaries.append(result.summary())
        print(json.dumps(result.summary(), sort_keys=True, default=str))
    (root / "manifest.json").write_text(
        json.dumps({
            "suite": args.suite,
            "search_space_size": len(space),
            "n_challenges": len(summaries),
            "summaries": summaries,
        }, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
