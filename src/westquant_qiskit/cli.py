from __future__ import annotations

import argparse
import json
from pathlib import Path

from .challenges import generate_suite
from .config import SearchSpace
from .io import save_search_result, save_sequential_result
from .search import DeterministicSearchEngine
from .sequential import SequentialSearchEngine, SequentialSearchSpace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WestQuant Qiskit representation search")
    parser.add_argument("--suite", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--search", choices=("grid", "sequential"), default="grid")
    parser.add_argument("--output", default="results/westquant-qiskit")
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []

    if args.search == "grid":
        space = SearchSpace.smoke() if args.suite == "smoke" else SearchSpace.training(seed_count=args.seed_count)
        engine = DeterministicSearchEngine(search_space=space)
    else:
        space = SequentialSearchSpace()
        engine = SequentialSearchEngine(search_space=space, beam_width=args.beam_width)

    for challenge in generate_suite(args.suite):
        common = dict(
            challenge_id=challenge.challenge_id,
            coupling_map=challenge.coupling_map,
            basis_gates=list(challenge.basis_gates),
            challenge_metadata=challenge.metadata,
        )
        if args.search == "grid":
            result = engine.search(challenge.circuit, max_candidates=args.max_candidates, **common)
            save_search_result(result, root / challenge.challenge_id)
        else:
            result = engine.search(challenge.circuit, **common)
            save_sequential_result(result, root / challenge.challenge_id)
        summaries.append(result.summary())
        print(json.dumps(result.summary(), sort_keys=True, default=str))

    manifest = {
        "suite": args.suite,
        "search": args.search,
        "search_space_id": space.search_space_id,
        "n_challenges": len(summaries),
        "summaries": summaries,
    }
    if args.search == "grid":
        manifest["search_space_size"] = len(space)
    else:
        manifest["beam_width"] = args.beam_width
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
