from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .search import SearchResult


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def save_search_result(result: SearchResult, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "summary.json", result.summary())
    _dump(out / "repgraph.json", result.graph.to_dict())
    records = result.training_records()
    with (out / "training.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return out


def save_sequential_result(result: Any, directory: str | Path) -> Path:
    """Persist a sequential policy-search trace in WQT-ready form."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "summary.json", result.summary())
    _dump(out / "repgraph.json", result.graph.to_dict())
    records = result.training_records()
    with (out / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return out
