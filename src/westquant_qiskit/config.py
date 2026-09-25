from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Iterator


@dataclass(frozen=True)
class PipelineConfig:
    """One deterministic Qiskit transpilation representation.

    The configuration is deliberately explicit.  A candidate identity is a
    stable hash of the representation choices and the transpiler seed, so the
    same search space can be regenerated later for RepGraph/WQT training.
    """

    optimization_level: int
    layout_method: str
    routing_method: str
    translation_method: str = "translator"
    seed_transpiler: int = 0
    approximation_degree: float = 1.0

    def __post_init__(self) -> None:
        if self.optimization_level not in (0, 1, 2, 3):
            raise ValueError("optimization_level must be 0, 1, 2, or 3")
        if not 0.0 <= self.approximation_degree <= 1.0:
            raise ValueError("approximation_degree must be in [0, 1]")

    @property
    def candidate_id(self) -> str:
        text = (
            f"opt={self.optimization_level}|layout={self.layout_method}|"
            f"routing={self.routing_method}|translation={self.translation_method}|"
            f"seed={self.seed_transpiler}|approx={self.approximation_degree:.9g}"
        )
        return "qiskit:" + sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "optimization_level": self.optimization_level,
            "layout_method": self.layout_method,
            "routing_method": self.routing_method,
            "translation_method": self.translation_method,
            "seed_transpiler": self.seed_transpiler,
            "approximation_degree": self.approximation_degree,
        }


@dataclass(frozen=True)
class SearchSpace:
    """Cartesian representation space searched by the deterministic engine."""

    optimization_levels: tuple[int, ...] = (0, 1, 2, 3)
    layout_methods: tuple[str, ...] = ("trivial", "dense", "sabre")
    routing_methods: tuple[str, ...] = ("basic", "lookahead", "sabre")
    translation_methods: tuple[str, ...] = ("translator", "synthesis")
    seeds: tuple[int, ...] = (0,)
    approximation_degrees: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if not all((self.optimization_levels, self.layout_methods, self.routing_methods,
                    self.translation_methods, self.seeds, self.approximation_degrees)):
            raise ValueError("all search-space dimensions must be non-empty")

    def __len__(self) -> int:
        return (
            len(self.optimization_levels)
            * len(self.layout_methods)
            * len(self.routing_methods)
            * len(self.translation_methods)
            * len(self.seeds)
            * len(self.approximation_degrees)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "optimization_levels": list(self.optimization_levels),
            "layout_methods": list(self.layout_methods),
            "routing_methods": list(self.routing_methods),
            "translation_methods": list(self.translation_methods),
            "seeds": list(self.seeds),
            "approximation_degrees": list(self.approximation_degrees),
        }

    @property
    def search_space_id(self) -> str:
        import json
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "qiskit-space:" + sha256(blob.encode("utf-8")).hexdigest()[:16]

    def iter_configs(self) -> Iterator[PipelineConfig]:
        for values in product(
            self.optimization_levels,
            self.layout_methods,
            self.routing_methods,
            self.translation_methods,
            self.seeds,
            self.approximation_degrees,
        ):
            yield PipelineConfig(
                optimization_level=values[0],
                layout_method=values[1],
                routing_method=values[2],
                translation_method=values[3],
                seed_transpiler=values[4],
                approximation_degree=values[5],
            )

    @classmethod
    def smoke(cls) -> "SearchSpace":
        return cls(
            optimization_levels=(0, 2),
            layout_methods=("trivial", "sabre"),
            routing_methods=("basic", "sabre"),
            translation_methods=("translator",),
            seeds=(0,),
        )

    @classmethod
    def training(cls, *, seed_count: int = 3) -> "SearchSpace":
        if seed_count < 1:
            raise ValueError("seed_count must be >= 1")
        return cls(seeds=tuple(range(seed_count)))
