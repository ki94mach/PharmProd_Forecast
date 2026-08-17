"""F2 experiment configuration (does not mutate F0 / F1 registries)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.features.demand_f2 import DEMAND_F2_FEATURE_NAMES
from pkg.research.features.human_f2 import HUMAN_F2_FEATURE_NAMES

Anchor = Literal["ts", "human"]

# Locked Analysis B PRIMARY WMAPEs from freeze-time (pkg.benchmark contract).
# F2 asserts current frozen backtest n/origins match; WMAPE may differ by XGB env.
LOCKED_F0_WMAPE = {
    "ts_xgb": 37.230140,
    "human_xgb": 36.694750,
    "n": 1877,
    "n_origins": 5,
}

F0_WMAPE_TOL = 0.05
F0_DRIFT_TOL = 0.05  # reproduced F0 vs F0 in the same F2 run

HIGH_VOLUME_WATCHLIST = (
    "Cinnatropin 10",
    "Cinnora AutoInjector",
    "Melitide",
    "FolicoGen",
    "Cinnal-f",
    "Paglino 10",
    "Zakaria",
)

CONCENTRATION_ONE_PRODUCT = 0.25
CONCENTRATION_TOP5 = 0.50

FILLNA_EXTRA = (
    "trend_log_3m",
    "trend_log_6m",
    "yoy_log_change",
)


@dataclass(frozen=True)
class F2Experiment:
    name: str
    groups: tuple[str, ...]  # "demand_f2" and/or "human_f2"
    anchors: tuple[Anchor, ...]
    train_universe: dict[str, str]

    def features_for(self, anchor: Anchor) -> tuple[str, ...]:
        if anchor == "ts":
            base = tuple(TS_RESID_FEATURES)
        elif anchor == "human":
            base = tuple(BUDGET_RESID_FEATURES)
        else:
            raise ValueError(f"anchor must be 'ts' or 'human', got {anchor!r}")
        extra: list[str] = []
        if "demand_f2" in self.groups:
            extra.extend(DEMAND_F2_FEATURE_NAMES)
        if "human_f2" in self.groups:
            extra.extend(HUMAN_F2_FEATURE_NAMES)
        seen: set[str] = set()
        out: list[str] = []
        for c in base + tuple(extra):
            if c not in seen:
                seen.add(c)
                out.append(c)
        return tuple(out)


F0 = F2Experiment(
    name="F0",
    groups=(),
    anchors=("ts", "human"),
    train_universe={"ts": "ts", "human": "budget"},
)
F2A = F2Experiment(
    name="F2A",
    groups=("demand_f2",),
    anchors=("ts", "human"),
    train_universe={"ts": "ts", "human": "budget"},
)
F2B = F2Experiment(
    name="F2B",
    groups=("human_f2",),
    anchors=("human",),  # Budget-only reliability; not injected into TS+ML
    train_universe={"human": "budget"},
)
F2C = F2Experiment(
    name="F2C",
    groups=("demand_f2", "human_f2"),
    anchors=("human",),
    train_universe={"human": "budget"},
)

F2_EXPERIMENTS: dict[str, F2Experiment] = {
    "F0": F0,
    "F2A": F2A,
    "F2B": F2B,
    "F2C": F2C,
}


def get_f2_experiment(name: str) -> F2Experiment:
    if name not in F2_EXPERIMENTS:
        raise KeyError(f"Unknown F2 experiment {name!r}; known={sorted(F2_EXPERIMENTS)}")
    return F2_EXPERIMENTS[name]


def f2_output_dir() -> Path:
    src_dir = Path(__file__).resolve().parents[3]
    out = src_dir / "data" / "results" / "f2"
    out.mkdir(parents=True, exist_ok=True)
    return out


def docs_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "docs"
