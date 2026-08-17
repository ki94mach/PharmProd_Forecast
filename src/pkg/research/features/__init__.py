"""Composable point-in-time feature groups for research experiments."""
from pkg.research.features import demand, human
from pkg.research.features.demand import DEMAND_FEATURE_NAMES, add_demand_features
from pkg.research.features.demand_f2 import DEMAND_F2_FEATURE_NAMES, add_demand_f2_features
from pkg.research.features.human import HUMAN_FEATURE_NAMES, add_human_features
from pkg.research.features.human_f2 import HUMAN_F2_FEATURE_NAMES, add_human_f2_features

FEATURE_GROUPS = {
    "demand": DEMAND_FEATURE_NAMES,
    "human": HUMAN_FEATURE_NAMES,
    "demand_f2": DEMAND_F2_FEATURE_NAMES,
    "human_f2": HUMAN_F2_FEATURE_NAMES,
    "price": (),
    "lifecycle": (),
    "commercial": (),
}

__all__ = [
    "FEATURE_GROUPS",
    "DEMAND_FEATURE_NAMES",
    "HUMAN_FEATURE_NAMES",
    "DEMAND_F2_FEATURE_NAMES",
    "HUMAN_F2_FEATURE_NAMES",
    "add_demand_features",
    "add_human_features",
    "add_demand_f2_features",
    "add_human_f2_features",
    "demand",
    "human",
]
