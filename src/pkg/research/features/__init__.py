"""Composable point-in-time feature groups for research experiments."""
from pkg.research.features import demand, human
from pkg.research.features.demand import DEMAND_FEATURE_NAMES, add_demand_features
from pkg.research.features.demand_f2 import DEMAND_F2_FEATURE_NAMES, add_demand_f2_features
from pkg.research.features.human import HUMAN_FEATURE_NAMES, add_human_features
from pkg.research.features.human_f2 import HUMAN_F2_FEATURE_NAMES, add_human_f2_features
from pkg.research.features.lifecycle import (
    FEATURE_NAMES as LIFECYCLE_FEATURE_NAMES,
    add_lifecycle_features,
)
from pkg.research.features.price import (
    FEATURE_NAMES as PRICE_FEATURE_NAMES,
    add_price_features,
)
from pkg.research.features.inventory import (
    FEATURE_NAMES as INVENTORY_FEATURE_NAMES,
    add_inventory_features,
)
from pkg.research.features.patient_consumption import (
    FEATURE_NAMES as PATIENT_CONSUMPTION_FEATURE_NAMES,
    add_patient_consumption_features,
)

FEATURE_GROUPS = {
    "demand": DEMAND_FEATURE_NAMES,
    "human": HUMAN_FEATURE_NAMES,
    "demand_f2": DEMAND_F2_FEATURE_NAMES,
    "human_f2": HUMAN_F2_FEATURE_NAMES,
    "price": PRICE_FEATURE_NAMES,
    "lifecycle": LIFECYCLE_FEATURE_NAMES,
    "inventory": INVENTORY_FEATURE_NAMES,
    "patient_consumption": PATIENT_CONSUMPTION_FEATURE_NAMES,
    "commercial": (),
}

__all__ = [
    "FEATURE_GROUPS",
    "DEMAND_FEATURE_NAMES",
    "HUMAN_FEATURE_NAMES",
    "DEMAND_F2_FEATURE_NAMES",
    "HUMAN_F2_FEATURE_NAMES",
    "LIFECYCLE_FEATURE_NAMES",
    "PRICE_FEATURE_NAMES",
    "INVENTORY_FEATURE_NAMES",
    "PATIENT_CONSUMPTION_FEATURE_NAMES",
    "add_demand_features",
    "add_human_features",
    "add_demand_f2_features",
    "add_human_f2_features",
    "add_lifecycle_features",
    "add_price_features",
    "add_inventory_features",
    "add_patient_consumption_features",
    "demand",
    "human",
]
