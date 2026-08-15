"""Research feature experiments on top of frozen benchmark v1.

Does not modify freeze panels, expected WMAPEs, or XGB hyperparameters.

Example::

    from pkg.research import compare_feature_experiments, get_experiment

    report = compare_feature_experiments()
    print(report["overall"])
"""
from pkg.research.experiments import (
    EXPERIMENTS,
    FeatureSet,
    enrich_dataset,
    get_experiment,
    make_residual_model,
)

# Lazy import helper to avoid ``python -m pkg.research.evaluate_features`` warning
def compare_feature_experiments(*args, **kwargs):
    from pkg.research.evaluate_features import compare_feature_experiments as _fn

    return _fn(*args, **kwargs)


__all__ = [
    "EXPERIMENTS",
    "FeatureSet",
    "compare_feature_experiments",
    "enrich_dataset",
    "get_experiment",
    "make_residual_model",
]
