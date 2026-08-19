"""M1 Optuna tuning — research-only XGBoost hyperparameter experiment.

Does NOT modify XGB_PARAMS, frozen benchmark files, or F0/F1/F2/F3* artifacts.
Tunes canonical F0 residual models on PRE-PRIMARY origins; evaluates frozen
params on PRIMARY.
"""
