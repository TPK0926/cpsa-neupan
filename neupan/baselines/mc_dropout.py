"""
DEPRECATED: Use input_perturbation.InputPerturbationWrapper instead.

This module is kept for backward compatibility. The original "MC Dropout"
implementation was input perturbation (not true MC Dropout since the
pretrained DUNE has no dropout layers). It has been renamed to
InputPerturbationWrapper for honesty.
"""

from neupan.baselines.input_perturbation import InputPerturbationWrapper as MCDropoutWrapper

# Provide the old class name for existing imports
__all__ = ['MCDropoutWrapper']
