"""Inspection and injection of tunable hyper-parameters for transfer learning methods."""

from __future__ import annotations

# Mapping of request attribute names to internal method private attributes.
TUNABLE_PARAMETERS = {
    "kernel_type": "_kernel_type",
    "regularization": "_regularization",
    "latent_dim": "_latent_dim",
    "mu": "_mu",
}


def apply_tunable_parameters(method: object, request: object) -> None:
    """Inject supported parameter values from the request object onto the method instance."""
    for field, attribute in TUNABLE_PARAMETERS.items():
        if hasattr(method, attribute):
            setattr(method, attribute, getattr(request, field))


def supported_parameters_for(method: object) -> frozenset[str]:
    """Return the set of tunable hyper-parameters consumed by the given method instance."""
    return frozenset(
        field for field, attribute in TUNABLE_PARAMETERS.items() if hasattr(method, attribute)
    )


__all__ = [
    "TUNABLE_PARAMETERS",
    "apply_tunable_parameters",
    "supported_parameters_for",
]
