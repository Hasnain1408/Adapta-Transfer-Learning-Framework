"""Which request fields a transfer-learning method will actually accept.

A method opts into a UI-configurable parameter simply by naming its
attribute ``_kernel_type`` / ``_regularization`` / ``_latent_dim`` / ``_mu``.
Both the injection (before a run) and the UI's capability check read that
same ``hasattr`` rule from here, so a control can never be enabled for a
parameter the method would silently ignore.

Note this makes method instances **stateful and mutated across runs** — the
values passed in ``main.build_window()`` are only the initial ones.
"""

from __future__ import annotations

#: Request field -> the private attribute a method must expose to receive it.
TUNABLE_PARAMETERS = {
    "kernel_type": "_kernel_type",
    "regularization": "_regularization",
    "latent_dim": "_latent_dim",
    "mu": "_mu",
}


def apply_tunable_parameters(method: object, request: object) -> None:
    """Push each supported field of *request* onto *method*."""
    for field, attribute in TUNABLE_PARAMETERS.items():
        if hasattr(method, attribute):
            setattr(method, attribute, getattr(request, field))


def supported_parameters_for(method: object) -> frozenset[str]:
    """Which tunable fields *method* consumes, by the same ``hasattr`` rule."""
    return frozenset(
        field for field, attribute in TUNABLE_PARAMETERS.items() if hasattr(method, attribute)
    )


__all__ = [
    "TUNABLE_PARAMETERS",
    "apply_tunable_parameters",
    "supported_parameters_for",
]
