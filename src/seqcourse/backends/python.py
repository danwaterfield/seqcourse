from __future__ import annotations

from typing import Any

from .base import Backend


class PythonBackend(Backend):
    name = "python"

    def compute_cost_matrix(self, *args: Any, **kwargs: Any) -> Any:
        from ..costs import _cost_matrix_impl

        return _cost_matrix_impl(*args, **kwargs)

    def compute_distance_matrix(self, *args: Any, **kwargs: Any) -> Any:
        from ..distances import _distance_matrix_impl

        return _distance_matrix_impl(*args, **kwargs)


_BACKENDS = {
    "python": PythonBackend(),
}


def get_backend(name: str | None = None) -> Backend:
    resolved = "python" if name is None else name.lower()
    if resolved not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Available backends: {', '.join(sorted(_BACKENDS))}.")
    return _BACKENDS[resolved]

