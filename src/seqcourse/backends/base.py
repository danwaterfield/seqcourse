from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    name: str

    @abstractmethod
    def compute_cost_matrix(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compute_distance_matrix(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

