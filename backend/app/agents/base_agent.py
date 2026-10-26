"""
Base interface for traffic signal control agents.
"""
from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: Any, deterministic: bool = False) -> int:
        """Select action 0 (KEEP) or 1 (SWITCH)."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Perform policy or value update step."""
        pass

    @abstractmethod
    def save(self, file_path: str):
        """Persist model parameters to disk."""
        pass

    @abstractmethod
    def load(self, file_path: str):
        """Restore model parameters from disk."""
        pass
