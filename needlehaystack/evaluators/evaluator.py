from typing import Dict
from abc import ABC, abstractmethod

class Evaluator(ABC):
    CRITERIA: Dict[str, str]

    @abstractmethod
    def evaluate_response(self, response: str) -> int: ...