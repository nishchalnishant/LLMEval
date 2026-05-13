from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, question: str, context: str, answer: str, generated: str) -> float:
        ...
