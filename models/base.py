from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        ...
