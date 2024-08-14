
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

class ModelProvider(ABC):
    @abstractmethod
    async def evaluate_model(self, prompt: str) -> str: ...

    @abstractmethod
    def generate_prompt(self, context: str, retrieval_question: str) -> str | List[Dict[str, str]]: ...

    @abstractmethod
    def encode_text_to_tokens(self, text: str) -> List[int]: ...

    @abstractmethod
    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str: ...