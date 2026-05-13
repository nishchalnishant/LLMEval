import httpx
from models.base import BaseModel

SYSTEM_PROMPT = (
    "You are a clinical research assistant. Given a PubMed abstract and a question, "
    "answer with 'yes', 'no', or 'maybe', followed by a one-sentence justification."
)


class MistralVLLMModel(BaseModel):
    name = "mistral-7b-clinical"

    def __init__(self, endpoint: str = "http://localhost:8000"):
        self._endpoint = endpoint
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            httpx.get(f"{self._endpoint}/health", timeout=2.0)
            return True
        except Exception:
            print(f"Warning: vLLM endpoint {self._endpoint} is unreachable. Skipping Mistral-7B.")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, question: str, context: str) -> str:
        if not self._available:
            raise RuntimeError("vLLM endpoint is not available")
        prompt = f"{SYSTEM_PROMPT}\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        response = httpx.post(
            f"{self._endpoint}/v1/completions",
            json={"model": "mistral-7b-clinical", "prompt": prompt, "max_tokens": 128},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
