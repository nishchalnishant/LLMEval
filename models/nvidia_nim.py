from __future__ import annotations

import os
import threading

from openai import OpenAI

from models.base import BaseModel

SYSTEM_PROMPT = (
    "You are a clinical research assistant. Given a PubMed abstract and a question, "
    "answer with 'yes', 'no', or 'maybe', followed by a one-sentence justification."
)

_CLIENT: OpenAI | None = None
_CLIENT_LOCK = threading.Lock()


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                api_key = os.environ.get("NVIDIA_API_KEY")
                if not api_key:
                    raise ValueError("NVIDIA_API_KEY not set")
                _CLIENT = OpenAI(
                    base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=api_key,
                )
    return _CLIENT


class NvidiaNIMModel(BaseModel):
    name = "nvidia-llama-3.3-70b"

    def __init__(self, model: str = "meta/llama-3.3-70b-instruct"):
        self._model = model

    def generate(self, question: str, context: str) -> str:
        client = _get_client()
        resp = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=128,
        )
        return resp.choices[0].message.content.strip()
