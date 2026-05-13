import os
from groq import Groq
from models.base import BaseModel

SYSTEM_PROMPT = (
    "You are a clinical research assistant. Given a PubMed abstract and a question, "
    "answer with 'yes', 'no', or 'maybe', followed by a one-sentence justification."
)


class GroqLlamaModel(BaseModel):
    name = "llama-3.3-70b"

    def __init__(self):
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def generate(self, question: str, context: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {question}"
        response = self._client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
