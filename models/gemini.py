import os
from google import genai
from google.genai import types
from models.base import BaseModel

SYSTEM_PROMPT = (
    "You are a clinical research assistant. Given a PubMed abstract and a question, "
    "answer with 'yes', 'no', or 'maybe', followed by a one-sentence justification."
)


class GeminiModel(BaseModel):
    name = "gemini-1.5-flash"

    def __init__(self):
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, question: str, context: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {question}"
        response = self._client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return response.text.strip()
