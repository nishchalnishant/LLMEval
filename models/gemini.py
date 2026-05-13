import os
import google.generativeai as genai
from models.base import BaseModel

SYSTEM_PROMPT = (
    "You are a clinical research assistant. Given a PubMed abstract and a question, "
    "answer with 'yes', 'no', or 'maybe', followed by a one-sentence justification."
)


class GeminiModel(BaseModel):
    name = "gemini-1.5-flash"

    def __init__(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )

    def generate(self, question: str, context: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {question}"
        response = self._model.generate_content(prompt)
        return response.text.strip()
