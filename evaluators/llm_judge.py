import json
import logging
import hashlib
import shelve
import os
from google import genai

logger = logging.getLogger(__name__)
_CACHE_PATH = os.environ.get("JUDGE_CACHE_PATH", ".eval_cache")

JUDGE_PROMPT = """You are evaluating a clinical QA system. Score the response on these dimensions:
- Correctness (0-10): Does it match the reference answer?
- Groundedness (0-10): Is it grounded in the provided context?
- Conciseness (0-10): Is it appropriately brief?

Question: {question}
Context: {context}
Reference answer: {long_answer}
Model response: {generated}

Return JSON only: {{"correctness": X, "groundedness": X, "conciseness": X, "overall": X}}"""


def _get_client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _parse_scores(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


def score(question: str, context: str, long_answer: str, generated: str) -> dict:
    cache_key = hashlib.md5((question + generated + long_answer).encode()).hexdigest()
    with shelve.open(_CACHE_PATH) as db:
        if cache_key in db:
            return db[cache_key]

    prompt = JUDGE_PROMPT.format(
        question=question, context=context, long_answer=long_answer, generated=generated
    )
    client = _get_client()

    for attempt in range(2):
        try:
            result = client.models.generate_content(model="gemini-1.5-flash", contents=prompt).text
            scores = _parse_scores(result)
            with shelve.open(_CACHE_PATH) as db:
                db[cache_key] = scores
            return scores
        except Exception as e:
            if attempt == 0:
                logger.warning(f"Judge parse failed, retrying: {e}")
            else:
                logger.warning(f"Judge failed after retry: {e}")
                return {"correctness": 0, "groundedness": 0, "conciseness": 0, "overall": 0}
