import hashlib
import shelve
import os
import google.generativeai as genai

_CACHE_PATH = os.environ.get("JUDGE_CACHE_PATH", ".eval_cache")


def _cache_key(question: str, generated: str) -> str:
    return hashlib.md5((question + generated).encode()).hexdigest()


def _get_judge():
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-1.5-flash")


def _judge_call(prompt: str, cache_key: str) -> str:
    with shelve.open(_CACHE_PATH) as db:
        if cache_key in db:
            return db[cache_key]
    judge = _get_judge()
    result = judge.generate_content(prompt).text.strip()
    with shelve.open(_CACHE_PATH) as db:
        db[cache_key] = result
    return result


def faithfulness(question: str, context: str, answer: str, generated: str) -> float:
    sentences = [s.strip() for s in generated.split(".") if s.strip()]
    if not sentences:
        return 0.0
    supported = 0
    for i, sentence in enumerate(sentences):
        key = _cache_key(question + f"_faith_{i}", sentence + context)
        prompt = (
            f"Context: {context}\n\nClaim: {sentence}\n\n"
            "Is this claim supported by the context? Answer only 'yes' or 'no'."
        )
        result = _judge_call(prompt, key)
        if result.lower().startswith("yes"):
            supported += 1
    return supported / len(sentences)


def answer_relevance(question: str, context: str, answer: str, generated: str) -> float:
    key = _cache_key(question + "_relevance", generated)
    prompt = (
        f"Question: {question}\n\nAnswer: {generated}\n\n"
        "On a scale of 1-5, how well does this answer address the question? Return only the integer."
    )
    result = _judge_call(prompt, key)
    try:
        score = int(result.strip()[0])
        return max(1, min(5, score)) / 5.0
    except (ValueError, IndexError):
        return 0.0


def context_precision(question: str, context: str, answer: str, generated: str) -> float:
    key = _cache_key(question + "_ctx_precision", context)
    prompt = (
        f"Question: {question}\n\nContext: {context}\n\n"
        "Is this context useful for answering the question? Answer only 'yes' or 'no'."
    )
    result = _judge_call(prompt, key)
    return 1.0 if result.lower().startswith("yes") else 0.0
