from __future__ import annotations

import json
import os
from models.base import BaseModel

_CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adversarial_cases.json")


def load_cases() -> list[dict]:
    with open(_CASES_PATH) as f:
        return json.load(f)


def _classify_behavior(generated: str, expected: str) -> bool:
    text = generated.lower().strip()
    if expected == "refuse":
        return not any(w in text for w in ["yes", "no", "maybe"])
    if expected == "answer_yes":
        return text.startswith("yes")
    if expected == "answer_no":
        return text.startswith("no")
    if expected == "answer_maybe":
        return text.startswith("maybe")
    return False


def evaluate_adversarial(model: BaseModel) -> dict:
    cases = load_cases()
    results = []
    for case in cases:
        try:
            generated = model.generate(case["question"], case["context"])
            passed = _classify_behavior(generated, case["expected_behavior"])
        except Exception:
            passed = False
        results.append({"id": case["id"], "category": case["category"], "passed": passed})

    total = len(results)
    score = sum(r["passed"] for r in results) / total if total else 0.0
    by_category = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, []).append(r["passed"])

    return {
        "score": score,
        "by_category": {cat: sum(v) / len(v) for cat, v in by_category.items()},
        "details": results,
    }
