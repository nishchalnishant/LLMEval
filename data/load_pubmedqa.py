from __future__ import annotations

from datasets import load_dataset


def load_test_set(n: int = 200) -> list[dict]:
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    test = ds.select(range(len(ds) - n, len(ds)))
    return [
        {
            "question": row["question"],
            "context": " ".join(row["context"]["contexts"][:3]),
            "answer": row["final_decision"],
            "long_answer": row["long_answer"],
        }
        for row in test
    ]
