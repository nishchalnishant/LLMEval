from __future__ import annotations

import argparse
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from data.load_pubmedqa import load_test_set
from models.base import BaseModel
from evaluators import ragas_metrics, llm_judge, adversarial

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, models: list[BaseModel], n_samples: int = 200, batch_size: int = 10, batch_sleep: float = 1.0):
        self.models = models
        self.dataset = load_test_set(n_samples)
        self.batch_size = batch_size
        self.batch_sleep = batch_sleep
        Path("results").mkdir(exist_ok=True)

    def _eval_sample(self, model: BaseModel, sample: dict) -> dict | None:
        try:
            start = time.time()
            generated = model.generate(sample["question"], sample["context"])
            latency_ms = (time.time() - start) * 1000

            faith = ragas_metrics.faithfulness(sample["question"], sample["context"], sample["answer"], generated)
            relevance = ragas_metrics.answer_relevance(sample["question"], sample["context"], sample["answer"], generated)
            precision = ragas_metrics.context_precision(sample["question"], sample["context"], sample["answer"], generated)
            judge_scores = llm_judge.score(sample["question"], sample["context"], sample["long_answer"], generated)

            return {
                "question": sample["question"],
                "generated": generated,
                "faithfulness": faith,
                "answer_relevance": relevance,
                "context_precision": precision,
                "llm_judge": judge_scores,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.warning(f"Sample failed for {model.name}: {e}")
            return None

    def _run_model(self, model: BaseModel) -> list[dict]:
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"results/raw_{model.name}_{timestamp}.jsonl"

        with open(out_path, "w") as f:
            for i in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset[i: i + self.batch_size]
                for sample in tqdm(batch, desc=f"{model.name} batch {i//self.batch_size+1}"):
                    result = self._eval_sample(model, sample)
                    results.append(result)
                    f.write(json.dumps(result) + "\n")
                if i + self.batch_size < len(self.dataset):
                    time.sleep(self.batch_sleep)

        return results

    def run(self) -> dict:
        summary = {}
        for model in self.models:
            raw = self._run_model(model)
            valid = [r for r in raw if r is not None]

            if not valid:
                continue

            adv = adversarial.evaluate_adversarial(model)

            def avg(key):
                return sum(r[key] for r in valid) / len(valid)

            summary[model.name] = {
                "faithfulness": avg("faithfulness"),
                "answer_relevance": avg("answer_relevance"),
                "context_precision": avg("context_precision"),
                "llm_judge_overall": sum(r["llm_judge"]["overall"] for r in valid) / len(valid),
                "adversarial_score": adv["score"],
                "adversarial_by_category": adv["by_category"],
                "avg_latency_ms": avg("latency_ms"),
                "n_valid": len(valid),
            }

        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    from models.gemini import GeminiModel
    from models.groq_llama import GroqLlamaModel
    from models.nvidia_nim import NvidiaNIMModel
    from models.mistral_vllm import MistralVLLMModel

    models = [GeminiModel(), GroqLlamaModel(), NvidiaNIMModel()]
    mistral = MistralVLLMModel()
    if mistral.available:
        models.append(mistral)

    runner = EvalRunner(models, n_samples=args.n_samples)
    results = runner.run()

    from pipeline.reporter import generate_reports
    generate_reports(results, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
