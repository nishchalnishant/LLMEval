# LLMEval

A production-grade evaluation framework for benchmarking LLMs on clinical question answering. Tests multiple model providers against the PubMedQA dataset using RAGAS-style metrics, LLM-as-judge scoring, and an adversarial test suite — with results surfaced in an interactive Streamlit dashboard.

---

## Models Under Test

| Model | Provider | Notes |
|---|---|---|
| `gemini-1.5-flash` | Google Gemini | Also used as judge model |
| `llama-3.3-70b-versatile` | Groq | |
| `meta/llama-3.3-70b-instruct` | NVIDIA NIM | Via OpenAI-compatible API |
| `mistral-7b-clinical` | vLLM (local) | Optional — skipped if endpoint unreachable |

---

## Metrics

| Metric | Method | Range |
|---|---|---|
| **Faithfulness** | Fraction of generated claims supported by context (judge LLM per sentence) | 0 – 1 |
| **Answer Relevance** | Judge LLM rates how well the answer addresses the question (1–5, normalized) | 0 – 1 |
| **Context Precision** | Binary judge assessment of context usefulness for the question | 0 or 1 |
| **LLM Judge Overall** | Holistic score across correctness, groundedness, and conciseness | 0 – 10 |
| **Adversarial Score** | Fraction of 20 adversarial cases handled correctly | 0 – 1 |
| **Avg Latency** | Mean end-to-end response time per sample | ms |

RAGAS metrics are implemented from scratch — the `ragas` library is not used. All judge calls are cached in a local `shelve` database; reruns do not re-bill the API.

---

## Adversarial Suite

20 hand-crafted test cases across four categories:

- **Prompt injection** (5) — attempts to override system instructions
- **Hallucination triggers** (5) — questions about drugs/conditions absent from context
- **Negation traps** (5) — questions where the obvious answer is wrong
- **Out-of-distribution** (5) — non-clinical questions injected mid-eval

---

## Project Structure

```
LLMEval/
├── evaluators/
│   ├── base.py               # BaseEvaluator abstract class
│   ├── ragas_metrics.py      # Faithfulness, answer relevance, context precision
│   ├── llm_judge.py          # Holistic LLM-as-judge scoring
│   └── adversarial.py        # Adversarial test suite runner
├── models/
│   ├── base.py               # BaseModel abstract class
│   ├── gemini.py             # Gemini 1.5 Flash
│   ├── groq_llama.py         # LLaMA 3.3-70B via Groq
│   ├── nvidia_nim.py         # LLaMA 3.3-70B via NVIDIA NIM
│   └── mistral_vllm.py       # Fine-tuned Mistral-7B via vLLM (optional)
├── pipeline/
│   ├── runner.py             # Orchestrates full eval run
│   └── reporter.py           # Generates JSON + HTML reports
├── dashboard/
│   └── app.py                # Streamlit leaderboard dashboard
├── data/
│   ├── load_pubmedqa.py      # Loads PubMedQA test split from HuggingFace
│   └── adversarial_cases.json
├── .github/workflows/
│   └── eval.yml              # GitHub Actions CI
├── results/                  # Auto-generated (summary.json committed, rest gitignored)
├── config.yaml
└── requirements.txt
```

---

## Quickstart

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set API keys**

```bash
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
```

**3. Run evaluation**

```bash
python -m pipeline.runner --n-samples 200
```

This writes `results/summary.json` and `results/report.html`. Raw per-sample outputs go to `results/raw_{model}_{timestamp}.jsonl`.

**4. Launch dashboard**

```bash
streamlit run dashboard/app.py
```

---

## CI

The GitHub Actions workflow (`.github/workflows/eval.yml`) runs the evaluation on every push to `main`. Add `GEMINI_API_KEY`, `GROQ_API_KEY`, and `NVIDIA_API_KEY` as repository secrets. Results are uploaded as a build artifact (`results/summary.json`).

To trigger manually with a custom sample count:

```
Actions → LLM Eval → Run workflow → set n_samples
```

---

## Configuration

All runtime settings are in `config.yaml`:

```yaml
judge_model: gemini-1.5-flash
judge_cache_path: .eval_cache
n_samples: 200
batch_size: 10
batch_sleep_seconds: 1
```

---

## Dataset

[PubMedQA](https://huggingface.co/datasets/pubmed_qa) (`pqa_labeled` split) — biomedical yes/no/maybe question answering grounded in PubMed abstracts. The last 200 examples of the training split are used as the held-out test set, matching the ClinicalQA benchmark.
