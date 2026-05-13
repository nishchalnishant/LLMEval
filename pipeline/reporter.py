import json
from datetime import datetime
from pathlib import Path

Path("results").mkdir(exist_ok=True)

METRICS = ["faithfulness", "answer_relevance", "context_precision", "llm_judge_overall", "adversarial_score", "avg_latency_ms"]

THRESHOLDS = {
    "faithfulness": (0.8, 0.6),
    "answer_relevance": (0.8, 0.6),
    "context_precision": (0.8, 0.6),
    "llm_judge_overall": (7.0, 5.0),
    "adversarial_score": (0.8, 0.6),
    "avg_latency_ms": (None, None),
}


def _cell_color(metric: str, value: float) -> str:
    hi, lo = THRESHOLDS.get(metric, (None, None))
    if hi is None:
        return "#ffffff"
    if metric == "avg_latency_ms":
        return "#ffffff"
    if value >= hi:
        return "#c6efce"
    if value >= lo:
        return "#ffeb9c"
    return "#ffc7ce"


def generate_reports(summary: dict, n_samples: int) -> None:
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": n_samples,
        "models": {
            name: {k: v for k, v in data.items() if k in METRICS}
            for name, data in summary.items()
        },
    }
    with open("results/summary.json", "w") as f:
        json.dump(output, f, indent=2)

    model_names = list(summary.keys())
    header_cells = "".join(f"<th>{m}</th>" for m in model_names)
    rows = ""
    for metric in METRICS:
        cells = ""
        for model in model_names:
            val = summary[model].get(metric, 0)
            color = _cell_color(metric, val)
            cells += f'<td style="background:{color};text-align:center">{val:.3f}</td>'
        rows += f"<tr><td><b>{metric}</b></td>{cells}</tr>"

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>LLMEval Report</title></head>
<body style="font-family:sans-serif;padding:20px">
<h1>LLMEval Report</h1>
<p>Generated: {output['timestamp']} &mdash; {n_samples} samples</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
  <thead><tr><th>Metric</th>{header_cells}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""

    with open("results/report.html", "w") as f:
        f.write(html)

    print("Reports written: results/summary.json, results/report.html")
