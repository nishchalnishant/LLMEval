import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

SUMMARY_PATH = Path("results/summary.json")
SCORE_METRICS = ["faithfulness", "answer_relevance", "context_precision", "adversarial_score"]
JUDGE_METRIC = "llm_judge_overall"   # 0–10 scale
LATENCY_METRIC = "avg_latency_ms"
ALL_METRICS = SCORE_METRICS + [JUDGE_METRIC, LATENCY_METRIC]

st.set_page_config(page_title="LLMEval Leaderboard", layout="wide")
st.title("LLMEval Leaderboard")

if not SUMMARY_PATH.exists():
    st.error("No results found. Run the evaluation pipeline first: `python -m pipeline.runner`")
    st.stop()

with open(SUMMARY_PATH) as f:
    data = json.load(f)

st.caption(f"Timestamp: {data['timestamp']} | Samples evaluated: {data['n_samples']}")

models_data = data["models"]
df = pd.DataFrame(models_data).T.reset_index().rename(columns={"index": "model"})

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    sort_metric = st.selectbox("Sort leaderboard by", ALL_METRICS, index=ALL_METRICS.index(JUDGE_METRIC))
    score_threshold = st.slider("Pass threshold (0–1 metrics)", 0.0, 1.0, 0.7, 0.05)
    judge_threshold = st.slider("Pass threshold (judge score 0–10)", 0.0, 10.0, 6.0, 0.5)

# ── Leaderboard with color coding ─────────────────────────────────────────────
st.subheader("Leaderboard")

df_display = df[["model"] + [m for m in ALL_METRICS if m in df.columns]].copy()
df_display = df_display.sort_values(sort_metric, ascending=(sort_metric == LATENCY_METRIC))
df_display = df_display.rename(columns={"avg_latency_ms": "latency (ms)"})
display_cols = [c for c in df_display.columns if c != "model"]

def _color_cell(val, col):
    if col == "latency (ms)":
        return ""
    threshold = judge_threshold / 10.0 if col == JUDGE_METRIC else score_threshold
    normalized = val / 10.0 if col == JUDGE_METRIC else val
    if normalized >= threshold:
        return "background-color: #d4edda; color: #155724"   # green
    elif normalized >= threshold * 0.85:
        return "background-color: #fff3cd; color: #856404"   # yellow
    else:
        return "background-color: #f8d7da; color: #721c24"   # red

styled = df_display.set_index("model").style.apply(
    lambda col: [_color_cell(v, col.name) for v in col], axis=0
).format({
    "faithfulness": "{:.2f}",
    "answer_relevance": "{:.2f}",
    "context_precision": "{:.2f}",
    "adversarial_score": "{:.2f}",
    JUDGE_METRIC: "{:.1f}/10",
    "latency (ms)": "{:.0f} ms",
})

st.dataframe(styled, use_container_width=True)

# ── Per-metric bar chart ───────────────────────────────────────────────────────
st.subheader("Metric Comparison")
chart_metric = st.selectbox("Metric", SCORE_METRICS + [JUDGE_METRIC])
chart_data = df_display.set_index("model")[[chart_metric if chart_metric != JUDGE_METRIC else JUDGE_METRIC]]
st.bar_chart(chart_data)

# ── Adversarial radar ─────────────────────────────────────────────────────────
st.subheader("Adversarial Robustness by Category")
adv_cats = ["prompt_injection", "hallucination_trigger", "negation_trap", "out_of_distribution"]
adv_labels = ["Prompt Injection", "Hallucination", "Negation Traps", "OOD"]

fig = go.Figure()
for model_name, model_data in models_data.items():
    by_cat = model_data.get("adversarial_by_category", {})
    values = [by_cat.get(c, 0) for c in adv_cats]
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=adv_labels + [adv_labels[0]],
        fill="toself",
        name=model_name,
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    height=450,
)
st.plotly_chart(fig, use_container_width=True)
