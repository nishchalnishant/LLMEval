import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

SUMMARY_PATH = Path("results/summary.json")
METRICS = ["faithfulness", "answer_relevance", "context_precision", "llm_judge_overall", "adversarial_score", "avg_latency_ms"]

st.set_page_config(page_title="LLMEval Leaderboard", layout="wide")
st.title("LLMEval Leaderboard")

if not SUMMARY_PATH.exists():
    st.error("No results found. Run the evaluation pipeline first.")
    st.stop()

with open(SUMMARY_PATH) as f:
    data = json.load(f)

st.caption(f"Timestamp: {data['timestamp']} | Samples: {data['n_samples']}")

models_data = data["models"]
df = pd.DataFrame(models_data).T.reset_index().rename(columns={"index": "model"})

with st.sidebar:
    st.header("Filters")
    sort_metric = st.selectbox("Sort by", METRICS, index=3)
    threshold = st.slider("Pass threshold (non-latency metrics)", 0.0, 1.0, 0.7, 0.05)

df_display = df[["model"] + [m for m in METRICS if m in df.columns]]
df_sorted = df_display.sort_values(sort_metric, ascending=(sort_metric == "avg_latency_ms"))

st.subheader("Leaderboard")
st.dataframe(df_sorted, use_container_width=True)

st.subheader("Metric Comparison")
score_metrics = [m for m in METRICS if m != "avg_latency_ms"]
chart_metric = st.selectbox("Metric", score_metrics)
chart_data = df_sorted[["model", chart_metric]].set_index("model")
st.bar_chart(chart_data)

st.subheader("Adversarial Radar")
adv_cats = ["prompt_injection", "hallucination_trigger", "negation_trap", "out_of_distribution"]
fig = go.Figure()
for model_name, model_data in models_data.items():
    by_cat = model_data.get("adversarial_by_category", {})
    values = [by_cat.get(c, 0) for c in adv_cats]
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=adv_cats + [adv_cats[0]], fill="toself", name=model_name))

fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
st.plotly_chart(fig, use_container_width=True)
