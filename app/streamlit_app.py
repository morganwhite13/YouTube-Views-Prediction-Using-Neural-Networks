"""Web UI for ytviews.  Run with:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ytviews.categories import CATEGORY_NAMES
from ytviews.predictor import Predictor, VideoInput

st.set_page_config(page_title="ytviews - YouTube view predictor", page_icon="📈", layout="wide")


def fmt(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


@st.cache_resource
def load_model(path: str) -> Predictor:
    return Predictor.load(path)


with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model directory", os.environ.get("YTVIEWS_MODEL", "models/latest"))
    try:
        predictor = load_model(model_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    meta = predictor.meta
    ds = meta.get("dataset", {})
    st.caption(
        f"Trained on {ds.get('videos', '?')} videos from {ds.get('channels', '?')} channels "
        f"· text encoder `{meta.get('text_encoder')}`"
    )
    ours = meta.get("metrics", {}).get("neural net (ours)")
    if ours:
        st.metric("Typical error (test set)", f"{ours['median_factor_error']:.2f}x")
        st.metric("Within 2x of actual", f"{ours['within_2x']:.0%}")

    st.header("Your channel")
    subscribers = st.number_input("Subscribers", min_value=0, value=100_000, step=1_000)
    recent_raw = st.text_area(
        "Views of your recent uploads (optional, one per line or comma-separated)",
        help="The single most useful input: how your last ~10 videos did.",
    )
    recent = [float(x) for x in recent_raw.replace(",", "\n").split() if x.strip()]
    video_count = st.number_input("Total videos on channel (optional)", min_value=0, value=0)

st.title("📈 How many views will this video get?")

cats = list(CATEGORY_NAMES)
col1, col2 = st.columns([2, 1])
with col1:
    title = st.text_input("Title", "I Survived 100 Days in Minecraft")
    description = st.text_area("Description", "", height=100)
    tags = st.text_input("Tags (comma-separated)", "")
with col2:
    category = st.selectbox(
        "Category", cats, index=cats.index("20"), format_func=lambda c: CATEGORY_NAMES[c]
    )
    minutes = st.number_input("Length (minutes)", min_value=0.0, value=12.0, step=0.5)
    days = st.slider("Views after N days", 1, 365, 7)
    thumb_file = st.file_uploader("Thumbnail (optional)", type=["jpg", "jpeg", "png", "webp"])

thumb_path = None
if thumb_file is not None:
    thumb_path = Path(tempfile.gettempdir()) / f"ytviews_thumb_{thumb_file.file_id}"
    thumb_path.write_bytes(thumb_file.getvalue())
    with col2:
        st.image(thumb_file, width=200)

base = VideoInput(
    title=title,
    description=description,
    tags=[t.strip() for t in tags.split(",") if t.strip()],
    category_id=category,
    duration_s=minutes * 60 if minutes else None,
    age_days=float(days),
    subscribers=float(subscribers) or None,
    channel_video_count=float(video_count) or None,
    recent_video_views=recent,
    thumbnail_path=str(thumb_path) if thumb_path else None,
)

predict_tab, compare_tab, suggest_tab, report_tab = st.tabs(
    ["Prediction", "Compare titles", "Title ideas", "Model report"]
)

with predict_tab:
    pred = predictor.predict(base)[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Expected views after {days} days", fmt(pred.median))
    c2.metric("Pessimistic (p10)", fmt(pred.low))
    c3.metric("Optimistic (p90)", fmt(pred.high))
    if not recent:
        st.info("Tip: add your recent uploads' views in the sidebar for a much sharper estimate.")
    traj = predictor.trajectory(base).set_index("days")
    st.subheader("Expected growth")
    st.line_chart(
        traj[["low", "median", "high"]],
        y_label="cumulative views",
        x_label="days",
        color=["#9db9e3", "#1f4e9c", "#9db9e3"],
    )
    st.subheader("Why this number?")
    ex = predictor.explain(base)
    st.caption(
        f"An average video after {days} days gets about {fmt(ex.baseline_views)} views. "
        "Each factor below multiplies that; together they give this prediction."
    )
    effects = ex.effects.assign(
        effect=lambda d: d["factor"].map(lambda f: f"×{f:.2f}"),
        direction=lambda d: d["log_effect"].map(lambda v: "raises" if v > 0 else "lowers"),
    )
    st.bar_chart(effects, x="group", y="log_effect", horizontal=True, sort="-log_effect")
    st.dataframe(effects[["group", "effect", "direction"]], hide_index=True, width="stretch")

with compare_tab:
    st.write("Enter alternative titles (one per line) to see which one the model prefers.")
    raw = st.text_area(
        "Candidate titles",
        "\n".join([title, "Minecraft Survival Tutorial - Part 3", "$1 vs $10,000 Minecraft Build"]),
        height=140,
    )
    titles = [t.strip() for t in raw.splitlines() if t.strip()]
    if titles:
        ranked = predictor.compare_titles(base, titles)
        st.bar_chart(ranked, x="title", y="median", horizontal=True, sort="-median")
        st.dataframe(
            ranked.assign(
                median=ranked["median"].map(fmt),
                low=ranked["low"].map(fmt),
                high=ranked["high"].map(fmt),
                vs_best=ranked["vs_best"].map("{:.0%}".format),
            ),
            hide_index=True,
            width="stretch",
        )

with suggest_tab:
    from ytviews.suggest import suggest_titles

    st.write(
        "Rewrites of your title using common YouTube patterns, ranked by the model. "
        "Check that the winner is still an honest description of the video."
    )
    use_llm = st.toggle(
        "Also ask Claude for ideas",
        value=False,
        help="Needs ANTHROPIC_API_KEY on the server; each click is a paid API call.",
    )
    if st.button("Suggest titles", type="primary"):
        with st.spinner("Generating and ranking titles..."):
            ideas = suggest_titles(predictor, base, use_llm=use_llm)
        st.dataframe(
            ideas.assign(
                median=ideas["median"].map(fmt),
                vs_best=ideas["vs_best"].map("{:.0%}".format),
            )[["title", "source", "median", "vs_best"]],
            hide_index=True,
            width="stretch",
        )

with report_tab:
    report = Path(model_path) / "report.md"
    image = Path(model_path) / "report.png"
    if image.exists():
        st.image(str(image))
    if report.exists():
        st.markdown(report.read_text())
    if meta.get("importance"):
        st.bar_chart(pd.Series(meta["importance"], name="MAE increase"), horizontal=True)
