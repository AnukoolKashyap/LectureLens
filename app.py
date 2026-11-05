"""Streamlit demo UI.

    streamlit run app.py

Left pane: the video and the pipeline settings actually in force.
Right pane: the generated summary, key points, timestamps, mind map,
evaluation metrics, safety screen results and the reviewer feedback form.

This is the screen to capture for the Review 2 slide.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from yt_summarizer.config import settings
from yt_summarizer.layer7_errors import PipelineError
from yt_summarizer.layer9_eval import record_feedback
from yt_summarizer.pipeline import run_pipeline
from yt_summarizer.prompts import USE_CASE_PROFILES
from yt_summarizer.stage1_ingest import default_sample
from yt_summarizer.stage6_output import to_markdown, to_mermaid

st.set_page_config(
    page_title="AI-Powered Video Summarization",
    page_icon="▶",
    layout="wide",
)

# --------------------------------------------------------------------------
# Sidebar — pipeline settings
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("Pipeline settings")
    st.caption("21CSE306P Applied Generative AI — course project")

    profile_key = st.selectbox(
        "Use-case profile",
        options=list(USE_CASE_PROFILES.keys()),
        format_func=lambda k: USE_CASE_PROFILES[k].label,
        help="Selects the prompt variation applied to the summarisation stage.",
    )

    st.divider()
    st.subheader("Stage 2 — chunking")
    settings.chunk_size = st.number_input("Chunk size (chars)", 300, 4000, settings.chunk_size, 100)
    settings.chunk_overlap = st.number_input("Overlap (chars)", 0, 1000, settings.chunk_overlap, 50)

    st.subheader("Stage 4 — retrieval")
    settings.top_k = st.slider("top-k", 1, 15, settings.top_k)
    settings.rerank = st.checkbox("Re-rank candidates", value=settings.rerank)

    st.divider()
    st.subheader("Stage 5 — model")
    if settings.has_llm:
        st.success("Mistral configured: `%s`" % settings.llm_model)
    else:
        st.warning(
            "No `MISTRAL_API_KEY` found. The offline extractive fallback will run "
            "instead — its output is copied from the transcript, not generated."
        )

    offline = st.checkbox("Force offline mode", value=settings.offline_mode)
    settings.offline_mode = offline

    st.divider()
    reference_text = st.text_area(
        "Reference summary (optional)",
        height=120,
        help="ROUGE needs a human-written reference. Leave this blank and ROUGE is "
             "reported as not computed rather than as a number.",
    )

# --------------------------------------------------------------------------
# Main pane — input
# --------------------------------------------------------------------------

st.title("AI-Powered Video Summarization for Smarter Learning")

col_url, col_btn = st.columns([5, 1])
with col_url:
    url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("Summarise", type="primary", use_container_width=True)

use_sample = st.checkbox(
    "Use the bundled sample transcript instead (no network needed)",
    value=False,
)

if run_clicked:
    if not url.strip() and not use_sample:
        st.error("Enter a YouTube URL, or tick the sample transcript box.")
        st.stop()

    status = st.status("Running the pipeline...", expanded=True)
    labels = {
        1: "Stage 1 — ingestion and transcript extraction",
        2: "Stage 2 — preprocessing and chunking",
        3: "Stage 3 — embedding and vector store",
        4: "Stage 4 — retrieval engine",
        5: "Stage 5 — summarisation engine",
        6: "Stage 6 — output assembly",
    }

    def progress(step: int, _label: str) -> None:
        status.write(labels.get(step, "Stage %d" % step))

    try:
        run = run_pipeline(
            url=url.strip(),
            local_path=default_sample() if use_sample else None,
            profile_key=profile_key,
            reference_summary=reference_text.strip() or None,
            progress=progress,
        )
    except PipelineError as exc:
        status.update(label="Pipeline stopped", state="error")
        st.error(exc.user_message)
        st.stop()

    status.update(label="Done", state="complete")
    st.session_state["run"] = run

# --------------------------------------------------------------------------
# Main pane — results
# --------------------------------------------------------------------------

run = st.session_state.get("run")
if run is None:
    st.info("Enter a video URL and press **Summarise** to run stages 1-6.")
    st.stop()

summary = run.summary
diag = summary.diagnostics

st.subheader(summary.title)
st.caption("%s · %s" % (summary.author, summary.url))

if not summary.generated:
    st.error(
        "**Offline fallback output.** These sentences were extracted from the "
        "transcript, not generated by a model. Do not present this as the system's "
        "generated output, and do not report the metrics below as project results."
    )
elif diag.get("embedder_degraded"):
    st.warning(
        "The hashing fallback embedder was used, so retrieval was lexical rather than "
        "semantic. Metrics are not representative of the real system."
    )

metric_cols = st.columns(4)
rows = run.evaluation.display_rows()
for col, (label, value) in zip(metric_cols, rows[:4]):
    col.metric(label, value)

if run.evaluation.rouge_1 is None:
    st.caption(
        "ROUGE is not shown because no reference summary was supplied. Paste one in "
        "the sidebar to compute it."
    )

tabs = st.tabs(
    ["Summary", "Key points", "Timestamps", "Mind map", "Retrieved context",
     "Safety screens", "Diagnostics", "Export"]
)

with tabs[0]:
    st.write(summary.summary or "_No summary returned._")
    question = st.text_input("Ask a question about this video")
    if question:
        with st.spinner("Retrieving and answering..."):
            st.info(run.ask(question, profile_key))

with tabs[1]:
    if summary.key_points:
        for point in summary.key_points:
            st.markdown("- %s" % point)
    else:
        st.write("_None returned._")

with tabs[2]:
    if summary.timestamps:
        st.table(
            [{"Time": t["time"], "Topic": t.get("topic", "")} for t in summary.timestamps]
        )
    else:
        st.write("_None returned._")

with tabs[3]:
    st.code(to_mermaid(summary.mind_map), language="text")
    st.caption("Mermaid mind-map source. Paste into any Mermaid renderer to view it.")

with tabs[4]:
    st.caption(
        "The %d chunks below are the only text the summariser was allowed to use."
        % len(run.retrieved)
    )
    for i, chunk in enumerate(run.retrieved, 1):
        with st.expander("[%d] %s — score %.3f" % (i, chunk.timestamp, chunk.final_score)):
            st.write(chunk.text)
            st.caption(
                "vector %.3f · lexical %.3f" % (chunk.vector_score, chunk.lexical_score)
            )

with tabs[5]:
    safety = summary.safety
    counts = safety["counts"]
    cols = st.columns(4)
    cols[0].metric("Prompt injection", counts["prompt_injection"])
    cols[1].metric("Content safety", counts["content_safety"])
    cols[2].metric("Bias / fairness", counts["bias_fairness"])
    cols[3].metric("Weakly grounded", counts["ungrounded_claims"])

    if safety["passed"]:
        st.success("All red-teaming screens passed.")
    for key, title in [
        ("prompt_injection", "Prompt injection attempts (neutralised before prompting)"),
        ("content_safety", "Content safety flags"),
        ("bias_fairness", "Possible unsupported generalisations"),
    ]:
        if safety[key]:
            st.markdown("**%s**" % title)
            for finding in safety[key]:
                st.markdown("- `%s` — %s" % (finding["label"], finding["excerpt"]))
    if safety["ungrounded_claims"]:
        st.markdown("**Sentences weakly supported by the retrieved context**")
        st.caption("Lexical proxy, not an entailment check.")
        for claim in safety["ungrounded_claims"]:
            st.markdown("- %s" % claim)

with tabs[6]:
    st.json(diag)

with tabs[7]:
    st.download_button(
        "Download Markdown",
        data=to_markdown(summary),
        file_name="%s_summary.md" % summary.video_id,
        mime="text/markdown",
    )
    if run.exports:
        st.caption("Also written to disk:")
        for kind, path in run.exports.items():
            st.code("%s: %s" % (kind, path))

st.divider()
st.subheader("Reviewer feedback")
st.caption(
    "Ratings are appended to a log a person reads later. The pipeline does not read "
    "this file at runtime and nothing here changes its behaviour automatically."
)
fb_cols = st.columns([1, 3, 1])
rating = fb_cols[0].select_slider("Rating", options=[1, 2, 3, 4, 5], value=4)
comment = fb_cols[1].text_input("Comment (optional)")
if fb_cols[2].button("Submit", use_container_width=True):
    path = record_feedback(summary.video_id, rating, comment)
    st.success("Recorded to %s" % path)
