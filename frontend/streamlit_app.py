"""Streamlit adaptive RAG chat UI."""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
ROUTE_BADGE_COLOR = {
    "LOCAL_RAG": "🟢",
    "WEB_SEARCH": "🌐",
    "HYBRID": "🔀",
    "UNKNOWN": "❓",
}

st.set_page_config(
    page_title="Adaptive Research Copilot",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Adaptive RAG")
    st.markdown("---")

    # Health check
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API ✅ — backend: `{health.get('services', {}).get('vector_backend', '?')}`")
    except Exception:
        st.error("API ❌ — is the backend running?")

    st.markdown("### Ingest Document")
    doc_path = st.text_input("File path (PDF/MD/HTML)", placeholder="/path/to/file.pdf")
    if st.button("Ingest", disabled=not doc_path):
        with st.spinner("Ingesting…"):
            try:
                r = httpx.post(f"{API_BASE}/ingest", json={"file_path": doc_path}, timeout=120)
                r.raise_for_status()
                data = r.json()
                st.success(f"✅ Stored {data['chunks_stored']} chunks")
            except httpx.HTTPStatusError as e:
                st.error(f"Error {e.response.status_code}: {e.response.text}")
            except Exception as exc:
                st.error(str(exc))

    st.markdown("---")
    st.markdown("### Session")
    if st.button("New session"):
        for key in ["session_id", "messages"]:
            st.session_state.pop(key, None)
        st.rerun()

    if session_id := st.session_state.get("session_id"):
        st.code(session_id, language=None)

    st.markdown("---")
    if st.button("Run Eval"):
        with st.spinner("Running LangSmith eval…"):
            try:
                r = httpx.post(f"{API_BASE}/eval/run", timeout=300)
                r.raise_for_status()
                report = r.json()
                st.json(report)
            except Exception as exc:
                st.error(str(exc))

# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("Adaptive Research Copilot")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            route = meta.get("route", "UNKNOWN")
            badge = ROUTE_BADGE_COLOR.get(route, "❓")
            cols = st.columns(4)
            cols[0].metric("Route", f"{badge} {route}")
            cols[1].metric("Confidence", f"{meta.get('confidence', 0):.0%}")
            cols[2].metric("Latency", f"{meta.get('latency_ms', 0):.0f} ms" if meta.get("latency_ms") else "—")
            usage = meta.get("token_usage", {})
            total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            cols[3].metric("Tokens", str(total_tokens) if total_tokens else "—")

            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"📎 {len(sources)} source(s)"):
                    for s in sources:
                        src_type = "📄" if s["type"] == "local" else "🌐"
                        st.markdown(f"**[{s['index']}]** {src_type} `{s['source']}`")
                        st.caption(s["snippet"])

if prompt := st.chat_input("Ask a research question…"):
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking…")

        payload: dict = {"question": prompt}
        if sid := st.session_state.get("session_id"):
            payload["session_id"] = sid

        try:
            start = time.perf_counter()
            response = httpx.post(f"{API_BASE}/chat", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            placeholder.error(f"API error {e.response.status_code}: {e.response.text}")
            st.stop()
        except Exception as exc:
            placeholder.error(str(exc))
            st.stop()

        st.session_state["session_id"] = data["session_id"]
        answer = data["answer"]
        placeholder.markdown(answer)

        # Metrics row
        route = data.get("route", "UNKNOWN")
        badge = ROUTE_BADGE_COLOR.get(route, "❓")
        cols = st.columns(4)
        cols[0].metric("Route", f"{badge} {route}")
        cols[1].metric("Confidence", f"{data.get('confidence', 0):.0%}")
        cols[2].metric("Latency", f"{data.get('latency_ms', 0):.0f} ms" if data.get("latency_ms") else "—")
        usage = data.get("token_usage", {})
        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        cols[3].metric("Tokens", str(total_tokens) if total_tokens else "—")

        sources = data.get("sources", [])
        if sources:
            with st.expander(f"📎 {len(sources)} source(s)"):
                for s in sources:
                    src_type = "📄" if s["type"] == "local" else "🌐"
                    st.markdown(f"**[{s['index']}]** {src_type} `{s['source']}`")
                    st.caption(s["snippet"])

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "meta": {
                    "route": route,
                    "confidence": data.get("confidence", 0),
                    "latency_ms": data.get("latency_ms"),
                    "token_usage": usage,
                    "sources": sources,
                },
            }
        )
