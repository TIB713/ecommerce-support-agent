"""Streamlit UI: ticket + order JSON → Resolve → structured JSON output."""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
import streamlit as st

DEFAULT_CONTEXT = """{
  "order_date": "2026-01-10",
  "delivery_date": "2026-01-14",
  "item_category": "electronics",
  "fulfillment_type": "standard",
  "shipping_region": "domestic",
  "order_status": "delivered",
  "payment_method": "credit_card"
}"""

st.set_page_config(
    page_title="Support Resolution Agent",
    page_icon="🛒",
    layout="wide",
)

st.title("E-commerce Support Resolution Agent")
st.caption("Multi-agent RAG · Groq · FAISS · HuggingFace embeddings")

api_base = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
st.sidebar.markdown("Start backend: `uvicorn backend.main:app --reload`")
st.sidebar.warning(
    "Set **GROQ_API_KEY** in the project `.env` (same folder as `backend/`) and **restart** uvicorn.",
)

ticket = st.text_area(
    "Customer ticket",
    height=180,
    placeholder="Describe the customer's issue...",
)

order_json = st.text_area("Order context (JSON)", value=DEFAULT_CONTEXT, height=220)

if st.button("Resolve", type="primary"):
    try:
        ctx = json.loads(order_json)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    url = api_base.rstrip("/") + "/query"
    with st.spinner("Running triage → retrieval → writer → compliance..."):
        try:
            r = httpx.post(url, json={"ticket": ticket, "order_context": ctx}, timeout=120.0)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.text
            try:
                body = e.response.json()
                if isinstance(body, dict) and "detail" in body:
                    detail = str(body["detail"])
            except Exception:
                pass
            st.error(f"Server error {e.response.status_code}: {detail}")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Structured output (JSON)")
    st.json(data)

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Classification", data.get("classification", "—"))
    c2.metric("Decision", data.get("decision", "—"))
    _conf = data.get("confidence", "—")
    if isinstance(_conf, (int, float)):
        _conf = f"{float(_conf):.2f}"
    c3.metric("Confidence", str(_conf))

    st.markdown("**Customer response**")
    st.info(data.get("customer_response", ""))

    st.markdown("**Rationale**")
    st.write(data.get("rationale", ""))

    st.markdown("**Citations**")
    st.write(data.get("citations", []))

    qs = data.get("clarifying_questions") or []
    if qs:
        st.markdown("**Clarifying questions (triage)**")
        for q in qs:
            st.write(f"- {q}")

    st.download_button(
        "Download JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="resolution.json",
        mime="application/json",
    )
