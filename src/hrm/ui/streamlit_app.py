"""Streamlit page: ask an HR policy question, see the cited answer and its sources.

Run with `hrm ui` (and `hrm serve` in another terminal). It only talks to the HTTP API.
"""

from typing import get_args

import streamlit as st

from hrm.core.config import get_settings
from hrm.rag.models import Category
from hrm.ui.client import ApiClient, ApiError

CATEGORY_LABELS: dict[Category | None, str] = {
    None: "All policies",
    "time_off": "Time off",
    "parental_leave": "Parental leave",
    "benefits": "Benefits (US)",
    "expenses": "Expenses & equipment",
    "conduct": "Code of conduct",
}
assert set(CATEGORY_LABELS) - {None} == set(get_args(Category))

st.set_page_config(page_title="HRM Copilot", page_icon="💬")
st.title("HRM Copilot")
st.caption("Answers from Sourcegraph's HR policies, with sources. Not legal advice.")

with st.form("ask"):
    question = st.text_input("Your question", placeholder="How much notice do I give before PTO?")
    category = st.selectbox(
        "Policy area", list(CATEGORY_LABELS), format_func=lambda c: CATEGORY_LABELS[c]
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Searching the policies..."):
        try:
            result = ApiClient(get_settings().api_url).ask(question.strip(), category)
        except ApiError as exc:
            st.error(str(exc))
            if exc.request_id:
                st.caption(f"Request id: {exc.request_id}")
            st.stop()

    if result.found:
        st.markdown(result.answer)
        with st.expander(f"Sources ({len(result.citations)})", expanded=True):
            for c in result.citations:
                st.markdown(f"**[{c.number}] {c.doc_id}** · {c.section}")
                st.caption(c.quote)
    else:
        st.info(result.answer)
    st.caption(
        f"Retrieval {result.timings.retrieve_ms} ms · answer {result.timings.generate_ms} ms "
        f"· total {result.timings.total_ms} ms"
    )
