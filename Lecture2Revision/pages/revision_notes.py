"""
Lecture To Revision - Revision Notes View
Displays grounded summaries, key definitions, detailed notes, and exam practice questions
with downloadable Markdown export functionality.
"""

import json
import streamlit as st
from database import get_all_lectures, get_learning_materials_by_lecture
from flashcard_service import compile_downloadable_notes
from evaluation import audit_material_grounding


def render_revision_notes():
    st.title("📖 Revision Notes & Key Concepts")
    st.caption("AI-generated, factual study notes grounded strictly in primary lecture texts.")

    lectures = get_all_lectures()
    if not lectures:
        st.info("No lecture materials available. Please upload a lecture PDF first.")
        return

    # Lecture Selector
    lecture_map = {f"{l['title']} ({l['topic']})": l for l in lectures}
    selected_name = st.selectbox("Select Lecture Material", options=list(lecture_map.keys()))
    selected_lec = lecture_map[selected_name]

    materials = get_learning_materials_by_lecture(selected_lec["id"])
    if not materials:
        st.warning("No generated materials found for this lecture.")
        return

    # Parse JSON fields safely
    defs = materials.get("key_definitions", [])
    if isinstance(defs, str):
        try:
            defs = json.loads(defs)
        except Exception:
            defs = []

    exam_q = materials.get("exam_questions", [])
    if isinstance(exam_q, str):
        try:
            exam_q = json.loads(exam_q)
        except Exception:
            exam_q = []

    # Export Button & Backend Info Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"**Generation Engine:** `{materials['backend_used']}` | **Subject:** {selected_lec['subject']}")
    with c2:
        doc_markdown = compile_downloadable_notes(selected_lec, {**materials, "key_definitions": defs, "exam_questions": exam_q})
        st.download_button(
            label="📥 Download Notes (.md)",
            data=doc_markdown,
            file_name=f"{selected_lec['topic'].replace(' ', '_')}_Revision_Notes.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Tabs for structured viewing
    tab_summary, tab_notes, tab_defs, tab_exam, tab_audit = st.tabs([
        "Executive Summary",
        "Detailed Notes",
        "Key Definitions",
        "Exam-Oriented Questions",
        "Grounding Audit",
    ])

    with tab_summary:
        st.subheader("Executive Lecture Summary")
        st.markdown(materials["summary"])

    with tab_notes:
        st.subheader("Detailed Revision Notes")
        st.markdown(materials["revision_notes"])

    with tab_defs:
        st.subheader("Important Concepts & Terminology")
        if defs:
            for d in defs:
                with st.container(border=True):
                    st.markdown(f"### {d.get('term', 'Term')}")
                    st.write(d.get("definition", ""))
                    if d.get("source_context"):
                        st.caption(f"**Source Grounding:** *\"{d.get('source_context')}\"*")
        else:
            st.caption("No specific definitions listed.")

    with tab_exam:
        st.subheader("Exam-Oriented Conceptual Questions")
        if exam_q:
            for i, eq in enumerate(exam_q, 1):
                diff = eq.get("difficulty", "Medium")
                with st.expander(f"Question {i} [{diff}]: {eq.get('question', '')}"):
                    st.markdown("**Suggested Model Answer:**")
                    st.info(eq.get("suggested_answer", "N/A"))
        else:
            st.caption("No exam questions generated.")

    with tab_audit:
        st.subheader("Empirical Grounding Audit")
        audit_res = audit_material_grounding(
            selected_lec.get("cleaned_text", ""),
            {**materials, "key_definitions": defs, "exam_questions": exam_q},
        )
        st.metric("Source Grounding Verification Score", f"{audit_res['grounding_score_pct']}%")
        for item in audit_res["checklist"]:
            icon = "✅" if item["passed"] else "⚠️"
            st.markdown(f"{icon} **{item['criterion']}:** {item['details']}")
