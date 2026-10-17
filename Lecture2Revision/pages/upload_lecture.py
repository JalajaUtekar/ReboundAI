"""
Lecture To Revision - Upload & Ingest Lecture View
Accepts PDF lecture slides/notes, extracts and cleans text with PyMuPDF,
chunks long texts, and triggers grounded material generation with the configured Transformer backend.
"""

import json
from pathlib import Path
import streamlit as st

from config import TRANSFORMER_BACKEND, UPLOAD_DIR
from database import (
    get_all_lectures,
    save_flashcards,
    save_learning_materials,
    save_lecture,
    save_quiz_questions,
)
from lecture_processor import (
    CorruptPDFError,
    NoExtractableTextError,
    chunk_document,
    extract_text_from_pdf,
)
from transformer_service import generate_learning_materials
from adaptive_engine import generate_adaptive_schedule


def render_upload_lecture():
    st.title("📤 Lecture Upload & Processing")
    st.caption("Convert digital lecture PDFs into grounded revision notes, flashcards, and quizzes.")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("1. Lecture Document & Metadata")
        uploaded_file = st.file_uploader("Upload Lecture PDF (Text-based slides or notes)", type=["pdf"])

        title = st.text_input("Lecture Title", placeholder="e.g., Computer Networks: Transport Layer")
        c_sub, c_top = st.columns(2)
        with c_sub:
            subject = st.text_input("Subject / Course", placeholder="e.g., Computer Science")
        with c_top:
            topic = st.text_input("Specific Topic", placeholder="e.g., TCP Congestion Control")

    with col2:
        st.subheader("2. Transformer AI Engine")
        st.markdown(f"**Default Backend:** `{TRANSFORMER_BACKEND}`")

        backend_choice = st.selectbox(
            "Select Inference Engine for this Lecture",
            options=["offline_extractive", "huggingface_api", "gemini_api", "openai_compatible"],
            index=["offline_extractive", "huggingface_api", "gemini_api", "openai_compatible"].index(
                TRANSFORMER_BACKEND if TRANSFORMER_BACKEND in ["offline_extractive", "huggingface_api", "gemini_api", "openai_compatible"] else "offline_extractive"
            ),
            help="Choose the model used to summarize, generate definitions, and construct questions.",
        )

        st.info(
            "ℹ️ **Grounding Guarantee:** All backends are strictly prompted and filtered to use only facts "
            "found in the uploaded text. 'offline_extractive' runs locally with zero external network calls."
        )

    submit_button = st.button("🚀 Process Lecture & Generate Revision Kit", type="primary", use_container_width=True)

    if submit_button:
        if not uploaded_file:
            st.error("Please upload a PDF file first.")
            return
        if not title.strip() or not topic.strip():
            st.error("Please provide both a Lecture Title and Topic.")
            return

        with st.status("Processing lecture...", expanded=True) as status:
            # Step 1: Save file to disk
            st.write("📁 Saving document to disk...")
            saved_pdf_path = UPLOAD_DIR / uploaded_file.name
            with open(saved_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Step 2: PyMuPDF Extraction & Cleaning
            st.write("🔍 Extracting text with PyMuPDF (fitz)...")
            try:
                full_text, pages_meta = extract_text_from_pdf(saved_pdf_path)
            except NoExtractableTextError as e:
                status.update(label="Extraction Failed: Scanned Document", state="error")
                st.error(f"❌ {e}")
                return
            except CorruptPDFError as e:
                status.update(label="Extraction Failed: Corrupt PDF", state="error")
                st.error(f"❌ {e}")
                return

            word_count = len(full_text.split())
            st.write(f"✅ Extracted {word_count} words across {len(pages_meta)} page(s).")

            # Step 3: Chunking
            st.write("✂️ Segmenting document into semantic context windows...")
            chunks = chunk_document(full_text, chunk_size_words=700, overlap_words=80)
            st.write(f"✅ Created {len(chunks)} overlapping chunk(s).")

            # Step 4: Save Lecture to SQLite
            st.write("💾 Registering lecture metadata in SQLite...")
            lecture_id = save_lecture(
                title=title.strip(),
                subject=subject.strip() or "General",
                topic=topic.strip(),
                filename=uploaded_file.name,
                file_path=str(saved_pdf_path),
                raw_text=full_text,
                cleaned_text=full_text,
                word_count=word_count,
                chunk_count=len(chunks),
            )

            # Step 5: Transformer Material Generation
            st.write(f"🧠 Synthesizing revision kit using `{backend_choice}`...")
            materials, actual_backend = generate_learning_materials(
                lecture_topic=topic.strip(),
                lecture_text=full_text,
                backend=backend_choice,
            )

            # Step 6: Persist structured outputs
            save_learning_materials(
                lecture_id=lecture_id,
                summary=materials["summary"],
                revision_notes=materials["revision_notes"],
                key_definitions=json.dumps(materials["key_definitions"]),
                exam_questions=json.dumps(materials["exam_questions"]),
                source_grounding_notes="Directly extracted from user uploaded PDF text.",
                backend_used=actual_backend,
            )

            fc_count = save_flashcards(lecture_id, topic.strip(), materials["flashcards"])
            q_count = save_quiz_questions(lecture_id, topic.strip(), materials["quiz_questions"])

            # Step 7: Update initial adaptive schedule
            generate_adaptive_schedule("demo_student")

            status.update(label="Revision Kit Successfully Generated!", state="complete")

        st.success(f"🎉 Lecture **'{title}'** processed! Generated **{fc_count} flashcards** and **{q_count} quiz questions** via `{actual_backend}`.")

        # Summary preview cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Words Extracted", word_count)
        with m2:
            st.metric("Flashcards Created", fc_count)
        with m3:
            st.metric("Quiz Questions Ready", q_count)

    st.markdown("---")
    st.subheader("📚 Previously Processed Lectures")
    lectures = get_all_lectures()
    if lectures:
        for lec in lectures:
            with st.expander(f"{lec['title']} ({lec['subject']} • {lec['topic']})"):
                st.write(f"**Filename:** `{lec['filename']}` | **Word Count:** {lec['word_count']} | **Chunks:** {lec['chunk_count']}")
                st.write(f"**Ingested:** {lec['created_at'][:19].replace('T', ' ')} UTC")
                st.text_area("Source Excerpt Preview", lec["cleaned_text"][:400] + "...", height=100, disabled=True)
    else:
        st.caption("No lectures uploaded yet.")
