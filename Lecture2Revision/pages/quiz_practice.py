"""
Lecture To Revision - Interactive Quiz Practice View
Delivers an interactive testing interface with 4-choice MCQs, instant evaluation,
detailed explanations, idempotency protection against duplicate saves, and retry capability.
"""

import time
import streamlit as st
from database import (
    get_all_lectures,
    get_quiz_history,
    get_quiz_questions_by_lecture,
    record_quiz_attempt,
)
from quiz_generator import evaluate_quiz_answers, generate_attempt_uuid
from adaptive_engine import generate_adaptive_schedule


def render_quiz_practice():
    st.title("🎯 Interactive Quiz Practice")
    st.caption("Test your comprehension with questions generated directly from lecture concepts.")

    lectures = get_all_lectures()
    if not lectures:
        st.info("No lecture materials available. Please upload a lecture PDF first.")
        return

    # Select topic/lecture
    lecture_map = {f"{l['title']} ({l['topic']})": l for l in lectures}
    selected_name = st.selectbox("Select Lecture for Quiz Practice", options=list(lecture_map.keys()))
    selected_lec = lecture_map[selected_name]

    questions = get_quiz_questions_by_lecture(selected_lec["id"])
    if not questions:
        st.warning(f"No quiz questions available for '{selected_lec['title']}'.")
        return

    # Initialize session state for quiz
    session_key_prefix = f"quiz_{selected_lec['id']}"
    uuid_key = f"{session_key_prefix}_uuid"
    start_time_key = f"{session_key_prefix}_start_time"
    submitted_key = f"{session_key_prefix}_submitted"
    results_key = f"{session_key_prefix}_results"
    answers_key = f"{session_key_prefix}_answers"

    if uuid_key not in st.session_state:
        st.session_state[uuid_key] = generate_attempt_uuid()
        st.session_state[start_time_key] = time.time()
        st.session_state[submitted_key] = False
        st.session_state[results_key] = None
        st.session_state[answers_key] = {}

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.write(f"**Topic:** {selected_lec['topic']} | **Questions:** {len(questions)}")
    with col_btn:
        if st.session_state[submitted_key]:
            if st.button("🔄 Retake Quiz", use_container_width=True):
                st.session_state[uuid_key] = generate_attempt_uuid()
                st.session_state[start_time_key] = time.time()
                st.session_state[submitted_key] = False
                st.session_state[results_key] = None
                st.session_state[answers_key] = {}
                st.rerun()

    # Form for taking quiz
    if not st.session_state[submitted_key]:
        with st.form(key=f"form_{selected_lec['id']}"):
            user_selections = {}
            for i, q in enumerate(questions, 1):
                st.markdown(f"#### Q{i}. {q['question_text']}")
                st.caption(f"Difficulty: {q.get('difficulty', 'Medium')}")

                options_display = {
                    "A": f"A) {q['option_a']}",
                    "B": f"B) {q['option_b']}",
                    "C": f"C) {q['option_c']}",
                    "D": f"D) {q['option_d']}",
                }

                choice = st.radio(
                    f"Select answer for Question {i}:",
                    options=["A", "B", "C", "D"],
                    format_func=lambda x: options_display[x],
                    index=None,
                    key=f"q_{q['id']}",
                )
                if choice:
                    user_selections[q["id"]] = choice
                st.markdown("---")

            submit_quiz = st.form_submit_button("Submit Quiz for Evaluation", type="primary", use_container_width=True)

            if submit_quiz:
                elapsed = max(1.0, time.time() - st.session_state[start_time_key])
                eval_results = evaluate_quiz_answers(questions, user_selections, time_spent_seconds=elapsed)

                # Persist to SQLite idempotently
                record_quiz_attempt(
                    attempt_uuid=st.session_state[uuid_key],
                    student_id="demo_student",
                    lecture_id=selected_lec["id"],
                    topic=selected_lec["topic"],
                    score_pct=eval_results["score_pct"],
                    total_questions=eval_results["total_questions"],
                    correct_count=eval_results["correct_count"],
                    incorrect_count=eval_results["incorrect_count"],
                    time_spent_seconds=eval_results["time_spent_seconds"],
                    responses=eval_results["responses"],
                )

                # Update adaptive revision plan with the fresh attempt
                generate_adaptive_schedule("demo_student")

                st.session_state[submitted_key] = True
                st.session_state[results_key] = eval_results
                st.session_state[answers_key] = user_selections
                st.rerun()

    else:
        # Results View
        res = st.session_state[results_key]
        score = res["score_pct"]

        st.subheader("Quiz Results & Performance Analysis")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Score", f"{score}%")
        with c2:
            st.metric("Correct Answers", f"{res['correct_count']} / {res['total_questions']}")
        with c3:
            st.metric("Incorrect Answers", res["incorrect_count"])
        with c4:
            st.metric("Time Spent", f"{res['time_spent_seconds']:.0f}s")

        if score >= 80.0:
            st.success("🌟 Mastery demonstrated! The Adaptive Engine has extended your SM-2 revision interval.")
        elif score >= 60.0:
            st.warning("⚠️ Passing score, but key concepts require reinforcement in your upcoming revision.")
        else:
            st.error("🚨 Needs revision! The Adaptive Engine has flagged this topic as high priority for review.")

        st.markdown("### Answer Review & Explanations")
        for i, r in enumerate(res["responses"], 1):
            with st.container(border=True):
                status_icon = "✅ Correct" if r["is_correct"] else "❌ Incorrect"
                st.markdown(f"**Q{i}: {r['question_text']}** — *{status_icon}*")
                st.write(f"• **Your Answer:** [{r['selected_option']}] {r['selected_text']}")
                if not r["is_correct"]:
                    st.write(f"• **Correct Answer:** [{r['correct_option']}] {r['correct_text']}")
                st.info(f"💡 **Explanation:** {r['explanation']}")

    st.markdown("---")
    st.subheader("📜 Recent Quiz History")
    history = get_quiz_history("demo_student")
    if history:
        import pandas as pd
        hist_df = pd.DataFrame([{
            "Date": h["completed_at"][:19].replace("T", " "),
            "Lecture": h["lecture_title"],
            "Topic": h["topic"],
            "Score (%)": f"{h['score_pct']:.1f}%",
            "Correct": f"{h['correct_count']}/{h['total_questions']}",
            "Time (s)": round(h["time_spent_seconds"], 1),
        } for h in history[:6]])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No quiz attempts recorded yet.")
