"""
Lecture To Revision - Adaptive Revision Plan View
Displays dynamically prioritized revision timetables with transparent explanations,
SM-2 spaced repetition dates, configurable scheduling parameters, and optional LSTM predictions.
"""

import pandas as pd
import streamlit as st
from adaptive_engine import calculate_topic_priority, generate_adaptive_schedule
from database import get_connection
from lstm_model import LSTMStudentPredictor


def render_revision_plan():
    st.title("🗓️ Adaptive Revision Timetable")
    st.caption("Personalized spaced repetition schedule dynamically weighted by your assessment history.")

    # Initialize predictor
    predictor = LSTMStudentPredictor()
    lstm_ready = predictor.is_ready

    # Header info on scheduler mode
    c1, c2 = st.columns([3, 1])
    with c1:
        if lstm_ready:
            st.success("🤖 **Dual Engine Active:** Combining SM-2 Rule-Based Urgency (70%) with LSTM Sequence Forecasts (30%).")
        else:
            st.info("⚙️ **Rule-Based Spaced Repetition Active:** Operating deterministically via SuperMemo SM-2 and empirical error analysis. (LSTM model untrained or unavailable).")
    with c2:
        if st.button("🔄 Recalculate Plan", type="primary", use_container_width=True):
            with st.spinner("Analyzing learning logs..."):
                generate_adaptive_schedule("demo_student", lstm_predictor=predictor if lstm_ready else None)
            st.rerun()

    # Query current schedules from SQLite
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, l.title as lecture_title
        FROM revision_schedules s
        LEFT JOIN lectures l ON s.lecture_id = l.id
        WHERE s.student_id = 'demo_student'
        ORDER BY s.priority_score DESC
    """)
    schedules = [dict(r) for r in cursor.fetchall()]
    conn.close()

    if not schedules:
        st.info("No active revision schedules found. Generate demo data or take a quiz to initialize your timetable.")
        return

    st.subheader("🎯 Prioritized Revision Queue")
    st.caption("Topics are ranked from highest to lowest urgency. High priority items should be revised immediately.")

    for i, s in enumerate(schedules, 1):
        score = s["priority_score"]
        # Determine priority badge color
        if score >= 65:
            badge_color = "red"
            priority_label = "HIGH PRIORITY"
        elif score >= 35:
            badge_color = "orange"
            priority_label = "MEDIUM PRIORITY"
        else:
            badge_color = "green"
            priority_label = "LOW PRIORITY (MASTERY)"

        with st.container(border=True):
            col_head, col_score = st.columns([4, 1])
            with col_head:
                st.markdown(f"### {i}. {s['topic']}")
                st.write(f"**Associated Lecture:** {s.get('lecture_title', 'General')} | **Scheduled Review Date:** `{s['scheduled_date']}`")
            with col_score:
                st.markdown(f":{badge_color}[**{priority_label}**]")
                st.metric("Urgency Score", f"{score:.1f} / 100")

            # Explanation section
            st.markdown(f"**Why this is prioritized:** {s['priority_reason']}")

            # Details row
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.caption(f"SM-2 Interval: **{s.get('sm2_interval_days', 1.0):.1f} days**")
            with d2:
                st.caption(f"SM-2 Ease Factor: **{s.get('sm2_ease_factor', 2.5):.2f}**")
            with d3:
                pred = s.get("lstm_predicted_score")
                st.caption(f"LSTM Forecast: **{f'{pred:.1f}%' if pred is not None else 'N/A (Untrained)'}**")
            with d4:
                status_txt = s.get("status", "pending").capitalize()
                st.caption(f"Status: **{status_txt}**")

    st.markdown("---")

    # Interactive What-If Simulation Box
    st.subheader("🔬 Interactive Demonstration: Performance vs Revision Priority")
    st.markdown(
        "Notice how poor quiz performance automatically elevates topic urgency, while high scores "
        "and consistent recall lengthen the revision interval:"
    )

    with st.expander("Explore Urgency Formula & Simulation", expanded=False):
        c_sim1, c_sim2 = st.columns(2)
        with c_sim1:
            sim_acc = st.slider("Simulated Recent Quiz Accuracy (%)", 0, 100, 40)
            sim_mistakes = st.slider("Simulated Incorrect Answers", 0, 10, 3)
        with c_sim2:
            sim_overdue = st.slider("Simulated Days Overdue", 0, 14, 2)
            sim_cards = st.slider("Simulated Flashcards Needing Practice", 0, 10, 2)

        sim_score, sim_reason = calculate_topic_priority(
            recent_accuracy=float(sim_acc),
            recent_incorrect_count=sim_mistakes,
            days_since_last_review=sim_overdue + 1,
            days_overdue=float(sim_overdue),
            cards_needing_practice=sim_cards,
        )

        st.metric("Calculated Urgency Score", f"{sim_score:.1f} / 100")
        st.info(f"**Generated Rationale:** {sim_reason}")
