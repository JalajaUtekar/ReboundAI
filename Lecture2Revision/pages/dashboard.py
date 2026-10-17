"""
Lecture To Revision - Dashboard View
Presents authentic student metrics, weak topics, upcoming revision sessions,
and learning progression charts. Features realistic onboarding empty state.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from database import get_student_dashboard_stats, get_topic_performance
from config import SDG_TITLE, SDG_STATEMENT


def render_dashboard():
    st.title("📚 Student Learning Dashboard")
    st.caption("AI-Powered Adaptive Revision Platform • Centered on UN SDG 4")

    stats = get_student_dashboard_stats("demo_student")

    # Empty State Onboarding if no lectures or quizzes
    if not stats["has_data"]:
        st.info("👋 **Welcome to Lecture To Revision!** No lecture materials or quiz records have been processed yet.")
        col_onboard1, col_onboard2 = st.columns([2, 1])
        with col_onboard1:
            st.markdown("""
            ### How to get started:
            1. **Upload a Lecture:** Head over to **Upload Lecture** in the sidebar to upload a PDF slide deck or lecture notes.
            2. **Review Generated Materials:** Study auto-generated summaries, key definitions, and exam questions.
            3. **Practice with Quizzes & Flashcards:** Test your recall. Every response feeds the **SM-2 Spaced Repetition** and **LSTM Adaptive Engine**.
            4. **Follow Your Adaptive Plan:** The engine prioritizes your revision intervals based on your actual performance.
            """)
        with col_onboard2:
            st.markdown("#### Quick Demonstration")
            st.write("Want to explore the platform with a pre-configured lecture on **Computer Networks** and synthetic quiz history?")
            if st.button("🚀 Load Sample Lecture & Demo Data", type="primary", use_container_width=True):
                with st.spinner("Seeding sample lecture and demo trajectory..."):
                    import subprocess
                    from config import PROJECT_ROOT
                    import sys
                    res = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_demo_data.py")], capture_output=True, text=True)
                    if res.returncode == 0:
                        st.success("Sample data loaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to seed demo data: {res.stderr}")
        return

    # Top Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Lectures Processed", stats["lectures_count"])
    with m2:
        st.metric("Quizzes Completed", stats["total_quizzes"])
    with m3:
        st.metric("Overall Average Score", f"{stats['avg_score']}%")
    with m4:
        st.metric("Topics Needing Revision", len(stats["weak_topics"]), delta=f"-{len(stats['weak_topics'])}" if stats["weak_topics"] else "0", delta_color="inverse")

    st.markdown("---")

    # Main Grid: Weak Topics & Upcoming Schedule
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("⚠️ Priority Topics Needing Revision")
        if stats["weak_topics"]:
            for wt in stats["weak_topics"]:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{wt['topic']}**")
                        st.caption(f"Based on {wt['attempts']} quiz attempt(s)")
                    with c2:
                        st.error(f"Avg: {wt['avg_score']:.1f}%")
        else:
            st.success("🎉 Excellent! All tested topics have an average score of 70% or higher.")

        # Performance trend chart if attempts exist
        perf_data = get_topic_performance("demo_student")
        if perf_data:
            st.subheader("📈 Chronological Quiz Score Trajectory")
            import pandas as pd
            df = pd.DataFrame(perf_data)
            df["attempt_num"] = range(1, len(df) + 1)
            fig = px.line(
                df,
                x="attempt_num",
                y="score_pct",
                color="topic",
                markers=True,
                labels={"attempt_num": "Attempt #", "score_pct": "Score (%)", "topic": "Topic"},
                title="Performance Across Chronological Attempts",
            )
            fig.update_layout(yaxis_range=[0, 105], margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📅 Upcoming Revision Sessions")
        if stats["upcoming_sessions"]:
            for s in stats["upcoming_sessions"]:
                with st.container(border=True):
                    st.markdown(f"**{s['topic']}**")
                    st.caption(f"Scheduled: `{s['scheduled_date']}` • Priority Score: `{s['priority_score']:.1f}/100`")
                    st.caption(f"Reason: {s['priority_reason']}")
        else:
            st.info("No upcoming scheduled sessions. Complete a quiz to initialize your adaptive timetable.")

        st.subheader("🕒 Recent Learning Activity")
        if stats["recent_activity"]:
            for a in stats["recent_activity"]:
                ts = a["timestamp"][:16].replace("T", " ")
                st.markdown(f"• **Quiz:** {a['title']} — `{a['score']:.1f}%` ({ts} UTC)")
        else:
            st.caption("No recent activity recorded.")

    # SDG 4 Impact Section
    st.markdown("---")
    with st.expander(f"🌐 {SDG_TITLE} — Alignment Statement"):
        st.markdown(SDG_STATEMENT)
        st.caption("Ethical AI Notice: Adaptive schedules and predictions are decision aids derived directly from logged student interactions and are not claims of scientifically proven cognitive mastery.")
