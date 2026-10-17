"""
Lecture To Revision - My Progress View
Visualizes student mastery across topics, quiz score trajectories,
accuracy distributions, and detailed attempt histories using Plotly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import get_connection, get_quiz_history, get_topic_performance


def render_my_progress():
    st.title("📊 My Learning Progress")
    st.caption("Comprehensive analytics tracking your subject mastery and revision efficiency.")

    history = get_quiz_history("demo_student")
    if not history:
        st.info("No learning records found yet. Complete a quiz to view your progress analytics.")
        return

    df = pd.DataFrame(history)

    # Summary Metrics
    total_attempts = len(df)
    avg_score = round(df["score_pct"].mean(), 1)
    total_time_min = round(df["time_spent_seconds"].sum() / 60.0, 1)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Quizzes Attempted", total_attempts)
    with m2:
        st.metric("Mean Assessment Score", f"{avg_score}%")
    with m3:
        st.metric("Total Practice Time", f"{total_time_min} mins")
    with m4:
        # Highest scoring topic
        topic_grp = df.groupby("topic")["score_pct"].mean()
        best_topic = topic_grp.idxmax() if not topic_grp.empty else "N/A"
        st.metric("Top Performing Topic", best_topic[:20] + "..." if len(best_topic) > 20 else best_topic)

    st.markdown("---")

    # Visualizations
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Topic-Wise Mean Mastery")
        topic_summary = df.groupby("topic").agg(
            avg_score=("score_pct", "mean"),
            attempts=("id", "count"),
        ).reset_index()

        fig_bar = px.bar(
            topic_summary,
            x="topic",
            y="avg_score",
            color="avg_score",
            color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
            labels={"avg_score": "Average Score (%)", "topic": "Topic"},
            text_auto=".1f",
            range_y=[0, 105],
        )
        fig_bar.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.subheader("Assessment Score Progression")
        df_sorted = df.sort_values("completed_at").reset_index(drop=True)
        df_sorted["Attempt #"] = range(1, len(df_sorted) + 1)

        fig_line = px.line(
            df_sorted,
            x="Attempt #",
            y="score_pct",
            color="topic",
            markers=True,
            labels={"score_pct": "Score (%)"},
        )
        fig_line.add_hline(y=70, line_dash="dash", line_color="gray", annotation_text="Target (70%)")
        fig_line.update_layout(yaxis_range=[0, 105], height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

    # Detailed Attempt History Table
    st.subheader("📋 Comprehensive Assessment History")
    display_df = df[[
        "completed_at", "topic", "lecture_title", "score_pct", "correct_count", "total_questions", "time_spent_seconds"
    ]].copy()
    display_df["completed_at"] = display_df["completed_at"].str.slice(0, 19).str.replace("T", " ")
    display_df.rename(columns={
        "completed_at": "Timestamp (UTC)",
        "topic": "Topic",
        "lecture_title": "Lecture Title",
        "score_pct": "Score (%)",
        "correct_count": "Correct",
        "total_questions": "Total",
        "time_spent_seconds": "Duration (s)",
    }, inplace=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
