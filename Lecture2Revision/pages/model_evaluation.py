"""
Lecture To Revision - Model Evaluation View
Presents authentic, verifiable evaluation metrics for:
1. TensorFlow/Keras LSTM student performance predictor (MAE, RMSE, Baseline Comparison).
2. Transformer-generated learning materials grounding audit.
3. Adaptive revision impact analytics (Score deltas and completion rate).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from evaluation import (
    audit_material_grounding,
    get_adaptive_progress_metrics,
    get_lstm_evaluation_summary,
)
from database import get_all_lectures, get_learning_materials_by_lecture
from lstm_model import LSTMStudentPredictor, TENSORFLOW_AVAILABLE
from config import DATA_DIR, DB_PATH


def render_model_evaluation():
    st.title("🔬 AI Pipeline Evaluation & Diagnostics")
    st.caption("Empirical verification of the Transformer learning materials generator and LSTM sequence model.")

    tab_lstm, tab_trans, tab_adaptive = st.tabs([
        "🧠 LSTM Sequential Performance",
        "📑 Transformer Grounding & Quality",
        "📈 Adaptive Revision Impact",
    ])

    # =========================================================================
    # TAB 1: LSTM Sequence Model Evaluation
    # =========================================================================
    with tab_lstm:
        st.subheader("LSTM Student Performance Prediction Evaluation")
        st.markdown(
            "The LSTM sequentially models student trajectories across past quizzes "
            "to forecast their next assessment score. Evaluation is performed chronologically "
            "with zero temporal leakage and compared against a Last-Observed Quiz Score baseline."
        )

        lstm_summary = get_lstm_evaluation_summary()

        col_train_info, col_train_btn = st.columns([3, 1])
        with col_train_info:
            if not TENSORFLOW_AVAILABLE:
                st.warning("⚠️ **TensorFlow Not Installed:** The LSTM pipeline requires TensorFlow/Keras. The app runs in rule-based mode.")
            elif lstm_summary["is_trained"]:
                st.success("✅ **Trained Artifacts Loaded:** Validated model weights and scaler active.")
            else:
                st.info("ℹ️ **Untrained:** Model has not been trained on sequential records yet.")

        with col_train_btn:
            if TENSORFLOW_AVAILABLE:
                if st.button("🚀 Train / Retrain LSTM", type="primary", use_container_width=True):
                    with st.spinner("Executing leakage-safe LSTM training pipeline..."):
                        predictor = LSTMStudentPredictor()
                        # Load records
                        synthetic_path = DATA_DIR / "synthetic_lstm_dataset.json"
                        if synthetic_path.exists():
                            with open(synthetic_path, "r", encoding="utf-8") as f:
                                records = json.load(f)
                            res = predictor.train_and_evaluate(records, test_ratio=0.25, epochs=25)
                            if res.get("success"):
                                st.success("LSTM Model successfully trained and evaluated!")
                                st.rerun()
                            else:
                                st.error(f"Training failed: {res.get('error')}")
                        else:
                            st.error("Synthetic demo dataset not found. Please click 'Load Sample Data' on Dashboard first.")

        if lstm_summary["is_trained"] and lstm_summary["metrics"]:
            m = lstm_summary["metrics"]
            st.markdown("#### Actual Quantitative Metrics")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("LSTM Test MAE", f"{m['lstm_mae']:.2f}%", delta=f"{m['mae_improvement_pct']:+.1f}% vs baseline", delta_color="inverse")
            with c2:
                st.metric("LSTM Test RMSE", f"{m['lstm_rmse']:.2f}%")
            with c3:
                st.metric("Baseline MAE", f"{m['baseline_mae']:.2f}%")
            with c4:
                st.metric("Test Sequences", f"{m['test_samples']} / {m['total_samples']}")

            # Comparison Bar Chart
            comp_df = pd.DataFrame([
                {"Model": "LSTM Neural Network", "Metric": "MAE (%)", "Value": m["lstm_mae"]},
                {"Model": "Last-Score Baseline", "Metric": "MAE (%)", "Value": m["baseline_mae"]},
                {"Model": "LSTM Neural Network", "Metric": "RMSE (%)", "Value": m["lstm_rmse"]},
                {"Model": "Last-Score Baseline", "Metric": "RMSE (%)", "Value": m["baseline_rmse"]},
            ])
            fig = px.bar(
                comp_df,
                x="Metric",
                y="Value",
                color="Model",
                barmode="group",
                title="LSTM Model vs Simple Moving Baseline Error Comparison",
                text_auto=".2f",
            )
            fig.update_layout(height=350, yaxis_title="Error Percentage (Lower is Better)")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed Pipeline Metadata & Limitations"):
                st.json(m)
                st.markdown("""
                **Methodological Integrity Note:**
                - *Data Partitioning:* Split is strictly chronological; future quiz sequences never contaminate training windows.
                - *Feature Scaling:* `StandardScaler` was fit exclusively on training sequences.
                - *Dataset Disclosure:* Evaluated on labeled synthetic student trajectories (`synthetic_lstm_dataset.json`). Real-world deployment requires longitudinal institutional student data.
                """)
        else:
            st.markdown(f"> *{lstm_summary['status_message']}*")

    # =========================================================================
    # TAB 2: Transformer Grounding & Quality Checklist
    # =========================================================================
    with tab_trans:
        st.subheader("Transformer Content Grounding Audit")
        st.markdown(
            "Verifies that all AI-generated revision notes, key concepts, definitions, and quizzes "
            "are empirically grounded in the uploaded lecture source without hallucinatory drift."
        )

        lectures = get_all_lectures()
        if not lectures:
            st.info("No lectures processed yet.")
        else:
            lec_options = {f"{l['title']} ({l['topic']})": l for l in lectures}
            sel_l_name = st.selectbox("Select Lecture to Audit", options=list(lec_options.keys()), key="audit_lec_select")
            aud_lec = lec_options[sel_l_name]

            materials = get_learning_materials_by_lecture(aud_lec["id"])
            if materials:
                audit_data = audit_material_grounding(aud_lec.get("cleaned_text", ""), materials)

                q1, q2, q3 = st.columns(3)
                with q1:
                    st.metric("Source Grounding Score", f"{audit_data['grounding_score_pct']}%")
                with q2:
                    st.metric("Verified Definitions", f"{audit_data['verified_definitions']} / {audit_data['total_definitions']}")
                with q3:
                    st.metric("Verified Quiz Questions", f"{audit_data['verified_quiz_questions']} / {audit_data['total_quiz_questions']}")

                st.markdown("#### Factual Review Checklist")
                for c in audit_data["checklist"]:
                    icon = "✅" if c["passed"] else "⚠️"
                    st.markdown(f"**{icon} {c['criterion']}**: {c['details']}")

                st.caption(f"**Inference Backend Used:** `{materials.get('backend_used', 'N/A')}`")

    # =========================================================================
    # TAB 3: Adaptive Revision Impact
    # =========================================================================
    with tab_adaptive:
        st.subheader("Adaptive Revision Impact & Learning Gains")
        prog_data = get_adaptive_progress_metrics("demo_student")

        if not prog_data["has_progress_data"]:
            st.info("No assessment progression data available yet.")
        else:
            a1, a2 = st.columns(2)
            with a1:
                st.metric("Revision Schedule Completion Rate", f"{prog_data['completion_rate_pct']}%")
            with a2:
                st.metric("Total Scheduled Sessions", prog_data["total_scheduled_sessions"])

            st.markdown("#### Score Deltas Across Repeated Topic Attempts")
            prog_df = pd.DataFrame(prog_data["topic_progression"])
            if not prog_df.empty:
                st.dataframe(prog_df, use_container_width=True, hide_index=True)

                fig_delta = px.bar(
                    prog_df,
                    x="topic",
                    y="delta",
                    color="delta",
                    color_continuous_scale=["#ef4444", "#22c55e"],
                    title="Score Improvement (Δ = Latest Score - Initial Score)",
                    labels={"delta": "Score Change (% points)", "topic": "Topic"},
                    text_auto=".1f",
                )
                fig_delta.update_layout(height=350)
                st.plotly_chart(fig_delta, use_container_width=True)
            else:
                st.caption("Multiple attempts per topic are required to compute longitudinal deltas.")
