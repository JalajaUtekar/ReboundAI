"""
Lecture To Revision - Main Application Entrypoint
A Streamlit-based personalized learning platform that converts lecture materials
into grounded revision kits and dynamically adapts revision schedules using
SuperMemo SM-2 and LSTM sequential forecasting.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRANSFORMER_BACKEND, SDG_TITLE
from database import init_db
from lstm_model import LSTMStudentPredictor, TENSORFLOW_AVAILABLE

# Import page renderers
from pages.dashboard import render_dashboard
from pages.upload_lecture import render_upload_lecture
from pages.revision_notes import render_revision_notes
from pages.quiz_practice import render_quiz_practice
from pages.flashcards import render_flashcards
from pages.my_progress import render_my_progress
from pages.revision_plan import render_revision_plan
from pages.model_evaluation import render_model_evaluation


# ==============================================================================
# Page Configuration & Modern Theme Styling
# ==============================================================================

st.set_page_config(
    page_title="Lecture To Revision",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal modern edtech stylesheet
st.markdown("""
<style>
    /* Clean layout adjustments */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    /* Restrained accent cards */
    div[data-testid="stMetric"] {
        background-color: rgba(240, 244, 248, 0.5);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    /* Sidebar badge styling */
    .sdg-badge {
        background: #0ea5e9;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Database tables if not already present
init_db()


# ==============================================================================
# Sidebar Navigation & System Diagnostics
# ==============================================================================

with st.sidebar:
    st.markdown("## 🎓 **Lecture To Revision**")
    st.caption("Convert Lectures Into Adaptive Revision")

    st.markdown("---")
    st.markdown("**Student Profile:** `Alex Taylor (Demo)`")

    # Main Page Navigation
    PAGES = {
        "📊 Dashboard": render_dashboard,
        "📤 Upload Lecture": render_upload_lecture,
        "📖 Revision Notes": render_revision_notes,
        "🎯 Quiz Practice": render_quiz_practice,
        "🗂️ Flashcards": render_flashcards,
        "📈 My Progress": render_my_progress,
        "🗓️ Adaptive Revision Plan": render_revision_plan,
        "🔬 Model Evaluation": render_model_evaluation,
    }

    selected_page_name = st.radio(
        "Navigation",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")

    # AI Engine Diagnostics in Sidebar
    st.markdown("#### ⚙️ AI Engine Status")
    st.caption(f"**Transformer:** `{TRANSFORMER_BACKEND}`")

    predictor = LSTMStudentPredictor()
    if not TENSORFLOW_AVAILABLE:
        st.caption("**LSTM Predictor:** `TensorFlow Not Installed (Rule-Based Active)`")
    elif predictor.is_ready:
        st.caption("**LSTM Predictor:** `Trained & Active (Keras)`")
    else:
        st.caption("**LSTM Predictor:** `Untrained (Rule-Based SM-2 Active)`")

    st.markdown("---")

    # Quick Demo Data Seeder
    if st.button("⚡ Reset / Seed Demo Data", use_container_width=True):
        import subprocess
        with st.spinner("Seeding Computer Networks lecture & demo logs..."):
            res = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_demo_data.py")], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Sample data loaded!")
                st.rerun()
            else:
                st.error(f"Error: {res.stderr}")

    st.markdown("---")
    st.markdown("<div class='sdg-badge'>UN SDG 4: Quality Education</div>", unsafe_allow_html=True)
    st.caption("Promoting inclusive, lifelong, personalized learning.")


# ==============================================================================
# Route Execution
# ==============================================================================

page_function = PAGES[selected_page_name]
page_function()
