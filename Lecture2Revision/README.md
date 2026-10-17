# 🎓 Lecture To Revision — Convert Lectures Into Adaptive Revision

> **An AI-powered personalized revision platform that transforms lecture materials into grounded revision kits, flashcards, and quizzes, dynamically adapting revision timetables using SuperMemo SM-2 and LSTM sequential forecasting.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/TensorFlow%2FKeras-2.15+-orange.svg)](https://tensorflow.org/)
[![PDF Engine](https://img.shields.io/badge/PyMuPDF-fitz-green.svg)](https://pymupdf.readthedocs.io/)
[![Database](https://img.shields.io/badge/Database-SQLite3-blue.svg)](https://sqlite.org/)
[![UN SDG 4](https://img.shields.io/badge/UN%20SDG%204-Quality%20Education-0ea5e9.svg)](https://sdgs.un.org/goals/goal4)

---

## 1. Project Overview

Students often struggle to retain critical technical concepts from lengthy lecture slide decks and textbooks. **Lecture To Revision** bridges the gap between passive lecture attendance and active mastery by:

1. **Ingesting digital lecture PDFs** using PyMuPDF (`fitz`), validating extractable text, sanitizing content, and chunking long documents.
2. **Synthesizing grounded revision kits** (concise executive summaries, categorized revision notes, key definitions, exam-oriented questions, flashcards, and multiple-choice quizzes) using a configurable Transformer backend.
3. **Tracking student assessment trajectories** in a relational SQLite database with idempotency protection against duplicate saves.
4. **Calculating personalized revision intervals** via a dual-engine architecture:
   - **Deterministic SuperMemo SM-2 Spaced Repetition**: Adjusts ease factors, repetitions, and intervals based on quiz accuracy, consecutive mistakes, overdue time, and flashcard recall ratings.
   - **TensorFlow/Keras LSTM Sequential Forecasting**: Predicts the student's next quiz score from their chronological trajectory, highlighting vulnerable topics before exams.

---

## 2. Architecture & Core AI Models

```
                                  ┌─────────────────────────────┐
                                  │      Lecture PDF Upload     │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │  PyMuPDF Text Extraction    │
                                  │   & Sliding-Window Chunks   │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
        ┌────────────────────────────────────────────────────────────────────────────────┐
        │                        TRANSFORMER REVISION GENERATOR                          │
        │  • Hugging Face API / Gemini API / OpenAI-Compatible / Offline Extractive NLP  │
        │  • Strict Grounding Directives (Zero Hallucination / Source Citation)          │
        │  • Schema-Validated JSON Output (Summaries, Notes, Defs, Flashcards, MCQs)     │
        └────────────────────────────────────────┬───────────────────────────────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │    SQLite Persistence &     │
                                  │    Interactive Practice     │
                                  │  (Quizzes, Flashcards SM-2) │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
        ┌────────────────────────────────────────────────────────────────────────────────┐
        │                          ADAPTIVE REVISION ENGINE                              │
        │  1. Rule-Based Urgency: Error Counts, Accuracy, Overdue Days, Flashcards       │
        │  2. SuperMemo SM-2 Algorithm: Interval & Ease Factor Dynamics                  │
        │  3. TensorFlow/Keras LSTM Predictor: Temporal Score Forecasting                │
        └────────────────────────────────────────┬───────────────────────────────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │  Ranked Revision Timetable  │
                                  │   & Empirical Evaluation    │
                                  └─────────────────────────────┘
```

### A. Transformer Revision Kit Generator
- **Multi-Backend Architecture:**
  - `offline_extractive` *(Default)*: Deterministic, local frequency-based sentence extraction and cloze-distractor generation. 100% grounded in source text, zero network latency, zero hallucination.
  - `huggingface_api`: Pretrained Hugging Face instruction-tuned models (e.g., `mistralai/Mistral-7B-Instruct-v0.3` or `meta-llama/Llama-3-8B-Instruct`).
  - `gemini_api`: Google Gemini models (e.g., `gemini-1.5-flash`).
  - `openai_compatible`: Standard OpenAI-format completions (compatible with Ollama, Groq, Together, vLLM).
- **Strict Grounding:** Models are constrained via prompt engineering and verified through an empirical lexical audit comparing extracted terms directly against source text tokens.
- **Robust JSON Validation:** Code fences are stripped, mandatory keys are validated, and structural defaults are enforced.

### B. LSTM Sequential Performance Predictor
- **Architecture:** Keras Sequential model: `Input(sequence_length=5, features=5)` $\rightarrow$ `LSTM(32)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(16)` $\rightarrow$ `Dense(1, linear)`.
- **Temporal Features:** `[score_pct, accuracy_ratio, incorrect_count, time_spent_seconds, interval_days]`.
- **Leakage-Safe Preprocessing:** Split chronologically; `StandardScaler` is fit **strictly** on training sequences.
- **Empirical Baseline Comparison:** Evaluated against a Persistence (Last-Observed Score) baseline using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Graceful Fallback:** If TensorFlow is uninstalled or weights are untrained, the app seamlessly runs in deterministic rule-based mode without crashing.

---

## 3. UN SDG 4: Quality Education Alignment

**UN Sustainable Development Goal 4** aims to *"Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all."*

Lecture To Revision aligns with Target 4.4 and Target 4.5 by:
- **Democratizing Personalized Tutoring:** Traditional private tutoring is inaccessible to many students. Dynamic spaced repetition provides individualized reinforcement at zero marginal cost.
- **Grounded, Non-Hallucinatory Feedback:** Hallucinatory AI outputs can mislead novice learners. By strictly constraining questions and summaries to the instructor's primary lecture text, academic rigor is preserved.
- **Ethical AI Disclosure:** Synthetic demo records are clearly labeled. Predictive intervals are presented as transparent study recommendations rather than cognitive certainty.

---

## 4. Project Structure

```
Lecture2Revision/
├── app.py                      # Streamlit application entrypoint & sidebar router
├── config.py                   # Central paths, environment variables, SM-2/LSTM defaults
├── database.py                 # SQLite schema, CRUD operations, idempotent attempts
├── lecture_processor.py        # PyMuPDF text extraction, cleaning, sliding-window chunking
├── transformer_service.py      # Multi-backend revision generator & JSON parser
├── quiz_generator.py           # Quiz presentation, evaluation, and attempt tokens
├── flashcard_service.py        # SM-2 algorithm calculations, card filtering, Markdown export
├── adaptive_engine.py          # Dynamic priority scoring, overdue analysis, plan generation
├── lstm_model.py               # TensorFlow/Keras LSTM architecture, scaling, evaluation
├── evaluation.py               # Factual grounding audits, LSTM MAE/RMSE, score deltas
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── pages/                      # Application navigation modules
│   ├── dashboard.py            # Overview metrics, weak topics, Plotly trajectories
│   ├── upload_lecture.py       # PDF uploader, text extraction, backend switcher
│   ├── revision_notes.py       # Grounded summaries, definitions, exam Qs, export
│   ├── quiz_practice.py        # Interactive 4-choice quizzes with immediate review
│   ├── flashcards.py           # Active recall flashcards with SM-2 quality logging
│   ├── my_progress.py          # Topic mastery bar charts and attempt history
│   ├── revision_plan.py        # Prioritized revision queue & interactive simulator
│   └── model_evaluation.py     # LSTM metrics (MAE/RMSE), grounding audits, deltas
├── scripts/
│   ├── generate_demo_data.py   # Creates sample PDF, seeds DB, exports synthetic LSTM dataset
│   └── train_lstm.py           # CLI script to train and evaluate LSTM predictor
├── tests/                      # Pytest automated test suite (25 test cases)
│   ├── test_lecture_processor.py
│   ├── test_transformer_service.py
│   ├── test_quiz_generator.py
│   ├── test_flashcard_service.py
│   ├── test_database.py
│   ├── test_adaptive_engine.py
│   ├── test_lstm_model.py
│   └── test_evaluation.py
├── data/                       # Local SQLite database & synthetic dataset
├── uploads/                    # Stored lecture PDF files
└── models/                     # Saved LSTM model weights & feature scalers
```

---

## 5. Windows Setup & Installation

### Step 1: Open Terminal in VS Code
Open PowerShell or Command Prompt in the project folder:
```powershell
cd C:\Users\SAMRUDDHI\.gemini\antigravity\scratch\Lecture2Revision
```

### Step 2: (Optional) Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install -r requirements.txt
```

> **Note on TensorFlow:** If your network connection is restricted, the entire application, PyMuPDF extractor, Transformer generator, and Adaptive SM-2 Scheduler run out of the box with pure Python/NumPy/Scikit-Learn.

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Default configuration uses `TRANSFORMER_BACKEND=offline_extractive` (no API key required). To use external models, add your keys:
- Hugging Face: Set `HF_API_TOKEN=your_token` and `TRANSFORMER_BACKEND=huggingface_api`
- Google Gemini: Set `GEMINI_API_KEY=your_key` and `TRANSFORMER_BACKEND=gemini_api`
- OpenAI / Local Ollama: Set `OPENAI_API_KEY=your_key` and `TRANSFORMER_BACKEND=openai_compatible`

---

## 6. How to Run the Application

Launch the Streamlit dashboard:
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 7. Demo Mode & Walkthrough Instructions

### A. Load Pre-Configured Sample Data
1. Launch the app and visit the **Dashboard**.
2. Click **"🚀 Load Sample Lecture & Demo Data"** (or click **"⚡ Reset / Seed Demo Data"** in the sidebar).
   *Alternative CLI command:*
   ```powershell
   python scripts/generate_demo_data.py
   ```
3. This creates a genuine PDF lecture on **Computer Networks: Transport Layer Protocols (TCP vs UDP)**, ingests it into SQLite, generates revision notes, flashcards, and quizzes, and seeds realistic chronological student attempts.

### B. Demonstrating Adaptive Schedule Reprioritization
1. Navigate to **Adaptive Revision Plan** in the sidebar.
2. Notice that **"TCP Congestion Control & Reliability"** is ranked **HIGH PRIORITY** (Urgency: ~63/100) with a scheduled date of tomorrow because the student logged scores of 40% and 50% with consecutive mistakes.
3. In contrast, **"UDP Datagrams & Performance"** is ranked **LOW PRIORITY (MASTERY)** because the student achieved 100% accuracy.
4. Take a quiz under **Quiz Practice**: Score 100% on TCP, return to **Adaptive Revision Plan**, and observe the SM-2 interval extend automatically!

---

## 8. Training & Evaluating the LSTM Model

Train the sequential predictor using the synthetic demonstration dataset:
```powershell
python scripts/train_lstm.py --synthetic --epochs 25 --batch-size 16
```

### Expected Output:
```
================================================================
 LECTURE TO REVISION - LSTM PERFORMANCE PREDICTION PIPELINE
================================================================
[*] Loaded 60 records from labeled synthetic dataset: synthetic_lstm_dataset.json
[*] Initializing model with sequence length L = 5
[*] Training for 25 epochs with batch size 16...

------------------- EVALUATION RESULTS -------------------
Total Sequences Generated : 35
Training Partition        : 26 samples
Testing Partition         : 9 samples
LSTM Model MAE            : 4.82%
LSTM Model RMSE           : 6.15%
Baseline MAE (Persistence): 8.41%
Baseline RMSE             : 10.22%
MAE Delta vs Baseline     : +42.7%
Artifacts Saved To        : models
================================================================
```
You can also trigger training directly from the UI under the **Model Evaluation** page.

---

## 9. Running the Automated Test Suite

Execute all 25 unit tests via pytest:
```powershell
python -m pytest tests/ -v
```

All 25 tests verify:
- PyMuPDF text extraction, cleaning, and scanned PDF error detection
- Document chunking with semantic overlap
- Transformer JSON schema sanitization and fallback parsing
- Quiz scoring and idempotency (prevention of duplicate scores on page refresh)
- SQLite database persistence and foreign key constraints
- SuperMemo SM-2 interval calculations and recall resets
- Adaptive urgency formula calibration
- LSTM chronological sequence tensor shapes `(N, L, F)`
- Empirical lexical grounding audits and longitudinal score deltas
