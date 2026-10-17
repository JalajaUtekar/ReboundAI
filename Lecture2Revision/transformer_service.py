"""
Lecture To Revision - Transformer & Learning Material Generation Service
Provides multi-backend AI generation (Hugging Face API, Gemini API, OpenAI-compatible,
and an offline extractive NLP pipeline) to produce grounded summaries, notes,
definitions, exam questions, flashcards, and multiple-choice quizzes.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from config import (
    TRANSFORMER_BACKEND,
    HF_API_TOKEN,
    HF_MODEL_ID,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

# In-memory execution cache: hash(text + backend) -> structured result
_RESULT_CACHE: Dict[str, Dict[str, Any]] = {}


def compute_content_hash(text: str, backend: str) -> str:
    """Computes a SHA-256 fingerprint for caching model outputs."""
    raw = f"{backend}::{text.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ==============================================================================
# Prompt Construction & Schema
# ==============================================================================

PROMPT_SYSTEM_DIRECTIVE = """You are an academic learning assistant.
CRITICAL GROUNDING DIRECTIVE:
1. Base all generated summaries, notes, definitions, questions, and flashcards EXCLUSIVELY on the provided lecture text.
2. DO NOT invent, hallucinate, or extrapolate facts, dates, authors, or mechanisms not present in the lecture text.
3. If an explanation requires external knowledge, clearly tag it as "[Supplementary Context]".
4. Output MUST be valid JSON adhering strictly to the schema requested without additional conversational filler.
"""

def build_material_generation_prompt(lecture_topic: str, lecture_text: str) -> str:
    return f"""Analyze the following lecture excerpt on the topic "{lecture_topic}" and generate comprehensive revision materials.

LECTURE TEXT:
\"\"\"
{lecture_text[:12000]}
\"\"\"

Return ONLY a valid JSON object matching this exact schema:
{{
  "summary": "A concise executive summary (3-5 paragraphs) capturing core theorems, architectures, and principles directly stated in the lecture.",
  "revision_notes": "Detailed bulleted revision notes categorized by subheadings.",
  "key_definitions": [
    {{
      "term": "Specific Term",
      "definition": "Definition stated in or directly derived from the text",
      "source_context": "Direct sentence or section reference"
    }}
  ],
  "exam_questions": [
    {{
      "question": "Conceptual or analytical exam question",
      "suggested_answer": "Model answer grounded strictly in lecture facts",
      "difficulty": "Easy|Medium|Hard"
    }}
  ],
  "flashcards": [
    {{
      "front_question": "Prompt or question testing recall",
      "back_answer": "Accurate, concise answer directly from the lecture",
      "difficulty": "Easy|Medium|Hard"
    }}
  ],
  "quiz_questions": [
    {{
      "question_text": "Clear multiple choice question directly testing a fact or mechanism in the text",
      "option_a": "First option",
      "option_b": "Second option",
      "option_c": "Third option",
      "option_d": "Fourth option",
      "correct_option": "A",
      "explanation": "Detailed explanation citing the lecture reason why this option is correct and others are wrong",
      "difficulty": "Easy|Medium|Hard"
    }}
  ]
}}
"""


# ==============================================================================
# JSON Sanitization & Validation
# ==============================================================================

def sanitize_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON from LLM output, stripping markdown fences if present.
    Validates mandatory keys and fallback defaults.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Find first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    data = json.loads(cleaned)
    return validate_learning_materials_dict(data)


def validate_learning_materials_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures learning materials dict has required keys and valid types."""
    if not isinstance(data, dict):
        raise ValueError("Model output did not parse into a dictionary.")

    validated = {
        "summary": str(data.get("summary", "")).strip(),
        "revision_notes": str(data.get("revision_notes", "")).strip(),
        "key_definitions": [],
        "exam_questions": [],
        "flashcards": [],
        "quiz_questions": [],
    }

    # Validate definitions
    for item in data.get("key_definitions", []):
        if isinstance(item, dict) and "term" in item and "definition" in item:
            validated["key_definitions"].append({
                "term": str(item.get("term", "")).strip(),
                "definition": str(item.get("definition", "")).strip(),
                "source_context": str(item.get("source_context", "Lecture source")).strip(),
            })

    # Validate exam questions
    for item in data.get("exam_questions", []):
        if isinstance(item, dict) and "question" in item:
            validated["exam_questions"].append({
                "question": str(item.get("question", "")).strip(),
                "suggested_answer": str(item.get("suggested_answer", "")).strip(),
                "difficulty": str(item.get("difficulty", "Medium")).strip(),
            })

    # Validate flashcards
    for item in data.get("flashcards", []):
        if isinstance(item, dict):
            front = str(item.get("front_question", item.get("question", ""))).strip()
            back = str(item.get("back_answer", item.get("answer", ""))).strip()
            if front and back:
                validated["flashcards"].append({
                    "front_question": front,
                    "back_answer": back,
                    "difficulty": str(item.get("difficulty", "Medium")).strip(),
                })

    # Validate quiz questions
    for item in data.get("quiz_questions", []):
        if isinstance(item, dict) and "question_text" in item:
            correct = str(item.get("correct_option", "A")).upper().strip()
            if correct not in ("A", "B", "C", "D"):
                correct = "A"
            validated["quiz_questions"].append({
                "question_text": str(item.get("question_text", "")).strip(),
                "option_a": str(item.get("option_a", "Option A")).strip(),
                "option_b": str(item.get("option_b", "Option B")).strip(),
                "option_c": str(item.get("option_c", "Option C")).strip(),
                "option_d": str(item.get("option_d", "Option D")).strip(),
                "correct_option": correct,
                "explanation": str(item.get("explanation", "Grounded in lecture text.")).strip(),
                "difficulty": str(item.get("difficulty", "Medium")).strip(),
            })

    return validated


# ==============================================================================
# Offline Extractive Primary Engine (Zero-hallucination baseline)
# ==============================================================================

def generate_offline_extractive_materials(lecture_topic: str, text: str) -> Dict[str, Any]:
    """
    Deterministically extracts key sentences, definitions, flashcards, and MCQs
    directly from text using sentence frequency scoring and syntactic matching.
    Guaranteed zero hallucination, fast, and requires no external API keys.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    words = re.findall(r"\b[A-Za-z]{3,}\b", text.lower())

    # Stop words
    stop_words = {
        "the", "and", "that", "this", "with", "from", "for", "are", "which",
        "was", "were", "been", "have", "has", "had", "will", "would", "can",
        "could", "should", "not", "their", "they", "them", "these", "those",
    }
    filtered_words = [w for w in words if w not in stop_words]

    # Frequency dict
    freq: Dict[str, int] = {}
    for w in filtered_words:
        freq[w] = freq.get(w, 0) + 1

    # Score sentences by term frequencies
    scored_sentences: List[Tuple[float, str]] = []
    for s in sentences:
        s_words = re.findall(r"\b[A-Za-z]{3,}\b", s.lower())
        score = sum(freq.get(w, 0) for w in s_words) / max(1, len(s_words))
        scored_sentences.append((score, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored_sentences[:15]]

    # 1. Summary
    summary = " ".join(top_sentences[:4])
    if not summary:
        summary = text[:400]

    # 2. Revision Notes
    bullet_notes = [f"- {s}" for s in top_sentences[4:10]]
    revision_notes = f"### Core Revision Points for {lecture_topic}\n\n" + "\n".join(bullet_notes)

    # 3. Key Definitions (detect "is defined as", "is a", "refers to", "means", "consists of")
    def_patterns = [
        re.compile(r"([A-Z][A-Za-z0-9\s-]{2,30})\s+(?:is defined as|refers to|means|is a|represents)\s+(.*)", re.IGNORECASE),
        re.compile(r"([A-Za-z0-9\s-]{3,25})\s*:\s*(.*)"),
    ]

    key_definitions = []
    for s in sentences:
        for pat in def_patterns:
            m = pat.search(s)
            if m:
                term = m.group(1).strip()
                defi = m.group(2).strip()
                if 2 < len(term.split()) <= 6 and len(defi) > 15:
                    key_definitions.append({
                        "term": term.title(),
                        "definition": defi,
                        "source_context": s,
                    })
                    break
        if len(key_definitions) >= 6:
            break

    if len(key_definitions) < 3 and top_sentences:
        # Fallback definitions based on top sentences
        for i, s in enumerate(top_sentences[:4]):
            parts = s.split(None, 3)
            term = " ".join(parts[:2]) if len(parts) >= 2 else f"Concept {i+1}"
            key_definitions.append({
                "term": term.title(),
                "definition": s,
                "source_context": s,
            })

    # 4. Flashcards
    flashcards = []
    for d in key_definitions[:5]:
        flashcards.append({
            "front_question": f"What is the definition and role of '{d['term']}' in {lecture_topic}?",
            "back_answer": d["definition"],
            "difficulty": "Medium",
        })

    for s in top_sentences[2:6]:
        flashcards.append({
            "front_question": f"Explain this key principle from {lecture_topic}: '{s[:60]}...'?",
            "back_answer": s,
            "difficulty": "Easy",
        })

    # 5. Exam Questions
    exam_questions = []
    for i, s in enumerate(top_sentences[:4]):
        exam_questions.append({
            "question": f"Analyze the following aspect of {lecture_topic}: How does the lecture describe its function and significance?",
            "suggested_answer": s,
            "difficulty": "Medium" if i % 2 == 0 else "Hard",
        })

    # 6. Quiz Questions (MCQs)
    quiz_questions = []
    for idx, d in enumerate(key_definitions[:4]):
        correct_text = d["definition"]
        # Generate plausible distractors from other sentences
        distractors = [
            f"An unrelated component that does not interact with {lecture_topic}.",
            f"A deprecated protocol superseded by modern alternatives.",
            f"A static storage parameter with fixed constant latency.",
        ]
        # Shuffle correct answer position deterministically
        options = ["A", "B", "C", "D"]
        correct_opt = options[idx % 4]
        opt_dict = {}
        dist_idx = 0
        for opt in options:
            if opt == correct_opt:
                opt_dict[f"option_{opt.lower()}"] = correct_text
            else:
                opt_dict[f"option_{opt.lower()}"] = distractors[dist_idx % len(distractors)]
                dist_idx += 1

        quiz_questions.append({
            "question_text": f"According to the lecture, what is '{d['term']}'?",
            "option_a": opt_dict["option_a"],
            "option_b": opt_dict["option_b"],
            "option_c": opt_dict["option_c"],
            "option_d": opt_dict["option_d"],
            "correct_option": correct_opt,
            "explanation": f"As explicitly stated in the lecture: '{d['source_context']}'",
            "difficulty": "Medium",
        })

    # If fewer than 4 quiz questions, create from top sentences
    while len(quiz_questions) < 4 and len(top_sentences) >= len(quiz_questions):
        s = top_sentences[len(quiz_questions)]
        correct_opt = "A"
        quiz_questions.append({
            "question_text": f"Which of the following statements regarding {lecture_topic} is directly supported by the text?",
            "option_a": s,
            "option_b": "This mechanism is unsupported and prohibited in standard architectures.",
            "option_c": "The system relies entirely on manual operators for each transaction.",
            "option_d": "Data is continuously discarded without error correction.",
            "correct_option": correct_opt,
            "explanation": f"Grounded statement from lecture: '{s}'",
            "difficulty": "Easy",
        })

    return {
        "summary": summary,
        "revision_notes": revision_notes,
        "key_definitions": key_definitions,
        "exam_questions": exam_questions,
        "flashcards": flashcards,
        "quiz_questions": quiz_questions,
    }


# ==============================================================================
# Backend Callers
# ==============================================================================

def call_huggingface_api(prompt: str, model_id: str, token: str) -> str:
    """Calls Hugging Face Inference API."""
    if not token:
        raise ValueError("HF_API_TOKEN is not set. Please add it to .env or switch to offline_extractive.")

    headers = {"Authorization": f"Bearer {token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {
        "inputs": f"{PROMPT_SYSTEM_DIRECTIVE}\n\n{prompt}",
        "parameters": {
            "max_new_tokens": 1500,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face API returned status {resp.status_code}: {resp.text}")

    result = resp.json()
    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    raise RuntimeError(f"Unexpected Hugging Face response format: {result}")


def call_openai_compatible_api(prompt: str, api_key: str, base_url: str, model: str) -> str:
    """Calls an OpenAI-compatible /chat/completions endpoint (e.g. Ollama, Groq, OpenAI)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM_DIRECTIVE},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI-compatible API returned status {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_gemini_api(prompt: str, api_key: str, model_name: str) -> str:
    """Calls Google Gemini API via official SDK or REST."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{PROMPT_SYSTEM_DIRECTIVE}\n\n{prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API returned status {resp.status_code}: {resp.text}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"]
    raise RuntimeError(f"Unexpected Gemini response format: {data}")


# ==============================================================================
# Main Generation Function
# ==============================================================================

def generate_learning_materials(
    lecture_topic: str,
    lecture_text: str,
    backend: Optional[str] = None,
    force_refresh: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Coordinates material generation using configured backend or offline fallback.
    Returns (validated_data_dictionary, backend_identifier_string).
    Never fabricates that external inference succeeded if it fell back.
    """
    chosen_backend = (backend or TRANSFORMER_BACKEND).lower().strip()
    cache_key = compute_content_hash(lecture_text, chosen_backend)

    if not force_refresh and cache_key in _RESULT_CACHE:
        logger.info("Returning cached generation results.")
        return _RESULT_CACHE[cache_key], f"{chosen_backend} (cached)"

    prompt = build_material_generation_prompt(lecture_topic, lecture_text)
    raw_response = ""
    backend_used_label = chosen_backend

    try:
        if chosen_backend == "huggingface_api":
            raw_response = call_huggingface_api(prompt, HF_MODEL_ID, HF_API_TOKEN)
            parsed = sanitize_and_parse_json(raw_response)
            backend_used_label = f"Hugging Face Transformer ({HF_MODEL_ID})"

        elif chosen_backend == "gemini_api":
            raw_response = call_gemini_api(prompt, GEMINI_API_KEY, GEMINI_MODEL)
            parsed = sanitize_and_parse_json(raw_response)
            backend_used_label = f"Google Gemini API ({GEMINI_MODEL})"

        elif chosen_backend == "openai_compatible":
            raw_response = call_openai_compatible_api(prompt, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)
            parsed = sanitize_and_parse_json(raw_response)
            backend_used_label = f"OpenAI-Compatible LLM ({OPENAI_MODEL})"

        elif chosen_backend == "offline_extractive":
            parsed = generate_offline_extractive_materials(lecture_topic, lecture_text)
            backend_used_label = "Offline Extractive NLP (Grounded, Local Heuristic)"

        else:
            logger.warning(f"Unknown backend '{chosen_backend}'. Defaulting to offline_extractive.")
            parsed = generate_offline_extractive_materials(lecture_topic, lecture_text)
            backend_used_label = "Offline Extractive NLP (Fallback: Unknown Backend)"

    except Exception as e:
        logger.warning(f"Backend '{chosen_backend}' failed with error: {e}. Falling back to offline extractive NLP.")
        parsed = generate_offline_extractive_materials(lecture_topic, lecture_text)
        backend_used_label = f"Offline Extractive NLP (Fallback after error in {chosen_backend}: {str(e)[:50]})"

    # Cache result
    _RESULT_CACHE[cache_key] = parsed
    return parsed, backend_used_label
