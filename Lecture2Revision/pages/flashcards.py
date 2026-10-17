"""
Lecture To Revision - Flashcards View
Interactive flashcard study interface with flip/reveal mechanism, search/filter controls,
and live SM-2 spaced repetition logging.
"""

from datetime import datetime, timezone
import streamlit as st
from database import get_all_lectures, get_flashcards, update_flashcard_review
from flashcard_service import calculate_sm2_update, filter_flashcards
from adaptive_engine import generate_adaptive_schedule


def render_flashcards():
    st.title("🗂️ Interactive Revision Flashcards")
    st.caption("Active recall powered by the SuperMemo SM-2 spaced repetition algorithm.")

    all_cards = get_flashcards()
    if not all_cards:
        st.info("No flashcards found. Please upload a lecture PDF or generate sample data.")
        return

    # Extract distinct topics
    distinct_topics = ["All"] + sorted(list({c.get("topic", "General") for c in all_cards if c.get("topic")}))

    # Controls row
    col_search, col_topic, col_status = st.columns([2, 1, 1])
    with col_search:
        search_query = st.text_input("🔍 Search Flashcards", placeholder="Filter by keyword...")
    with col_topic:
        topic_filter = st.selectbox("Topic Filter", options=distinct_topics)
    with col_status:
        status_filter = st.selectbox("Status", options=["All", "Needs Practice", "Due Today"])

    filtered = filter_flashcards(all_cards, search_query=search_query, topic_filter=topic_filter, status_filter=status_filter)

    st.write(f"Showing **{len(filtered)}** flashcard(s)")

    if not filtered:
        st.caption("No flashcards match the selected filters.")
        return

    # Carousel-style card navigation or card grid
    card_index = st.session_state.get("flashcard_index", 0)
    card_index = min(max(0, card_index), len(filtered) - 1)
    card = filtered[card_index]

    reveal_key = f"reveal_fc_{card['id']}"
    if reveal_key not in st.session_state:
        st.session_state[reveal_key] = False

    # Card Navigation Bar
    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("⬅️ Previous Card", disabled=(card_index == 0), use_container_width=True):
            st.session_state["flashcard_index"] = card_index - 1
            st.rerun()
    with nav2:
        st.markdown(f"<div style='text-align: center; font-weight: 500;'>Card {card_index + 1} of {len(filtered)} • Topic: {card['topic']}</div>", unsafe_allow_html=True)
    with nav3:
        if st.button("Next Card ➡️", disabled=(card_index >= len(filtered) - 1), use_container_width=True):
            st.session_state["flashcard_index"] = card_index + 1
            st.rerun()

    # Flashcard Render Box
    with st.container(border=True):
        st.caption(f"Difficulty: **{card.get('difficulty', 'Medium')}** | Repetitions: `{card.get('repetitions', 0)}` | SM-2 Interval: `{card.get('interval_days', 1.0):.1f} days`")
        st.markdown(f"### ❓ {card['front_question']}")

        st.markdown("---")

        if not st.session_state[reveal_key]:
            if st.button("👁️ Reveal Answer", type="primary", use_container_width=True):
                st.session_state[reveal_key] = True
                st.rerun()
        else:
            st.markdown(f"### 💡 Answer:\n{card['back_answer']}")
            st.markdown("---")
            st.markdown("**How well did you recall this concept?**")

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("⚠️ Needs Practice (Hard / Forgot)", use_container_width=True):
                    update_dict = calculate_sm2_update(
                        current_interval=card.get("interval_days", 1.0),
                        current_ease=card.get("ease_factor", 2.5),
                        current_repetitions=card.get("repetitions", 0),
                        rating="needs_practice",
                    )
                    update_flashcard_review(
                        flashcard_id=card["id"],
                        student_id="demo_student",
                        rating="needs_practice",
                        interval_days=update_dict["interval_days"],
                        ease_factor=update_dict["ease_factor"],
                        repetitions=update_dict["repetitions"],
                        next_review_date=update_dict["next_review_date"],
                    )
                    generate_adaptive_schedule("demo_student")
                    st.toast("Logged as 'Needs Practice'. Interval reset to 1 day.")
                    st.session_state[reveal_key] = False
                    if card_index < len(filtered) - 1:
                        st.session_state["flashcard_index"] = card_index + 1
                    st.rerun()

            with b_col2:
                if st.button("✅ Remembered (Easy / Recalled)", type="primary", use_container_width=True):
                    update_dict = calculate_sm2_update(
                        current_interval=card.get("interval_days", 1.0),
                        current_ease=card.get("ease_factor", 2.5),
                        current_repetitions=card.get("repetitions", 0),
                        rating="remembered",
                    )
                    update_flashcard_review(
                        flashcard_id=card["id"],
                        student_id="demo_student",
                        rating="remembered",
                        interval_days=update_dict["interval_days"],
                        ease_factor=update_dict["ease_factor"],
                        repetitions=update_dict["repetitions"],
                        next_review_date=update_dict["next_review_date"],
                    )
                    generate_adaptive_schedule("demo_student")
                    st.toast(f"Logged! Next review in {update_dict['interval_days']} days ({update_dict['next_review_date']}).")
                    st.session_state[reveal_key] = False
                    if card_index < len(filtered) - 1:
                        st.session_state["flashcard_index"] = card_index + 1
                    st.rerun()
