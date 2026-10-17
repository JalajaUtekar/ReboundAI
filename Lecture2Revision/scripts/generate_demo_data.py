"""
Lecture To Revision - Demo Data Generator Script
Creates:
1. A genuine sample PDF lecture on Computer Networks: Transport Layer Protocols.
2. Ingests the lecture into SQLite.
3. Generates learning materials, flashcards, and quizzes.
4. Populates chronological quiz attempts (clearly identified as synthetic demo records).
5. Exports a synthetic sequential trajectory dataset for LSTM pipeline testing.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF

from config import DATA_DIR, DB_PATH, UPLOAD_DIR
from database import (
    get_connection,
    init_db,
    record_quiz_attempt,
    save_flashcards,
    save_learning_materials,
    save_lecture,
    save_quiz_questions,
)
from lecture_processor import extract_text_from_pdf, chunk_document
from transformer_service import generate_learning_materials
from adaptive_engine import generate_adaptive_schedule


SAMPLE_LECTURE_TEXT = """
COMPUTER NETWORKS: TRANSPORT LAYER PROTOCOLS (TCP AND UDP)
Department of Computer Science & Engineering - Lecture Notes

1. INTRODUCTION TO THE TRANSPORT LAYER
The Transport Layer is Layer 4 in the OSI reference model and TCP/IP protocol suite. It provides logical communication between application processes running on distinct hosts. Unlike the Network Layer, which handles host-to-host delivery using IP addressing, the Transport Layer manages process-to-process delivery using transport service access points known as port numbers. A socket is defined as the combination of an IP address and a port number (e.g., 192.168.1.10:8080). The primary protocols operating at this layer are the Transmission Control Protocol (TCP) and the User Datagram Protocol (UDP).

2. TRANSMISSION CONTROL PROTOCOL (TCP) ARCHITECTURE
TCP is a connection-oriented, reliable, byte-stream transport protocol specified in RFC 793. Connection establishment requires a Three-Way Handshake mechanism:
1. SYN: The client transmits a segment with the SYN control flag enabled and an Initial Sequence Number (ISN_client).
2. SYN-ACK: The server replies with both SYN and ACK flags set, acknowledging the client's sequence number (ISN_client + 1) and specifying its own Initial Sequence Number (ISN_server).
3. ACK: The client acknowledges the server's sequence number (ISN_server + 1). Once completed, bidirectional data transfer commences.

Connection termination employs a Four-Way Handshake utilizing FIN and ACK flags. The host terminating the session enters the TIME_WAIT state for 2 MSL (Maximum Segment Lifetime) to ensure lingering duplicate segments are discarded and the final ACK is received by the peer.

3. TCP RELIABILITY AND FLOW CONTROL
Reliability in TCP is achieved through cumulative acknowledgments, sequence numbering, checksum verification, and retransmission timers. TCP calculates the Retransmission TimeOut (RTO) adaptively using exponential smoothing of Round-Trip Time (RTT) estimates:
EstimatedRTT = (1 - alpha) * EstimatedRTT + alpha * SampleRTT (typically alpha = 0.125).

Flow Control protects the receiver buffer from being overwhelmed by a fast sender. The receiver advertises its available buffer space through the 16-bit Receive Window (rwnd) field in the TCP header. The sender restricts its unacknowledged bytes to not exceed min(cwnd, rwnd). When rwnd reaches zero, the sender pauses transmissions and periodically dispatches 1-byte probe packets to determine if window space has reopened, avoiding deadlock.

4. TCP CONGESTION CONTROL MECHANISMS
Congestion Control prevents network link saturation and packet drop collapses across intermediate routers. TCP maintains a Congestion Window (cwnd). The four core congestion control algorithms are:
A. Slow Start: cwnd starts at 1 MSS (Maximum Segment Size) and increases exponentially by doubling every RTT (i.e., incrementing by 1 MSS for every ACK received) until it reaches the Slow Start Threshold (ssthresh).
B. Congestion Avoidance: Once cwnd >= ssthresh, growth transitions to linear incrementation (Additive Increase: cwnd increases by 1 MSS per RTT).
C. Fast Retransmit: Upon receiving three duplicate ACKs for the same segment, TCP infers packet loss without waiting for the RTO timer to expire, immediately retransmitting the missing segment.
D. Fast Recovery: cwnd is set to ssthresh + 3 MSS, and increases for further duplicate ACKs. Upon receiving a fresh ACK, TCP returns to congestion avoidance (Multiplicative Decrease).

5. USER DATAGRAM PROTOCOL (UDP) ARCHITECTURE
UDP is a lightweight, connectionless, unreliable transport protocol defined in RFC 768. UDP operates on discrete packets known as datagrams without establishing a handshake or verifying segment delivery. The UDP header comprises only 8 bytes: Source Port (16 bits), Destination Port (16 bits), Length (16 bits), and Checksum (16 bits). The checksum is optional in IPv4 but mandatory in IPv6. UDP introduces minimal transmission overhead, has no retransmission delay, and maintains no connection state. Consequently, UDP is the protocol of choice for latency-critical applications such as VoIP, DNS lookups, video streaming (RTP), and online multiplayer games.

6. COMPARATIVE ANALYSIS: TCP VS UDP
- Connection: TCP is connection-oriented; UDP is connectionless.
- Reliability: TCP guarantees in-order delivery and zero packet loss; UDP provides best-effort delivery with no delivery guarantees.
- Overhead: TCP header is 20-60 bytes; UDP header is fixed at 8 bytes.
- Flow & Congestion Control: TCP incorporates dynamic rwnd and cwnd regulation; UDP possesses neither flow nor congestion control mechanisms.
"""


def generate_sample_pdf(pdf_path: Path) -> Path:
    """Creates a sample PDF document using PyMuPDF."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    # Split into sections across pages
    sections = SAMPLE_LECTURE_TEXT.strip().split("\n\n")

    current_page = doc.new_page(width=595, height=842)  # A4 size
    rect = fitz.Rect(50, 50, 545, 792)
    pno = 1

    content_buffer = []
    for s in sections:
        content_buffer.append(s)
        joined_text = "\n\n".join(content_buffer)
        rc = current_page.insert_textbox(rect, joined_text, fontsize=10, fontname="helv")
        if rc < 0:  # Text overflowed page
            content_buffer.pop()  # remove last section from this page
            current_page.insert_textbox(rect, "\n\n".join(content_buffer), fontsize=10, fontname="helv")
            current_page = doc.new_page(width=595, height=842)
            content_buffer = [s]
            current_page.insert_textbox(rect, s, fontsize=10, fontname="helv")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def seed_database_and_demo():
    """Initializes DB and seeds sample lecture, generated materials, and synthetic student records."""
    print("[1/5] Initializing SQLite database...")
    init_db()

    pdf_path = UPLOAD_DIR / "sample_computer_networks_lecture.pdf"
    print(f"[2/5] Creating sample lecture PDF at {pdf_path}...")
    generate_sample_pdf(pdf_path)

    print("[3/5] Extracting text and chunking with PyMuPDF...")
    cleaned_text, page_records = extract_text_from_pdf(pdf_path)
    chunks = chunk_document(cleaned_text, chunk_size_words=600, overlap_words=80)
    word_count = len(cleaned_text.split())

    # Save Lecture record
    lecture_id = save_lecture(
        title="Transport Layer Protocols (TCP vs UDP)",
        subject="Computer Networks",
        topic="TCP Congestion Control & Reliability",
        filename=pdf_path.name,
        file_path=str(pdf_path),
        raw_text=cleaned_text,
        cleaned_text=cleaned_text,
        word_count=word_count,
        chunk_count=len(chunks),
    )
    print(f"      Saved lecture (ID={lecture_id}, Words={word_count}, Chunks={len(chunks)})")

    print("[4/5] Generating learning materials, flashcards, and quizzes...")
    materials, backend_used = generate_learning_materials(
        lecture_topic="TCP Congestion Control & Reliability",
        lecture_text=cleaned_text,
        backend="offline_extractive",
    )

    save_learning_materials(
        lecture_id=lecture_id,
        summary=materials["summary"],
        revision_notes=materials["revision_notes"],
        key_definitions=json.dumps(materials["key_definitions"]),
        exam_questions=json.dumps(materials["exam_questions"]),
        source_grounding_notes="Directly extracted from RFC 793 and RFC 768 lecture texts.",
        backend_used=backend_used,
    )

    save_flashcards(lecture_id, "TCP Congestion Control & Reliability", materials["flashcards"])
    save_quiz_questions(lecture_id, "TCP Congestion Control & Reliability", materials["quiz_questions"])

    # Create a secondary lecture on UDP for multi-topic comparison
    lec2_text = """
    USER DATAGRAM PROTOCOL (UDP) AND REAL-TIME NETWORKING
    UDP is connectionless and does not perform retransmissions. It is suitable for DNS, DHCP, and RTP streaming.
    Because there is no handshake, latency is minimal. Port numbers differentiate application sockets.
    Checksum calculation detects corrupted segments. If corrupted, datagrams are discarded by the host.
    """
    lec2_path = UPLOAD_DIR / "udp_realtime_protocols.pdf"
    generate_sample_pdf(lec2_path)
    lec2_cleaned, _ = extract_text_from_pdf(lec2_path)
    lec2_chunks = chunk_document(lec2_cleaned, chunk_size_words=400)
    lec2_id = save_lecture(
        title="UDP & Real-Time Applications",
        subject="Computer Networks",
        topic="UDP Datagrams & Performance",
        filename=lec2_path.name,
        file_path=str(lec2_path),
        raw_text=lec2_cleaned,
        cleaned_text=lec2_cleaned,
        word_count=len(lec2_cleaned.split()),
        chunk_count=len(lec2_chunks),
    )
    mat2, b_used2 = generate_learning_materials("UDP Datagrams & Performance", lec2_cleaned, "offline_extractive")
    save_learning_materials(
        lecture_id=lec2_id,
        summary=mat2["summary"],
        revision_notes=mat2["revision_notes"],
        key_definitions=json.dumps(mat2["key_definitions"]),
        exam_questions=json.dumps(mat2["exam_questions"]),
        source_grounding_notes="Grounded in RFC 768 lecture segment.",
        backend_used=b_used2,
    )
    save_flashcards(lec2_id, "UDP Datagrams & Performance", mat2["flashcards"])
    save_quiz_questions(lec2_id, "UDP Datagrams & Performance", mat2["quiz_questions"])

    print("[5/5] Seeding chronological student quiz history (synthetic demonstration data)...")
    now = datetime.now(timezone.utc)

    # Synthetic learning history for demo_student showing:
    # 1. Weak performance on "TCP Congestion Control & Reliability" (e.g., 40% score, 3 incorrect answers)
    # 2. Strong performance on "UDP Datagrams & Performance" (e.g., 100% score)
    # This demonstrates that the Adaptive Engine elevates TCP as high priority and relaxes UDP!

    demo_attempts = [
        # Past attempt 1 (7 days ago) - TCP
        {
            "uuid": "demo-attempt-001",
            "lecture_id": lecture_id,
            "topic": "TCP Congestion Control & Reliability",
            "score_pct": 50.0,
            "total_questions": 4,
            "correct_count": 2,
            "incorrect_count": 2,
            "time_spent": 120.0,
            "dt": now - timedelta(days=7),
        },
        # Past attempt 2 (5 days ago) - TCP (struggled again)
        {
            "uuid": "demo-attempt-002",
            "lecture_id": lecture_id,
            "topic": "TCP Congestion Control & Reliability",
            "score_pct": 40.0,
            "total_questions": 4,
            "correct_count": 1,
            "incorrect_count": 3,
            "time_spent": 140.0,
            "dt": now - timedelta(days=5),
        },
        # Past attempt 3 (3 days ago) - UDP (mastered)
        {
            "uuid": "demo-attempt-003",
            "lecture_id": lec2_id,
            "topic": "UDP Datagrams & Performance",
            "score_pct": 100.0,
            "total_questions": 4,
            "correct_count": 4,
            "incorrect_count": 0,
            "time_spent": 45.0,
            "dt": now - timedelta(days=3),
        },
        # Past attempt 4 (yesterday) - TCP (slight improvement, still needs work)
        {
            "uuid": "demo-attempt-004",
            "lecture_id": lecture_id,
            "topic": "TCP Congestion Control & Reliability",
            "score_pct": 60.0,
            "total_questions": 4,
            "correct_count": 2,
            "incorrect_count": 2,
            "time_spent": 110.0,
            "dt": now - timedelta(days=1),
        },
    ]

    for att in demo_attempts:
        record_quiz_attempt(
            attempt_uuid=att["uuid"],
            student_id="demo_student",
            lecture_id=att["lecture_id"],
            topic=att["topic"],
            score_pct=att["score_pct"],
            total_questions=att["total_questions"],
            correct_count=att["correct_count"],
            incorrect_count=att["incorrect_count"],
            time_spent_seconds=att["time_spent"],
            responses=[],
        )

    # Generate Synthetic Sequential Trajectory Dataset for LSTM pipeline training
    # Generates 25 chronological learning sequences across 5 hypothetical students
    synthetic_lstm_records = []
    student_ids = [f"student_synth_{i:02d}" for i in range(1, 6)]
    topics = ["TCP Congestion Control & Reliability", "UDP Datagrams & Performance", "IP Addressing", "Routing Algorithms"]

    import random
    rng = random.Random(42)  # reproducible seed

    for sid in student_ids:
        base_ability = rng.uniform(45.0, 75.0)
        curr_score = base_ability
        for step in range(12):  # 12 sequential quizzes per student
            # Simulating learning curve with noise and interval degradation
            interval = rng.choice([1.0, 2.0, 3.0, 5.0])
            study_gain = rng.uniform(2.0, 8.0)
            decay = interval * 1.5
            curr_score = max(20.0, min(100.0, curr_score + study_gain - decay + rng.gauss(0, 4)))
            total_q = 10
            corr = int(round((curr_score / 100.0) * total_q))
            inc = total_q - corr
            time_spent = rng.uniform(40.0, 150.0)

            rec = {
                "student_id": sid,
                "step": step,
                "topic": rng.choice(topics),
                "score_pct": round(curr_score, 1),
                "total_questions": total_q,
                "correct_count": corr,
                "incorrect_count": inc,
                "time_spent_seconds": round(time_spent, 1),
                "interval_days": interval,
                "completed_at": (now - timedelta(days=20 - step)).isoformat(),
            }
            synthetic_lstm_records.append(rec)

    synthetic_file = DATA_DIR / "synthetic_lstm_dataset.json"
    with open(synthetic_file, "w", encoding="utf-8") as f:
        json.dump(synthetic_lstm_records, f, indent=2)
    print(f"      Saved {len(synthetic_lstm_records)} synthetic LSTM training records to {synthetic_file}")

    # Run adaptive schedule calculation
    print("      Calculating initial adaptive revision schedule...")
    generate_adaptive_schedule("demo_student")

    print("\n[SUCCESS] Seeded sample lecture, generated materials, synthetic quiz history, and revision schedule.")


if __name__ == "__main__":
    seed_database_and_demo()
