/**
 * Shared types for the Rebound (Lecture2Revision) workspace, mirroring the
 * exact response shape documented in `api/rebound/process_lecture.py`.
 */

export type KeyDefinition = {
  term: string;
  definition: string;
  source_context: string;
};

export type ExamQuestion = {
  question: string;
  suggested_answer: string;
  difficulty: string;
};

export type Flashcard = {
  id: string;
  front_question: string;
  back_answer: string;
  difficulty: string;
};

export type QuizQuestion = {
  id: string;
  question_text: string;
  option_a: string;
  option_b: string;
  option_c: string;
  option_d: string;
  correct_option: "A" | "B" | "C" | "D";
  explanation: string;
  difficulty: string;
};

export type RevisionKit = {
  lecture_id: string;
  summary: string;
  revision_notes: string;
  key_definitions: KeyDefinition[];
  exam_questions: ExamQuestion[];
  flashcards: Flashcard[];
  quiz_questions: QuizQuestion[];
  backend_used: string;
};

export type AdaptiveScheduleEntry = {
  subject_slug: string;
  subject_name: string;
  student_subject_id: string;
  priority_score: number;
  priority_reason: string;
  scheduled_date: string;
  risk_probability: number;
  recent_score: number;
  days_overdue: number;
  cards_needing_practice: number;
  lstm_predicted_score: number | null;
  status: string;
};
