"use client";

import { useState } from "react";
import type { QuizQuestion } from "./types";

const OPTION_KEYS: Array<"A" | "B" | "C" | "D"> = ["A", "B", "C", "D"];

function optionText(q: QuizQuestion, key: "A" | "B" | "C" | "D"): string {
  switch (key) {
    case "A":
      return q.option_a;
    case "B":
      return q.option_b;
    case "C":
      return q.option_c;
    case "D":
      return q.option_d;
  }
}

/**
 * Client-side-only scoring: process_lecture's quiz_questions already carry
 * correct_option + explanation, and there's no quiz-attempt-persistence
 * endpoint yet (spec out of scope for this task), so this is immediate
 * feedback only -- nothing is sent to the backend here.
 */
export default function QuizPractice({ questions }: { questions: QuizQuestion[] }) {
  const [answers, setAnswers] = useState<Record<string, "A" | "B" | "C" | "D">>({});

  if (questions.length === 0) {
    return <p className="text-sm text-black/50 dark:text-white/50">No quiz questions generated.</p>;
  }

  const answeredCount = Object.keys(answers).length;
  const correctCount = questions.filter((q) => answers[q.id] === q.correct_option).length;

  return (
    <div className="space-y-6">
      <div className="rounded-md border border-black/10 bg-black/[0.02] px-3 py-2 text-sm dark:border-white/10 dark:bg-white/[0.03]">
        Score: {correctCount} / {answeredCount} answered ({questions.length} total)
        <span className="ml-2 text-xs text-black/40 dark:text-white/40">
          (scored locally — not saved to your account yet)
        </span>
      </div>

      <ol className="space-y-5">
        {questions.map((q, i) => {
          const selected = answers[q.id];
          const isAnswered = selected !== undefined;
          return (
            <li key={q.id} className="rounded-lg border border-black/10 p-4 dark:border-white/10">
              <p className="mb-3 text-sm font-medium">
                {i + 1}. {q.question_text}
              </p>
              <div className="space-y-2">
                {OPTION_KEYS.map((key) => {
                  const isSelected = selected === key;
                  const isCorrect = key === q.correct_option;
                  let tone = "border-black/15 dark:border-white/15";
                  if (isAnswered && isSelected && isCorrect) {
                    tone = "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/40";
                  } else if (isAnswered && isSelected && !isCorrect) {
                    tone = "border-red-500 bg-red-50 dark:bg-red-950/40";
                  } else if (isAnswered && isCorrect) {
                    tone = "border-emerald-500/60";
                  }
                  return (
                    <button
                      key={key}
                      type="button"
                      disabled={isAnswered}
                      onClick={() => setAnswers((a) => ({ ...a, [q.id]: key }))}
                      className={`block w-full rounded-md border px-3 py-2 text-left text-sm disabled:cursor-default ${tone}`}
                    >
                      <span className="mr-2 font-medium">{key}.</span>
                      {optionText(q, key)}
                    </button>
                  );
                })}
              </div>
              {isAnswered && (
                <div
                  className={`mt-3 rounded-md px-3 py-2 text-sm ${
                    selected === q.correct_option
                      ? "bg-emerald-50 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200"
                      : "bg-red-50 text-red-800 dark:bg-red-950/40 dark:text-red-200"
                  }`}
                >
                  {selected === q.correct_option ? "Correct. " : `Incorrect — correct answer is ${q.correct_option}. `}
                  {q.explanation}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
