"use client";

import { useState } from "react";
import { callApi } from "@/lib/api";
import type { Flashcard } from "./types";

type ReviewState = "idle" | "saving" | "saved" | "error";

export default function FlashcardDeck({ flashcards }: { flashcards: Flashcard[] }) {
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [reviewState, setReviewState] = useState<ReviewState>("idle");
  const [reviewError, setReviewError] = useState<string | null>(null);

  if (flashcards.length === 0) {
    return <p className="text-sm text-black/50 dark:text-white/50">No flashcards generated.</p>;
  }

  const card = flashcards[index];

  const goTo = (next: number) => {
    setIndex(((next % flashcards.length) + flashcards.length) % flashcards.length);
    setRevealed(false);
    setReviewState("idle");
    setReviewError(null);
  };

  const submitReview = async (rating: "remembered" | "needs_practice") => {
    setReviewState("saving");
    setReviewError(null);
    try {
      await callApi("/api/rebound/sm2_review", {
        method: "POST",
        body: JSON.stringify({ flashcard_id: card.id, rating }),
      });
      setReviewState("saved");
    } catch (err) {
      setReviewState("error");
      setReviewError(err instanceof Error ? err.message : "Failed to save review.");
    }
  };

  return (
    <div>
      <div className="mb-3 flex items-center justify-between text-xs text-black/50 dark:text-white/50">
        <span>
          Card {index + 1} of {flashcards.length}
        </span>
        <span className="rounded-full bg-black/5 px-2 py-0.5 dark:bg-white/10">{card.difficulty}</span>
      </div>

      <button
        type="button"
        onClick={() => setRevealed((r) => !r)}
        className="flex min-h-[160px] w-full flex-col items-center justify-center rounded-lg border border-black/10 p-6 text-center transition hover:border-black/30 dark:border-white/10 dark:hover:border-white/30"
      >
        <p className="text-sm font-medium">{revealed ? card.back_answer : card.front_question}</p>
        <p className="mt-3 text-xs text-black/40 dark:text-white/40">
          {revealed ? "Click to show question" : "Click to reveal answer"}
        </p>
      </button>

      {revealed && (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => submitReview("needs_practice")}
            disabled={reviewState === "saving"}
            className="rounded-md border border-amber-400/60 bg-amber-50 px-3 py-1.5 text-sm font-medium text-amber-800 disabled:opacity-50 dark:border-amber-400/30 dark:bg-amber-950/40 dark:text-amber-200"
          >
            Needs practice
          </button>
          <button
            type="button"
            onClick={() => submitReview("remembered")}
            disabled={reviewState === "saving"}
            className="rounded-md border border-emerald-400/60 bg-emerald-50 px-3 py-1.5 text-sm font-medium text-emerald-800 disabled:opacity-50 dark:border-emerald-400/30 dark:bg-emerald-950/40 dark:text-emerald-200"
          >
            Remembered
          </button>
          {reviewState === "saved" && (
            <span className="text-xs text-emerald-600 dark:text-emerald-400">Saved — SM-2 schedule updated.</span>
          )}
          {reviewState === "error" && (
            <span className="text-xs text-red-600 dark:text-red-400">{reviewError}</span>
          )}
        </div>
      )}

      <div className="mt-4 flex gap-2">
        <button
          type="button"
          onClick={() => goTo(index - 1)}
          className="rounded-md border border-black/15 px-3 py-1.5 text-sm dark:border-white/15"
        >
          Previous
        </button>
        <button
          type="button"
          onClick={() => goTo(index + 1)}
          className="rounded-md border border-black/15 px-3 py-1.5 text-sm dark:border-white/15"
        >
          Next
        </button>
      </div>
    </div>
  );
}
