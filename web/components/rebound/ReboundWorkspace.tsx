"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import UploadLectureForm from "./UploadLectureForm";
import RevisionKitViewer from "./RevisionKitViewer";
import FlashcardDeck from "./FlashcardDeck";
import QuizPractice from "./QuizPractice";
import AdaptiveScheduleSection from "./AdaptiveScheduleSection";
import type { RevisionKit } from "./types";

type Tab = "upload" | "kit" | "flashcards" | "quiz" | "plan";

export default function ReboundWorkspace({ subjectId }: { subjectId: string }) {
  const { session, profile, loading } = useAuth();
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("upload");
  const [kit, setKit] = useState<RevisionKit | null>(null);

  useEffect(() => {
    if (!loading && !session) {
      router.replace("/login");
    }
  }, [loading, session, router]);

  if (loading) {
    return <p className="px-6 py-8 text-sm text-black/50 dark:text-white/50">Loading…</p>;
  }

  if (!session) {
    // useEffect above is already redirecting to /login.
    return <p className="px-6 py-8 text-sm text-black/50 dark:text-white/50">Redirecting to sign in…</p>;
  }

  if (!profile) {
    return (
      <div className="px-6 py-8">
        <p className="text-sm text-black/60 dark:text-white/60">
          We couldn&apos;t load your student profile. Please finish registration first.
        </p>
      </div>
    );
  }

  const subject = profile.student_subjects.find((s) => s.id === subjectId);

  if (!subject) {
    return (
      <div className="px-6 py-8">
        <div className="rounded-lg border border-amber-300/60 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/30 dark:bg-amber-950/40 dark:text-amber-200">
          This isn&apos;t one of your subjects — Rebound only shows the subjects linked to your own
          account.
        </div>
      </div>
    );
  }

  const tabs: Array<{ id: Tab; label: string; disabled?: boolean }> = [
    { id: "upload", label: "Upload lecture" },
    { id: "kit", label: "Revision kit", disabled: !kit },
    { id: "flashcards", label: "Flashcards", disabled: !kit || kit.flashcards.length === 0 },
    { id: "quiz", label: "Quiz practice", disabled: !kit || kit.quiz_questions.length === 0 },
    { id: "plan", label: "Adaptive plan" },
  ];

  return (
    <div className="px-6 py-8">
      <div className="mb-6">
        <h2 className="text-lg font-semibold">Rebound — {subject.subject_name}</h2>
        <p className="text-sm text-black/60 dark:text-white/60">
          ScholarTrack flagged this subject at risk. Upload a lecture to generate a revision kit, then
          practice with flashcards and a quiz.
        </p>
      </div>

      <div className="mb-6 flex flex-wrap gap-1 border-b border-black/10 dark:border-white/10">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            disabled={t.disabled}
            onClick={() => setTab(t.id)}
            className={`-mb-px rounded-t-md border-b-2 px-3 py-2 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40 ${
              tab === t.id
                ? "border-black text-black dark:border-white dark:text-white"
                : "border-transparent text-black/50 hover:text-black/80 dark:text-white/50 dark:hover:text-white/80"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "upload" && (
        <UploadLectureForm
          studentSubjectId={subject.id}
          onProcessed={(newKit) => {
            setKit(newKit);
            setTab("kit");
          }}
        />
      )}

      {tab === "kit" && kit && <RevisionKitViewer kit={kit} />}
      {tab === "flashcards" && kit && <FlashcardDeck flashcards={kit.flashcards} />}
      {tab === "quiz" && kit && <QuizPractice questions={kit.quiz_questions} />}
      {tab === "plan" && <AdaptiveScheduleSection />}
    </div>
  );
}
