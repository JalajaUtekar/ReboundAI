"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import { fetchRiskPredictions, type SubjectPredictionResult } from "@/lib/predictions";
import { SubjectRiskCard } from "@/components/SubjectRiskCard";

export default function DashboardPage() {
  const router = useRouter();
  const { session, profile, loading } = useAuth();

  const [results, setResults] = useState<Record<string, SubjectPredictionResult>>({});
  const [loadingPredictions, setLoadingPredictions] = useState(true);

  useEffect(() => {
    if (!loading && !session) {
      router.replace("/login");
    }
  }, [loading, session, router]);

  useEffect(() => {
    const subjects = profile?.student_subjects ?? [];
    if (subjects.length === 0) {
      setLoadingPredictions(false);
      return;
    }

    let cancelled = false;
    setLoadingPredictions(true);
    fetchRiskPredictions(subjects.map((s) => s.id)).then((byId) => {
      if (!cancelled) {
        setResults(byId);
        setLoadingPredictions(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [profile]);

  if (loading || !session) {
    return (
      <div className="flex min-h-full flex-1 items-center justify-center">
        <p className="text-sm text-black/50 dark:text-white/50">Loading your dashboard…</p>
      </div>
    );
  }

  const subjects = profile?.student_subjects ?? [];
  const flaggedCount = subjects.filter((s) => {
    const result = results[s.id];
    return result?.status === "ok" && result.prediction.is_flagged;
  }).length;

  return (
    <div className="flex min-h-full flex-col">
      <header className="border-b border-black/10 dark:border-white/10">
        <div className="mx-auto flex max-w-5xl items-center gap-4 px-6 py-5">
          <Image
            src="/mmcoe-logo.png"
            alt="Marathwada Mitramandal College of Engineering"
            width={48}
            height={48}
            className="shrink-0"
          />
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Rebound AI</h1>
            <p className="text-sm text-black/60 dark:text-white/60">
              Fall behind less. Bounce back faster. — Department of Computer Engineering, MMCOE
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
        <div className="mb-6 flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold">Your subjects</h2>
            {profile?.student_id && (
              <p className="text-xs text-black/40 dark:text-white/40">PRN {profile.student_id}</p>
            )}
          </div>
          {!loadingPredictions && subjects.length > 0 && (
            <p className="text-sm text-black/50 dark:text-white/50">
              {flaggedCount > 0
                ? `${flaggedCount} of ${subjects.length} subject${subjects.length === 1 ? "" : "s"} flagged`
                : "No subjects currently flagged — keep it up"}
            </p>
          )}
        </div>

        {subjects.length === 0 ? (
          <div className="rounded-lg border border-black/10 px-4 py-6 text-center text-sm text-black/50 dark:border-white/10 dark:text-white/50">
            No subjects are set up on your profile yet.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {subjects.map((subject) => (
              <SubjectRiskCard key={subject.id} subject={subject} result={results[subject.id]} />
            ))}
          </div>
        )}
      </main>

      <footer className="border-t border-black/10 px-6 py-4 text-center text-xs text-black/40 dark:border-white/10 dark:text-white/40">
        Rebound AI — CE24MDM305 Foundations of GenAI
      </footer>
    </div>
  );
}
