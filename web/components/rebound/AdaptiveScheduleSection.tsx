"use client";

import { useEffect, useState } from "react";
import { callApi } from "@/lib/api";
import type { AdaptiveScheduleEntry } from "./types";

type LoadState = "loading" | "loaded" | "error";

export default function AdaptiveScheduleSection() {
  const [state, setState] = useState<LoadState>("loading");
  const [plan, setPlan] = useState<AdaptiveScheduleEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    callApi<AdaptiveScheduleEntry[]>("/api/rebound/adaptive_schedule")
      .then((data) => {
        if (!isMounted) return;
        setPlan(data);
        setState("loaded");
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err instanceof Error ? err.message : "Failed to load the adaptive schedule.");
        setState("error");
      });
    return () => {
      isMounted = false;
    };
  }, []);

  if (state === "loading") {
    return <p className="text-sm text-black/50 dark:text-white/50">Loading your adaptive revision plan…</p>;
  }

  if (state === "error") {
    return (
      <div className="rounded-md border border-red-300/60 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-400/30 dark:bg-red-950/40 dark:text-red-300">
        {error}
      </div>
    );
  }

  if (plan.length === 0) {
    return (
      <p className="text-sm text-black/50 dark:text-white/50">
        No flagged subjects currently need an adaptive revision plan.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-black/60 dark:text-white/60">
        Prioritized across all of your currently at-risk subjects, highest priority first.
      </p>
      {plan.map((entry) => (
        <div
          key={entry.student_subject_id}
          className="rounded-lg border border-black/10 p-4 dark:border-white/10"
        >
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold">{entry.subject_name}</h4>
            <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700 dark:bg-red-950/50 dark:text-red-300">
              Priority {entry.priority_score.toFixed(1)}
            </span>
          </div>
          <p className="mt-2 text-sm text-black/70 dark:text-white/70">{entry.priority_reason}</p>
          <div className="mt-3 flex flex-wrap gap-4 text-xs text-black/50 dark:text-white/50">
            <span>Scheduled: {entry.scheduled_date}</span>
            <span>Risk: {(entry.risk_probability * 100).toFixed(0)}%</span>
            <span>Cards needing practice: {entry.cards_needing_practice}</span>
            <span>Days overdue: {entry.days_overdue}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
