"use client";

import { useState } from "react";
import Link from "next/link";
import type { SubjectPredictionResult } from "@/lib/predictions";
import { humanizeFeatureName, type RiskFactor } from "@/lib/riskFactors";

export type SubjectSummary = {
  id: string;
  subject_slug: string;
  subject_name: string;
  subject_type: string;
};

export function SubjectRiskCard({
  subject,
  result,
}: {
  subject: SubjectSummary;
  result: SubjectPredictionResult | undefined;
}) {
  const [showWhy, setShowWhy] = useState(false);

  return (
    <div className="rounded-lg border border-black/10 p-4 dark:border-white/10">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-medium">{subject.subject_name}</h3>
          <p className="text-xs uppercase tracking-wide text-black/40 dark:text-white/40">
            {subject.subject_type}
          </p>
        </div>
        <StatusBadges result={result} />
      </div>

      {!result && (
        <p className="mt-3 text-sm text-black/50 dark:text-white/50">Loading risk assessment…</p>
      )}

      {result?.status === "error" && (
        <p className="mt-3 text-sm text-amber-700 dark:text-amber-300">
          Couldn&apos;t load a risk assessment for this subject: {result.message}
        </p>
      )}

      {result?.status === "ok" && result.prediction.is_flagged && (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setShowWhy((v) => !v)}
            className="text-sm font-medium text-red-700 underline-offset-2 hover:underline dark:text-red-300"
          >
            {showWhy ? "Hide" : "Why flagged?"}
          </button>

          {showWhy && <WhyFlagged prediction={result.prediction} />}

          <div className="mt-3">
            <Link
              href={`/rebound/${subject.id}`}
              className="inline-flex items-center rounded-md bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700 dark:bg-red-500 dark:hover:bg-red-600"
            >
              Open your Rebound plan for {subject.subject_name}
            </Link>
          </div>
        </div>
      )}

      {result?.status === "ok" && !result.prediction.is_flagged && (
        <p className="mt-3 text-sm text-black/60 dark:text-white/60">
          {result.prediction.human_readable_summary}
        </p>
      )}
    </div>
  );
}

function StatusBadges({ result }: { result: SubjectPredictionResult | undefined }) {
  if (!result) {
    return (
      <span className="rounded-full bg-black/5 px-2 py-0.5 text-xs font-medium text-black/40 dark:bg-white/10 dark:text-white/40">
        Checking…
      </span>
    );
  }

  if (result.status === "error") {
    return (
      <span className="rounded-full bg-black/5 px-2 py-0.5 text-xs font-medium text-black/40 dark:bg-white/10 dark:text-white/40">
        Unavailable
      </span>
    );
  }

  const { is_flagged, is_failing } = result.prediction;

  if (!is_flagged) {
    return (
      <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">
        On track
      </span>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-950/50 dark:text-amber-300">
        At risk of falling behind
      </span>
      {is_failing && (
        <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700 dark:bg-red-950/50 dark:text-red-300">
          Failing
        </span>
      )}
    </div>
  );
}

function WhyFlagged({ prediction }: { prediction: { human_readable_summary: string; top_risk_factors: RiskFactor[]; protective_factors: RiskFactor[]; risk_probability: number } }) {
  return (
    <div className="mt-2 rounded-md bg-black/5 p-3 text-sm dark:bg-white/5">
      <p className="text-black/80 dark:text-white/80">{prediction.human_readable_summary}</p>

      {prediction.top_risk_factors.length > 0 && (
        <div className="mt-2">
          <p className="text-xs font-medium uppercase tracking-wide text-black/40 dark:text-white/40">
            Main risk factors
          </p>
          <ul className="mt-1 list-inside list-disc text-black/70 dark:text-white/70">
            {prediction.top_risk_factors.map((factor) => (
              <li key={factor.feature}>{humanizeFeatureName(factor.feature)}</li>
            ))}
          </ul>
        </div>
      )}

      {prediction.protective_factors.length > 0 && (
        <div className="mt-2">
          <p className="text-xs font-medium uppercase tracking-wide text-black/40 dark:text-white/40">
            Working in your favour
          </p>
          <ul className="mt-1 list-inside list-disc text-black/70 dark:text-white/70">
            {prediction.protective_factors.map((factor) => (
              <li key={factor.feature}>{humanizeFeatureName(factor.feature)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
