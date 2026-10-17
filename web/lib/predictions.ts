import { callApi } from "@/lib/api";
import type { RiskFactor } from "@/lib/riskFactors";

/** POST /api/predict_risk response shape (api/predict_risk.py). */
export type RiskPrediction = {
  id: string;
  student_subject_id: string;
  checkpoint_index: number;
  risk_probability: number;
  is_flagged: boolean;
  is_failing: boolean;
  top_risk_factors: RiskFactor[];
  protective_factors: RiskFactor[];
  human_readable_summary: string;
  model_version: string;
  predicted_at: string;
};

export type SubjectPredictionResult =
  | { status: "ok"; prediction: RiskPrediction }
  | { status: "error"; message: string };

/**
 * Requests a fresh risk assessment for one student_subject.
 * Never throws -- failures (e.g. a subject with no checkpoints yet, a 404)
 * are captured per-subject so one bad subject doesn't blank the whole
 * dashboard.
 */
export async function fetchRiskPrediction(studentSubjectId: string): Promise<SubjectPredictionResult> {
  try {
    const prediction = await callApi<RiskPrediction>("/api/predict_risk", {
      method: "POST",
      body: JSON.stringify({ student_subject_id: studentSubjectId }),
    });
    return { status: "ok", prediction };
  } catch (err) {
    return {
      status: "error",
      message: err instanceof Error ? err.message : "Could not load this subject's risk assessment.",
    };
  }
}

/** Fetches risk predictions for several subjects concurrently (up to 6 for a TY CE student). */
export async function fetchRiskPredictions(
  studentSubjectIds: string[]
): Promise<Record<string, SubjectPredictionResult>> {
  const results = await Promise.all(
    studentSubjectIds.map(async (id) => [id, await fetchRiskPrediction(id)] as const)
  );
  return Object.fromEntries(results);
}
