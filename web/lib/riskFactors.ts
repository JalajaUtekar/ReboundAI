/**
 * Human-readable labels for the raw ScholarTrack feature names
 * (`api/_lib/scholartrack.py` / `scripts/feature_engineering/extraction.py`'s
 * `FEATURE_NAMES`), used as a display fallback under "Why flagged?" in case a
 * raw feature name ever leaks through `top_risk_factors`/`protective_factors`
 * (the model's `human_readable_summary` is the primary, already-narrated
 * explanation and is shown above these).
 */
const FEATURE_LABELS: Record<string, string> = {
  internal_marks_pct: "Internal marks",
  assignment_marks_pct: "Assignment marks",
  attendance_pct: "Attendance",
  on_time_submission_ratio: "On-time submission rate",
  avg_days_late: "Average days late on submissions",
  attendance_velocity: "Attendance trend",
  marks_velocity: "Marks trend",
  delay_acceleration: "Submission-delay trend",
  attendance_deficit_margin: "Attendance vs. 75% eligibility line",
  marks_deficit_margin: "Aggregate vs. 70% risk line",
};

export function humanizeFeatureName(raw: string): string {
  if (FEATURE_LABELS[raw]) return FEATURE_LABELS[raw];
  return raw
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export type RiskFactor = {
  feature: string;
  shap_value: number;
  feature_value: number;
};
