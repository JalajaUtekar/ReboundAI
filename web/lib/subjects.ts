/**
 * Subject catalog for Rebound AI's TY Computer Engineering cohort (MMCOE).
 *
 * Mirrors `scripts/data_generation/subjects.py` exactly (names, grouping,
 * and component weights). This file is hand-kept in sync with that module
 * rather than generated/shared at build time -- if the Python catalog ever
 * changes (new MDM/OEL/Elective options, renamed subjects), this file must
 * be updated to match by hand.
 */

const PR_COMPONENTS = { TH: 40, ETE: 60, PR: 25 };
const TW_COMPONENTS = { TH: 40, ETE: 60, TW: 25 };

export const FIXED_SUBJECTS = [
  { slug: "DSML", name: "Data Science and Machine Learning", components: PR_COMPONENTS },
  { slug: "CN", name: "Computer Networks", components: TW_COMPONENTS },
  { slug: "SPOS", name: "System Programming and Operating Systems", components: PR_COMPONENTS },
] as const;

export const MDM_OPTIONS = [
  "GenAI",
  "Privacy and Security Engineering",
  "Robotics and Automation",
  "Electric Vehicle Technology",
  "Sustainable and Smart Systems",
  "Embedded Systems",
] as const;

export const OEL_OPTIONS = [
  "Practical Japanese Communication",
  "Workplace Ethics in German Language",
  "Product and Brand Management",
  "Strategic Management",
  "Operations and Supply Chain Management",
  "Organizational Behaviour",
] as const;

export const ELECTIVE_OPTIONS = [
  "Cloud Computing",
  "Information Systems",
  "Prompt Engineering",
  "Game Development",
] as const;

export const BRANCHES = ["AI&DS", "CE", "IT", "E&TC", "ME", "EE"] as const;
export const ENABLED_BRANCH: (typeof BRANCHES)[number] = "CE";

export const ACADEMIC_YEARS = ["FY", "SY", "TY", "BE"] as const;
export const ENABLED_ACADEMIC_YEAR: (typeof ACADEMIC_YEARS)[number] = "TY";

/** Converts a human-readable subject name into a stable lowercase, underscore-separated slug. Mirrors `scripts/data_generation/subjects.py::subject_slug`. */
export function subjectSlug(name: string): string {
  return name.trim().toLowerCase().replace(/ /g, "_").replace(/&/g, "and");
}
