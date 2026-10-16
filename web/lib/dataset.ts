import fs from "fs";
import path from "path";

export type SubjectOutcome = {
  ete_score: number;
  aggregate_pct: number;
  final_attendance_pct: number;
  at_risk: boolean;
};

export type Subject = {
  subject_slug: string;
  subject_name: string;
  th_score: number;
  pr_tw_score: number;
  outcome: SubjectOutcome;
};

export type Student = {
  student_id: string;
  name: string;
  prn: string;
  email: string;
  branch: string;
  academic_year: string;
  subjects: Subject[];
};

/** Reads the generated synthetic cohort. Server-only (uses the filesystem). */
export function loadStudents(): Student[] {
  const filePath = path.join(process.cwd(), "data", "synthetic_students.json");
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as Student[];
}

export function flaggedSubject(student: Student): Subject | undefined {
  return student.subjects.find((subject) => subject.outcome.at_risk);
}
