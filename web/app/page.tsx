import Image from "next/image";
import { loadStudents, flaggedSubject, type Student } from "@/lib/dataset";

export default function Home() {
  const students = loadStudents();
  const atRisk = students.filter((s) => flaggedSubject(s));
  const atRiskRate = ((atRisk.length / students.length) * 100).toFixed(1);

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
        <div className="mb-6 rounded-lg border border-amber-300/60 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/30 dark:bg-amber-950/40 dark:text-amber-200">
          Showing the generated demo cohort&apos;s ground-truth labels (used to{" "}
          <em>train</em> ScholarTrack). The risk model itself hasn&apos;t been built yet —
          these aren&apos;t predictions.
        </div>

        <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <StatCard label="Students" value={students.length.toString()} />
          <StatCard label="Flagged at risk" value={atRisk.length.toString()} tone="risk" />
          <StatCard label="At-risk rate" value={`${atRiskRate}%`} />
        </div>

        <h2 className="mb-3 text-sm font-medium text-black/60 dark:text-white/60">
          TY — Computer Engineering
        </h2>
        <div className="overflow-hidden rounded-lg border border-black/10 dark:border-white/10">
          <table className="w-full text-sm">
            <thead className="bg-black/5 text-left dark:bg-white/5">
              <tr>
                <th className="px-4 py-2 font-medium">Student</th>
                <th className="px-4 py-2 font-medium">PRN</th>
                <th className="px-4 py-2 font-medium">Subjects</th>
                <th className="px-4 py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {students.map((student) => (
                <StudentRow key={student.student_id} student={student} />
              ))}
            </tbody>
          </table>
        </div>
      </main>

      <footer className="border-t border-black/10 px-6 py-4 text-center text-xs text-black/40 dark:border-white/10 dark:text-white/40">
        Rebound AI — CE24MDM305 Foundations of GenAI
      </footer>
    </div>
  );
}

function StatCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "risk";
}) {
  return (
    <div className="rounded-lg border border-black/10 px-4 py-3 dark:border-white/10">
      <div className="text-xs text-black/50 dark:text-white/50">{label}</div>
      <div
        className={`text-2xl font-semibold ${
          tone === "risk" ? "text-red-600 dark:text-red-400" : ""
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function StudentRow({ student }: { student: Student }) {
  const flagged = flaggedSubject(student);
  return (
    <tr className="border-t border-black/10 dark:border-white/10">
      <td className="px-4 py-2">{student.name}</td>
      <td className="px-4 py-2 text-black/60 dark:text-white/60">{student.prn}</td>
      <td className="px-4 py-2 text-black/60 dark:text-white/60">
        {student.subjects.length} subjects
        {flagged && <span className="ml-1 text-red-600 dark:text-red-400">({flagged.subject_name})</span>}
      </td>
      <td className="px-4 py-2">
        {flagged ? (
          <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700 dark:bg-red-950/50 dark:text-red-300">
            At risk
          </span>
        ) : (
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">
            On track
          </span>
        )}
      </td>
    </tr>
  );
}
