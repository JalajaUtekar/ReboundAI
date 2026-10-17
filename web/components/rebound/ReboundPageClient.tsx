"use client";

import Image from "next/image";
import Link from "next/link";
import ReboundWorkspace from "./ReboundWorkspace";

export default function ReboundPageClient({ subjectId }: { subjectId: string }) {
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
            <h1 className="text-xl font-semibold tracking-tight">
              <Link href="/dashboard">Rebound AI</Link>
            </h1>
            <p className="text-sm text-black/60 dark:text-white/60">
              Lecture2Revision — adaptive revision for your flagged subjects
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1">
        <ReboundWorkspace subjectId={subjectId} />
      </main>

      <footer className="border-t border-black/10 px-6 py-4 text-center text-xs text-black/40 dark:border-white/10 dark:text-white/40">
        Rebound AI — CE24MDM305 Foundations of GenAI
      </footer>
    </div>
  );
}
