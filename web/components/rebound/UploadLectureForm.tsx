"use client";

import { useState } from "react";
import { callApi } from "@/lib/api";
import type { RevisionKit } from "./types";

/** Reads a File as a base64 string, stripping the `data:...;base64,` prefix
 * so the payload matches process_lecture.py's `_decode_pdf_base64` (which
 * also tolerates the prefix, but sending the raw base64 keeps the request
 * body smaller and the intent explicit). */
function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const commaIndex = result.indexOf(",");
      resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
    };
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read file."));
    reader.readAsDataURL(file);
  });
}

export default function UploadLectureForm({
  studentSubjectId,
  onProcessed,
}: {
  studentSubjectId: string;
  onProcessed: (kit: RevisionKit) => void;
}) {
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Choose a PDF file first.");
      return;
    }
    if (!title.trim()) {
      setError("Give the lecture a title.");
      return;
    }
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }

    setStatus("uploading");
    setError(null);
    try {
      const pdf_base64 = await readFileAsBase64(file);
      const kit = await callApi<RevisionKit>("/api/rebound/process_lecture", {
        method: "POST",
        body: JSON.stringify({
          student_subject_id: studentSubjectId,
          title: title.trim(),
          pdf_base64,
        }),
      });
      setStatus("idle");
      onProcessed(kit);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Failed to process the lecture.");
    }
  };

  const uploading = status === "uploading";

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-lg border border-black/10 p-5 dark:border-white/10"
    >
      <h3 className="mb-1 text-sm font-semibold">Upload a lecture PDF</h3>
      <p className="mb-4 text-sm text-black/60 dark:text-white/60">
        We&apos;ll extract the text and generate a revision kit: a summary, revision notes,
        key definitions, exam questions, flashcards, and a practice quiz.
      </p>

      <div className="mb-4">
        <label htmlFor="lecture-title" className="mb-1 block text-xs font-medium text-black/70 dark:text-white/70">
          Lecture title
        </label>
        <input
          id="lecture-title"
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Transport Layer Protocols"
          disabled={uploading}
          className="w-full rounded-md border border-black/15 bg-transparent px-3 py-2 text-sm outline-none focus:border-black/40 disabled:opacity-50 dark:border-white/15 dark:focus:border-white/40"
        />
      </div>

      <div className="mb-4">
        <label htmlFor="lecture-pdf" className="mb-1 block text-xs font-medium text-black/70 dark:text-white/70">
          Lecture PDF
        </label>
        <input
          id="lecture-pdf"
          type="file"
          accept="application/pdf"
          disabled={uploading}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="block w-full text-sm disabled:opacity-50"
        />
      </div>

      {error && (
        <div className="mb-4 rounded-md border border-red-300/60 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-400/30 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={uploading}
        className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-white dark:text-black"
      >
        {uploading ? "Generating revision kit…" : "Generate revision kit"}
      </button>
      {uploading && (
        <p className="mt-2 text-xs text-black/50 dark:text-white/50">
          This can take a few seconds — extracting the PDF and generating notes, flashcards, and a quiz.
        </p>
      )}
    </form>
  );
}
