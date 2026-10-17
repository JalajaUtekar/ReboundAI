"use client";

/**
 * Rebound AI registration (spec section 2's 4-step flow).
 *
 * Implemented as a single page with local step-index state (rather than
 * `step-N/page.tsx` sub-routes) because the whole flow is one logical
 * form submitted as one POST /api/register at the end -- there's no need
 * for each step to be independently linkable/bookmarkable, and keeping
 * all the in-progress answers in one component's state is simpler than
 * threading them through routed sub-pages (query params, sessionStorage,
 * etc.) for a 4-step wizard this small.
 *
 * A Supabase Auth session is required before the 4-step wizard itself
 * starts (per the task brief: registration completes a profile for an
 * already-authenticated auth user). If there's no session yet, this page
 * first shows a lightweight sign-up gate (email/password sign-up, or
 * Google) before stepping into the wizard.
 */

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { callApi } from "@/lib/api";
import {
  ACADEMIC_YEARS,
  BRANCHES,
  ELECTIVE_OPTIONS,
  ENABLED_ACADEMIC_YEAR,
  ENABLED_BRANCH,
  FIXED_SUBJECTS,
  MDM_OPTIONS,
  OEL_OPTIONS,
  subjectSlug,
} from "@/lib/subjects";
import { supabase } from "@/lib/supabaseClient";

const TOTAL_STEPS = 4;

export default function RegisterPage() {
  const { session, loading } = useAuth();

  if (loading) {
    return <CenteredMessage>Loading...</CenteredMessage>;
  }

  if (!session) {
    return <SignUpGate />;
  }

  return <RegistrationWizard email={session.user.email ?? ""} />;
}

function CenteredMessage({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-full flex-1 items-center justify-center px-6 py-12 text-sm text-black/60 dark:text-white/60">
      {children}
    </div>
  );
}

function SignUpGate() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleGoogleSignIn() {
    setError(null);
    const { error: oauthError } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/register` },
    });
    if (oauthError) setError(oauthError.message);
  }

  async function handleSignUp(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setInfo(null);
    setSubmitting(true);
    try {
      const { data, error: signUpError } = await supabase.auth.signUp({ email, password });
      if (signUpError) {
        setError(signUpError.message);
        return;
      }
      // If email confirmation is required by the Supabase project's auth
      // settings, there's no session yet -- tell the user to confirm, then
      // come back. If confirmation is off, `data.session` is already set
      // and AuthContext will pick it up on the next render via
      // onAuthStateChange, moving straight into the wizard.
      if (!data.session) {
        setInfo("Check your email to confirm your account, then come back to this page to finish registering.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-full flex-1 flex-col items-center justify-center px-6 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-8 flex flex-col items-center gap-3 text-center">
          <Image
            src="/mmcoe-logo.png"
            alt="Marathwada Mitramandal College of Engineering"
            width={56}
            height={56}
          />
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Create your account</h1>
            <p className="text-sm text-black/60 dark:text-white/60">Rebound AI — MMCOE, Department of Computer Engineering</p>
          </div>
        </div>

        <button
          type="button"
          onClick={handleGoogleSignIn}
          className="mb-4 w-full rounded-lg border border-black/10 px-4 py-2.5 text-sm font-medium transition hover:bg-black/5 dark:border-white/15 dark:hover:bg-white/5"
        >
          Sign up with Google
        </button>

        <div className="mb-4 flex items-center gap-3 text-xs text-black/40 dark:text-white/40">
          <div className="h-px flex-1 bg-black/10 dark:bg-white/10" />
          or sign up with email
          <div className="h-px flex-1 bg-black/10 dark:bg-white/10" />
        </div>

        <form onSubmit={handleSignUp} className="flex flex-col gap-3">
          <input
            type="email"
            required
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="rounded-lg border border-black/10 bg-transparent px-3 py-2 text-sm outline-none focus:border-black/30 dark:border-white/15 dark:focus:border-white/30"
          />
          <input
            type="password"
            required
            minLength={6}
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="rounded-lg border border-black/10 bg-transparent px-3 py-2 text-sm outline-none focus:border-black/30 dark:border-white/15 dark:focus:border-white/30"
          />

          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
          {info && <p className="text-sm text-emerald-600 dark:text-emerald-400">{info}</p>}

          <button
            type="submit"
            disabled={submitting}
            className="mt-1 w-full rounded-lg bg-black px-4 py-2.5 text-sm font-medium text-white transition hover:bg-black/85 disabled:opacity-50 dark:bg-white dark:text-black dark:hover:bg-white/85"
          >
            {submitting ? "Creating account..." : "Create account"}
          </button>
        </form>
      </div>
    </div>
  );
}

type FormState = {
  name: string;
  prn: string;
  photoFileName: string | null;
  branch: string | null;
  academicYear: string | null;
  mdmName: string | null;
  oelName: string | null;
  electiveName: string | null;
};

function RegistrationWizard({ email }: { email: string }) {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [form, setForm] = useState<FormState>({
    name: "",
    prn: "",
    photoFileName: null,
    branch: null,
    academicYear: null,
    mdmName: null,
    oelName: null,
    electiveName: null,
  });
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function next() {
    setStep((s) => Math.min(TOTAL_STEPS, s + 1));
  }
  function back() {
    setStep((s) => Math.max(1, s - 1));
  }

  async function handleFinalSubmit() {
    setSubmitError(null);
    setSubmitting(true);
    try {
      await callApi("/api/register", {
        method: "POST",
        body: JSON.stringify({
          name: form.name,
          prn: form.prn,
          email,
          branch: form.branch,
          academic_year: form.academicYear,
          mdm_slug: subjectSlug(form.mdmName!),
          oel_slug: subjectSlug(form.oelName!),
          elective_slug: subjectSlug(form.electiveName!),
        }),
      });
      router.push("/dashboard");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-full flex-1 flex-col items-center px-6 py-12">
      <div className="w-full max-w-lg">
        <div className="mb-6 flex items-center gap-3">
          <Image src="/mmcoe-logo.png" alt="MMCOE" width={40} height={40} />
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Complete your registration</h1>
            <p className="text-xs text-black/60 dark:text-white/60">Step {step} of {TOTAL_STEPS}</p>
          </div>
        </div>

        <div className="mb-6 h-1.5 w-full overflow-hidden rounded-full bg-black/10 dark:bg-white/10">
          <div
            className="h-full bg-black transition-all dark:bg-white"
            style={{ width: `${(step / TOTAL_STEPS) * 100}%` }}
          />
        </div>

        <div className="rounded-lg border border-black/10 p-6 dark:border-white/10">
          {step === 1 && (
            <StepBasicDetails
              email={email}
              name={form.name}
              prn={form.prn}
              photoFileName={form.photoFileName}
              onChange={(patch) => setForm((f) => ({ ...f, ...patch }))}
              onNext={next}
            />
          )}
          {step === 2 && (
            <StepBranch
              branch={form.branch}
              onChange={(branch) => setForm((f) => ({ ...f, branch }))}
              onNext={next}
              onBack={back}
            />
          )}
          {step === 3 && (
            <StepAcademicYear
              academicYear={form.academicYear}
              onChange={(academicYear) => setForm((f) => ({ ...f, academicYear }))}
              onNext={next}
              onBack={back}
            />
          )}
          {step === 4 && (
            <StepSubjects
              mdmName={form.mdmName}
              oelName={form.oelName}
              electiveName={form.electiveName}
              onChange={(patch) => setForm((f) => ({ ...f, ...patch }))}
              onBack={back}
              onSubmit={handleFinalSubmit}
              submitting={submitting}
              submitError={submitError}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <label className="mb-1 block text-xs font-medium text-black/60 dark:text-white/60">{children}</label>;
}

function WizardNav({
  onBack,
  onNext,
  nextDisabled,
  nextLabel = "Continue",
}: {
  onBack?: () => void;
  onNext: () => void;
  nextDisabled?: boolean;
  nextLabel?: string;
}) {
  return (
    <div className="mt-6 flex items-center justify-between">
      {onBack ? (
        <button
          type="button"
          onClick={onBack}
          className="rounded-lg px-4 py-2 text-sm font-medium text-black/60 transition hover:bg-black/5 dark:text-white/60 dark:hover:bg-white/5"
        >
          Back
        </button>
      ) : (
        <span />
      )}
      <button
        type="button"
        onClick={onNext}
        disabled={nextDisabled}
        className="rounded-lg bg-black px-5 py-2 text-sm font-medium text-white transition hover:bg-black/85 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-white dark:text-black dark:hover:bg-white/85"
      >
        {nextLabel}
      </button>
    </div>
  );
}

function StepBasicDetails({
  email,
  name,
  prn,
  photoFileName,
  onChange,
  onNext,
}: {
  email: string;
  name: string;
  prn: string;
  photoFileName: string | null;
  onChange: (patch: Partial<FormState>) => void;
  onNext: () => void;
}) {
  return (
    <div>
      <h2 className="mb-4 text-sm font-semibold">Basic details</h2>
      <div className="flex flex-col gap-4">
        <div>
          <FieldLabel>Name</FieldLabel>
          <input
            type="text"
            value={name}
            onChange={(e) => onChange({ name: e.target.value })}
            placeholder="Full name"
            className="w-full rounded-lg border border-black/10 bg-transparent px-3 py-2 text-sm outline-none focus:border-black/30 dark:border-white/15 dark:focus:border-white/30"
          />
        </div>
        <div>
          <FieldLabel>PRN</FieldLabel>
          <input
            type="text"
            value={prn}
            onChange={(e) => onChange({ prn: e.target.value })}
            placeholder="Permanent Registration Number"
            className="w-full rounded-lg border border-black/10 bg-transparent px-3 py-2 text-sm outline-none focus:border-black/30 dark:border-white/15 dark:focus:border-white/30"
          />
        </div>
        <div>
          <FieldLabel>Email</FieldLabel>
          <input
            type="email"
            value={email}
            disabled
            className="w-full rounded-lg border border-black/10 bg-black/5 px-3 py-2 text-sm text-black/60 outline-none dark:border-white/15 dark:bg-white/5 dark:text-white/60"
          />
        </div>
        <div>
          <FieldLabel>Photograph</FieldLabel>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => onChange({ photoFileName: e.target.files?.[0]?.name ?? null })}
            className="w-full text-sm"
          />
          {/* Known simplification: the file is accepted client-side (its
              name is kept in local state for display) but never actually
              uploaded to storage -- POST /api/register has no photo_url
              field wired up, so students.photo_url stays NULL for now.
              Wiring real photo upload (Supabase Storage bucket + signed
              URL) is a follow-up, not needed for the hackathon demo. */}
          {photoFileName && <p className="mt-1 text-xs text-black/50 dark:text-white/50">Selected: {photoFileName} (not uploaded yet — demo simplification)</p>}
        </div>
      </div>
      <WizardNav onNext={onNext} nextDisabled={!name.trim() || !prn.trim()} />
    </div>
  );
}

function StepBranch({
  branch,
  onChange,
  onNext,
  onBack,
}: {
  branch: string | null;
  onChange: (branch: string) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  return (
    <div>
      <h2 className="mb-4 text-sm font-semibold">Branch</h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {BRANCHES.map((b) => {
          const enabled = b === ENABLED_BRANCH;
          const selected = branch === b;
          return (
            <button
              key={b}
              type="button"
              disabled={!enabled}
              onClick={() => enabled && onChange(b)}
              className={`relative rounded-lg border px-3 py-3 text-sm font-medium transition ${
                selected
                  ? "border-black bg-black text-white dark:border-white dark:bg-white dark:text-black"
                  : enabled
                    ? "border-black/10 hover:bg-black/5 dark:border-white/15 dark:hover:bg-white/5"
                    : "cursor-not-allowed border-black/5 text-black/30 dark:border-white/10 dark:text-white/30"
              }`}
            >
              {b}
              {!enabled && (
                <span className="absolute -right-1.5 -top-1.5 rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-950/60 dark:text-amber-300">
                  Coming soon
                </span>
              )}
            </button>
          );
        })}
      </div>
      <WizardNav onBack={onBack} onNext={onNext} nextDisabled={!branch} />
    </div>
  );
}

function StepAcademicYear({
  academicYear,
  onChange,
  onNext,
  onBack,
}: {
  academicYear: string | null;
  onChange: (year: string) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  return (
    <div>
      <h2 className="mb-4 text-sm font-semibold">Academic year</h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {ACADEMIC_YEARS.map((y) => {
          const enabled = y === ENABLED_ACADEMIC_YEAR;
          const selected = academicYear === y;
          return (
            <button
              key={y}
              type="button"
              disabled={!enabled}
              onClick={() => enabled && onChange(y)}
              className={`relative rounded-lg border px-3 py-3 text-sm font-medium transition ${
                selected
                  ? "border-black bg-black text-white dark:border-white dark:bg-white dark:text-black"
                  : enabled
                    ? "border-black/10 hover:bg-black/5 dark:border-white/15 dark:hover:bg-white/5"
                    : "cursor-not-allowed border-black/5 text-black/30 dark:border-white/10 dark:text-white/30"
              }`}
            >
              {y}
              {!enabled && (
                <span className="absolute -right-1.5 -top-1.5 rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-950/60 dark:text-amber-300">
                  Coming soon
                </span>
              )}
            </button>
          );
        })}
      </div>
      <WizardNav onBack={onBack} onNext={onNext} nextDisabled={!academicYear} />
    </div>
  );
}

function StepSubjects({
  mdmName,
  oelName,
  electiveName,
  onChange,
  onBack,
  onSubmit,
  submitting,
  submitError,
}: {
  mdmName: string | null;
  oelName: string | null;
  electiveName: string | null;
  onChange: (patch: Partial<FormState>) => void;
  onBack: () => void;
  onSubmit: () => void;
  submitting: boolean;
  submitError: string | null;
}) {
  const canSubmit = Boolean(mdmName && oelName && electiveName) && !submitting;

  return (
    <div>
      <h2 className="mb-4 text-sm font-semibold">Subjects — TY Computer Engineering</h2>

      <div className="mb-4">
        <FieldLabel>Fixed subjects</FieldLabel>
        <div className="flex flex-wrap gap-2">
          {FIXED_SUBJECTS.map((s) => (
            <span
              key={s.slug}
              className="rounded-full bg-black/5 px-3 py-1 text-xs font-medium dark:bg-white/10"
            >
              {s.name}
            </span>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <SubjectSelect
          label="MDM (Multidisciplinary Minor)"
          options={MDM_OPTIONS}
          value={mdmName}
          onSelect={(v) => onChange({ mdmName: v })}
        />
        <SubjectSelect
          label="OEL (Open Elective)"
          options={OEL_OPTIONS}
          value={oelName}
          onSelect={(v) => onChange({ oelName: v })}
        />
        <SubjectSelect
          label="Elective"
          options={ELECTIVE_OPTIONS}
          value={electiveName}
          onSelect={(v) => onChange({ electiveName: v })}
        />
      </div>

      {submitError && <p className="mt-4 text-sm text-red-600 dark:text-red-400">{submitError}</p>}

      <WizardNav onBack={onBack} onNext={onSubmit} nextDisabled={!canSubmit} nextLabel={submitting ? "Submitting..." : "Finish"} />
    </div>
  );
}

function SubjectSelect({
  label,
  options,
  value,
  onSelect,
}: {
  label: string;
  options: readonly string[];
  value: string | null;
  onSelect: (value: string) => void;
}) {
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <select
        value={value ?? ""}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full rounded-lg border border-black/10 bg-transparent px-3 py-2 text-sm outline-none focus:border-black/30 dark:border-white/15 dark:focus:border-white/30"
      >
        <option value="" disabled>
          Select...
        </option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}
