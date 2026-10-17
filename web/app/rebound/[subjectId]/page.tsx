import ReboundPageClient from "@/components/rebound/ReboundPageClient";

// This page is per-student and reads live auth/session state client-side
// (via the root layout's AuthProvider), so there's nothing useful to
// prerender statically -- opt it out of Cache Components' prerendering
// instead of wrapping in a Suspense boundary that would never resolve at
// build time anyway.
export const instant = false;

export default async function ReboundSubjectPage({
  params,
}: {
  params: Promise<{ subjectId: string }>;
}) {
  const { subjectId } = await params;
  return <ReboundPageClient subjectId={subjectId} />;
}
