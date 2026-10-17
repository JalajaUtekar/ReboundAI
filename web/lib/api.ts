import { supabase } from "./supabaseClient";

/**
 * Calls a same-origin backend API route (e.g. `/api/me`, `/api/predict_risk`)
 * with the current Supabase session's access token attached as a Bearer
 * token, matching every `api/*.py` endpoint's `Authorization` expectation
 * (see `api/_lib/auth.py`).
 *
 * Paths are same-origin (no absolute URL needed) because the root
 * `vercel.json` deploys the Next.js app and the Python API routes together.
 *
 * Throws an `Error` whose message is the backend's `{"error": "..."}` body
 * on any non-2xx response, so callers can just `try/catch` and show
 * `error.message`.
 */
export async function callApi<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const { data } = await supabase.auth.getSession();
  const accessToken = data.session?.access_token;

  const headers = new Headers(options.headers);
  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(path, { ...options, headers });

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    // Non-JSON body (e.g. an empty 204) -- leave payload as null.
  }

  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload
        ? String(payload.error)
        : `Request to ${path} failed with status ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}
