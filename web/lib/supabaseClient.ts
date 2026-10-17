import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  // Fails loudly at import time rather than producing confusing runtime
  // errors deep inside supabase-js if the env vars are missing.
  throw new Error(
    "NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY must be set (see web/.env.local)."
  );
}

/** Shared browser/server Supabase client, used across the app for auth and (if ever needed) direct table reads. */
export const supabase = createClient(supabaseUrl, supabaseAnonKey);
