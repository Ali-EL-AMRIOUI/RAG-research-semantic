/**
 * Centralized API client.
 *
 * IMPORTANT LIMITATION: NEXT_PUBLIC_API_KEY is inlined into the browser
 * JavaScript bundle at build time — anyone can read it via devtools
 * (Network tab or the bundled JS itself). This is NOT a secret from the
 * end user's perspective. It stops anonymous bots/scanners hitting the
 * API directly (the common case for an exposed backend with no auth at
 * all), but a motivated user of the frontend can still extract it and
 * call the API themselves.
 *
 * For real protection against an authenticated app's own users abusing
 * endpoints, the key must never reach the browser at all — route these
 * calls through a Next.js server-side API route that attaches the real
 * key itself, or move to per-user auth (login) instead of a shared
 * secret. Combined with the ingress rate limit (kubernetes/ingress.yaml),
 * this is a reasonable baseline for a single-operator/internal tool, not
 * a public multi-user product.
 */
import axios from 'axios';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: API_KEY ? { 'X-API-Key': API_KEY } : {},
});