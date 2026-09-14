import { createHmac } from 'node:crypto';
import { API_BASE_URL, FRONTEND_SHARED_SECRET } from '$env/static/private';
import type { Session } from '@auth/sveltekit';

/**
 * Server-side proxy to the Python API.
 *
 * The browser never talks to api.overversion.com directly: this signs a
 * short-lived identity header with a secret shared with the backend, so
 * Google's tokens stay here and the backend stays free of OAuth.
 */
function identityHeader(email: string, name: string): string {
	const expires = Math.floor(Date.now() / 1000) + 300;
	const safeName = (name || '').replace(/[|\n]/g, ' ');
	const msg = `${email}\n${safeName}\n${expires}`;
	const sig = createHmac('sha256', FRONTEND_SHARED_SECRET).update(msg).digest('hex');
	return `${email}|${safeName}|${expires}|${sig}`;
}

export function apiHeaders(session: Session | null): Record<string, string> {
	const headers: Record<string, string> = { 'Content-Type': 'application/json' };
	const email = session?.user?.email;
	if (email) headers['X-Overversion-Identity'] = identityHeader(email, session?.user?.name ?? '');
	return headers;
}

export const apiUrl = (path: string) => `${API_BASE_URL.replace(/\/$/, '')}${path}`;

export async function api<T>(
	session: Session | null,
	path: string,
	init: RequestInit = {}
): Promise<T> {
	const res = await fetch(apiUrl(path), {
		...init,
		headers: { ...apiHeaders(session), ...(init.headers ?? {}) }
	});
	if (!res.ok) {
		const body = await res.text();
		throw new Error(`api ${res.status}: ${body.slice(0, 300)}`);
	}
	return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}
