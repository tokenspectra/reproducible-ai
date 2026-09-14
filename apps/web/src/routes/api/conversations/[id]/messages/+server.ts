import { error } from '@sveltejs/kit';
import { apiHeaders, apiUrl } from '$lib/server/api';
import type { RequestHandler } from './$types';

/** Pipes the backend's server-sent events straight through to the browser. */
export const POST: RequestHandler = async (event) => {
	const session = await event.locals.auth();
	if (!session?.user) error(401, 'not signed in');

	const upstream = await fetch(apiUrl(`/v1/conversations/${event.params.id}/messages`), {
		method: 'POST',
		headers: apiHeaders(session),
		body: await event.request.text()
	});

	if (!upstream.ok || !upstream.body) {
		error(upstream.status, (await upstream.text()).slice(0, 300));
	}

	return new Response(upstream.body, {
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			Connection: 'keep-alive'
		}
	});
};
