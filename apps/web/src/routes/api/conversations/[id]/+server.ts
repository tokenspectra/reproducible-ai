import { json, error } from '@sveltejs/kit';
import { api } from '$lib/server/api';
import type { Conversation } from '$lib/types';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async (event) => {
	const session = await event.locals.auth();
	if (!session?.user) error(401, 'not signed in');
	return json(await api<Conversation>(session, `/v1/conversations/${event.params.id}`));
};

export const DELETE: RequestHandler = async (event) => {
	const session = await event.locals.auth();
	if (!session?.user) error(401, 'not signed in');
	await api<void>(session, `/v1/conversations/${event.params.id}`, { method: 'DELETE' });
	return new Response(null, { status: 204 });
};
