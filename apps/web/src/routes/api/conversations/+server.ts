import { json } from '@sveltejs/kit';
import { error } from '@sveltejs/kit';
import { api } from '$lib/server/api';
import type { Conversation } from '$lib/types';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async (event) => {
	const session = await event.locals.auth();
	if (!session?.user) error(401, 'not signed in');
	return json(await api<{ conversations: Conversation[] }>(session, '/v1/conversations'));
};

export const POST: RequestHandler = async (event) => {
	const session = await event.locals.auth();
	if (!session?.user) error(401, 'not signed in');
	const body = await event.request.text();
	try {
		return json(
			await api<Conversation>(session, '/v1/conversations', { method: 'POST', body }),
			{ status: 201 }
		);
	} catch (e) {
		error(400, (e as Error).message);
	}
};
