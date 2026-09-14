import { json, error } from '@sveltejs/kit';
import { api } from '$lib/server/api';
import type { Conversation } from '$lib/types';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async (event) => {
	const session = await event.locals.auth();
	if (!session?.user) error(401, 'not signed in');
	return json(
		await api<Conversation>(session, `/v1/conversations/${event.params.id}/fork`, {
			method: 'POST'
		}),
		{ status: 201 }
	);
};
