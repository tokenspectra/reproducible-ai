import { json, error } from '@sveltejs/kit';
import { api } from '$lib/server/api';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async (event) => {
	const session = await event.locals.auth();
	if (!session?.user) error(401, 'not signed in');
	try {
		return json(
			await api<{ token: string; payload: Record<string, unknown> }>(
				session,
				`/v1/conversations/${event.params.id}/token`
			)
		);
	} catch (e) {
		error(409, (e as Error).message);
	}
};
