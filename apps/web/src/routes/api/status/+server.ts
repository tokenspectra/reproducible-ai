import { json } from '@sveltejs/kit';
import { api } from '$lib/server/api';
import type { StatusResponse } from '$lib/types';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async (event) => {
	const session = await event.locals.auth();
	try {
		return json(await api<StatusResponse>(session, '/v1/status'));
	} catch {
		return json({ engines: [], default_engine: null } satisfies StatusResponse);
	}
};
