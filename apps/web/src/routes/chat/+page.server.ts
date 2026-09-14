import { redirect } from '@sveltejs/kit';
import { api } from '$lib/server/api';
import type { Conversation, StatusResponse } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async (event) => {
	const session = await event.locals.auth();
	if (!session?.user) redirect(303, '/');

	const [status, list] = await Promise.all([
		api<StatusResponse>(session, '/v1/status').catch(
			() => ({ engines: [], default_engine: null }) as StatusResponse
		),
		api<{ conversations: Conversation[] }>(session, '/v1/conversations').catch(() => ({
			conversations: []
		}))
	]);

	return { session, status, conversations: list.conversations };
};
