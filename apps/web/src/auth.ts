import { SvelteKitAuth } from '@auth/sveltekit';
import Google from '@auth/sveltekit/providers/google';
import { AUTH_GOOGLE_ID, AUTH_GOOGLE_SECRET, AUTH_SECRET } from '$env/static/private';

export const { handle, signIn, signOut } = SvelteKitAuth({
	providers: [Google({ clientId: AUTH_GOOGLE_ID, clientSecret: AUTH_GOOGLE_SECRET })],
	secret: AUTH_SECRET,
	trustHost: true,
	callbacks: {
		session({ session, token }) {
			if (session.user && token.sub) session.user.id = token.sub;
			return session;
		}
	}
});
