<script lang="ts">
	import { signIn } from '@auth/sveltekit/client';
</script>

<svelte:head><title>overversion</title></svelte:head>

<main>
	<div class="card">
		<h1>overversion</h1>
		<p class="lede">
			A chat where the same question, with the same seed, gives you back exactly the same answer.
			Byte for byte, no matter what else the GPU is doing at the time.
		</p>
		<p class="detail">
			Fork a conversation and both copies answer identically. Copy its token, hand it to someone
			else, and they resume from your exact state.
		</p>
		<button class="primary" onclick={() => signIn('google', { callbackUrl: '/chat' })}>
			Continue with Google
		</button>
	</div>
</main>

<style>
	main {
		min-height: 100dvh;
		display: grid;
		place-items: center;
		padding: 24px;
	}
	.card {
		max-width: 460px;
		background: var(--panel);
		border: 1px solid var(--line);
		border-radius: 16px;
		padding: 32px;
	}
	h1 { margin: 0 0 14px; font-size: 26px; letter-spacing: -0.02em; }
	.lede { margin: 0 0 12px; }
	.detail { margin: 0 0 24px; color: var(--muted); font-size: 14px; }
	.primary { background: var(--accent); color: #0d1512; border-color: var(--accent); font-weight: 600; width: 100%; padding: 11px; }
</style>
