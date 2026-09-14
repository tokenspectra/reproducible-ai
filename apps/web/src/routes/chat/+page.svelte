<script lang="ts">
	import { signOut } from '@auth/sveltekit/client';
	import Composer from '$lib/components/Composer.svelte';
	import StatusPill from '$lib/components/StatusPill.svelte';
	import { ENGINE_LABELS, type Conversation, type EngineId, type StatusResponse } from '$lib/types';

	let { data } = $props();

	let status = $state<StatusResponse>(data.status);
	let conversations = $state<Conversation[]>(data.conversations);
	let current = $state<Conversation | null>(null);
	let engine = $state<EngineId>(data.status.default_engine ?? 'openrouter');
	let draft = $state('');
	let resumeToken = $state('');
	let busy = $state(false);
	let streamed = $state('');
	let errorText = $state('');
	let token = $state<string | null>(null);
	let copied = $state(false);
	let transcript = $state<HTMLDivElement | null>(null);

	// The GPU box is switched on and off by hand between sessions, so the pill
	// and the selector have to keep up without a reload.
	$effect(() => {
		const timer = setInterval(async () => {
			const res = await fetch('/api/status');
			if (!res.ok) return;
			status = await res.json();
			const live = status.engines.find((e) => e.id === engine);
			if (live && !live.up) engine = status.default_engine ?? 'openrouter';
		}, 10000);
		return () => clearInterval(timer);
	});

	$effect(() => {
		if ((streamed || current) && transcript) transcript.scrollTop = transcript.scrollHeight;
	});

	async function refreshList() {
		const res = await fetch('/api/conversations');
		if (res.ok) conversations = (await res.json()).conversations;
	}

	async function open(id: string) {
		errorText = '';
		token = null;
		const res = await fetch(`/api/conversations/${id}`);
		if (res.ok) current = await res.json();
	}

	/** Back to the empty state: the next message starts a fresh conversation. */
	function startNew() {
		current = null;
		token = null;
		errorText = '';
		draft = '';
		resumeToken = '';
	}

	async function fork() {
		if (!current) return;
		const res = await fetch(`/api/conversations/${current.id}/fork`, { method: 'POST' });
		if (!res.ok) {
			errorText = await res.text();
			return;
		}
		current = await res.json();
		token = null;
		await refreshList();
	}

	async function showToken() {
		if (!current) return;
		errorText = '';
		copied = false;
		const res = await fetch(`/api/conversations/${current.id}/token`);
		if (res.ok) token = (await res.json()).token;
		else errorText = 'No deterministic turn yet, so there is no state to resume.';
	}

	async function copyToken() {
		if (!token) return;
		await navigator.clipboard.writeText(token);
		copied = true;
		setTimeout(() => (copied = false), 1500);
	}

	/**
	 * One entry point for sending. With no conversation open it creates one
	 * first, so the first message starts the conversation rather than making
	 * the person press "new" beforehand.
	 */
	async function send() {
		const content = draft.trim();
		if (!content || busy) return;
		busy = true;
		errorText = '';
		streamed = '';
		token = null;

		try {
			if (!current) {
				const body = resumeToken.trim() ? { resume_token: resumeToken.trim() } : {};
				const created = await fetch('/api/conversations', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify(body)
				});
				if (!created.ok) {
					errorText = (await created.text()).slice(0, 300);
					return;
				}
				current = await created.json();
				resumeToken = '';
			}

			draft = '';
			current!.turns = [
				...(current!.turns ?? []),
				{
					index: current!.next_turn,
					role: 'user',
					content,
					engine: null,
					envelope_id: null,
					seed: null,
					finish_reason: null,
					evidence_kind: null,
					steps: 0,
					hash: null
				}
			];

			const res = await fetch(`/api/conversations/${current!.id}/messages`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ content, engine })
			});
			if (!res.ok || !res.body) {
				errorText = (await res.text()).slice(0, 300);
				return;
			}

			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';
			for (;;) {
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				const parts = buffer.split('\n\n');
				buffer = parts.pop() ?? '';
				for (const part of parts) {
					const line = part.split('\n').find((l) => l.startsWith('data:'));
					if (!line) continue;
					const payload = JSON.parse(line.slice(5).trim());
					if (payload.text) streamed += payload.text;
					if (payload.message) errorText = payload.message;
				}
			}
		} catch (e) {
			errorText = (e as Error).message;
		} finally {
			busy = false;
			streamed = '';
			if (current) await open(current.id);
			await refreshList();
		}
	}

	const empty = $derived(!current);
</script>

<svelte:head><title>overversion</title></svelte:head>

<div class="shell">
	<aside>
		<div class="brand">
			<strong>overversion</strong>
			<button class="ghost" onclick={startNew} title="Start a new conversation">New</button>
		</div>

		<nav>
			{#each conversations as c (c.id)}
				<button class="row" class:active={current?.id === c.id} onclick={() => open(c.id)}>
					<span class="row-title">{c.title}</span>
					{#if c.imported_from_token}
						<span class="badge" title="started from a resume token">token</span>
					{:else if c.forked_from}
						<span class="badge" title="forked">fork</span>
					{/if}
				</button>
			{:else}
				<p class="empty-list">Nothing yet. Your first message starts a conversation.</p>
			{/each}
		</nav>

		<div class="who">
			<span class="mono">{data.session?.user?.email}</span>
			<button class="ghost" onclick={() => signOut({ callbackUrl: '/' })}>Sign out</button>
		</div>
	</aside>

	<main>
		<header>
			<div class="head-left">
				{#if current}
					<span class="title">{current.title}</span>
					<button class="ghost" onclick={fork} title="Same seed, same history, separate conversation">
						Fork
					</button>
					<button class="ghost" onclick={showToken}>Token</button>
					{#if current.seed_source === 'external'}
						<span class="badge" title="seed came from a token or was chosen">seeded</span>
					{/if}
				{/if}
			</div>
			<StatusPill {status} />
		</header>

		{#if token}
			<div class="token">
				<div class="token-head">
					<span>
						Resume token. Paste it into an empty conversation, here or on someone else's
						account, to continue from this exact state.
					</span>
					<button class="ghost" onclick={copyToken}>{copied ? 'Copied' : 'Copy'}</button>
				</div>
				<code class="token-body">{token}</code>
			</div>
		{/if}

		{#if empty}
			<div class="hero">
				<h1>Same question, same answer.</h1>
				<p class="lede">
					Ask anything. In deterministic mode the reply is byte-identical every time for the same
					seed, no matter what else the GPU is doing.
				</p>
				<div class="hero-composer">
					<Composer {status} bind:engine bind:draft bind:resumeToken {busy} centered onsend={send} />
				</div>
				{#if errorText}<p class="error">{errorText}</p>{/if}
			</div>
		{:else}
			<div class="transcript" bind:this={transcript}>
				{#each current?.turns ?? [] as turn (turn.index)}
					<article class="turn {turn.role}">
						<div class="who-line">
							<span>{turn.role === 'user' ? 'You' : 'Assistant'}</span>
							{#if turn.engine}
								<span class="badge" class:good={turn.engine === 'reproducible'}>
									{ENGINE_LABELS[turn.engine]}
								</span>
								{#if turn.engine === 'reproducible'}
									<span class="meta mono">seed {turn.seed} · {turn.steps} steps hashed</span>
								{:else}
									<span class="meta">no guarantee</span>
								{/if}
							{/if}
						</div>
						<div class="body">{turn.content}</div>
					</article>
				{/each}
				{#if busy}
					<article class="turn assistant">
						<div class="who-line">
							<span>Assistant</span>
							<span class="badge">{ENGINE_LABELS[engine]}</span>
							<span class="meta">streaming…</span>
						</div>
						<div class="body">{streamed}</div>
					</article>
				{/if}
			</div>

			{#if errorText}<p class="error">{errorText}</p>{/if}

			<footer>
				<Composer {status} bind:engine bind:draft bind:resumeToken {busy} onsend={send} />
			</footer>
		{/if}
	</main>
</div>

<style>
	.shell { display: grid; grid-template-columns: 264px 1fr; height: 100dvh; }
	aside {
		display: flex;
		flex-direction: column;
		border-right: 1px solid var(--line);
		background: var(--panel);
		min-width: 0;
	}
	.brand {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 14px 14px 10px;
	}
	nav { flex: 1; overflow-y: auto; padding: 0 8px; display: flex; flex-direction: column; gap: 2px; }
	.row {
		display: flex;
		align-items: center;
		gap: 6px;
		width: 100%;
		text-align: left;
		background: transparent;
		border: 1px solid transparent;
		padding: 8px 10px;
		font-size: 13.5px;
	}
	.row:hover { background: var(--panel-2); }
	.row.active { background: var(--panel-2); border-color: var(--line); }
	.row-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
	.empty-list { color: var(--muted); font-size: 13px; padding: 8px 10px; }
	.who {
		border-top: 1px solid var(--line);
		padding: 10px 12px;
		display: flex;
		flex-direction: column;
		gap: 8px;
		font-size: 12px;
		color: var(--muted);
	}
	main { display: flex; flex-direction: column; min-width: 0; }
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		padding: 12px 18px;
		border-bottom: 1px solid var(--line);
		min-height: 56px;
	}
	.head-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
	.title { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.ghost { background: transparent; font-size: 13px; padding: 5px 10px; }
	.hero {
		flex: 1;
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		gap: 14px;
		padding: 24px;
	}
	.hero h1 { margin: 0; font-size: 27px; letter-spacing: -0.02em; text-align: center; }
	.lede { margin: 0; color: var(--muted); max-width: 520px; text-align: center; font-size: 14.5px; }
	.hero-composer { width: min(720px, 100%); margin-top: 8px; }
	.token { margin: 12px 18px 0; border: 1px solid var(--accent-dim); border-radius: var(--radius); background: var(--panel); }
	.token-head {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		align-items: center;
		padding: 9px 12px;
		font-size: 12.5px;
		color: var(--muted);
		border-bottom: 1px solid var(--line);
	}
	.token-body { display: block; padding: 10px 12px; word-break: break-all; font-size: 11.5px; color: var(--accent); }
	.transcript { flex: 1; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 16px; }
	.turn { max-width: 760px; }
	.turn.user { align-self: flex-end; }
	.who-line { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); margin-bottom: 5px; }
	.badge {
		font-size: 10.5px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		border: 1px solid var(--line);
		border-radius: 4px;
		padding: 1px 5px;
		color: var(--muted);
	}
	.badge.good { color: var(--accent); border-color: var(--accent-dim); }
	.meta { font-size: 11.5px; }
	.body { white-space: pre-wrap; background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 12px 14px; }
	.turn.user .body { background: var(--panel-2); }
	.error { margin: 10px 18px 0; color: var(--bad); font-size: 13px; }
	footer { padding: 12px 18px 16px; }
	@media (max-width: 720px) {
		.shell { grid-template-columns: 1fr; }
		aside { display: none; }
	}
</style>
