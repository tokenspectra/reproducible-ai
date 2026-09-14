<script lang="ts">
	import EngineSelector from '$lib/components/EngineSelector.svelte';
	import type { EngineId, StatusResponse } from '$lib/types';

	let {
		status,
		engine = $bindable(),
		draft = $bindable(),
		resumeToken = $bindable(),
		tokenMode = $bindable(),
		busy = false,
		centered = false,
		onsend,
		onresume
	}: {
		status: StatusResponse;
		engine: EngineId;
		draft: string;
		resumeToken: string;
		tokenMode: 'seed' | 'state';
		busy?: boolean;
		centered?: boolean;
		onsend: () => void;
		onresume?: () => void;
	} = $props();

	let showToken = $state(false);

	function onKey(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			if (draft.trim() && !busy) onsend();
		}
	}
</script>

<div class="composer" class:centered>
	<textarea
		rows={centered ? 3 : 2}
		placeholder="Ask something…"
		bind:value={draft}
		onkeydown={onKey}
		disabled={busy}
	></textarea>

	<div class="controls">
		<EngineSelector {status} bind:selected={engine} />
		<div class="right">
			{#if centered}
				<button
					type="button"
					class="ghost"
					aria-pressed={showToken}
					onclick={() => (showToken = !showToken)}
					title="Continue from someone else's exact state"
				>
					{resumeToken ? 'Token set' : 'Resume token'}
				</button>
			{/if}
			<button class="send" onclick={onsend} disabled={busy || !draft.trim()}>
				{busy ? 'Sending…' : 'Send'}
			</button>
		</div>
	</div>

	{#if centered && showToken}
		<div class="token-row">
			<input bind:value={resumeToken} placeholder="rai1.…  (leave empty for a fresh random seed)" />
		</div>
		<div class="token-modes" role="radiogroup" aria-label="What to take from the token">
			<button
				type="button"
				role="radio"
				aria-checked={tokenMode === 'seed'}
				class="mode"
				class:on={tokenMode === 'seed'}
				onclick={() => (tokenMode = 'seed')}
			>
				Same seed, empty chat
			</button>
			<button
				type="button"
				role="radio"
				aria-checked={tokenMode === 'state'}
				class="mode"
				class:on={tokenMode === 'state'}
				onclick={() => (tokenMode = 'state')}
			>
				Continue that conversation
			</button>
			<button
				type="button"
				class="ghost"
				disabled={!resumeToken.trim() || busy}
				onclick={() => onresume?.()}
			>
				Open now
			</button>
		</div>
		<p class="token-hint">
			{tokenMode === 'seed'
				? 'Starts an empty conversation on the same seed. Ask the same first question and you get the same answer, word for word.'
				: 'Copies the whole history and carries on from where the token was taken.'}
		</p>
	{/if}
</div>

<style>
	.composer {
		border: 1px solid var(--line);
		border-radius: 14px;
		background: var(--panel);
		padding: 10px 10px 9px;
		display: flex;
		flex-direction: column;
		gap: 9px;
	}
	.composer.centered { padding: 14px; }
	textarea {
		resize: none;
		border: none;
		background: transparent;
		padding: 4px 4px 0;
	}
	textarea:focus { outline: none; }
	.controls {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		flex-wrap: wrap;
	}
	.right { display: flex; align-items: center; gap: 8px; }
	.ghost { background: transparent; font-size: 13px; padding: 5px 10px; }
	.ghost[aria-pressed='true'] { border-color: var(--accent-dim); color: var(--accent); }
	.send { background: var(--accent); color: #0d1512; border-color: var(--accent); font-weight: 600; }
	.token-row {
		border-top: 1px solid var(--line);
		padding-top: 9px;
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.token-row input { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
	.token-modes { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
	.mode {
		padding: 4px 10px;
		font-size: 12.5px;
		border-radius: 999px;
		background: transparent;
	}
	.mode.on { background: var(--accent-dim); border-color: var(--accent); }
	.token-modes .ghost { margin-left: auto; white-space: nowrap; }
	.token-hint { margin: 0; font-size: 11.5px; color: var(--muted); }
</style>
