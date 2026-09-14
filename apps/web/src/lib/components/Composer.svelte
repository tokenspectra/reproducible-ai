<script lang="ts">
	import EngineSelector from '$lib/components/EngineSelector.svelte';
	import type { EngineId, StatusResponse } from '$lib/types';

	let {
		status,
		engine = $bindable(),
		draft = $bindable(),
		resumeToken = $bindable(),
		busy = false,
		centered = false,
		onsend
	}: {
		status: StatusResponse;
		engine: EngineId;
		draft: string;
		resumeToken: string;
		busy?: boolean;
		centered?: boolean;
		onsend: () => void;
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
	.token-row { border-top: 1px solid var(--line); padding-top: 9px; }
	.token-row input { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
</style>
