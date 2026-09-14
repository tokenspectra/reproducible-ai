<script lang="ts">
	import type { StatusResponse } from '$lib/types';
	let { status }: { status: StatusResponse } = $props();

	const box = $derived(status.engines.find((e) => e.id === 'reproducible'));
	const envelope = $derived(box?.envelope_id ?? null);
</script>

<div class="pill" class:up={box?.up} title={box?.detail || ''}>
	<span class="dot"></span>
	<span>{box?.up ? 'GPU box up' : 'GPU box down'}</span>
	{#if box?.up}
		<span class="sep">·</span>
		<span class="mono">{box.model}</span>
		{#if envelope}
			<span class="sep">·</span>
			<span class="mono env" title="envelope: the hardware and software this runs on">
				{envelope}
			</span>
		{/if}
	{/if}
</div>

<style>
	.pill {
		display: inline-flex;
		align-items: center;
		gap: 7px;
		padding: 5px 11px;
		border: 1px solid var(--line);
		border-radius: 999px;
		background: var(--panel-2);
		font-size: 12.5px;
		color: var(--muted);
	}
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--bad);
	}
	.up .dot { background: var(--accent); }
	.up { color: var(--text); }
	.sep { opacity: 0.4; }
	.env { color: var(--muted); }
</style>
