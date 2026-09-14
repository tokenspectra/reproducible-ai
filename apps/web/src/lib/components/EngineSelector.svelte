<script lang="ts">
	import { ENGINE_BLURBS, ENGINE_LABELS, type EngineId, type StatusResponse } from '$lib/types';

	let {
		status,
		selected = $bindable()
	}: { status: StatusResponse; selected: EngineId } = $props();

	const order: EngineId[] = ['reproducible', 'standard', 'openrouter'];
	const byId = $derived(new Map(status.engines.map((e) => [e.id, e])));

	function reason(id: EngineId): string {
		const e = byId.get(id);
		if (!e) return 'not configured';
		if (e.up) return ENGINE_BLURBS[id];
		return id === 'openrouter'
			? `Unavailable: ${e.detail}`
			: `The GPU box is off. ${e.detail ? `(${e.detail})` : ''}`;
	}
</script>

<div class="engines" role="radiogroup" aria-label="Inference engine">
	{#each order as id (id)}
		{@const engine = byId.get(id)}
		{@const disabled = !engine?.up}
		<button
			type="button"
			role="radio"
			aria-checked={selected === id}
			class="engine"
			class:on={selected === id}
			{disabled}
			title={reason(id)}
			onclick={() => (selected = id)}
		>
			<span class="name">{ENGINE_LABELS[id]}</span>
			{#if id === 'reproducible'}<span class="tag">evidence</span>{/if}
			{#if disabled}<span class="tag off">off</span>{/if}
		</button>
	{/each}
</div>

<style>
	.engines { display: flex; gap: 6px; flex-wrap: wrap; }
	.engine {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 5px 11px;
		font-size: 13px;
		border-radius: 999px;
		background: transparent;
	}
	.engine.on { background: var(--accent-dim); border-color: var(--accent); }
	.name { white-space: nowrap; }
	.tag {
		font-size: 10.5px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--muted);
		border: 1px solid var(--line);
		border-radius: 4px;
		padding: 0 4px;
	}
	.tag.off { color: var(--warn); }
</style>
