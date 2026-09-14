export type EngineId = 'reproducible' | 'standard' | 'openrouter';

export interface EngineStatus {
	id: EngineId;
	up: boolean;
	detail: string;
	model: string | null;
	envelope_id: string | null;
	checked_at: number;
	capabilities: {
		deterministic: boolean;
		accepts_seed: boolean;
		returns_evidence: boolean;
		supports_min_p: boolean;
	};
}

export interface StatusResponse {
	engines: EngineStatus[];
	default_engine: EngineId | null;
}

export interface Turn {
	index: number;
	role: 'user' | 'assistant';
	content: string;
	engine: EngineId | null;
	envelope_id: string | null;
	seed: number | null;
	finish_reason: string | null;
	evidence_kind: string | null;
	steps: number;
	hash: string | null;
}

export interface Conversation {
	id: string;
	title: string;
	next_turn: number;
	seed_source: string;
	forked_from: string | null;
	forked_at_turn: number | null;
	imported_from_token: boolean;
	created_at: number;
	updated_at: number;
	turns?: Turn[];
}

export const ENGINE_LABELS: Record<EngineId, string> = {
	reproducible: 'Deterministic',
	standard: 'Baseline',
	openrouter: 'OpenRouter'
};

export const ENGINE_BLURBS: Record<EngineId, string> = {
	reproducible: 'Same seed, same answer, whatever else the GPU is doing. Evidence recorded.',
	standard: 'Same model and weights on the same box, determinism switched off.',
	openrouter: 'Qwen3-8B through OpenRouter. Always available, guarantees nothing.'
};
