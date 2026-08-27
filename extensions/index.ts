/**
 * pi-in4m-ai — registers the IN4M AI server (https://ai.in4m.au) as a pi
 * provider.
 *
 * The server speaks the OpenAI Chat Completions API (`POST /v1/chat/completions`,
 * `GET /v1/models`). This extension registers it via pi's config-form
 * `registerProvider` with `api: "openai-completions"` — pi wires its built-in
 * OpenAI-completions streaming (streamSimple etc.) itself, so there is no
 * `createProvider` / `openAICompletionsApi` runtime import here.
 *
 * Auth: the `IN4M_AI_API_KEY` env var, resolved by pi as the Bearer key
 * (`apiKey: "$IN4M_AI_API_KEY"`, `authHeader: true`). Set it in your shell or
 * pi config: `export IN4M_AI_API_KEY=...`.
 *
 * Models: discovered dynamically from `GET /v1/models`. Nothing model-related
 * is hardcoded — context window, max output, cost, reasoning support and compat
 * all come from the gateway (the source of truth), which reads them from the
 * same env the llm-server enforces. Change a server setting, refresh in pi, and
 * the catalog updates with no edit to this file.
 *
 * Thinking is model-dictated and binary (Nemotron `/think` vs `/no_think`
 * message markers). pi's 7 thinking levels collapse to off + on:
 * `thinkingLevelMap` hides everything except `off` and `high` (the user's
 * default level), and a `before_provider_request` hook injects the matching
 * marker into the last user message each turn so pi's thinking toggle actually
 * drives the gateway (the gateway's `thinking_enabled()` parses these
 * markers).
 *
 * Only type-only imports are used (erased at runtime), so the extension has no
 * runtime dependency on pi-ai — it loads cleanly under pi's jiti loader, which
 * aliases pi-ai to a bundled module for the root/compat specifiers only (deep
 * subpaths like `/api/openai-completions.lazy` are not aliased and fail to
 * resolve on a fresh install).
 */

import type { RefreshModelsContext } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

const PROVIDER_ID = "in4m-ai";
const PROVIDER_NAME = "IN4M AI";
/** Gateway base URL. Override with IN4M_AI_BASE_URL (e.g. for local testing). */
const BASE_URL = (
	process.env.IN4M_AI_BASE_URL ?? "https://ai.in4m.au/v1"
).replace(/\/+$/u, "");
const ENV_KEY = "IN4M_AI_API_KEY";

/** Subset of the gateway's enriched /v1/models entry. All fields optional
 * except id; the gateway always reports context/cost/reasoning/compat now. */
interface RemoteModel {
	id: string;
	name?: string;
	model_type?: string;
	context_window?: number;
	context_length?: number;
	max_tokens?: number;
	max_output_tokens?: number;
	reasoning?: { supported?: boolean; default_on?: boolean; levels?: string[] };
	cost?: {
		input_per_million_usd?: number;
		output_per_million_usd?: number;
		input_per_1k_cents?: number;
		output_per_1k_cents?: number;
	};
	compat?: {
		supports_reasoning_effort?: boolean;
		supports_developer_role?: boolean;
		max_tokens_field?: string;
	};
}

interface ModelsResponse {
	data?: RemoteModel[];
	models?: RemoteModel[];
}

/** Binary thinking map: only `off` and `high` are offered. `off` -> off
 * (no thinking), `high` -> on (thinking enabled). Every other level is hidden
 * (null) — Nemotron has no minimal/low/medium/xhigh/max tiers (reasoning levels
 * are model-dictated, not a setting we invent). `high` is kept (not a lower
 * level) so a user whose defaultThinkingLevel is "high" lands on "on". */
const BINARY_THINKING_MAP = {
	off: "off",
	minimal: null,
	low: null,
	medium: null,
	high: "on",
	xhigh: null,
	max: null,
} as const;

/** Turn a raw model id into a human-friendly display name. */
function prettyName(id: string): string {
	if (/nemotron/i.test(id)) {
		const m = id.match(/nemotron[-_]?([a-z0-9]+)?/i);
		if (m && m[1]) return `Nemotron ${m[1].replace(/\b\w/g, (c) => c.toUpperCase())}`;
		return "Nemotron";
	}
	const parts = id.split(/[\/_\-]/).filter(Boolean);
	if (parts.length <= 1) return id;
	return parts
		.map((p) => p.charAt(0).toUpperCase() + p.slice(1))
		.join(" ")
		.slice(0, 60);
}

/** Fetch the live model catalog from the server using the resolved credential
 * (or the IN4M_AI_API_KEY env var as a fallback). Every capability field is
 * read from the gateway (the source of truth). */
async function refreshIn4mModels(
	context: RefreshModelsContext,
): Promise<ProviderModelConfig[]> {
	const key =
		(context.credential?.type === "api_key" ? context.credential.key : undefined) ??
		process.env[ENV_KEY];
	if (!key) return [];

	const response = await fetch(`${BASE_URL}/models`, {
		method: "GET",
		headers: {
			Authorization: `Bearer ${key}`,
			Accept: "application/json",
		},
		signal: context.signal ?? undefined,
	});

	if (!response.ok) {
		let detail = "";
		try {
			detail = (await response.text()).slice(0, 200);
		} catch {}
		throw new Error(
			`IN4M AI /v1/models returned ${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`,
		);
	}

	const payload = (await response.json()) as ModelsResponse;
	const list = payload.data ?? payload.models ?? [];

	const models: ProviderModelConfig[] = list
		.filter(
			(m) =>
				m &&
				typeof m.id === "string" &&
				m.id.length > 0 &&
				// Only expose chat-capable models. The gateway lists STT (parakeet)
				// and TTS (chatterbox) alongside the LLM; those aren't chat models.
				(!m.model_type || m.model_type === "chat"),
		)
		.map((m) => {
			const reasoningSupported = m.reasoning?.supported === true;
			// pi cost is per-million-token USD (rates.input/1e6 * tokens).
			const c = m.cost ?? {};
			const compat = m.compat ?? {};
			return {
				id: m.id,
				name: m.name ?? prettyName(m.id),
				// Reasoning support is model-dictated and reported by the gateway.
				reasoning: reasoningSupported,
				// Binary model: collapse pi's 7 levels to off / on.
				...(reasoningSupported ? { thinkingLevelMap: BINARY_THINKING_MAP } : {}),
				input: ["text" as const],
				cost: {
					input: c.input_per_million_usd ?? 0,
					output: c.output_per_million_usd ?? 0,
					cacheRead: 0,
					cacheWrite: 0,
				},
				contextWindow: m.context_window ?? m.context_length ?? 131072,
				maxTokens: m.max_output_tokens ?? m.max_tokens ?? 8192,
				// Compat from the gateway where it reports it; conservative defaults
				// for the pi-internal fields the gateway doesn't speak.
				compat: {
					supportsStore: false,
					supportsDeveloperRole: compat.supports_developer_role ?? false,
					supportsReasoningEffort: compat.supports_reasoning_effort ?? false,
					supportsStrictMode: false,
					supportsUsageInStreaming: true,
					maxTokensField: compat.max_tokens_field === "max_completion_tokens" ? "max_completion_tokens" : "max_tokens",
					requiresToolResultName: false,
					thinkingFormat: "openai" as const,
				},
			} satisfies ProviderModelConfig;
		});

	// De-duplicate by id (some proxies repeat entries).
	const seen = new Set<string>();
	return models.filter((m) => (seen.has(m.id) ? false : (seen.add(m.id), true)));
}

export default function (pi: ExtensionAPI) {
	// Config-form registration: pi builds the provider and wires the built-in
	// "openai-completions" streaming (streamSimple) itself. Auth is the
	// IN4M_AI_API_KEY env var (Bearer). Models are discovered dynamically.
	pi.registerProvider(PROVIDER_ID, {
		name: PROVIDER_NAME,
		baseUrl: BASE_URL,
		apiKey: `$${ENV_KEY}`,
		authHeader: true,
		api: "openai-completions",
		models: [],
		refreshModels: refreshIn4mModels,
	});

	// Drive the gateway's binary thinking from pi's thinking toggle. The
	// gateway parses /think /no_think from user/system messages, so inject the
	// marker for the current level into the last user message of each request.
	pi.on("before_provider_request", (event, ctx) => {
		if (ctx?.model?.provider !== PROVIDER_ID) return;
		const level = pi.getThinkingLevel();
		const marker = level === "off" ? "/no_think" : "/think";
		const body = event.payload as {
			messages?: Array<{ role: string; content: unknown }>;
			temperature?: number;
			top_p?: number;
		} | undefined;
		if (body) {
			// Sampling is the platform's responsibility (the inference server is
			// sampling-agnostic). Nemotron's validated tool-call band is
			// 0.5 / 0.95 (40/40 reliable calls); set it as pi's default. A user's
			// explicit /temperature or per-request value always wins (only set
			// when absent).
			if (body.temperature === undefined) body.temperature = 0.5;
			if (body.top_p === undefined) body.top_p = 0.95;
		}
		const msgs = body?.messages;
		if (!Array.isArray(msgs)) return;
		for (let i = msgs.length - 1; i >= 0; i--) {
			const m = msgs[i];
			if (m.role !== "user") continue;
			if (typeof m.content === "string") {
				m.content = `${marker} ${m.content}`;
			} else if (Array.isArray(m.content)) {
				// multimodal: prepend a text part carrying the marker
				m.content = [{ type: "text", text: marker }, ...m.content];
			}
			break;
		}
		return event.payload;
	});

	// Small status command: `/in4m-ai` reports auth + discovered models.
	pi.registerCommand("in4m-ai", {
		description: "Show IN4M AI provider status (auth source + available models)",
		handler: async (_args, ctx) => {
			const registry = ctx.modelRegistry;
			const auth = await registry.getProviderAuth(PROVIDER_ID);
			if (!auth) {
				ctx.ui.notify(
					`IN4M AI: not configured. Set ${ENV_KEY} (e.g. export ${ENV_KEY}=sk-...).`,
					"warning",
				);
				return;
			}
			const all = await registry.getAvailable();
			const ids = all.filter((m) => m.provider === PROVIDER_ID).map((m) => m.id);
			ctx.ui.notify(
				ids.length
					? `IN4M AI: configured (${auth.source}). Models: ${ids.join(", ")}`
					: `IN4M AI: configured (${auth.source}). No models yet — run /models --refresh.`,
				"info",
			);
		},
	});
}