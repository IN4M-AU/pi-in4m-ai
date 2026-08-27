/**
 * pi-in4m-ai — registers the IN4M AI server (https://ai.in4m.au) as a pi
 * provider.
 *
 * The server speaks the OpenAI Chat Completions API
 * (`POST /v1/chat/completions`, `GET /v1/models`), so this extension uses
 * pi-ai's built-in `openAICompletionsApi()` streaming implementation — no
 * custom streaming code, no reimplementation. Tool calling, streaming, and
 * usage accounting flow through pi's standard OpenAI-completions path.
 *
 * Auth: interactive API-key login via `/login in4m-ai` (prompted for a
 * secret), with an `IN4M_AI_API_KEY` env-var fallback.
 *
 * Models: discovered dynamically from `GET /v1/models` after the key is
 * configured. The model list is persisted by pi's ModelsStore, so it survives
 * restarts and is available to `pi --list-models`.
 *
 * Nothing model-related is hardcoded here — context window, max output, cost,
 * reasoning support and compat all come from the gateway's /v1/models (which
 * reads them from the same env the llm-server enforces). Change a server
 * setting, refresh in pi, and the catalog updates with no edit to this file.
 *
 * Thinking is model-dictated and binary (Nemotron /think vs /no_think message
 * markers). pi's 7 thinking levels collapse to off + on: `thinkingLevelMap`
 * hides everything except `off` and `high` (the user's default level), and a
 * `before_provider_request` hook injects the matching marker into the last
 * user message each turn so pi's thinking toggle actually drives the gateway
 * (the gateway's thinking_enabled() parses these markers).
 */

import {
	createProvider,
	type Model,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PROVIDER_ID = "in4m-ai";
const PROVIDER_NAME = "IN4M AI";
/** Gateway base URL. Override with IN4M_AI_BASE_URL (e.g. for local testing). */
const BASE_URL = (
	process.env.IN4M_AI_BASE_URL ?? "https://ai.in4m.au/v1"
).replace(/\/+$/u, "");
const ENV_KEY = "IN4M_AI_API_KEY";

type OpenAIModel = Model<"openai-completions">;

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

/** Fetch the live model catalog from the server using the resolved credential.
 * Every capability field is read from the gateway (the source of truth). */
async function fetchIn4mModels(
	context: RefreshModelsContext,
): Promise<readonly OpenAIModel[]> {
	const key = context.credential?.type === "api_key" ? context.credential.key : undefined;
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

	const models: OpenAIModel[] = list
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
				api: "openai-completions" as const,
				provider: PROVIDER_ID,
				baseUrl: BASE_URL,
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
			} satisfies OpenAIModel;
		});

	// De-duplicate by id (some proxies repeat entries).
	const seen = new Set<string>();
	return models.filter((m) => (seen.has(m.id) ? false : (seen.add(m.id), true)));
}

export default function (pi: ExtensionAPI) {
	const provider = createProvider({
		id: PROVIDER_ID,
		name: PROVIDER_NAME,
		baseUrl: BASE_URL,
		api: openAICompletionsApi(),
		models: [],
		fetchModels: fetchIn4mModels,
		auth: {
			apiKey: {
				name: "IN4M AI API key",
				async login(interaction) {
					interaction.notify({
						type: "info",
						message: "Get an API key from the IN4M AI console at https://ai.in4m.au/console",
						links: [{ url: "https://ai.in4m.au/console", label: "IN4M AI console" }],
					});
					const key = await interaction.prompt({
						type: "secret",
						message: "Paste your IN4M AI API key:",
						placeholder: "sk-...",
					});
					if (!key || !key.trim()) throw new Error("No API key entered");
					return { type: "api_key", key: key.trim() };
				},
				async resolve({ ctx, credential }) {
					const stored = credential?.type === "api_key" ? credential.key : undefined;
					if (stored) {
						return { auth: { apiKey: stored }, source: "stored API key" };
					}
					const envKey = await ctx.env(ENV_KEY);
					if (envKey) {
						return { auth: { apiKey: envKey }, source: ENV_KEY };
					}
					return undefined;
				},
				async check({ ctx, credential }) {
					const stored = credential?.type === "api_key" ? credential.key : undefined;
					if (stored) return { type: "api_key", source: "stored API key" };
					const envKey = await ctx.env(ENV_KEY);
					if (envKey) return { type: "api_key", source: ENV_KEY };
					return undefined;
				},
			},
		},
	});

	pi.registerProvider(provider);

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
					`IN4M AI: not configured. Run /login in4m-ai or set ${ENV_KEY}.`,
					"warning",
				);
				return;
			}
			const all = await registry.getAvailable();
			const ids = all.filter((m) => m.provider === PROVIDER_ID).map((m) => m.id);
			const head = ids.slice(0, 6).join(", ");
			const more = ids.length > 6 ? ` (+${ids.length - 6} more)` : "";
			ctx.ui.notify(
				`IN4M AI: ${auth.source ?? "configured"} • ${ids.length} model${ids.length === 1 ? "" : "s"}${head ? `: ${head}` : ""}${more}`,
				"info",
			);
		},
	});
}