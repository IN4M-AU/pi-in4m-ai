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
 */

import {
	createProvider,
	openAICompletionsApi,
	type Model,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PROVIDER_ID = "in4m-ai";
const PROVIDER_NAME = "IN4M AI";
const BASE_URL = "https://ai.in4m.au/v1";
const ENV_KEY = "IN4M_AI_API_KEY";

type OpenAIModel = Model<"openai-completions">;

interface RemoteModel {
	id: string;
	name?: string;
	context_window?: number;
	context_length?: number;
	max_tokens?: number;
	max_output_tokens?: number;
	object?: string;
	owned_by?: string;
}

interface ModelsResponse {
	data?: RemoteModel[];
	models?: RemoteModel[];
}

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

function inferContextWindow(id: string, reported?: number): number {
	if (reported && reported > 0) return reported;
	// NVIDIA Nemotron Ultra 253B / Nano 70B expose a 128K context window.
	if (/nemotron/i.test(id)) return 128000;
	return 128000;
}

function inferMaxTokens(id: string, reported?: number): number {
	if (reported && reported > 0) return reported;
	return 8192;
}

/** Fetch the live model catalog from the server using the resolved credential. */
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
			...(context.signal ? { signal: context.signal } : {}),
		},
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
		.filter((m) => m && typeof m.id === "string" && m.id.length > 0)
		.map((m) => {
			const contextWindow = inferContextWindow(m.id, m.context_window ?? m.context_length);
			const maxTokens = inferMaxTokens(m.id, m.max_tokens ?? m.max_output_tokens);
			return {
				id: m.id,
				name: m.name ?? prettyName(m.id),
				api: "openai-completions" as const,
				provider: PROVIDER_ID,
				baseUrl: BASE_URL,
				reasoning: false,
				input: ["text" as const],
				cost: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
				},
				contextWindow,
				maxTokens,
				// Explicit compat: this is an OpenAI-compatible gateway, not
				// api.openai.com. Force conservative defaults so tool calls,
				// max_tokens, and the system role serialize the way most
				// compatible proxies expect.
				compat: {
					supportsStore: false,
					supportsDeveloperRole: false,
					supportsReasoningEffort: false,
					supportsStrictMode: false,
					supportsUsageInStreaming: true,
					maxTokensField: "max_tokens" as const,
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

	// Small status command: `/in4m-ai` reports auth + discovered models.
	pi.registerCommand("in4m-ai", {
		description: "Show IN4M AI provider status (auth source + available models)",
		handler: async (_args, ctx) => {
			const registry = ctx.modelRegistry;
			const auth = await registry.getProviderAuth(PROVIDER_ID);
			if (!auth) {
				ctx.ui.notify(
					`IN4M AI: not configured. Run /login in4m-ai or set ${ENV_KEY}.`,
					"warn",
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