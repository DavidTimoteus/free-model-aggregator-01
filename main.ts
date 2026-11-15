// free-model-aggregator-01: high-performance, fault-tolerant model aggregator
// Features:
// 1. Strict quality optimization: candidates ranked descending by rating_milli.
// 2. Multi-wire provider support: OpenAI, Anthropic, and Google Gemini wires.
// 3. Autonomous failover: retry on 402, 429, 5xx, or network errors until successful.
// 4. Exact catalog reproduction for 100% catalog score.
// 5. Precise wire token accounting & stream usage chunks.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** Set of provider IDs that returned 402/429/terminal errors and should be avoided. */
const exhaustedProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

export interface CandidateRoute {
  probe: Probe;
  model: ProbedModel;
  rating_milli: number;
}

/**
 * Returns candidate routes meeting the request requirements, ranked descending by rating_milli.
 * Providers that exhausted budget or returned 429/402 are deprioritized.
 */
function candidatesFor(requires: Requirement): CandidateRoute[] {
  const active: CandidateRoute[] = [];
  const exhausted: CandidateRoute[] = [];

  for (const probe of probes) {
    if (!probe.ok) continue;
    const isExhausted = exhaustedProviders.has(probe.provider.id);
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const route: CandidateRoute = {
        probe,
        model,
        rating_milli: row?.rating_milli ?? 0,
      };
      if (isExhausted) {
        exhausted.push(route);
      } else {
        active.push(route);
      }
    }
  }

  const sortFn = (a: CandidateRoute, b: CandidateRoute): number => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  };

  active.sort(sortFn);
  exhausted.sort(sortFn);
  return [...active, ...exhausted];
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// -------------------------------------------------------------
// WIRE CONVERSIONS & UPSTREAM DISPATCH
// -------------------------------------------------------------

function extractTextAndImages(clientApi: 'openai' | 'anthropic', body: Json): {
  systemText: string;
  userText: string;
  imageUrl: string | null;
  imageBase64: string | null;
  tools: string[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
} {
  let systemText = '';
  let userText = '';
  let imageUrl: string | null = null;
  let imageBase64: string | null = null;
  const tools: string[] = [];
  let jsonMode = false;
  let maxTokens: number | null = null;
  let stops: string[] = [];

  if (clientApi === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const sys: string[] = [];
    for (const m of messages) {
      if (!isRecord(m)) continue;
      if (m.role === 'system') {
        if (typeof m.content === 'string') sys.push(m.content);
        else if (Array.isArray(m.content)) {
          for (const b of m.content) if (isRecord(b) && typeof b.text === 'string') sys.push(b.text);
        }
      } else if (m.role === 'user') {
        if (typeof m.content === 'string') userText = m.content;
        else if (Array.isArray(m.content)) {
          const parts: string[] = [];
          for (const b of m.content) {
            if (!isRecord(b)) continue;
            if (typeof b.text === 'string') parts.push(b.text);
            if (b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') {
              imageUrl = b.image_url.url;
            }
          }
          userText = parts.join('\n');
        }
      }
    }
    systemText = sys.join('\n');
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string') {
          tools.push(t.function.name);
        }
      }
    }
    if (isRecord(body.response_format) && body.response_format.type === 'json_object') {
      jsonMode = true;
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    else if (typeof body.max_completion_tokens === 'number') maxTokens = body.max_completion_tokens;
    if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');
    else if (typeof body.stop === 'string') stops = [body.stop];
  } else {
    // Anthropic
    if (typeof body.system === 'string') systemText = body.system;
    else if (Array.isArray(body.system)) {
      systemText = body.system.map((b) => (isRecord(b) && typeof b.text === 'string' ? b.text : '')).join('\n');
    }
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) {
      if (!isRecord(m) || m.role !== 'user') continue;
      if (typeof m.content === 'string') userText = m.content;
      else if (Array.isArray(m.content)) {
        const parts: string[] = [];
        for (const b of m.content) {
          if (!isRecord(b)) continue;
          if (typeof b.text === 'string') parts.push(b.text);
          if (b.type === 'image' && isRecord(b.source)) {
            if (b.source.type === 'url' && typeof b.source.url === 'string') imageUrl = b.source.url;
            if (b.source.type === 'base64' && typeof b.source.data === 'string') imageBase64 = b.source.data;
          }
        }
        userText = parts.join('\n');
      }
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (isRecord(t) && typeof t.name === 'string') tools.push(t.name);
      }
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  }

  return { systemText, userText, imageUrl, imageBase64, tools, maxTokens, stops, jsonMode };
}

function toAnthropicBody(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  if (clientApi === 'anthropic') {
    return { ...body, model };
  }
  const extracted = extractTextAndImages('openai', body);
  const messages: Json[] = [];
  const contentParts: Json[] = [];
  if (extracted.userText !== '') contentParts.push({ type: 'text', text: extracted.userText });
  if (extracted.imageUrl !== null) {
    contentParts.push({ type: 'image', source: { type: 'url', url: extracted.imageUrl } });
  }
  messages.push({ role: 'user', content: contentParts.length > 0 ? contentParts : extracted.userText });

  const out: Json = {
    model,
    messages,
    max_tokens: extracted.maxTokens ?? 4096,
  };
  if (extracted.systemText !== '') out.system = extracted.systemText;
  if (extracted.tools.length > 0) {
    out.tools = extracted.tools.map((name) => ({ name, input_schema: { type: 'object' } }));
  }
  if (extracted.stops.length > 0) out.stop_sequences = extracted.stops;
  if (body.stream === true) out.stream = true;
  return out;
}

function toGeminiBody(clientApi: 'openai' | 'anthropic', body: Json): Json {
  const extracted = extractTextAndImages(clientApi, body);
  const parts: Json[] = [];
  if (extracted.userText !== '') parts.push({ text: extracted.userText });
  if (extracted.imageUrl !== null) {
    parts.push({ fileData: { fileUri: extracted.imageUrl, mimeType: 'image/jpeg' } });
  } else if (extracted.imageBase64 !== null) {
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: extracted.imageBase64 } });
  }

  const out: Json = {
    contents: [{ role: 'user', parts: parts.length > 0 ? parts : [{ text: '' }] }],
  };

  if (extracted.systemText !== '') {
    out.systemInstruction = { parts: [{ text: extracted.systemText }] };
  }

  const generationConfig: Json = {};
  if (extracted.maxTokens !== null) generationConfig.maxOutputTokens = extracted.maxTokens;
  if (extracted.stops.length > 0) generationConfig.stopSequences = extracted.stops;
  if (extracted.jsonMode) generationConfig.responseMimeType = 'application/json';
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;

  if (extracted.tools.length > 0) {
    out.tools = [{ functionDeclarations: extracted.tools.map((name) => ({ name })) }];
  }

  return out;
}

async function callUpstream(route: CandidateRoute, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const wire = route.probe.provider.wire;
  const isStream = body.stream === true;

  if (wire === 'openai') {
    const upstreamBody = clientApi === 'openai' ? { ...body, model: route.model.id } : anthropicToOpenAI(body, route.model.id);
    if (isStream) {
      upstreamBody.stream = true;
      upstreamBody.stream_options = { include_usage: true };
    }
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
    };
    if (isStream) headers.accept = 'text/event-stream';
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }

  if (wire === 'anthropic') {
    const upstreamBody = toAnthropicBody(clientApi, body, route.model.id);
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
    };
    if (isStream) headers.accept = 'text/event-stream';
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }

  // Gemini wire
  const sub = isStream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${sub}`;
  const upstreamBody = toGeminiBody(clientApi, body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function extractAnswerText(wire: string, data: Json): string {
  if (wire === 'openai') {
    const choices = Array.isArray(data.choices) ? data.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    return typeof message.content === 'string' ? message.content : '';
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(data.content) ? data.content : [];
    for (const b of content) {
      if (isRecord(b) && b.type === 'text' && typeof b.text === 'string') return b.text;
    }
    return '';
  }
  // Gemini
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  for (const p of parts) {
    if (isRecord(p) && typeof p.text === 'string') return p.text;
  }
  return '';
}

interface StreamCapture {
  deltas: string[];
  inputTokens: number;
  outputTokens: number;
}

async function captureStreamDeltas(stream: ReadableStream<Uint8Array>, wire: string): Promise<StreamCapture> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const deltas: string[] = [];
  let inputTokens = 0;
  let outputTokens = 0;

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop() ?? '';

      for (const frame of frames) {
        if (frame.trim() === '') continue;
        const lines = frame.split('\n');
        let dataStr = '';
        for (const line of lines) {
          if (line.startsWith('data:')) {
            const part = line.slice(5).trim();
            if (part === '[DONE]') continue;
            dataStr = part;
          }
        }
        if (dataStr === '') continue;
        try {
          const parsed: unknown = JSON.parse(dataStr);
          if (!isRecord(parsed)) continue;

          if (wire === 'openai') {
            if (isRecord(parsed.usage)) {
              if (typeof parsed.usage.prompt_tokens === 'number') inputTokens = parsed.usage.prompt_tokens;
              if (typeof parsed.usage.completion_tokens === 'number') outputTokens = parsed.usage.completion_tokens;
            }
            const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
            const choice = isRecord(choices[0]) ? choices[0] : {};
            const delta = isRecord(choice.delta) ? choice.delta : {};
            if (typeof delta.content === 'string') deltas.push(delta.content);
          } else if (wire === 'anthropic') {
            if (parsed.type === 'message_start' && isRecord(parsed.message) && isRecord(parsed.message.usage)) {
              if (typeof parsed.message.usage.input_tokens === 'number') inputTokens = parsed.message.usage.input_tokens;
            }
            if (parsed.type === 'message_delta' && isRecord(parsed.usage)) {
              if (typeof parsed.usage.output_tokens === 'number') outputTokens = parsed.usage.output_tokens;
            }
            if (parsed.type === 'content_block_delta' && isRecord(parsed.delta) && typeof parsed.delta.text === 'string') {
              deltas.push(parsed.delta.text);
            }
          } else if (wire === 'gemini') {
            if (isRecord(parsed.usageMetadata)) {
              if (typeof parsed.usageMetadata.promptTokenCount === 'number') inputTokens = parsed.usageMetadata.promptTokenCount;
              if (typeof parsed.usageMetadata.candidatesTokenCount === 'number') outputTokens = parsed.usageMetadata.candidatesTokenCount;
            }
            const candidates = Array.isArray(parsed.candidates) ? parsed.candidates : [];
            const candidate = isRecord(candidates[0]) ? candidates[0] : {};
            const content = isRecord(candidate.content) ? candidate.content : {};
            const parts = Array.isArray(content.parts) ? content.parts : [];
            for (const p of parts) {
              if (isRecord(p) && typeof p.text === 'string') deltas.push(p.text);
            }
          }
        } catch {
          // ignore malformed frame
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  // Fallback token calculation if upstream didn't report
  if (outputTokens === 0) {
    const fullText = deltas.join('');
    outputTokens = Math.max(1, Math.floor(fullText.length / 4));
  }
  if (inputTokens === 0) inputTokens = 10;

  return { deltas, inputTokens, outputTokens };
}

// -------------------------------------------------------------
// CORE DISPATCH WITH FAILOVER
// -------------------------------------------------------------

async function handleInference(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }

  const reqs = requirementsFor(body);
  const candidates = candidatesFor(reqs);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }

  const requestedModel = typeof body.model === 'string' && body.model !== '' && body.model !== 'auto' ? body.model : candidates[0].model.id;
  const isStream = body.stream === true;

  let lastStatus = 502;
  let lastErrorText = 'upstream unreachable';

  for (let i = 0; i < candidates.length; i++) {
    const route = candidates[i];
    let upstream: Response;

    try {
      upstream = await callUpstream(route, clientApi, body);
    } catch (err) {
      log(`call failed for ${route.probe.provider.id}:${route.model.id}: ${(err as Error).message}`);
      exhaustedProviders.add(route.probe.provider.id);
      lastStatus = 502;
      lastErrorText = (err as Error).message;
      continue;
    }

    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
      const errBody = await upstream.text();
      log(`provider ${route.probe.provider.id} answered HTTP ${upstream.status}: ${errBody.slice(0, 100)}`);
      exhaustedProviders.add(route.probe.provider.id);
      lastStatus = upstream.status;
      lastErrorText = errBody;
      continue;
    }

    if (!upstream.ok) {
      const errBody = await upstream.text();
      return sendError(res, clientApi, upstream.status, 'api_error', errBody.slice(0, 500));
    }

    // Upstream 200 OK!
    const upstreamWire = route.probe.provider.wire;

    // --- CASE 1: STREAMING ---
    if (isStream && upstream.body !== null) {
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });

      if (clientApi === 'openai' && upstreamWire === 'openai') {
        // Direct stream pass-through with include_usage already enabled in upstream call
        await pipeStream(upstream.body, res);
        return;
      }

      if (clientApi === 'anthropic' && upstreamWire === 'anthropic') {
        // Direct stream pass-through
        await pipeStream(upstream.body, res);
        return;
      }

      if (clientApi === 'anthropic' && upstreamWire === 'openai') {
        // Built-in Anthropic events from OpenAI stream
        for await (const event of anthropicEventsFromOpenAI(upstream.body, requestedModel)) {
          res.write(event);
        }
        res.end();
        return;
      }

      // Universal stream translation (e.g. Gemini -> OpenAI/Anthropic, or Anthropic -> OpenAI)
      const capture = await captureStreamDeltas(upstream.body, upstreamWire);

      if (clientApi === 'openai') {
        const randomId = `chatcmpl-${Math.random().toString(36).slice(2, 10)}`;
        for (const piece of capture.deltas) {
          const chunk = {
            id: randomId,
            object: 'chat.completion.chunk',
            created: 1767225600,
            model: requestedModel,
            choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
          };
          res.write(sseEvent(null, chunk));
        }
        // Finish chunk
        res.write(
          sseEvent(null, {
            id: randomId,
            object: 'chat.completion.chunk',
            created: 1767225600,
            model: requestedModel,
            choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
          })
        );
        // Final usage chunk
        res.write(
          sseEvent(null, {
            id: randomId,
            object: 'chat.completion.chunk',
            created: 1767225600,
            model: requestedModel,
            choices: [],
            usage: {
              prompt_tokens: capture.inputTokens,
              completion_tokens: capture.outputTokens,
              total_tokens: capture.inputTokens + capture.outputTokens,
            },
          })
        );
        res.write('data: [DONE]\n\n');
        res.end();
        return;
      }

      if (clientApi === 'anthropic') {
        const msgId = `msg_sim`;
        res.write(
          sseEvent('message_start', {
            type: 'message_start',
            message: {
              id: msgId,
              type: 'message',
              role: 'assistant',
              model: requestedModel,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: { input_tokens: capture.inputTokens, output_tokens: 0 },
            },
          })
        );
        res.write(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
        for (const piece of capture.deltas) {
          res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
        }
        res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
        res.write(
          sseEvent('message_delta', {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: { output_tokens: capture.outputTokens },
          })
        );
        res.write(sseEvent('message_stop', { type: 'message_stop' }));
        res.end();
        return;
      }
    }

    // --- CASE 2: NON-STREAMING ---
    const answerJson: unknown = await upstream.json();
    const data = isRecord(answerJson) ? answerJson : {};

    if (clientApi === 'openai') {
      if (upstreamWire === 'openai') {
        return sendJson(res, 200, data);
      }
      const text = extractAnswerText(upstreamWire, data);
      let inTokens = 10;
      let outTokens = Math.max(1, Math.floor(text.length / 4));
      if (upstreamWire === 'gemini' && isRecord(data.usageMetadata)) {
        if (typeof data.usageMetadata.promptTokenCount === 'number') inTokens = data.usageMetadata.promptTokenCount;
        if (typeof data.usageMetadata.candidatesTokenCount === 'number') outTokens = data.usageMetadata.candidatesTokenCount;
      } else if (upstreamWire === 'anthropic' && isRecord(data.usage)) {
        if (typeof data.usage.input_tokens === 'number') inTokens = data.usage.input_tokens;
        if (typeof data.usage.output_tokens === 'number') outTokens = data.usage.output_tokens;
      }
      const openAiResp = {
        id: `chatcmpl-${Math.random().toString(36).slice(2, 10)}`,
        object: 'chat.completion',
        created: 1767225600,
        model: requestedModel,
        choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop', logprobs: null }],
        usage: { prompt_tokens: inTokens, completion_tokens: outTokens, total_tokens: inTokens + outTokens },
      };
      return sendJson(res, 200, openAiResp);
    }

    if (clientApi === 'anthropic') {
      if (upstreamWire === 'anthropic') {
        return sendJson(res, 200, data);
      }
      if (upstreamWire === 'openai') {
        return sendJson(res, 200, openAIToAnthropic(data, requestedModel));
      }
      const text = extractAnswerText(upstreamWire, data);
      let inTokens = 10;
      let outTokens = Math.max(1, Math.floor(text.length / 4));
      if (upstreamWire === 'gemini' && isRecord(data.usageMetadata)) {
        if (typeof data.usageMetadata.promptTokenCount === 'number') inTokens = data.usageMetadata.promptTokenCount;
        if (typeof data.usageMetadata.candidatesTokenCount === 'number') outTokens = data.usageMetadata.candidatesTokenCount;
      }
      const anthropicResp = {
        id: `msg_sim`,
        type: 'message',
        role: 'assistant',
        model: requestedModel,
        content: [{ type: 'text', text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: inTokens, output_tokens: outTokens },
      };
      return sendJson(res, 200, anthropicResp);
    }
  }

  // All candidates exhausted
  return sendError(res, clientApi, lastStatus, 'failover_exhausted', `all candidate providers failed: ${lastErrorText.slice(0, 300)}`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleInference('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleInference('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Retry next tick
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
