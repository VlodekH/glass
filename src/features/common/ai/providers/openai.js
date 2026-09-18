const OpenAI = require('openai');
const WebSocket = require('ws');

const OPENAI_API_BASE_URL = 'https://api.openai.com/v1';
const OPENAI_REALTIME_URL = 'wss://api.openai.com/v1/realtime';
const STT_SESSION_READY_TIMEOUT_MS = 15000;

function normalizeApiKey(key) {
  return typeof key === 'string' ? key.trim() : '';
}

function usesModernChatParameters(model) {
  return /^(?:gpt-[56](?:[.-]|$)|chat-latest$|o\d(?:-|$))/i.test(model || '');
}

function buildChatCompletionRequest({ model, messages, temperature, maxTokens, stream = false }) {
  const request = {
    model,
    messages,
    stream,
  };

  if (usesModernChatParameters(model)) {
    // Current reasoning model families reject legacy sampling parameters.
    request.max_completion_tokens = maxTokens;
  } else {
    request.temperature = temperature;
    request.max_tokens = maxTokens;
  }

  return request;
}

async function getApiError(response, providerName = 'OpenAI') {
  const errorData = await response.json().catch(() => ({}));
  const message = errorData.error?.message || response.statusText || 'Unknown API error';
  const error = new Error(`${providerName} API error (${response.status}): ${message}`);
  error.status = response.status;
  error.code = errorData.error?.code;
  error.type = errorData.error?.type;
  return error;
}

function buildTranscriptionSessionUpdate({
  model = 'gpt-live-transcribe',
  language,
  prompt = '',
} = {}) {
  const transcription = { model };
  if (prompt) transcription.prompt = prompt;

  // The live model accepts a list of expected languages. Older transcription
  // models use the singular language hint. Omit both to enable auto-detection.
  if (language) {
    if (model === 'gpt-live-transcribe') {
      transcription.languages = [language];
      transcription.delay = 'low';
    } else {
      transcription.language = language;
    }
  }

  return {
    type: 'session.update',
    session: {
      type: 'transcription',
      audio: {
        input: {
          format: {
            type: 'audio/pcm',
            rate: 24000,
          },
          transcription,
          turn_detection: {
            type: 'server_vad',
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
          },
          noise_reduction: {
            type: 'near_field',
          },
        },
      },
    },
  };
}


class OpenAIProvider {
    static async validateApiKey(key) {
        const normalizedKey = normalizeApiKey(key);
        if (!normalizedKey.startsWith('sk-')) {
            return { success: false, error: 'Invalid OpenAI API key format.' };
        }

        try {
            const response = await fetch(`${OPENAI_API_BASE_URL}/models`, {
                headers: { 'Authorization': `Bearer ${normalizedKey}` },
                signal: AbortSignal.timeout(15000),
            });

            if (response.ok) {
                return { success: true };
            }

            const error = await getApiError(response);
            // Restricted project keys can be valid while lacking permission to list
            // models. Save those keys and let the actual model request report any
            // missing Chat Completions or Realtime permission precisely.
            if (response.status === 403) {
                return { success: true, warning: error.message };
            }
            return { success: false, error: error.message };
        } catch (error) {
            console.error(`[OpenAIProvider] Network error during key validation:`, error);
            const message = error.name === 'TimeoutError'
              ? 'OpenAI API key validation timed out.'
              : 'A network error occurred during OpenAI API key validation.';
            return { success: false, error: message };
        }
    }
}


/**
 * Creates an OpenAI STT session
 * @param {object} opts - Configuration options
 * @param {string} opts.apiKey - OpenAI API key
 * @param {string} [opts.language='en'] - Language code
 * @param {object} [opts.callbacks] - Event callbacks
 * @param {boolean} [opts.usePortkey=false] - Whether to use Portkey
 * @param {string} [opts.portkeyVirtualKey] - Portkey virtual key
 * @returns {Promise<object>} STT session
 */
async function createSTT({ apiKey, model = 'gpt-live-transcribe', language, callbacks = {}, usePortkey = false, portkeyVirtualKey, ...config }) {
  const keyType = usePortkey ? 'vKey' : 'apiKey';
  const key = normalizeApiKey(usePortkey ? (portkeyVirtualKey || apiKey) : apiKey);

  const wsUrl = keyType === 'apiKey'
    ? OPENAI_REALTIME_URL
    : 'wss://api.portkey.ai/v1/realtime';

  const headers = keyType === 'apiKey'
    ? {
        'Authorization': `Bearer ${key}`,
      }
    : {
        'x-portkey-api-key': 'gRv2UGRMq6GGLJ8aVEB4e7adIewu',
        'x-portkey-virtual-key': key,
      };

  const ws = new WebSocket(wsUrl, { headers });

  return new Promise((resolve, reject) => {
    let settled = false;
    const readyTimeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      ws.close();
      reject(new Error('OpenAI transcription session did not become ready in time.'));
    }, STT_SESSION_READY_TIMEOUT_MS);

    const finishWithError = error => {
      callbacks.onerror?.(error);
      if (!settled) {
        settled = true;
        clearTimeout(readyTimeout);
        reject(error);
      }
    };

    const createSessionHandle = () => ({
      sendRealtimeInput: audioData => {
        if (ws.readyState !== WebSocket.OPEN) {
          throw new Error('OpenAI transcription session is not connected.');
        }
        ws.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: audioData,
        }));
      },
      keepAlive: () => {
        if (ws.readyState === WebSocket.OPEN) ws.ping();
      },
      close: () => {
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close(1000, 'Client initiated close.');
        }
      },
    });

    ws.on('open', () => {
      console.log('[OpenAI STT] WebSocket connected; configuring transcription session.');
      ws.send(JSON.stringify(buildTranscriptionSessionUpdate({
        model,
        language,
        prompt: config.prompt || '',
      })));
    });

    ws.on('message', data => {
      // ── 종료·하트비트 패킷 필터링 ──────────────────────────────
      const raw = data?.toString();
      if (!raw || raw === 'null' || raw === '[DONE]') return;

      let msg;
      try { msg = JSON.parse(raw); }
      catch { return; }                       // JSON 파싱 실패 무시

      if (!msg || typeof msg !== 'object') return;

      if (msg.type === 'error' || msg.error) {
        const apiError = msg.error || msg;
        const error = new Error(apiError.message || 'OpenAI transcription session error.');
        error.code = apiError.code;
        finishWithError(error);
        return;
      }

      if (!settled && (msg.type === 'session.updated' || msg.type === 'transcription_session.updated')) {
        settled = true;
        clearTimeout(readyTimeout);
        console.log(`[OpenAI STT] Transcription session ready with model ${model}.`);
        resolve(createSessionHandle());
      }

      msg.provider = 'openai';                // ← 항상 명시
      callbacks.onmessage?.(msg);
    });

    ws.on('error', error => {
      console.error('WebSocket error:', error.message);
      finishWithError(error);
    });

    ws.on('close', (code, reasonBuffer) => {
      const reason = reasonBuffer?.toString() || '';
      console.log(`WebSocket closed: ${code} ${reason}`);
      callbacks.onclose?.({ code, reason });
      if (!settled) {
        finishWithError(new Error(`OpenAI transcription connection closed before setup (${code} ${reason}).`));
      }
    });
  });
}

/**
 * Creates an OpenAI LLM instance
 * @param {object} opts - Configuration options
 * @param {string} opts.apiKey - OpenAI API key
 * @param {string} [opts.model='gpt-4.1'] - Model name
 * @param {number} [opts.temperature=0.7] - Temperature
 * @param {number} [opts.maxTokens=2048] - Max tokens
 * @param {boolean} [opts.usePortkey=false] - Whether to use Portkey
 * @param {string} [opts.portkeyVirtualKey] - Portkey virtual key
 * @returns {object} LLM instance
 */
function createLLM({ apiKey, model = 'gpt-4.1', temperature = 0.7, maxTokens = 2048, usePortkey = false, portkeyVirtualKey, ...config }) {
  const client = new OpenAI({ apiKey: normalizeApiKey(apiKey) });
  
  const callApi = async (messages) => {
    if (!usePortkey) {
      const response = await client.chat.completions.create(buildChatCompletionRequest({
        model,
        messages,
        temperature,
        maxTokens,
      }));
      return {
        content: response.choices[0]?.message?.content?.trim() || '',
        raw: response
      };
    } else {
      const fetchUrl = 'https://api.portkey.ai/v1/chat/completions';
      const response = await fetch(fetchUrl, {
        method: 'POST',
        headers: {
            'x-portkey-api-key': 'gRv2UGRMq6GGLJ8aVEB4e7adIewu',
            'x-portkey-virtual-key': portkeyVirtualKey || apiKey,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(buildChatCompletionRequest({
          model,
          messages,
          temperature,
          maxTokens,
        })),
      });

      if (!response.ok) {
        throw await getApiError(response, 'Portkey');
      }

      const result = await response.json();
      return {
        content: result.choices[0]?.message?.content?.trim() || '',
        raw: result
      };
    }
  };

  return {
    generateContent: async (parts) => {
      const messages = [];
      let systemPrompt = '';
      let userContent = [];
      
      for (const part of parts) {
        if (typeof part === 'string') {
          if (systemPrompt === '' && part.includes('You are')) {
            systemPrompt = part;
          } else {
            userContent.push({ type: 'text', text: part });
          }
        } else if (part.inlineData) {
          userContent.push({
            type: 'image_url',
            image_url: { url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}` }
          });
        }
      }
      
      if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
      if (userContent.length > 0) messages.push({ role: 'user', content: userContent });
      
      const result = await callApi(messages);

      return {
        response: {
          text: () => result.content
        },
        raw: result.raw
      };
    },
    
    // For compatibility with chat-style interfaces
    chat: async (messages) => {
      return await callApi(messages);
    }
  };
}

/** 
 * Creates an OpenAI streaming LLM instance
 * @param {object} opts - Configuration options
 * @param {string} opts.apiKey - OpenAI API key
 * @param {string} [opts.model='gpt-4.1'] - Model name
 * @param {number} [opts.temperature=0.7] - Temperature
 * @param {number} [opts.maxTokens=2048] - Max tokens
 * @param {boolean} [opts.usePortkey=false] - Whether to use Portkey
 * @param {string} [opts.portkeyVirtualKey] - Portkey virtual key
 * @returns {object} Streaming LLM instance
 */
function createStreamingLLM({ apiKey, model = 'gpt-4.1', temperature = 0.7, maxTokens = 2048, usePortkey = false, portkeyVirtualKey, ...config }) {
  const normalizedKey = normalizeApiKey(apiKey);
  return {
    streamChat: async (messages) => {
      const fetchUrl = usePortkey 
        ? 'https://api.portkey.ai/v1/chat/completions'
        : `${OPENAI_API_BASE_URL}/chat/completions`;
      
      const headers = usePortkey
        ? {
            'x-portkey-api-key': 'gRv2UGRMq6GGLJ8aVEB4e7adIewu',
            'x-portkey-virtual-key': portkeyVirtualKey || apiKey,
            'Content-Type': 'application/json',
          }
        : {
            Authorization: `Bearer ${normalizedKey}`,
            'Content-Type': 'application/json',
          };

      const response = await fetch(fetchUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(buildChatCompletionRequest({
          model,
          messages,
          temperature,
          maxTokens,
          stream: true,
        })),
      });

      if (!response.ok) {
        throw await getApiError(response, usePortkey ? 'Portkey' : 'OpenAI');
      }

      return response;
    }
  };
}

module.exports = {
    OpenAIProvider,
    buildChatCompletionRequest,
    buildTranscriptionSessionUpdate,
    createSTT,
    createLLM,
    createStreamingLLM
};
