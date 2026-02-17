# LLM Model Queries

A collection of Python wrapper functions for querying large language models from multiple providers through a unified, consistent interface. Each function handles client initialization, parameter validation, token usage tracking, and cost estimation, so that switching between providers requires minimal code changes.

## Supported Providers and Models

| Provider | Function | Default Model | Available Models |
|----------|----------|---------------|------------------|
| OpenAI | `openai_query()` | `gpt-5.2` | `gpt-5.2`, `gpt-5.2-pro`, `gpt-5-mini`, `o1-pro` |
| Perplexity | `perplexity_query()` | `sonar` | `sonar`, `sonar-pro`, `sonar-reasoning-pro`, `sonar-deep-research` |
| DeepSeek | `deepseek_query()` | `deepseek-chat` | `deepseek-chat`, `deepseek-reasoner` |
| xAI (Grok) | `grok_query()` | `grok-4-1-fast` | `grok-4`, `grok-4-fast`, `grok-4-1-fast`, `grok-3`, `grok-3-mini`, plus non-reasoning variants |
| Moonshot (Kimi) | `kimi_query()` | `kimi-k2.5` | `kimi-k2.5`, `kimi-k2-0905`, `kimi-k2-thinking`, `kimi-k2-thinking-turbo`, `kimi-k2-turbo` |

## Prerequisites

### What is an API Key?

An API key is a unique string of characters that authenticates your requests to a provider's servers. It is essentially a password that identifies you as an authorized user. Each provider issues its own key, and you will need a separate key for each provider you want to use. API keys are associated with a billing account: each query you make will incur a small cost based on the number of tokens processed.

### Obtaining API Keys

To obtain an API key, create a developer account with the relevant provider. The links below will direct you to each provider's API platform:

- **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Perplexity**: [https://www.perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)
- **DeepSeek**: [https://platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
- **xAI (Grok)**: [https://console.x.ai](https://console.x.ai)
- **Moonshot (Kimi)**: [https://platform.moonshot.ai](https://platform.moonshot.ai)

You do not need keys for all providers. Only obtain keys for the providers you intend to use.

## Installation

### 1. Install Python Dependencies

```bash
pip install openai perplexity python-dotenv numpy xai-sdk
```

Note: `xai-sdk` is only required if you plan to use the Grok wrapper. The DeepSeek and Kimi wrappers use the `openai` package with a custom base URL, so no additional SDK is needed for those providers.

### 2. Set Up Your API Keys

API keys should **never** be hard-coded into your scripts or committed to version control. Instead, store them in a `.env` file that is loaded at runtime.

**Step 1.** In the same directory as `LLM_model_queries.py`, create a file named `.env` (note the leading dot):

```
OPENAI_API_KEY=sk-your-openai-key-here
PERPLEXITY_API_KEY=pplx-your-perplexity-key-here
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
XAI_API_KEY=your-xai-key-here
MOONSHOT_API_KEY=sk-your-moonshot-key-here
```

Replace the placeholder values with your actual keys. Only include the keys for the providers you plan to use.

**Step 2.** If you are using Git, add `.env` to your `.gitignore` file to prevent accidentally committing your keys:

```
# .gitignore
.env
```

**How this works:** The script calls `load_dotenv()` at import time, which reads the `.env` file and sets each line as an environment variable. When a wrapper function runs, it retrieves the appropriate key via `os.environ["PROVIDER_API_KEY"]`. If the key is missing, Python will raise a `KeyError`.

### 3. Import the Functions

```python
from LLM_model_queries import openai_query, perplexity_query, deepseek_query, grok_query, kimi_query
```

## Usage

### Basic Queries

Every wrapper function requires at minimum a `prompt` string and returns a dictionary containing the model's response text, token usage, and an estimated cost.

```python
# Query OpenAI GPT-5.2 (default model)
result = openai_query("What is the photoelectric effect?")
print(result['text'])

# Query DeepSeek
result = deepseek_query("Explain the Fourier transform in simple terms.")
print(result['text'])

# Query Perplexity (includes web search by default)
result = perplexity_query("What were the major physics breakthroughs in 2025?")
print(result['text'])
print(result['citations'])  # URLs of sources used

# Query Grok
result = grok_query("Summarize the latest developments in quantum computing.")
print(result['text'])

# Query Kimi
result = kimi_query("Derive the Euler-Lagrange equation from first principles.")
print(result['text'])
```

### Selecting a Model

Pass the `model` parameter to use a specific model variant. Each function will raise a `ValueError` if an invalid model name is provided.

```python
result = openai_query("Solve this integral.", model='gpt-5-mini')
result = deepseek_query("Prove this theorem.", model='deepseek-reasoner')
result = grok_query("Explain general relativity.", model='grok-4')
result = kimi_query("Analyze this dataset.", model='kimi-k2-0905')
```

### System Prompts

A system prompt provides persistent instructions that guide the model's behavior throughout the interaction. In the OpenAI wrapper, this parameter is called `instructions`; in all other wrappers it is `system_prompt`.

```python
result = openai_query(
    "What is dark matter?",
    instructions="You are a physics professor. Explain concepts rigorously but accessibly."
)

result = deepseek_query(
    "What is dark matter?",
    system_prompt="You are a physics professor. Explain concepts rigorously but accessibly."
)
```

### Reasoning / Thinking Mode

Several providers offer models that perform explicit chain-of-thought reasoning before producing their final answer. The mechanism varies by provider.

```python
# OpenAI: control reasoning effort (low, medium, high, xhigh)
result = openai_query("Solve this problem step by step.", reasoning_effort='high')

# DeepSeek: toggle thinking on/off
result = deepseek_query("Prove that the square root of 2 is irrational.", thinking=True)
print(result['reasoning'])  # the chain-of-thought trace
print(result['text'])       # the final answer

# Kimi: thinking is enabled by default
result = kimi_query("Derive the wave equation.", thinking=True)
print(result['reasoning'])

# Grok: reasoning effort is only supported on grok-3-mini
result = grok_query("Solve this puzzle.", model='grok-3-mini', reasoning_effort='high')
```

### Streaming

All wrappers support streaming, which prints the response incrementally as it is generated. This is useful for long responses where you do not want to wait for the full completion.

```python
result = openai_query("Write a long essay on thermodynamics.", stream=True)
# Output is printed token-by-token as it arrives.
# The returned dict contains the full text in result['text'].
```

Note: When streaming is enabled, token usage statistics and cost estimates are generally not available.

### Verbose Output

By default, all wrappers print the response text, token usage, and cost estimate to the console. Set `verbose=False` to suppress this output, which is useful when running queries programmatically or in batch.

```python
result = openai_query("What is entropy?", verbose=False)
# Nothing is printed; access result['text'], result['cost_estimate'], etc.
```

### Web Search

Some providers support tool-augmented generation with web search.

```python
# OpenAI: pass the tool parameter
result = openai_query("What happened at SPIE Photonics West 2026?", tool='web_search')

# Perplexity: web search is built in; control context size and recency
result = perplexity_query(
    "Latest results from the LHC",
    search_context_size='high',
    search_recency_filter='week'
)

# Perplexity: restrict search to specific domains
result = perplexity_query(
    "New results on gravitational waves",
    search_domain_filter=['arxiv.org', 'nature.com']
)

# Kimi: use the built-in web search tool
result = kimi_query(
    "What is the current price of gold?",
    tools=[{"type": "builtin_function", "function": {"name": "$web_search"}}]
)
```

### Temperature and Sampling

Temperature and `top_p` control the randomness of the model's output. Lower temperature values produce more deterministic responses; higher values increase diversity. Not all models support these parameters (reasoning models generally do not), and the wrappers will raise a `ValueError` if you attempt to set them on an unsupported model.

```python
# More deterministic output
result = deepseek_query("Classify this image.", temperature=0.2)

# More creative output
result = kimi_query("Write a poem about photons.", thinking=False, temperature=1.2)
```

### Structured Output

Some wrappers support requesting structured responses (e.g., JSON).

```python
result = openai_query(
    "List the planets in the solar system with their masses.",
    response_format={"type": "json_object"}
)
```

For Grok, structured output uses Pydantic models:

```python
from pydantic import BaseModel

class Planet(BaseModel):
    name: str
    mass_kg: float

result = grok_query("What is the mass of Jupiter?", response_format=Planet)
print(result['parsed'])  # a Planet instance
```

## Return Values

All wrapper functions return a dictionary. The exact keys vary slightly by provider, but the common structure is:

| Key | Type | Description |
|-----|------|-------------|
| `text` | `str` | The model's response text. |
| `input_tokens` / `prompt_tokens` | `int` | Number of tokens in the input prompt. |
| `output_tokens` / `completion_tokens` | `int` | Number of tokens in the model's response. |
| `cached_tokens` | `int` | Number of input tokens served from cache (reduces cost). |
| `total_tokens` | `int` | Total tokens consumed. |
| `cost_estimate` | `float` | Estimated cost in USD for the query. |
| `response` | object | The raw API response object for advanced use. |
| `reasoning` | `str` or `None` | Chain-of-thought trace (DeepSeek, Kimi only). |
| `citations` | `list` | Source URLs (Perplexity only). |

## Cost Estimation

Each wrapper includes hardcoded per-token pricing and computes an estimated cost for every query. These estimates account for cache hits (cheaper) versus cache misses, and for per-request fees where applicable (e.g., Perplexity). The cost estimate is printed when `verbose=True` and is always available in the returned dictionary under `cost_estimate`.

**Important:** Pricing data is embedded in the source code and may become outdated. Verify current pricing at the provider's documentation before relying on these estimates for budgeting.

## Error Handling

All wrappers perform input validation before making any API call. Invalid model names, unsupported parameter combinations (e.g., setting `temperature` on a reasoning model), and unrecognized tool names will raise a `ValueError` with a descriptive message. Missing API keys will raise a `KeyError`.

```python
try:
    result = openai_query("Hello", model='nonexistent-model')
except ValueError as e:
    print(e)  # "Invalid model 'nonexistent-model'. Must be one of [...]"
```

## Internal Structure of the Query Functions

All five wrapper functions follow the same sequential architecture. Understanding this shared structure makes the codebase easier to read, modify, and extend to new providers.

### 1. Model Configuration Dictionary

Each function defines a `MODEL_CONFIG` dictionary at the top of its body. This dictionary maps model name strings to a nested dictionary of metadata: per-token pricing (input, cached input, and output), context window length, and flags indicating which parameters the model supports (e.g., whether `temperature` is allowed, whether the model is a reasoning variant). All validation and cost computation later in the function references this dictionary.

```python
MODEL_CONFIG = {
    'gpt-5.2': {
        'reasoning_options': ['low', 'medium', 'high', 'xhigh'],
        'supports_temperature': False,
        'cost_in': 1.75 / 1e6,
        'cost_cached': 0.175 / 1e6,
        'cost_out': 14 / 1e6,
    },
    # ...
}
```

Pricing is stored in cost-per-token (i.e., the per-million-token rate divided by 1e6), so that the cost calculation later reduces to a simple product of token counts and per-token rates.

### 2. Input Validation

Before any network call is made, the function validates all user-supplied parameters against `MODEL_CONFIG`. This includes checking that the model name is a valid key, that the reasoning effort level (if applicable) is among the options the model supports, and that parameters like `temperature` or `frequency_penalty` are not being passed to models that ignore or reject them. Invalid inputs raise a `ValueError` with a descriptive message. This fail-fast approach ensures that errors are caught locally rather than surfacing as opaque API errors from the provider.

### 3. Client Initialization

Each function constructs an API client using the appropriate SDK. The API key is read from the environment via `os.environ["PROVIDER_API_KEY"]`, which was populated by the `load_dotenv()` call at module import time. Several providers (DeepSeek, Kimi) use OpenAI-compatible APIs, so their wrappers instantiate the `OpenAI` client with a custom `base_url` pointing to that provider's endpoint. Grok uses its own `xai_sdk.Client`.

```python
# DeepSeek example: OpenAI client with a custom base URL
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
```

### 4. Request Construction

Each function builds a keyword-arguments dictionary (`kwargs`) that will be unpacked into the API call. Required parameters (model name, prompt/messages, stream flag) are set unconditionally. Optional parameters (system prompt, temperature, tools, max tokens, etc.) are added to the dictionary only if the caller provided a non-`None` value. This conditional insertion pattern keeps the API request minimal by default and avoids sending parameters the user did not explicitly set.

For providers using the chat completions interface (DeepSeek, Perplexity, Kimi), the prompt and optional system prompt are assembled into a `messages` list in the standard `[{"role": "system", ...}, {"role": "user", ...}]` format. OpenAI's Responses API uses a flat `input` field instead.

### 5. API Call

The constructed `kwargs` dictionary is passed to the provider's SDK method:

```python
response = client.responses.create(**kwargs)       # OpenAI Responses API
response = client.chat.completions.create(**kwargs) # DeepSeek, Perplexity, Kimi
response = chat.sample()                            # Grok (xai_sdk)
```

### 6. Response Handling (Streaming vs. Non-Streaming)

Every function branches on the `stream` parameter immediately after the API call.

**Streaming path:** The function iterates over chunks from the response generator, accumulating text (and reasoning tokens, where applicable) into a list. If `verbose=True`, each chunk is printed to the console as it arrives using `flush=True` to ensure immediate output. Once the stream is exhausted, the collected fragments are joined and returned. Token usage and cost estimates are generally not available in streaming mode, so the returned dictionary contains only `text` (and `reasoning`, where applicable) along with the raw `response` object.

**Non-streaming path:** The function extracts the complete response text from the response object, then proceeds to token extraction and cost computation.

### 7. Token Extraction and Cost Computation

In the non-streaming path, the function reads token counts from the response's `usage` object. The exact field names vary by provider (e.g., `input_tokens` vs. `prompt_tokens`, `cached_tokens` vs. `prompt_cache_hit_tokens`), but each function maps these to a consistent set of local variables. Where the provider reports cache hit/miss breakdowns, these are extracted via `getattr` with a fallback of `0` to guard against missing fields.

The estimated cost is then computed as a weighted sum of the token counts and the per-token rates from `MODEL_CONFIG`:

```
cost = (cache_miss_tokens * cost_per_input_token)
     + (cached_tokens * cost_per_cached_token)
     + (output_tokens * cost_per_output_token)
```

Some providers add additional cost components. Perplexity includes a per-request fee that varies with `search_context_size`, and its `sonar-deep-research` model adds charges for citation tokens, search queries, and reasoning tokens. Grok charges reasoning tokens at the output token rate. These are folded into the same `cost` variable.

### 8. Verbose Output

If `verbose=True` (the default), the function prints the response text, token usage breakdown, and estimated cost to the console. For models that produce chain-of-thought reasoning (DeepSeek, Kimi), the reasoning trace is printed separately under a `--- Reasoning ---` header before the final answer. Perplexity additionally prints the list of citation URLs.

### 9. Return Dictionary

Finally, the function returns a dictionary containing the response text, all token counts, the cost estimate, and the raw response object. The raw response is included to allow advanced users to access provider-specific fields not surfaced by the wrapper (e.g., finish reason, logprobs, or tool call results). The keys are not perfectly identical across providers due to differences in their APIs, but the structure is consistent enough that switching between wrappers requires only minor key name adjustments.

## License

Creative Commons Zero v1.0 Universal

## Contact

Owen Root  
_Physics PhD Candidate, Quantum Information & Bio-Photonics_  
Email: owenbroughallroot@gmail.com
GitHub: [github.com/yagoiroot](https://github.com/yoagoiroot)

