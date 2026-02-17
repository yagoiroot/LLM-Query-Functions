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

## License

[Specify your license here.]
