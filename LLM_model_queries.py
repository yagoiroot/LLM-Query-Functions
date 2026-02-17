import os
from openai import OpenAI
from perplexity import Perplexity
import numpy as np
from dotenv import load_dotenv
import difflib
import math
import re

load_dotenv()

def openai_query(prompt, model='gpt-5.2', reasoning_effort='medium',  tool=None, instructions=None,
                 max_output_tokens=None, temperature=None, top_p=None, response_format=None, stream=False,
                 verbose=True):
    """
    Query the OpenAI Responses API with configurable parameters.

    Parameters
    ----------
    prompt : str
        The user prompt to send to the model.
    model : str
        Model identifier. Must be a key in MODEL_CONFIG.
    reasoning_effort : str
        Reasoning effort level. Valid options depend on the model.
    tool : str or None
        Optional tool to enable (e.g., 'web_search').
    instructions : str or None
        System-level instructions (system prompt) to guide model behavior.
    max_output_tokens : int or None
        Maximum number of tokens in the response.
    temperature : float or None
        Sampling temperature. Not supported by all models.
    top_p : float or None
        Nucleus sampling parameter. Typically use one of temperature or top_p, not both.
    response_format : dict or None
        Response format specification, e.g. {"type": "json_object"}.
    stream : bool
        If True, stream the response incrementally.
    verbose : bool
        If True, print the response text, usage stats, and cost estimate.

    Returns
    -------
    dict
        Keys: 'text', 'input_tokens', 'cached_tokens', 'output_tokens',
        'total_tokens', 'cost_estimate', 'response'.
    """

    MODEL_CONFIG = {
        'gpt-5.2': {
            'reasoning_options': ['low', 'medium', 'high', 'xhigh'],
            'supports_temperature': False,
            'cost_in': 1.75 / 1e6,
            'cost_cached': 0.175 / 1e6,
            'cost_out': 14 / 1e6,
        },
        'gpt-5.2-pro': {
            'reasoning_options': ['low', 'medium', 'high', 'xhigh'],
            'supports_temperature': False,
            'cost_in': 21 / 1e6,
            'cost_cached': 0 / 1e6,
            'cost_out': 168 / 1e6,
        },
        'gpt-5-mini': {
            'reasoning_options': ['low', 'medium', 'high', 'xhigh'],
            'supports_temperature': False,
            'cost_in': 0.25 / 1e6,
            'cost_cached': 0.025 / 1e6,
            'cost_out': 2 / 1e6,
        },
        'o1-pro': {
            'reasoning_options': ['medium'],
            'supports_temperature': False,
            'cost_in': 150 / 1e6,
            'cost_cached': 0 / 1e6,
            'cost_out': 600 / 1e6,
        },
    }

    TOOL_OPTIONS = ['web_search']

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Validate reasoning effort ---
    if reasoning_effort not in cfg['reasoning_options']:
        raise ValueError(
            f"Invalid reasoning_effort '{reasoning_effort}' for model '{model}'. "
            f"Must be one of {cfg['reasoning_options']}"
        )

    # --- Validate tool ---
    if tool is not None and tool not in TOOL_OPTIONS:
        raise ValueError(f"Invalid tool '{tool}'. Must be one of {TOOL_OPTIONS}")

    # --- Validate temperature support ---
    if temperature is not None and not cfg['supports_temperature']:
        raise ValueError(
            f"Model '{model}' does not support the temperature parameter."
        )

    # --- Build API kwargs ---
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    kwargs = {
        'model': model,
        'input': prompt,
        'reasoning': {'effort': reasoning_effort},
        'stream': stream,
    }

    if instructions is not None:
        kwargs['instructions'] = instructions
    if max_output_tokens is not None:
        kwargs['max_output_tokens'] = max_output_tokens
    if temperature is not None:
        kwargs['temperature'] = temperature
    if top_p is not None:
        kwargs['top_p'] = top_p
    if response_format is not None:
        kwargs['text'] = {'format': response_format}

    if tool == 'web_search':
        kwargs['tools'] = [{'type': 'web_search'}]
        kwargs['include'] = ['web_search_call.action.sources']

    # --- Call the API ---
    response = client.responses.create(**kwargs)

    # --- Handle streaming vs. non-streaming ---
    if stream:
        collected_text = []
        for event in response:
            if hasattr(event, 'delta') and event.delta:
                if verbose:
                    print(event.delta, end='', flush=True)
                collected_text.append(event.delta)
        if verbose:
            print()  # newline after stream
        output_text = ''.join(collected_text)
        # Note: usage stats may not be available in all streaming modes
        return {'text': output_text, 'response': response}

    # --- Extract usage and compute cost ---
    input_tokens = response.usage.input_tokens
    cached_tokens = response.usage.input_tokens_details.cached_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = response.usage.total_tokens

    non_cached_input = input_tokens - cached_tokens
    cost = (
        non_cached_input * cfg['cost_in']
        + cached_tokens * cfg['cost_cached']
        + output_tokens * cfg['cost_out']
    )

    if verbose:
        print(response.output_text)
        print(f"\nTokens — input: {input_tokens} (cached: {cached_tokens}), "
              f"output: {output_tokens}, total: {total_tokens}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': response.output_text,
        'input_tokens': input_tokens,
        'cached_tokens': cached_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
    }

def perplexity_query(prompt, model='sonar', system_prompt=None, search_context_size='low', search_recency_filter=None,
                     search_domain_filter=None, return_related_questions=False, max_tokens=None, temperature=None,
                     top_p=None, stream=False, verbose=True):
    """
    Query the Perplexity Sonar API via the official Python SDK.

    Parameters
    ----------
    prompt : str
        The user prompt to send to the model.
    model : str
        Model identifier. Must be a key in MODEL_CONFIG.
    system_prompt : str or None
        System-level instruction prepended to the conversation.
    search_context_size : str
        How much web context to retrieve: 'low', 'medium', or 'high'.
        Affects both result quality and the per-request cost component.
    search_recency_filter : str or None
        Restrict search results by recency: 'hour', 'day', 'week', or 'month'.
    search_domain_filter : list[str] or None
        Whitelist or blacklist domains. Prefix with '-' to exclude
        (e.g., ['-reddit.com', 'nature.com']).
    return_related_questions : bool
        If True, the response includes suggested follow-up questions.
    max_tokens : int or None
        Maximum number of tokens in the response.
    temperature : float or None
        Sampling temperature (0.0-2.0). Not supported by all models.
    top_p : float or None
        Nucleus sampling parameter. Use one of temperature or top_p, not both.
    stream : bool
        If True, stream the response incrementally.
    verbose : bool
        If True, print the response text, citations, usage stats, and cost estimate.

    Returns
    -------
    dict
        Keys: 'text', 'citations', 'search_results', 'prompt_tokens',
        'completion_tokens', 'total_tokens', 'cost_estimate', 'response'.
    """

    MODEL_CONFIG = {
        'sonar': {
            'supports_temperature': True,
            'cost_in': 1 / 1e6,
            'cost_out': 1 / 1e6,
            'request_cost_per_1k': {'low': 5, 'medium': 8, 'high': 12},
        },
        'sonar-pro': {
            'supports_temperature': True,
            'cost_in': 3 / 1e6,
            'cost_out': 15 / 1e6,
            'request_cost_per_1k': {'low': 6, 'medium': 10, 'high': 14},
        },
        'sonar-reasoning-pro': {
            'supports_temperature': True,
            'cost_in': 2 / 1e6,
            'cost_out': 8 / 1e6,
            'request_cost_per_1k': {'low': 6, 'medium': 10, 'high': 14},
        },
        'sonar-deep-research': {
            'supports_temperature': False,
            'cost_in': 2 / 1e6,
            'cost_out': 8 / 1e6,
            'cost_citation': 2 / 1e6,
            'cost_search_queries': 5 / 1e3,
            'cost_reasoning': 3 / 1e6,
            'request_cost_per_1k': None,  # no per-request fee
        },
    }

    SEARCH_CONTEXT_OPTIONS = ['low', 'medium', 'high']
    RECENCY_OPTIONS = ['hour', 'day', 'week', 'month']

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Validate search parameters ---
    if search_context_size not in SEARCH_CONTEXT_OPTIONS:
        raise ValueError(
            f"Invalid search_context_size '{search_context_size}'. "
            f"Must be one of {SEARCH_CONTEXT_OPTIONS}"
        )

    if search_recency_filter is not None and search_recency_filter not in RECENCY_OPTIONS:
        raise ValueError(
            f"Invalid search_recency_filter '{search_recency_filter}'. "
            f"Must be one of {RECENCY_OPTIONS}"
        )

    if temperature is not None and not cfg['supports_temperature']:
        raise ValueError(f"Model '{model}' does not support the temperature parameter.")

    # --- Build messages ---
    messages = []
    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    # --- Build API kwargs ---
    client = Perplexity(api_key=os.environ.get("PERPLEXITY_API_KEY"))

    kwargs = {
        'model': model,
        'messages': messages,
        'stream': stream,
        'web_search_options': {'search_context_size': search_context_size},
    }

    if search_recency_filter is not None:
        kwargs['search_recency_filter'] = search_recency_filter
    if search_domain_filter is not None:
        kwargs['search_domain_filter'] = search_domain_filter
    if return_related_questions:
        kwargs['return_related_questions'] = True
    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens
    if temperature is not None:
        kwargs['temperature'] = temperature
    if top_p is not None:
        kwargs['top_p'] = top_p

    # --- Call the API ---
    response = client.chat.completions.create(**kwargs)

    # --- Handle streaming ---
    if stream:
        collected_text = []
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text_piece = chunk.choices[0].delta.content
                if verbose:
                    print(text_piece, end='', flush=True)
                collected_text.append(text_piece)
        if verbose:
            print()
        return {'text': ''.join(collected_text), 'response': response}

    # --- Extract response content ---
    output_text = response.choices[0].message.content
    citations = getattr(response, 'citations', []) or []
    search_results = getattr(response, 'search_results', []) or []

    # --- Extract usage and compute cost ---
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Token cost
    cost = prompt_tokens * cfg['cost_in'] + completion_tokens * cfg['cost_out']

    # Per-request cost (varies by search_context_size)
    if cfg['request_cost_per_1k'] is not None:
        cost += cfg['request_cost_per_1k'][search_context_size] / 1000

    # Deep Research has additional cost components
    if model == 'sonar-deep-research':
        # The API response includes these in the usage object when available
        citation_tokens = getattr(usage, 'citation_tokens', 0) or 0
        search_queries = getattr(usage, 'search_queries', 0) or 0
        reasoning_tokens = getattr(usage, 'reasoning_tokens', 0) or 0
        cost += citation_tokens * cfg['cost_citation']
        cost += search_queries * cfg['cost_search_queries']
        cost += reasoning_tokens * cfg['cost_reasoning']

    if verbose:
        print(output_text)
        if citations:
            print(f"\nCitations ({len(citations)}):")
            for i, url in enumerate(citations, 1):
                print(f"  [{i}] {url}")
        print(f"\nTokens -- prompt: {prompt_tokens}, completion: {completion_tokens}, "
              f"total: {total_tokens}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': output_text,
        'citations': citations,
        'search_results': search_results,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
    }

def deepseek_query(prompt, model='deepseek-chat', system_prompt=None, thinking=False,
                    max_tokens=None, temperature=None, top_p=None, frequency_penalty=None,
                    presence_penalty=None, response_format=None, tools=None, stream=False,
                    verbose=True):
    """
    Query the DeepSeek API via the OpenAI-compatible chat completions endpoint.

    DeepSeek's API uses the OpenAI SDK with a different base_url. Both models
    (deepseek-chat and deepseek-reasoner) are backed by DeepSeek-V3.2:
    deepseek-chat is the non-thinking mode, deepseek-reasoner is the thinking mode.

    Parameters
    ----------
    prompt : str
        The user prompt to send to the model.
    model : str
        Model identifier. Must be a key in MODEL_CONFIG.
    thinking : bool
        If True, enable thinking (chain-of-thought) mode. This is equivalent to
        using model='deepseek-reasoner', but can also be set independently via
        extra_body when using 'deepseek-chat'. When thinking is enabled,
        temperature/top_p/frequency_penalty/presence_penalty are silently ignored
        by the API (no error, but no effect).
    system_prompt : str or None
        System-level instruction prepended to the conversation.
    max_tokens : int or None
        Maximum number of tokens in the response (including CoT if thinking).
        Default 32K, maximum 64K for thinking mode; 8K for non-thinking.
    temperature : float or None
        Sampling temperature (0.0-2.0). Only effective in non-thinking mode.
    top_p : float or None
        Nucleus sampling parameter. Only effective in non-thinking mode.
    frequency_penalty : float or None
        Frequency penalty (-2.0 to 2.0). Only effective in non-thinking mode.
    presence_penalty : float or None
        Presence penalty (-2.0 to 2.0). Only effective in non-thinking mode.
    response_format : dict or None
        Response format specification, e.g. {"type": "json_object"}.
    tools : list or None
        List of tool/function definitions for function calling.
    stream : bool
        If True, stream the response incrementally.
    verbose : bool
        If True, print the response text, reasoning (if any), usage stats,
        and cost estimate.

    Returns
    -------
    dict
        Keys: 'text', 'reasoning', 'prompt_tokens', 'cached_tokens',
        'completion_tokens', 'total_tokens', 'cost_estimate', 'response'.
        For streaming, only 'text', 'reasoning', and 'response' are returned.

    Notes
    -----
    Pricing is from the official DeepSeek API docs (per 1M tokens):
      - deepseek-chat:     $0.07 (cache hit) / $0.27 (cache miss) / $1.10 (output)
      - deepseek-reasoner: $0.14 (cache hit) / $0.55 (cache miss) / $2.19 (output)

    Context caching is automatic on DeepSeek's side; prompts sharing the same
    prefix with recent requests will benefit from cache hits.
    """

    MODEL_CONFIG = {
        'deepseek-chat': {
            'supports_temperature': True,
            'context_length': 128_000,
            'max_output_default': 4_000,
            'max_output': 8_000,
            'cost_in_hit': 0.028 / 1e6,
            'cost_in_miss': 0.28 / 1e6,
            'cost_out': 0.42 / 1e6,
        },
        'deepseek-reasoner': {
            'supports_temperature': False,  # silently ignored, not an error
            'context_length': 128_000,
            'max_output_default': 32_000,
            'max_output': 64_000,
            'cost_in_hit': 0.028 / 1e6,
            'cost_in_miss': 0.28 / 1e6,
            'cost_out': 0.42 / 1e6,
        },
    }

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Warn about temperature in thinking mode ---
    effective_thinking = thinking or model == 'deepseek-reasoner'
    if effective_thinking and temperature is not None:
        import warnings
        warnings.warn(
            "temperature has no effect in thinking mode (silently ignored by the API).",
            stacklevel=2,
        )

    # --- Build messages ---
    messages = []
    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    # --- Build API kwargs ---
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    kwargs = {
        'model': model,
        'messages': messages,
        'stream': stream,
    }

    # Enable thinking mode via extra_body if model is deepseek-chat but thinking=True
    if thinking and model == 'deepseek-chat':
        kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}

    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens
    if temperature is not None:
        kwargs['temperature'] = temperature
    if top_p is not None:
        kwargs['top_p'] = top_p
    if frequency_penalty is not None:
        kwargs['frequency_penalty'] = frequency_penalty
    if presence_penalty is not None:
        kwargs['presence_penalty'] = presence_penalty
    if response_format is not None:
        kwargs['response_format'] = response_format
    if tools is not None:
        kwargs['tools'] = tools

    # --- Call the API ---
    response = client.chat.completions.create(**kwargs)

    # --- Handle streaming ---
    if stream:
        collected_reasoning = []
        collected_text = []
        for chunk in response:
            delta = chunk.choices[0].delta
            # In thinking mode, reasoning_content comes first, then content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                collected_reasoning.append(delta.reasoning_content)
                if verbose:
                    print(delta.reasoning_content, end='', flush=True)
            if delta.content:
                if collected_reasoning and not collected_text:
                    # Transition from reasoning to content
                    if verbose:
                        print("\n\n--- Answer ---\n", flush=True)
                collected_text.append(delta.content)
                if verbose:
                    print(delta.content, end='', flush=True)
        if verbose:
            print()
        return {
            'text': ''.join(collected_text),
            'reasoning': ''.join(collected_reasoning) if collected_reasoning else None,
            'response': response,
        }

    # --- Extract content ---
    message = response.choices[0].message
    output_text = message.content
    reasoning = getattr(message, 'reasoning_content', None)

    # --- Extract usage and compute cost ---
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # DeepSeek reports cache hit/miss in usage details
    cached_tokens = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
    cache_miss_tokens = getattr(usage, 'prompt_cache_miss_tokens', 0) or 0

    # Use the appropriate cost tier if thinking is active
    if effective_thinking:
        cost_cfg = MODEL_CONFIG['deepseek-reasoner']
    else:
        cost_cfg = cfg

    cost = (
        cached_tokens * cost_cfg['cost_in_hit']
        + cache_miss_tokens * cost_cfg['cost_in_miss']
        + completion_tokens * cost_cfg['cost_out']
    )

    if verbose:
        if reasoning:
            print("--- Reasoning ---")
            print(reasoning)
            print("\n--- Answer ---")
        print(output_text)
        print(f"\nTokens -- prompt: {prompt_tokens} (cache hit: {cached_tokens}, "
              f"cache miss: {cache_miss_tokens}), completion: {completion_tokens}, "
              f"total: {total_tokens}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': output_text,
        'reasoning': reasoning,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
    }

def grok_query(prompt, model='grok-4-1-fast', system_prompt=None, reasoning_effort=None,
               tools=None, max_tokens=None, temperature=None, top_p=None,
               frequency_penalty=None, presence_penalty=None, response_format=None,
               stream=False, timeout=900, verbose=True):
    """
    Query the xAI Grok API via the official xai_sdk Python library.

    The xai_sdk is a gRPC-based SDK (not REST/OpenAI-compatible). It uses a
    stateful chat object pattern: create a chat, append messages, then sample.

    Parameters
    ----------
    prompt : str
        The user prompt to send to the model.
    model : str
        Model identifier. Must be a key in MODEL_CONFIG. Reasoning vs.
        non-reasoning variants exist for grok-4-fast and grok-4.1-fast
        (e.g., 'grok-4-1-fast' is reasoning, 'grok-4-1-fast-non-reasoning'
        is the non-reasoning variant).
    system_prompt : str or None
        System-level instruction prepended to the conversation.
    reasoning_effort : str or None
        Reasoning effort level: 'low', 'medium', or 'high'. Only supported
        by grok-3-mini. Specifying this for grok-4 or grok-4-fast models
        will raise an API error.
    tools : list or None
        List of xai_sdk server-side tool instances (e.g., from
        xai_sdk.tools.web_search(), xai_sdk.tools.x_search(), etc.).
    max_tokens : int or None
        Maximum number of tokens in the response.
    temperature : float or None
        Sampling temperature. Not supported by reasoning models (grok-4,
        grok-4-fast reasoning variants, grok-3, grok-3-mini).
    top_p : float or None
        Nucleus sampling parameter.
    frequency_penalty : float or None
        Frequency penalty. Not supported by reasoning models (grok-4,
        grok-4-fast reasoning variants).
    presence_penalty : float or None
        Presence penalty. Not supported by reasoning models (grok-4,
        grok-4-fast reasoning variants).
    response_format : type or None
        A Pydantic BaseModel class for structured output. When provided,
        uses chat.parse() instead of chat.sample().
    stream : bool
        If True, stream the response incrementally.
    timeout : int
        Timeout in seconds for the API request. Default 900 (15 min).
        Increase for reasoning models which can take longer.
    verbose : bool
        If True, print the response text, usage stats, and cost estimate.

    Returns
    -------
    dict
        Keys: 'text', 'reasoning_tokens', 'prompt_tokens', 'cached_tokens',
        'completion_tokens', 'total_tokens', 'cost_estimate', 'response'.
        For streaming, only 'text' and 'response' are returned.
        For structured output, 'parsed' contains the Pydantic model instance.

    Notes
    -----
    Pricing is per 1M tokens (from xAI docs and consistent third-party sources).
    Verify current pricing at https://docs.x.ai/developers/models.

    Cached prompt tokens are automatically handled by xAI (prefix-based caching).
    Reasoning tokens are charged at the output token rate.

    Tool invocations incur additional per-call fees:
      - web_search, x_search, code_execution: $5 / 1k calls
      - collections_search: $2.50 / 1k calls
      - attachment_search: $10 / 1k calls
    These are NOT included in the cost_estimate returned here.
    """
    from xai_sdk import Client
    from xai_sdk.chat import system, user

    MODEL_CONFIG = {
        'grok-4': {
            'is_reasoning': True,
            'supports_temperature': False,
            'supports_penalties': False,
            'supports_reasoning_effort': False,
            'context_length': 256_000,
            'cost_in': 3.00 / 1e6,
            'cost_cached': 0.75 / 1e6,
            'cost_out': 15.00 / 1e6,
        },
        'grok-4-fast': {
            'is_reasoning': True,
            'supports_temperature': False,
            'supports_penalties': False,
            'supports_reasoning_effort': False,
            'context_length': 2_000_000,
            'cost_in': 0.20 / 1e6,
            'cost_cached': 0.05 / 1e6,
            'cost_out': 0.50 / 1e6,
        },
        'grok-4-fast-non-reasoning': {
            'is_reasoning': False,
            'supports_temperature': True,
            'supports_penalties': True,
            'supports_reasoning_effort': False,
            'context_length': 2_000_000,
            'cost_in': 0.20 / 1e6,
            'cost_cached': 0.05 / 1e6,
            'cost_out': 0.50 / 1e6,
        },
        'grok-4-1-fast': {
            'is_reasoning': True,
            'supports_temperature': False,
            'supports_penalties': False,
            'supports_reasoning_effort': False,
            'context_length': 2_000_000,
            'cost_in': 0.20 / 1e6,
            'cost_cached': 0.05 / 1e6,
            'cost_out': 0.50 / 1e6,
        },
        'grok-4-1-fast-non-reasoning': {
            'is_reasoning': False,
            'supports_temperature': True,
            'supports_penalties': True,
            'supports_reasoning_effort': False,
            'context_length': 2_000_000,
            'cost_in': 0.20 / 1e6,
            'cost_cached': 0.05 / 1e6,
            'cost_out': 0.50 / 1e6,
        },
        'grok-3': {
            'is_reasoning': True,
            'supports_temperature': False,
            'supports_penalties': False,
            'supports_reasoning_effort': False,
            'context_length': 131_072,
            'cost_in': 3.00 / 1e6,
            'cost_cached': 0.75 / 1e6,
            'cost_out': 15.00 / 1e6,
        },
        'grok-3-mini': {
            'is_reasoning': True,
            'supports_temperature': False,
            'supports_penalties': False,
            'supports_reasoning_effort': True,
            'context_length': 131_072,
            'cost_in': 0.30 / 1e6,
            'cost_cached': 0.075 / 1e6,
            'cost_out': 0.50 / 1e6,
        },
    }

    REASONING_EFFORT_OPTIONS = ['low', 'medium', 'high']

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Validate reasoning_effort ---
    if reasoning_effort is not None:
        if not cfg['supports_reasoning_effort']:
            raise ValueError(
                f"Model '{model}' does not support reasoning_effort. "
                f"Only grok-3-mini supports this parameter."
            )
        if reasoning_effort not in REASONING_EFFORT_OPTIONS:
            raise ValueError(
                f"Invalid reasoning_effort '{reasoning_effort}'. "
                f"Must be one of {REASONING_EFFORT_OPTIONS}"
            )

    # --- Validate temperature/penalty support ---
    if temperature is not None and not cfg['supports_temperature']:
        raise ValueError(
            f"Model '{model}' does not support the temperature parameter. "
            f"Use a non-reasoning variant (e.g., 'grok-4-1-fast-non-reasoning')."
        )
    if frequency_penalty is not None and not cfg['supports_penalties']:
        raise ValueError(
            f"Model '{model}' does not support frequency_penalty. "
            f"Use a non-reasoning variant."
        )
    if presence_penalty is not None and not cfg['supports_penalties']:
        raise ValueError(
            f"Model '{model}' does not support presence_penalty. "
            f"Use a non-reasoning variant."
        )

    # --- Build client and chat ---
    client = Client(timeout=timeout)

    chat_kwargs = {'model': model}

    # Build initial messages list
    messages = []
    if system_prompt is not None:
        messages.append(system(system_prompt))
    if messages:
        chat_kwargs['messages'] = messages

    if reasoning_effort is not None:
        chat_kwargs['reasoning_effort'] = reasoning_effort
    if tools is not None:
        chat_kwargs['tools'] = tools
    if max_tokens is not None:
        chat_kwargs['max_tokens'] = max_tokens
    if temperature is not None:
        chat_kwargs['temperature'] = temperature
    if top_p is not None:
        chat_kwargs['top_p'] = top_p
    if frequency_penalty is not None:
        chat_kwargs['frequency_penalty'] = frequency_penalty
    if presence_penalty is not None:
        chat_kwargs['presence_penalty'] = presence_penalty

    chat = client.chat.create(**chat_kwargs)
    chat.append(user(prompt))

    # --- Handle structured output ---
    if response_format is not None:
        response, parsed = chat.parse(response_format)
        output_text = response.content

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        reasoning_tokens = getattr(usage, 'reasoning_tokens', 0) or 0
        cached_tokens = getattr(usage, 'cached_prompt_tokens', 0) or 0
        total_tokens = getattr(usage, 'total_tokens', 0) or 0

        non_cached_input = prompt_tokens - cached_tokens
        cost = (
            non_cached_input * cfg['cost_in']
            + cached_tokens * cfg['cost_cached']
            + (completion_tokens + reasoning_tokens) * cfg['cost_out']
        )

        if verbose:
            print(output_text)
            print(f"\nParsed: {parsed}")
            print(f"\nTokens -- prompt: {prompt_tokens} (cached: {cached_tokens}), "
                  f"completion: {completion_tokens}, reasoning: {reasoning_tokens}, "
                  f"total: {total_tokens}")
            print(f"Estimated cost: ${cost:.6f}")

        return {
            'text': output_text,
            'parsed': parsed,
            'reasoning_tokens': reasoning_tokens,
            'prompt_tokens': prompt_tokens,
            'cached_tokens': cached_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'cost_estimate': cost,
            'response': response,
        }

    # --- Handle streaming ---
    if stream:
        collected_text = []
        final_response = None
        for response, chunk in chat.stream():
            final_response = response
            if chunk.content:
                if verbose:
                    print(chunk.content, end='', flush=True)
                collected_text.append(chunk.content)
        if verbose:
            print()
        return {
            'text': ''.join(collected_text),
            'response': final_response,
        }

    # --- Non-streaming ---
    response = chat.sample()
    output_text = response.content

    # --- Extract usage and compute cost ---
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    reasoning_tokens = getattr(usage, 'reasoning_tokens', 0) or 0
    cached_tokens = getattr(usage, 'cached_prompt_tokens', 0) or 0
    total_tokens = getattr(usage, 'total_tokens', 0) or 0

    non_cached_input = prompt_tokens - cached_tokens
    # Reasoning tokens are charged at the output token rate
    cost = (
        non_cached_input * cfg['cost_in']
        + cached_tokens * cfg['cost_cached']
        + (completion_tokens + reasoning_tokens) * cfg['cost_out']
    )

    if verbose:
        print(output_text)
        print(f"\nTokens -- prompt: {prompt_tokens} (cached: {cached_tokens}), "
              f"completion: {completion_tokens}, reasoning: {reasoning_tokens}, "
              f"total: {total_tokens}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': output_text,
        'reasoning_tokens': reasoning_tokens,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
    }

def kimi_query(prompt, model='kimi-k2.5', thinking=True, system_prompt=None,
               max_tokens=None, temperature=None, top_p=None,
               response_format=None, tools=None, stream=False, verbose=True):
    """
    Query the Moonshot AI Kimi API via the OpenAI-compatible chat completions endpoint.

    Moonshot's API is fully compatible with the OpenAI SDK. Both Kimi K2 and
    K2.5 are backed by a 1T-parameter MoE architecture (32B activated). The API
    supports thinking (chain-of-thought) and instant (non-thinking) modes,
    toggled via extra_body.

    Parameters
    ----------
    prompt : str
        The user prompt to send to the model.
    model : str
        Model identifier. Must be a key in MODEL_CONFIG.
    thinking : bool
        If True (default), enable thinking mode (chain-of-thought reasoning).
        The model will emit reasoning_content before the final answer.
        If False, use instant (non-thinking) mode.
        Recommended temperatures: 1.0 for thinking, 0.6 for instant.
    system_prompt : str or None
        System-level instruction prepended to the conversation.
    max_tokens : int or None
        Maximum number of tokens in the response.
    temperature : float or None
        Sampling temperature. If None, defaults are applied based on mode:
        1.0 for thinking, 0.6 for instant (per Moonshot's recommendations).
    top_p : float or None
        Nucleus sampling parameter. Moonshot recommends 0.95.
    response_format : dict or None
        Response format specification, e.g. {"type": "json_object"}.
    tools : list or None
        List of tool/function definitions for function calling.
        Moonshot also supports a built-in web search tool via:
        tools=[{"type": "builtin_function", "function": {"name": "$web_search"}}]
    stream : bool
        If True, stream the response incrementally.
    verbose : bool
        If True, print the response text, reasoning (if any), usage stats,
        and cost estimate.

    Returns
    -------
    dict
        Keys: 'text', 'reasoning', 'prompt_tokens', 'cached_tokens',
        'completion_tokens', 'total_tokens', 'cost_estimate', 'response'.
        For streaming, only 'text', 'reasoning', and 'response' are returned.

    Notes
    -----
    Pricing is per 1M tokens from Moonshot's platform and corroborating sources.
    Automatic context caching is enabled server-side (prefix-based); cache hits
    reduce input costs by ~75%.

    Figures below are compiled from multiple consistent third-party sources and
    Moonshot's blog. Verify at https://platform.moonshot.ai if in doubt.

    Web search tool invocations cost ~$0.005 per call (not included in the
    cost_estimate returned here).
    """

    MODEL_CONFIG = {
        # --- Kimi K2.5 (latest, multimodal, Jan 2026) ---
        'kimi-k2.5': {
            'context_length': 256_000,
            'cost_in_miss': 0.60 / 1e6,
            'cost_in_hit': 0.10 / 1e6,
            'cost_out': 3.00 / 1e6,
        },
        # --- Kimi K2 0905 (Sep 2025 release, text-only) ---
        'kimi-k2-0905': {
            'context_length': 256_000,
            'cost_in_miss': 0.60 / 1e6,
            'cost_in_hit': 0.15 / 1e6,
            'cost_out': 2.50 / 1e6,
        },
        # --- Thinking-specific model identifiers ---
        'kimi-k2-thinking': {
            'context_length': 256_000,
            'cost_in_miss': 0.60 / 1e6,
            'cost_in_hit': 0.15 / 1e6,
            'cost_out': 2.50 / 1e6,
        },
        'kimi-k2-thinking-turbo': {
            'context_length': 256_000,
            'cost_in_miss': 1.15 / 1e6,
            'cost_in_hit': 0.29 / 1e6,
            'cost_out': 8.00 / 1e6,
        },
        # --- Turbo variant (lower latency, higher cost) ---
        'kimi-k2-turbo': {
            'context_length': 256_000,
            'cost_in_miss': 1.15 / 1e6,
            'cost_in_hit': 0.29 / 1e6,
            'cost_out': 8.00 / 1e6,
        },
    }

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Build messages ---
    messages = []
    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    # --- Build API kwargs ---
    client = OpenAI(
        api_key=os.environ["MOONSHOT_API_KEY"],
        base_url="https://api.moonshot.ai/v1",
    )

    kwargs = {
        'model': model,
        'messages': messages,
        'stream': stream,
    }

    # Toggle thinking/instant mode
    if thinking:
        kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}
    else:
        kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}

    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens
    if temperature is not None:
        kwargs['temperature'] = temperature
    if top_p is not None:
        kwargs['top_p'] = top_p
    if response_format is not None:
        kwargs['response_format'] = response_format
    if tools is not None:
        kwargs['tools'] = tools

    # --- Call the API ---
    response = client.chat.completions.create(**kwargs)

    # --- Handle streaming ---
    if stream:
        collected_reasoning = []
        collected_text = []
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                collected_reasoning.append(delta.reasoning_content)
                if verbose:
                    print(delta.reasoning_content, end='', flush=True)
            if delta.content:
                if collected_reasoning and not collected_text:
                    if verbose:
                        print("\n\n--- Answer ---\n", flush=True)
                collected_text.append(delta.content)
                if verbose:
                    print(delta.content, end='', flush=True)
        if verbose:
            print()
        return {
            'text': ''.join(collected_text),
            'reasoning': ''.join(collected_reasoning) if collected_reasoning else None,
            'response': response,
        }

    # --- Extract content ---
    message = response.choices[0].message
    output_text = message.content
    reasoning = getattr(message, 'reasoning_content', None)

    # --- Extract usage and compute cost ---
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Moonshot reports cache hit/miss tokens in usage details
    cached_tokens = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
    cache_miss_tokens = getattr(usage, 'prompt_cache_miss_tokens', 0) or 0

    # Fall back to full prompt_tokens as cache miss if breakdown unavailable
    if cached_tokens == 0 and cache_miss_tokens == 0:
        cache_miss_tokens = prompt_tokens

    cost = (
        cached_tokens * cfg['cost_in_hit']
        + cache_miss_tokens * cfg['cost_in_miss']
        + completion_tokens * cfg['cost_out']
    )

    if verbose:
        if reasoning:
            print("--- Reasoning ---")
            print(reasoning)
            print("\n--- Answer ---")
        print(output_text)
        print(f"\nTokens -- prompt: {prompt_tokens} (cache hit: {cached_tokens}, "
              f"cache miss: {cache_miss_tokens}), completion: {completion_tokens}, "
              f"total: {total_tokens}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': output_text,
        'reasoning': reasoning,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
    }