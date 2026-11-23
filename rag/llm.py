"""LLM providers, shared by answering (query.py), query rewriting (retrieve.py) and the eval judge.

Each provider streams text pieces and fills in a `stats` dict with token counts.
- ollama:   local qwen2.5:3b, offline, slow on CPU
- deepseek: DeepSeek API (deepseek-flash), fast, needs DEEPSEEK_API_KEY
"""

import os
from collections.abc import Iterator

import ollama

from rag import config


def stream_ollama(model: str, messages: list[dict], stats: dict) -> Iterator[str]:
    stream = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": config.LLM_TEMPERATURE, "num_ctx": config.LLM_NUM_CTX},
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]
        if part.get("done"):  # the last message carries token counts and timings (in ns)
            stats.update(
                prompt_tokens=part["prompt_eval_count"],
                prompt_seconds=part["prompt_eval_duration"] / 1e9,
                answer_tokens=part["eval_count"],
                answer_seconds=part["eval_duration"] / 1e9,
            )


def stream_deepseek(model: str, messages: list[dict], stats: dict) -> Iterator[str]:
    # DeepSeek speaks the OpenAI chat API, so we use the openai client with DeepSeek's URL.
    from openai import OpenAI  # imported here so the Ollama path works without the package

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set in this shell (try: source ~/.bashrc)")
    client = OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL, timeout=120.0)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        stream=True,
        stream_options={"include_usage": True},  # token counts arrive in the last chunk
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        if chunk.usage:
            details = chunk.usage.completion_tokens_details
            stats.update(
                prompt_tokens=chunk.usage.prompt_tokens,
                answer_tokens=chunk.usage.completion_tokens,
                # deepseek-flash "thinks" before answering; those tokens are billed but not shown
                reasoning_tokens=details.reasoning_tokens if details else 0,
            )


PROVIDERS = {"ollama": stream_ollama, "deepseek": stream_deepseek}


def complete(messages: list[dict], llm: str) -> str:
    """Non-streaming call: the whole reply as one string."""
    # Retry: one slow or dropped API call should not kill a long eval run.
    for attempt in range(3):
        try:
            return "".join(PROVIDERS[llm](config.LLM_MODELS[llm], messages, {}))
        except Exception as err:
            # Retry timeouts, rate limits and server errors; a 4xx like "insufficient balance" will not fix itself.
            status = getattr(err, "status_code", None)
            if attempt == 2 or (status is not None and 400 <= status < 500 and status != 429):
                raise
            print(f"  (LLM call failed: {type(err).__name__}; retrying)", flush=True)
