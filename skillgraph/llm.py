"""LLM calling utilities. Supports OpenAI and Anthropic directly."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("skillgraph.llm")

# Models that do NOT support temperature / top_p
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


@dataclass
class LLMResponse:
    text: str
    data: dict | list | None = None
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


def call_llm(
    prompt: str,
    *,
    system: str = "",
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    retries: int = 2,
) -> LLMResponse:
    """Call an LLM and return structured response."""
    from skillgraph.config import MODEL

    model = model or MODEL

    for attempt in range(retries + 1):
        try:
            if model.startswith("claude"):
                resp = _call_anthropic(
                    prompt,
                    system=system,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                resp = _call_openai(
                    prompt,
                    system=system,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return resp
        except Exception as e:
            logger.warning(
                "LLM call failed (attempt %d/%d): %s",
                attempt + 1,
                retries + 1,
                e,
            )
            if attempt < retries:
                time.sleep(2**attempt)
            else:
                raise


def call_llm_json(
    prompt: str,
    *,
    system: str = "",
    model: str = "",
    temperature: float = 0.2,
) -> tuple[dict | list, float]:
    """Call LLM and return parsed JSON + cost. Raises on parse failure."""
    resp = call_llm(prompt, system=system, model=model, temperature=temperature)
    if resp.data is None:
        raise ValueError(
            f"Failed to extract JSON from LLM response:\n{resp.text[:500]}"
        )
    return resp.data, resp.cost


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _call_anthropic(
    prompt: str,
    *,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system if system else anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    return LLMResponse(
        text=text,
        data=_extract_json(text),
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
    )


def _call_openai(
    prompt: str,
    *,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = openai.OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {"model": model, "messages": messages}
    if model.startswith(_NO_TEMPERATURE_PREFIXES):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return LLMResponse(
        text=text,
        data=_extract_json(text),
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | list | None:
    """Extract JSON from LLM response text."""
    # Try fenced JSON block first
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON (find first { or [)
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break

    return None
