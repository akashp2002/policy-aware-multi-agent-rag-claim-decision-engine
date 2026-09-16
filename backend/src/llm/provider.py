"""LLM provider abstraction layer.

Supports:
  1. Groq (free tier, llama-3.1-8b-instant or similar) — requires GROQ_API_KEY.
  2. OpenAI-compatible providers via OPENAI_API_KEY + OPENAI_BASE_URL.
  3. Local/rule-based deterministic fallback when no API key is available.

The fallback is NOT a conversational LLM; it applies structured
heuristic rules against the retrieval evidence. This keeps the
pipeline running for evaluation without any external service, while
the full LLM path provides higher-quality reasoning when a key is
available.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

USE_ENV_VARS = True


def _get_groq_key() -> Optional[str]:
    if not USE_ENV_VARS:
        return None
    return os.environ.get("GROQ_API_KEY") or None


def _get_openai_key() -> Optional[str]:
    if not USE_ENV_VARS:
        return None
    return os.environ.get("OPENAI_API_KEY") or None


@dataclass
class LLMCallResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "fallback"
    provider: str = "local"
    latency_ms: float = 0.0


@dataclass
class LLMProvider:
    provider: str = "auto"
    temperature: float = 0.0
    max_tokens: int = 1200
    _groq_client: Any = field(default=None, repr=False)
    _openai_client: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.provider == "auto":
            if _get_groq_key():
                self.provider = "groq"
                logger.info("LLM provider: Groq (GROQ_API_KEY detected)")
            elif _get_openai_key():
                self.provider = "openai"
                logger.info("LLM provider: OpenAI")
            else:
                self.provider = "fallback"
                logger.warning("No API key found; using deterministic heuristic fallback (no LLM calls)")

    def _get_groq_client(self):
        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=_get_groq_key())
        return self._groq_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            base = os.environ.get("OPENAI_BASE_URL")
            self._openai_client = OpenAI(api_key=_get_openai_key(), base_url=base if base else None)
        return self._openai_client

    # ------------------------------------------------------------------
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMCallResult:
        """Send a single-turn completion. Raises on API errors."""
        t0 = time.time()
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "groq":
            return self._groq_call(system_prompt, user_prompt, temp, max_tok, t0)
        elif self.provider == "openai":
            return self._openai_call(system_prompt, user_prompt, temp, max_tok, t0)
        else:
            return LLMCallResult(
                text="",
                model="none",
                provider="fallback",
                latency_ms=(time.time() - t0) * 1000,
            )

    def _groq_call(self, sys: str, usr: str, temp: float, max_tok: int, t0: float) -> LLMCallResult:
        client = self._get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
            temperature=temp,
            max_tokens=max_tok,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMCallResult(
            text=text.strip(),
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            model=resp.model,
            provider="groq",
            latency_ms=(time.time() - t0) * 1000,
        )

    def _openai_call(self, sys: str, usr: str, temp: float, max_tok: int, t0: float) -> LLMCallResult:
        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
            temperature=temp,
            max_tokens=max_tok,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMCallResult(
            text=text.strip(),
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            model=resp.model,
            provider="openai",
            latency_ms=(time.time() - t0) * 1000,
        )

    @property
    def is_fallback(self) -> bool:
        return self.provider == "fallback"

    @property
    def model_id(self) -> str:
        if self.provider == "groq":
            return "groq/llama-3.1-8b-instant"
        elif self.provider == "openai":
            return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return "local/heuristic-v1"