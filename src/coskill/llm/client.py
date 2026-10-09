"""Minimal LLM interfaces and an OpenAI-compatible HTTP client."""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.models import TokenUsage


@dataclass(slots=True)
class Message:
    """One role-tagged message sent to a chat-completion endpoint."""

    role: str
    content: str


@dataclass(slots=True)
class LLMResponse:
    """Normalized model output, usage accounting, and raw provider response."""

    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when an LLM request or response cannot be used."""

    pass


class LLMClient(ABC):
    """Async interface implemented by device- and cloud-side LLM clients."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError

    async def complete_json(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        retries: int = 2,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Complete with JSON mode and retry transient malformed model output.

        Some OpenAI-compatible servers accept JSON mode but still occasionally
        return truncated or syntactically invalid JSON, especially under long
        prompts or concurrent load.  Retry only parsing failures; transport and
        HTTP errors should still fail immediately.  Token usage is accumulated
        across all attempts so experiment accounting remains truthful.
        """
        if retries < 0:
            raise ValueError("retries must be non-negative")
        base_messages = list(messages)
        total_usage = TokenUsage()
        last_error: LLMError | None = None
        for attempt in range(retries + 1):
            attempt_messages = base_messages
            if attempt:
                attempt_messages = [
                    *base_messages,
                    Message(
                        "user",
                        "Your previous response was not valid JSON. Return only "
                        "one valid JSON object matching the requested schema; "
                        "do not include markdown, commentary, or trailing text.",
                    ),
                ]
            response = await self.complete(
                attempt_messages, json_mode=True, temperature=temperature
            )
            total_usage = total_usage + response.usage
            try:
                value = extract_json_object(response.content)
            except LLMError as exc:
                last_error = exc
                if attempt == retries:
                    raise
                continue
            response.usage = total_usage
            return value, response
        assert last_error is not None
        raise last_error


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON even when a model wraps it in a Markdown fence."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("Model response did not contain a JSON object")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("Expected a JSON object from model")
    return value


class OpenAICompatibleClient(LLMClient):
    """Small dependency-free client for /v1/chat/completions APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}

    @classmethod
    def from_env(cls, prefix: str) -> "OpenAICompatibleClient":
        """Create a client from environment variables using the given prefix."""
        prefix = prefix.upper().rstrip("_")
        required = {
            "base_url": os.getenv(f"{prefix}_BASE_URL", ""),
            "api_key": os.getenv(f"{prefix}_API_KEY", ""),
            "model": os.getenv(f"{prefix}_MODEL", ""),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            names = ", ".join(f"{prefix}_{name.upper()}" for name in missing)
            raise ValueError(f"Missing environment variables: {names}")
        max_tokens_raw = os.getenv(f"{prefix}_MAX_TOKENS", "").strip()
        max_tokens = int(max_tokens_raw) if max_tokens_raw else None
        return cls(**required, max_tokens=max_tokens)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Run the blocking HTTP request off the asyncio event loop."""
        return await asyncio.to_thread(
            self._complete_sync, messages, json_mode, temperature
        )

    def _complete_sync(
        self,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM HTTP {exc.code}: {detail[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response: {raw}") from exc
        usage = raw.get("usage") or {}
        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            raw=raw,
        )
