"""OpenAI-compatible LLM client for the trader (OpenCode Go / DeepSeek V4 Flash).

This uses the **OpenCode Go** hosted endpoint that the ``OPENCODE_GO_KEY``
grants access to. The Go plan's ``deepseek-v4-flash`` is the 2x usage tier;
the model identifier is ``deepseek-v4-flash`` and we enable max thinking in
the request body (``reasoning_effort: "max"`` + ``thinking.type: enabled``).

Key: read from the ``OPENCODE_GO_KEY`` environment variable. On GitHub
you set it as a repository Actions secret (Settings -> Secrets and
variables -> Actions -> "New repository secret", name ``OPENCODE_GO_KEY``);
locally export it or put it in a gitignored ``secrets.env``.

Base URL + model are configurable. The bot never commits a key -- it only
reads it from the environment.
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
KEY_ENV = "OPENCODE_GO_KEY"


class LLMClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL,
                 api_key: str | None = None,
                 timeout: float = 60.0,
                 session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get(KEY_ENV, "")
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str,
                 thinking: bool = True, temperature: float = 0.2,
                 max_tokens: int = 700) -> dict[str, Any]:
        """Call the model and return {'reasoning':..., 'content':...}.

        Raises a clear error if no API key is set, so a misconfigured bot
        fails loudly instead of silently trading on guesses.
        """
        if not self.configured:
            raise RuntimeError(
                f"No {KEY_ENV} set. Add it as a GitHub Actions secret "
                f"(or export locally) before enabling llm_trader.")
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking:
            # Max-effort reasoning (the "thinking max" the Go plan supports)
            payload["reasoning_effort"] = "max"
            payload["thinking"] = {"type": "enabled"}
        resp = self.session.post(
            url, json=payload, timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"})
        if resp.status_code != 200:
            raise RuntimeError(
                f"LLM API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message", {})
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        content = msg.get("content") or ""
        return {"reasoning": reasoning, "content": content}


def parse_decision_json(content: str) -> dict:
    """Extract a JSON decision object from the model's text reply."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in LLM reply: {content[:200]!r}")
    return json.loads(text[start:end + 1])
