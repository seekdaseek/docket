"""A minimal Anthropic client. Standard library only, so the project keeps its
zero-dependency property and a judge can read the whole call path.

Two behaviours matter more than the plumbing:

  A model that returns something unparseable is a MEASUREMENT FAILURE, not an
  empty answer. `complete_json` raises rather than returning [], because a
  session the model refused to read and a session that genuinely contained no
  claims must never end up looking the same in the checkpoint.

  Overload and rate limits are retried with backoff and the server's own
  Retry-After honoured. 940 sessions through a rate limit is exactly the run
  that dies at hour two otherwise.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)


class LLMError(RuntimeError):
    """The model could not be reached, or answered in a shape we cannot use."""


def load_env(path: str) -> dict:
    """Read KEY=value lines. The file lives OUTSIDE the repo on purpose."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def strip_fences(text: str) -> str:
    """Remove a markdown code fence if the model wrapped its JSON in one."""
    body = text.strip()
    if not body.startswith("```"):
        return body
    lines = body.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


class Anthropic:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 max_tokens: int = 4000, timeout: float = 120,
                 max_retries: int = 5, opener=None, sleeper=None):
        if not api_key:
            raise LLMError(
                "no API key. Put ANTHROPIC_API_KEY in ~/docket/.env, which "
                "sits outside the repo so the tree never holds a credential.")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = 0
        self.retries = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleeper or time.sleep

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        attempt = 0
        while True:
            req = urllib.request.Request(API_URL, data=data, method="POST", headers={
                "x-api-key": self.api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            })
            try:
                with self._opener(req, timeout=self.timeout) as r:
                    body = json.loads(r.read().decode())
                self.calls += 1
                usage = body.get("usage") or {}
                self.input_tokens += int(usage.get("input_tokens") or 0)
                self.output_tokens += int(usage.get("output_tokens") or 0)
                return body
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                if e.code not in RETRY_STATUS or attempt >= self.max_retries:
                    raise LLMError(f"http {e.code}: {detail}") from None
                after = e.headers.get("retry-after") if e.headers else None
                delay = float(after) if after and str(after).isdigit() else (
                    min(30.0, (2 ** attempt) + random.random()))
            except urllib.error.URLError as e:
                if attempt >= self.max_retries:
                    raise LLMError(f"transport: {e.reason}") from None
                delay = min(30.0, (2 ** attempt) + random.random())
            attempt += 1
            self.retries += 1
            self._sleep(delay)

    def complete_json(self, system: str, user: str) -> list | dict:
        """Ask for JSON and return it parsed, or raise.

        Never returns a default on bad output. The caller records the failure
        as unmeasured, which is a different fact from an empty result.
        """
        body = self._post({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        })
        parts = [b.get("text", "") for b in (body.get("content") or [])
                 if b.get("type") == "text"]
        text = strip_fences("".join(parts))
        if not text:
            raise LLMError(
                f"model returned no text (stop_reason={body.get('stop_reason')})")
        try:
            return json.loads(text)
        except ValueError:
            raise LLMError(
                f"model output was not JSON (stop_reason="
                f"{body.get('stop_reason')}): {text[:200]!r}") from None
