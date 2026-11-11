"""Provider-agnostic LLM client with an on-disk cache and a call log.

* Providers (env LLM_PROVIDER): "gemini" (GEMINI_API_KEY, Google AI Studio), "anthropic"
  (ANTHROPIC_API_KEY), "openai" (any
  OpenAI-compatible endpoint: OPENAI_BASE_URL + OPENAI_API_KEY, which also covers a local
  Ollama at http://localhost:11434/v1) and "stub" (development only, see below).
* Temperature is always 0.
* Every call is cached under LLM_CACHE_DIR by sha256(model, system, user, params). A cache
  hit costs nothing and works offline, so the demo can run from the shipped cache.
* LLM_OFFLINE=1 makes a cache miss an error instead of a network call.
* A hard cap (LLM_MAX_LIVE_CALLS, default 200) stops runaway loops, and the number of live
  (uncached) calls is counted and printed by the eval scripts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODELS = {"gemini": "gemini-3.5-flash", "anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini",
                  "stub": "stub-extractive"}


def load_env(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE lines). Real environment variables win."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class CacheMiss(RuntimeError):
    """Offline mode and the answer is not in the cache."""


class LiveCallCapReached(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    cached: bool
    model: str
    provider: str


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 cache_dir: str | Path | None = None, offline: bool | None = None,
                 max_live_calls: int | None = None):
        load_env()
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "stub").lower()
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"unknown LLM_PROVIDER {self.provider!r}")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self.cache_dir = Path(cache_dir or os.getenv("LLM_CACHE_DIR") or ROOT / "cache" / "llm")
        self.offline = offline if offline is not None else os.getenv("LLM_OFFLINE", "") == "1"
        self.max_live_calls = max_live_calls or int(os.getenv("LLM_MAX_LIVE_CALLS", "200"))
        self.live_calls = 0
        self.cache_hits = 0

    @property
    def is_stub(self) -> bool:
        return self.provider == "stub"

    # --- cache -------------------------------------------------------------------

    def _key(self, system: str, user: str, max_tokens: int) -> str:
        payload = json.dumps(
            {"model": self.model, "system": system, "user": user, "max_tokens": max_tokens, "temperature": 0},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _log(self, record: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- public API ----------------------------------------------------------------

    def complete(self, system: str, user: str, max_tokens: int = 500) -> LLMResponse:
        key = self._key(system, user, max_tokens)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            text = json.loads(path.read_text(encoding="utf-8"))["text"]
            self._log({"ts": time.time(), "key": key, "model": self.model, "cached": True})
            return LLMResponse(text, True, self.model, self.provider)
        if self.offline:
            raise CacheMiss(f"offline mode and no cached answer for key {key[:12]}")
        if self.live_calls >= self.max_live_calls:
            raise LiveCallCapReached(f"live call cap of {self.max_live_calls} reached")
        self.live_calls += 1
        text = self._call_provider(system, user, max_tokens)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": self.model, "system": system, "user": user, "text": text}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        self._log({"ts": time.time(), "key": key, "model": self.model, "cached": False})
        return LLMResponse(text, False, self.model, self.provider)

    def usage_line(self) -> str:
        return f"LLM calls: {self.live_calls} live, {self.cache_hits} from cache (model {self.model})"

    # --- providers -----------------------------------------------------------------

    def _call_provider(self, system: str, user: str, max_tokens: int) -> str:
        if self.provider == "stub":
            return _stub_answer(user)
        import requests  # imported here so tests and the offline path never need the network stack

        if self.provider == "gemini":
            return self._call_gemini(requests, system, user, max_tokens)

        if self.provider == "anthropic":
            url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages"
            headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                    "messages": [{"role": "user", "content": user}]}
            data = _post(requests, url, headers, body)
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()

        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        headers = {"authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'none')}", "content-type": "application/json"}
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        data = _post(requests, base + "/chat/completions", headers, body)
        return data["choices"][0]["message"]["content"].strip()


    def _call_gemini(self, requests, system: str, user: str, max_tokens: int) -> str:
        """Google Gemini REST API (generateContent). Key from GEMINI_API_KEY (or GOOGLE_API_KEY)."""
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY in .env (see .env.example)")
        base = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        # Gemini 3.x "thinks" before answering and thinking tokens count against maxOutputTokens,
        # so we leave extra room; otherwise a short answer can come back empty.
        gen = {"temperature": 0, "maxOutputTokens": max_tokens + 2048}
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen}
        headers = {"x-goog-api-key": key, "content-type": "application/json"}
        data = _post(requests, f"{base}/models/{self.model}:generateContent", headers, body)
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            reason = data.get("promptFeedback") or data.get("candidates", [{}])[0].get("finishReason")
            raise RuntimeError(f"Gemini returned no answer ({reason})") from None
        return "".join(p.get("text", "") for p in parts).strip()


_LAST_CALL = [0.0]


def _post(requests, url: str, headers: dict, body: dict, retries: int = 6) -> dict:
    """POST with spacing and retries on transient failures.

    - LLM_MIN_INTERVAL seconds (default 4.5) between live calls keeps us under a 15 requests/minute cap.
    - Read timeouts and dropped connections are retried (LLM_TIMEOUT seconds per attempt, default 180).
    - HTTP 429: a per-day quota stops at once with a clear message; a per-minute limit waits
      (as long as the server says, at least 20 s) and retries.
    A retry is safe: requests are deterministic and the disk cache only stores finished answers.
    """
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))
    gap = float(os.getenv("LLM_MIN_INTERVAL", "4.5"))
    transient = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
    for attempt in range(retries + 1):
        wait = _LAST_CALL[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except transient:
            _LAST_CALL[0] = time.time()
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        _LAST_CALL[0] = time.time()
        if r.status_code == 429:
            if "PerDay" in r.text:
                raise RuntimeError("LLM daily quota exhausted (HTTP 429, per-day). Finished answers are cached; "
                                   "rerun after the quota resets or use another model/billing.")
            if attempt < retries:
                m = re.search(r"retry in ([0-9.]+)s", r.text)
                time.sleep(min(max(float(m.group(1)) if m else 0.0, 20.0), 90.0))
                continue
        elif r.status_code in (500, 502, 503, 529) and attempt < retries:
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def _stub_answer(user: str) -> str:
    """DEVELOPMENT STUB, not a language model: quotes the first sentence of passage [P1].

    It lets the app and tests run with no key. It is refused by the eval scripts so it can
    never produce a reported number.
    """
    m = re.search(r"\[P1\][^\n]*\n(.+)", user)
    if not m:
        return "I cannot answer without a language model (stub provider)."
    sentence = re.split(r"(?<=[.!?])\s", m.group(1).replace("[FIXTURE] ", "").strip())[0]
    return f"{sentence} [P1]"
