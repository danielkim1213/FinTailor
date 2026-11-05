"""OpenAI-compatible chat client. Works with llama.cpp server, vLLM, Ollama, LM Studio and most
hosted APIs. Configure via .env: LLM_BASE_URL (e.g. http://localhost:8080/v1), LLM_MODEL, LLM_API_KEY.
A separate TEACHER_* set is used for dataset labeling."""
from __future__ import annotations

import json
import re
import time

import requests
from pydantic import BaseModel, ValidationError

from ..config import env, get_config
from ..utils import log

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def extract_json(text: str):
    t = _THINK.sub("", text or "").strip()
    t = _FENCE.sub("", t).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        s, e = t.find(open_), t.rfind(close)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except ValueError:
                continue
    raise ValueError("no JSON found in model output")


class LLMClient:
    def __init__(self, prefix: str = "LLM", base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 120):
        self.base_url = (base_url or env(f"{prefix}_BASE_URL") or "").rstrip("/")
        self.model = model or env(f"{prefix}_MODEL")
        self.api_key = api_key or env(f"{prefix}_API_KEY", "none")
        self.timeout = timeout
        self.cfg = get_config()["llm"]
        self.latencies: list[float] = []

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int = 800,
             json_mode: bool = False) -> str:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": self.cfg["temperature"] if temperature is None else temperature}
        body.update(self.cfg.get("extra_body", {}))
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code == 400 and json_mode:  # some servers reject response_format / extra keys
            body.pop("response_format", None)
            for k in self.cfg.get("extra_body", {}):
                body.pop(k, None)
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        self.latencies.append(time.time() - t0)
        return r.json()["choices"][0]["message"]["content"] or ""

    def chat_json(self, messages: list[dict], schema: type[BaseModel] | None = None, max_tokens: int = 800,
                  retries: int | None = None):
        retries = self.cfg["max_retries"] if retries is None else retries
        err = None
        for _ in range(retries + 1):
            try:
                out = extract_json(self.chat(messages, max_tokens=max_tokens, json_mode=True))
                return schema.model_validate(out) if schema else out
            except (ValueError, ValidationError, requests.RequestException) as e:
                err = e
                messages = messages + [{"role": "user", "content": f"Your last output was invalid ({str(e)[:200]}). "
                                                                   "Reply with valid JSON only."}]
        log.warning("LLM JSON failed after retries: %s", err)
        return None
