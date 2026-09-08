"""Chat providers: OpenAI-compatible endpoints (DeepSeek, Zhipu GLM) and a
deterministic mock for offline development and testing."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

__all__ = ["PROVIDERS", "chat_completion", "load_decoys"]

PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env": "ZHIPU_API_KEY",
        "default_model": "glm-4.6",
    },
    "mock": {"base_url": None, "env": None, "default_model": "mock-1"},
}


def _mock_response(arm: str, task_id: str, gold: float | None, provider: str) -> str:
    """Deterministic pseudo-model: correct ~55% on real tasks, hedges ~30%
    of decoys, confabulates confident numbers otherwise."""
    seed = int.from_bytes(
        hashlib.sha256(f"{provider}:{arm}:{task_id}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    if gold is not None:
        if rng.random() < 0.55:
            answer, conf = gold, round(float(rng.uniform(0.55, 0.98)), 2)
        else:
            answer = gold + float(rng.integers(1, 9)) * 10
            conf = round(float(rng.uniform(0.4, 0.95)), 2)
        text = f"Let me work through it step by step.\nAnswer: {answer:g}\nConfidence: {conf}"
        if arm == "open" and rng.random() < 0.03:
            text += "\n(insufficient information was considered)"
        return text
    draw = rng.random()
    if draw < 0.30:
        return (
            "The question does not provide enough information to compute a "
            "value.\nAnswer: insufficient information\n"
            f"Confidence: {round(float(rng.uniform(0.6, 0.95)), 2)}"
        )
    if draw < 0.35:
        return (
            "Hmm, this one is unclear.\nConfidence: "
            f"{round(float(rng.uniform(0.3, 0.6)), 2)}"
        )
    confab = round(float(rng.uniform(20, 400)), 1)
    return (
        "Computing from the given numbers.\n"
        f"Answer: {confab:g}\nConfidence: {round(float(rng.uniform(0.8, 0.99)), 2)}"
    )


def chat_completion(
    provider: str,
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 600,
    task_id: str | None = None,
    gold: float | None = None,
    log_path: str | Path | None = None,
) -> str:
    """One chat completion; every real call is appended to ``log_path`` as
    a JSONL audit record (prompt, response, latency, status)."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")
    spec = PROVIDERS[provider]
    model = model or spec["default_model"]
    if provider == "mock":
        return _mock_response(
            arm=_arm_from_user(user), task_id=task_id or "unknown", gold=gold,
            provider=provider,
        )

    import os

    import requests

    api_key = os.environ.get(spec["env"])
    if not api_key:
        raise RuntimeError(f"set {spec['env']} to use the {provider} provider")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{spec['base_url']}/chat/completions"

    last_error = None
    for attempt in range(4):
        started = time.time()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=180)
            latency = round(time.time() - started, 2)
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if log_path is not None:
                _append_log(
                    log_path,
                    {
                        "task_id": task_id,
                        "provider": provider,
                        "model": model,
                        "system": system,
                        "user": user,
                        "response": content,
                        "latency_s": latency,
                        "status": response.status_code,
                    },
                )
            return content
        except requests.RequestException as err:
            last_error = str(err)
            time.sleep(2**attempt)
    raise RuntimeError(f"{provider} failed after retries: {last_error}")


def _arm_from_user(user: str) -> str:
    return "open" if "insufficient information" in user else "closed"


def _append_log(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_decoys() -> list[dict]:
    """The hand-written decoy task set shipped with the package."""
    from importlib.resources import files

    path = files("agent_overclaim").joinpath("data/decoys.jsonl")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
