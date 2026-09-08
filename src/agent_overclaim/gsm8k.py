"""GSM8K loading, caching, and deterministic sampling."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

__all__ = ["GSM8K_TEST_URL", "load_gsm8k"]

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)

_CACHE_ENV = "AGENT_OVERCLAIM_CACHE"


def _cache_dir() -> Path:
    override = os.environ.get(_CACHE_ENV)
    root = Path(override) if override else Path.home() / ".cache" / "agent-overclaim"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _extract_gold(answer_text: str) -> float:
    match = re.search(r"####\s*\$?(-?\d[\d,]*(?:\.\d+)?)", answer_text)
    if match is None:
        raise ValueError(f"no final answer marker in: {answer_text[:120]!r}")
    return float(match.group(1).replace(",", ""))


def load_gsm8k(n: int = 200, seed: int = 0, source: str | Path | None = None) -> list[dict]:
    """Deterministically sampled GSM8K test items as graded tasks.

    Each item: ``{"id", "question", "gold"}``. ``source`` is a local
    JSONL path or a URL; defaults to the official test split (cached
    under ``~/.cache/agent-overclaim``).
    """
    source = Path(source) if source is not None else _cache_dir() / "gsm8k_test.jsonl"
    if not Path(source).exists() and not str(source).startswith("http"):
        raise FileNotFoundError(
            f"{source} not found; pass a local file or let the loader download"
        )
    if Path(source).exists():
        lines = Path(source).read_text(encoding="utf-8").strip().splitlines()
    else:
        import urllib.request

        try:
            urllib.request.urlretrieve(str(source), Path(source))
        except OSError as err:
            raise RuntimeError(
                f"could not download {source}; place the file manually and retry"
            ) from err
        lines = Path(source).read_text(encoding="utf-8").strip().splitlines()

    records = []
    for i, line in enumerate(lines):
        item = json.loads(line)
        records.append(
            {
                "id": f"gsm8k-{i:04d}",
                "question": item["question"].strip(),
                "gold": _extract_gold(item["answer"]),
            }
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(records), size=min(n, len(records)), replace=False)
    return [records[i] for i in sorted(chosen)]
