# agent-overclaim

**How often do LLM agents confidently answer unanswerable questions?**

![CI](https://github.com/LAZARUSj/agent-overclaim/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-pre--release-orange)

An empirical study built on [falsify](https://github.com/LAZARUSj/falsify):
**decoy tasks** — math word problems that look perfectly ordinary but are
missing exactly one piece of information — measure whether a model
confabulates a definitive number (overclaims) or honestly declines.

## Design

| Component | What |
| --- | --- |
| **Real tasks** | GSM8K test items (deterministic sample), graded strictly against gold answers |
| **Decoy tasks** | 60 hand-written GSM8K-style problems, each verified unanswerable, in six failure categories (missing operand / missing base / underdetermined / missing referent / missing rate / scope gap) |
| **Arms** | `closed` — pipeline-style prompting that demands `Answer: <number>`; `open` — identical plus an explicit `Answer: insufficient information` escape hatch |
| **Models** | any OpenAI-compatible chat endpoint (DeepSeek, Zhipu GLM shipped; mock for offline runs) |
| **Metrics** | accuracy (Wilson CI), decoy overclaim / hedge / evasive rates (Wilson CI), confident-overclaim share (confidence ≥ 0.8), Brier score and ECE of self-reported confidence |

The headline quantity is the **honesty gain**: how much the overclaim rate
drops when the model is *told* it may decline — and whether that drop costs
accuracy on the real tasks.

## Quickstart

Offline smoke test (no API keys):

```bash
pip install -e ".[test]"
python -m agent_overclaim.run_study --provider mock --out runs/mock
python -m agent_overclaim.analyze runs/mock
```

Real run:

```bash
export DEEPSEEK_API_KEY=sk-...
python -m agent_overclaim.run_study --provider deepseek --out runs/deepseek --sleep 0.2

export ZHIPU_API_KEY=...
python -m agent_overclaim.run_study --provider zhipu --model glm-4.6 --out runs/zhipu --sleep 0.2

python -m agent_overclaim.analyze runs/deepseek --report results/deepseek.md
```

Every real API call is appended to `runs/<name>/calls.jsonl` with prompt,
response, latency, and status — full audit trail.

## What this is not

- Not a leaderboard: 60 decoys, 2 arms, temperature 0, single run per cell.
  The claim is the existence and rough magnitude of overclaiming, with
  Wilson intervals — not model rankings.
- Decoys are hand-written by the authors; the set ships in
  `src/agent_overclaim/data/decoys.jsonl` with a per-item rationale and is
  released for reuse and criticism.
- GSM8K has known contamination; this affects the accuracy arm (models may
  have seen items), not the decoy arm, which is novel by construction.

## Repository layout

- `src/agent_overclaim/gsm8k.py` — loader, cache, deterministic sampling
- `src/agent_overclaim/data/decoys.jsonl` — the decoy task set
- `src/agent_overclaim/prompts.py` — the two arms
- `src/agent_overclaim/provider.py` — DeepSeek / Zhipu / mock clients + JSONL audit log
- `src/agent_overclaim/grade.py` — parsing and grading via falsify primitives
- `src/agent_overclaim/run_study.py`, `analyze.py` — CLI runner and report generator
- `paper/paper.md` — short-report draft (results filled by `analyze`)

## License

MIT © Wenjie Zhang. GSM8K is distributed by its authors under the MIT license.
