"""Run the overclaiming experiment.

Example (offline smoke test):
    python -m agent_overclaim.run_study --provider mock --out runs/mock

Real run (needs DEEPSEEK_API_KEY or ZHIPU_API_KEY in the environment):
    python -m agent_overclaim.run_study --provider deepseek --out runs/deepseek
    python -m agent_overclaim.run_study --provider zhipu --model glm-4.6 --out runs/zhipu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_overclaim.grade import classify_decoy, grade_real, parse_confidence, parse_final_answer
from agent_overclaim.gsm8k import load_gsm8k
from agent_overclaim.prompts import ARMS, build_prompt
from agent_overclaim.provider import chat_completion, load_decoys


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=["mock", "deepseek", "zhipu"])
    parser.add_argument("--model", default=None, help="model name (provider default if omitted)")
    parser.add_argument("--arms", nargs="+", default=["closed", "open"], choices=sorted(ARMS))
    parser.add_argument("--n-real", type=int, default=200)
    parser.add_argument("--n-decoy", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0, help="GSM8K sampling seed")
    parser.add_argument("--gsm8k", default=None, help="local GSM8K test JSONL (skip download)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = load_gsm8k(n=args.n_real, seed=args.seed, source=args.gsm8k)
    decoys = load_decoys()[: args.n_decoy]

    responses_path = out / "responses.jsonl"
    calls_path = out / "calls.jsonl"
    import time

    with responses_path.open("w", encoding="utf-8") as sink:
        for arm in args.arms:
            for task_type, tasks in (("real", real), ("decoy", decoys)):
                for task in tasks:
                    system, user = build_prompt(arm, task["question"])
                    text = chat_completion(
                        args.provider,
                        system,
                        user,
                        model=args.model,
                        task_id=task["id"],
                        gold=task.get("gold") if task_type == "real" else None,
                        log_path=calls_path if args.provider != "mock" else None,
                    )
                    if task_type == "real":
                        label = "correct" if grade_real(
                            text, task["id"], task["question"], task["gold"]
                        ) else "incorrect"
                        label = label if parse_final_answer(text) is not None else "no-answer"
                    else:
                        label = classify_decoy(text, task["id"], task["question"])
                    record = {
                        "task_type": task_type,
                        "task_id": task["id"],
                        "arm": arm,
                        "provider": args.provider,
                        "model": args.model or f"{args.provider}-default",
                        "question": task["question"],
                        "response": text,
                        "answer": parse_final_answer(text)
                        if not isinstance(parse_final_answer(text), str)
                        else "insufficient information",
                        "confidence": parse_confidence(text),
                        "label": label,
                    }
                    if task_type == "real":
                        record["gold"] = task["gold"]
                    else:
                        record["category"] = task.get("category", "")
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if args.sleep:
                        time.sleep(args.sleep)
    print(f"wrote {responses_path}")
    return responses_path


if __name__ == "__main__":
    main()
