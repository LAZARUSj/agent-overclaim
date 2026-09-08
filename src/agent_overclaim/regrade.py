"""Re-classify decoy labels in existing responses.jsonl files.

Grading rules improve as real responses reveal unanticipated wordings;
raw responses are the ground truth, labels are derived data. Point this at
any run directory to refresh its labels without re-calling any API:

    python -m agent_overclaim.regrade runs/deepseek [runs/other ...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_overclaim.grade import classify_decoy


def regrade(run_dir: Path) -> Counter[str]:
    path = run_dir / "responses.jsonl"
    records = []
    counter: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["task_type"] == "decoy":
            record["label"] = classify_decoy(
                record["response"], record["task_id"], record.get("question", "")
            )
        counter[record["label"]] += 1
        records.append(record)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return counter


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument(
        "--set-model",
        default=None,
        help="also rewrite the model field to this name (e.g. a served-model fix)",
    )
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        path = Path(run_dir)
        if args.set_model:
            records = [
                {**json.loads(line), "model": args.set_model}
                for line in (path / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            (path / "responses.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                encoding="utf-8",
            )
        counter = regrade(path)
        print(f"{run_dir}: {dict(counter)}")


if __name__ == "__main__":
    main()
