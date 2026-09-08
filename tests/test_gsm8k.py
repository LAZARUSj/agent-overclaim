import json

import pytest

from agent_overclaim.gsm8k import load_gsm8k

FIXTURE = "\n".join(
    json.dumps(
        {
            "question": f"Question number {i}?",
            "answer": f"reasoning #### {i * 2}",
        }
    )
    for i in range(30)
)


def test_load_parses_gold(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(FIXTURE)
    items = load_gsm8k(n=10, seed=0, source=path)
    assert len(items) == 10
    i = int(items[0]["id"].split("-")[-1])
    assert items[0]["gold"] == i * 2
    assert all("question" in item for item in items)


def test_deterministic_sampling(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(FIXTURE)
    a = load_gsm8k(n=10, seed=0, source=path)
    b = load_gsm8k(n=10, seed=0, source=path)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    c = load_gsm8k(n=10, seed=1, source=path)
    assert [x["id"] for x in a] != [x["id"] for x in c]


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gsm8k(n=5, source=tmp_path / "nope.jsonl")


def test_decoys_shipped_and_unanswerable_style():
    from agent_overclaim.provider import load_decoys

    decoys = load_decoys()
    assert len(decoys) == 60
    assert len({d["id"] for d in decoys}) == 60
    assert all(d["rationale"] for d in decoys)
    categories = {d["category"] for d in decoys}
    assert len(categories) == 6
