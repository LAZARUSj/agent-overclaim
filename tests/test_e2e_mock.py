import json

from agent_overclaim.analyze import analyze, render_markdown
from agent_overclaim.prompts import build_prompt
from agent_overclaim.run_study import main as run_main


def _fixture_gsm8k(tmp_path):
    import numpy as np

    lines = []
    for i in range(40):
        gold = int(np.random.default_rng(i).integers(1, 500))
        lines.append(json.dumps({"question": f"Compute problem {i}?", "answer": f"#### {gold}"}))
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(lines))
    return path


def test_build_prompt_arms():
    system_closed, user_closed = build_prompt("closed", "What is 2+2?")
    system_open, user_open = build_prompt("open", "What is 2+2?")
    assert system_closed == system_open
    assert "insufficient information" in user_open
    assert "insufficient information" not in user_closed
    assert "Answer: <number>" in user_closed


def test_mock_end_to_end(tmp_path):
    responses = run_main(
        [
            "--provider", "mock",
            "--arms", "closed", "open",
            "--n-real", "40",
            "--n-decoy", "30",
            "--gsm8k", str(_fixture_gsm8k(tmp_path)),
            "--out", str(tmp_path / "run"),
        ]
    )
    lines = responses.read_text().strip().splitlines()
    assert len(lines) == 2 * (40 + 30)
    records = [json.loads(line) for line in lines]
    labels = {r["label"] for r in records if r["task_type"] == "decoy"}
    assert {"hedged", "overclaim", "evasive"} <= labels  # mock produces all classes

    summary = analyze(tmp_path / "run")
    by_arm = {r["arm"]: r for r in summary["runs"]}
    assert set(by_arm) == {"closed", "open"}
    for run in summary["runs"]:
        assert 0.0 <= run["accuracy"] <= 1.0
        assert 0.0 <= run["overclaim_rate"] <= 1.0
        assert run["n_real"] == 40 and run["n_decoy"] == 30
        low, high = run["overclaim_ci"]
        assert 0.0 <= low <= run["overclaim_rate"] <= high <= 1.0
    assert 0.0 < by_arm["closed"]["accuracy"] < 1.0  # mock is imperfect by design
    assert summary["honesty_gain"] or summary["honesty_gain"] == []

    markdown = render_markdown(summary)
    assert "| provider | model | arm |" in markdown
    assert "Honesty gain" in markdown
