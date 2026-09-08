# Changelog

## 0.0.1 — 2026-09-08

Initial release.

- GSM8K loader with caching and deterministic sampling
- 60 hand-written decoy tasks (6 categories, per-item rationale)
- Closed/open prompt arms; DeepSeek / Zhipu / mock providers with JSONL audit logs
- Grading via falsify primitives (HiddenTest, DecoyTask) + Brier/ECE
- Wilson CIs on accuracy and overclaim rates; honesty-gain summary
- Offline end-to-end test via the mock provider
