# Data

This directory should contain the following two benchmark dataset files:

| File | Source | Description |
|------|--------|-------------|
| `longmemeval_s_cleaned.json` | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 500 QA items with haystack conversations, grouped by `question_type` field |
| `locomo10.json` | [LoCoMo](https://github.com/snap-stanford/LoCoMo) | 10 multi-session conversations with QA pairs in `qa` field |

## How to Obtain

Both datasets are publicly available:

1. **LongMemEval**: https://github.com/xiaowu0162/LongMemEval
2. **LoCoMo**: https://github.com/snap-stanford/LoCoMo

Download and place the processed JSON files in this directory. The pipeline scripts expect:
- `data/longmemeval_s_cleaned.json` — a JSON array of 500 objects, each with keys: `question_id`, `question_type`, `question`, `answer`, `haystack_sessions`, `haystack_dates`, `haystack_session_ids`
- `data/locomo10.json` — a JSON array of 10 objects, each with keys: `sample_id`, `conversation`, `qa`, `event_summary`, `observation`, `session_summary`
