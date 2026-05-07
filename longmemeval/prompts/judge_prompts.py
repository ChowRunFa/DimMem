
from __future__ import annotations

from typing import Any, Dict


LONGMEMEVAL_JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.
The point of the question is to ask about something one user should know about the other.
The gold answer will usually be a concise and short answer that includes the referenced user based on their prior conversations.
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading -
as long as it touches on the same topic as the gold answer, it should be counted as
CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The
generated answer might be much longer or use relative time references (like "last
Tuesday" or "next month"), but you should be generous with your grading - as long as
it refers to the same date or time period as the gold answer, it should be counted as
CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT
if it's the same date.

First, provide a short (one sentence) explanation of your reasoning, then finish with
CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation
script.
Just return the label CORRECT or WRONG in a json format with the key as "label".


Now it's time for the real question:
Question: {{query}}
Gold answer: {{gold_answer}}
Generated answer: {{generated_answer}}
"""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def build_judge_prompt(*, query: str, gold_answer: Any, model_answer: Any) -> str:
    return (
        LONGMEMEVAL_JUDGE_PROMPT.replace("{{query}}", _clean(query))
        .replace("{{gold_answer}}", _clean(gold_answer))
        .replace("{{generated_answer}}", _clean(model_answer))
    )


def build_judge_payload(*, query: str, gold_answer: Any, model_answer: Any) -> Dict[str, Any]:
    return {
        "query": _clean(query),
        "gold_answer": _clean(gold_answer),
        "model_answer": _clean(model_answer),
        "prompt": build_judge_prompt(
            query=query,
            gold_answer=gold_answer,
            model_answer=model_answer,
        ),
    }


__all__ = [
    "LONGMEMEVAL_JUDGE_PROMPT",
    "build_judge_prompt",
    "build_judge_payload",
]
