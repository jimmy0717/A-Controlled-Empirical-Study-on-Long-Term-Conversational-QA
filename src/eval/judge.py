"""LLM-as-a-judge evaluator.

Wraps an :class:`LLM` instance and the ``prompts/judge.txt`` template to
return a boolean ``correct`` flag for each prediction.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from ..generator.llm import LLM
from ..prompting import fill_template

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "judge.txt"
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


class Judge:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.template = PROMPT_PATH.read_text(encoding="utf-8")

    def _parse(self, text: str) -> dict:
        m = _JSON_RE.search(text)
        if not m:
            return {"correct": False, "reason": "judge produced no JSON"}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"correct": False, "reason": "judge JSON malformed"}
        return {
            "correct": bool(obj.get("correct", False)),
            "reason": str(obj.get("reason", "")),
        }

    def grade(self, question: str, gold: str, pred: str) -> dict:
        prompt = fill_template(self.template, question=question, gold=gold, pred=pred)
        out = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=128,
            temperature=0.0,
        )
        return self._parse(out)

    def grade_batch(self, items: List[dict]) -> List[dict]:
        return [self.grade(it["question"], it["gold"], it["pred"]) for it in items]
