"""Filters for `humaneval_instruct_norm`.

Keeps lm-eval's original code-path semantics (cut at the first code fence,
concatenate with ``doc["prompt"]``, score with HF ``evaluate``) and only adds
one deterministic repair: the continuation's first non-empty line is aligned
to the indentation level implied by the prompt's last line.

The repair changes nothing when the model already continues correctly, and it
never touches a continuation whose first line is a top-level definition.
"""

import evaluate as hf_evaluate


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    compute_ = hf_evaluate.load("code_eval")
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(references=references, predictions=predictions, k=k)
    return res[0]


def _last_line_indent(text: str) -> int:
    for line in reversed(text.split("\n")):
        if line.strip():
            return len(line) - len(line.lstrip(" "))
    return 0


def align_continuation(prompt: str, continuation: str) -> str:
    """Repair a partially indented first continuation line; preserve top-level definitions."""
    target = _last_line_indent(prompt)
    if target <= 0:
        return continuation
    lines = continuation.split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        stripped = line.lstrip(" ")
        current = len(line) - len(stripped)
        if 0 < current < target:
            lines[index] = " " * (target - current) + line
        return "\n".join(lines)
    return continuation


def build_predictions_instruct(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    predictions = []
    for resp, doc in zip(resps, docs):
        prompt = doc["prompt"]
        candidates = []
        for r in resp:
            body = r if r.find("```") == -1 else r[: r.find("```")]
            candidates.append(prompt + align_continuation(prompt, body))
        predictions.append(candidates)
    return predictions
