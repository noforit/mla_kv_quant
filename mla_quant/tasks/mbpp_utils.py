"""Reuse the upstream MBPP tests and examples with literal stop normalization."""


def normalize_response(text):
    ends = [text.find(marker) for marker in ("[DONE]", "[END]") if marker in text]
    return text[: min(ends)] if ends else text


def pass_at_1(references, predictions):
    from lm_eval.tasks.mbpp.utils import pass_at_1 as upstream

    if isinstance(predictions[0], str):
        predictions = [[normalize_response(text)] for text in predictions]
    else:
        predictions = [[normalize_response(text) for text in row] for row in predictions]
    return upstream(references, predictions)


def list_fewshot_samples():
    from lm_eval.tasks.mbpp.utils import list_fewshot_samples as upstream

    return upstream()
