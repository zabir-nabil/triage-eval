import json
import re


def _answer_text(text):
    if text.lstrip().startswith(("[", "{", "```")):
        return text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for marker in (
        "<|channel|>final<|message|>",
        "<|start_header_id|>assistant<|end_header_id|>",
    ):
        if marker in text:
            text = text.rsplit(marker, 1)[1]
    for marker in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|return|>"):
        text = text.replace(marker, "")
    return text.strip()


def _recover_array(text):
    for start, character in enumerate(text):
        if character != "[":
            continue
        fragment = text[start:]
        end = fragment.rfind("}")
        while end >= 0:
            candidate = fragment[: end + 1].rstrip().rstrip(",") + "]"
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, list) and value:
                return value
            end = fragment.rfind("}", 0, end)
    return None


def _decode(text, recovery):
    warnings = []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        if isinstance(value, dict) and recovery and "id" in value:
            return [value], ["wrapped a single plan entry"]
        return value, warnings
    decoder = json.JSONDecoder()
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates.append(text)
    for candidate in candidates:
        for start, character in enumerate(candidate):
            if character != "[":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                return value, ["extracted a JSON array"]
    if recovery:
        for candidate in candidates:
            value = _recover_array(candidate)
            if value is not None:
                return value, ["retained complete entries from a truncated array"]
        for candidate in candidates:
            for start, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(candidate[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and "id" in value:
                    return [value], ["wrapped a single plan entry"]
    return None, warnings


def parse_plan(text, valid_ids, recovery=True):
    result = {"ok": False, "plan": [], "parse_attempts": 1, "parse_warnings": []}
    if not isinstance(text, str):
        result["parse_warnings"] = ["response is not text"]
        return result
    value, warnings = _decode(_answer_text(text), recovery)
    if not isinstance(value, list):
        result["parse_warnings"] = ["no recoverable JSON plan array"]
        return result
    valid = {str(problem_id) for problem_id in valid_ids}
    seen = set()
    cleaned = []
    for entry in value:
        if not isinstance(entry, dict):
            warnings.append("discarded a non-object entry")
            continue
        if entry.get("id") is None:
            warnings.append("discarded an entry without an id")
            continue
        problem_id = str(entry["id"])
        if problem_id not in valid:
            warnings.append(f"discarded unknown id: {problem_id}")
            continue
        if problem_id in seen:
            warnings.append(f"discarded duplicate id: {problem_id}")
            continue
        seen.add(problem_id)
        try:
            allocation = max(0, int(entry.get("tokens", 0)))
        except (TypeError, ValueError, OverflowError):
            allocation = 0
            warnings.append(f"set invalid tokens to zero: {problem_id}")
        cleaned.append(
            {
                "id": problem_id,
                "tokens": allocation,
                "reason": str(entry.get("reason", "")),
            }
        )
    result.update(ok=True, plan=cleaned, parse_warnings=warnings)
    return result
