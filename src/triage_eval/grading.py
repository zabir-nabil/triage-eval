import json
import re
import subprocess
import sys
from pathlib import Path


def extract_boxed(text):
    matches = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            matches.append(text[start : index - 1].strip())
    return matches[-1] if matches else None


def normalize_number(value):
    value = str(value).strip().replace(",", "").replace(" ", "")
    fraction = re.match(r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", value)
    if fraction:
        try:
            return float(fraction.group(1)) / float(fraction.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return None


def extract_letter(text):
    text = text.strip()
    match = re.search(
        r"(?:the\s+answer\s+is|answer:\s*|correct\s+answer\s+is)\s*[:\s]*\(?([A-E])\b\)?",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    tail = text[-200:]
    matches = re.findall(r"\b([A-E])\b(?:\s*[.)\]]|\s*$)", tail)
    if matches:
        return matches[-1].upper()
    match = re.search(r"\b([A-E])\s*[.!]?\s*$", tail)
    return match.group(1).upper() if match else None


def extract_final_answer(text):
    text = text.strip()
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.(?!\d)|$)",
        r"(?:^|\n)\s*(?:answer|Answer|ANSWER)[:\s]+(.+?)(?:\.(?!\d)|$)",
        r"\\boxed\{(.+?)\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[-1] if lines else text


def normalize_text(value):
    value = str(value).strip()
    value = re.sub(
        r"^(The answer is|Answer:|answer:)\s*", "", value, flags=re.IGNORECASE
    )
    value = value.strip(" .\"'")
    value = re.sub(r"\s+", " ", value)
    value = value.replace("\\left(", "(").replace("\\right)", ")")
    value = value.replace("\\left[", "[").replace("\\right]", "]")
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\$+", "", value)
    value = re.sub(r"\\[,(;:!)\s]", " ", value)
    return value.strip().lower()


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    text = text.strip()
    if text.startswith(("import ", "from ", "def ", "class ")):
        return text
    return None


def grade_aime(response, metadata):
    expected = normalize_number(metadata.get("answer", ""))
    boxed = extract_boxed(response)
    if expected is None or boxed is None:
        return False
    got = normalize_number(boxed)
    return got is not None and abs(got - expected) < 0.01


def grade_gpqa(response, metadata):
    expected = str(metadata.get("correct_letter", "")).upper()
    return bool(expected) and extract_letter(response) == expected


def grade_hle(response, metadata):
    answer_type = metadata.get("answer_type", "short_answer")
    expected = str(metadata.get("answer", ""))
    if answer_type in {"multipleChoice", "multiple_choice"}:
        return (
            len(expected.strip()) == 1
            and extract_letter(response) == expected.strip().upper()
        )
    if not expected:
        return False
    extracted = extract_final_answer(response)
    normalized_expected = normalize_text(expected)
    normalized_got = normalize_text(extracted)
    if normalized_expected == normalized_got:
        return True
    expected_number = normalize_number(expected)
    got_number = normalize_number(extracted)
    if expected_number is not None and got_number is not None:
        return abs(expected_number - got_number) < 0.01
    return len(normalized_expected) >= 2 and normalized_expected in normalized_got


def grade_livecodebench_entry(response, problem_id, test_cases_db):
    if test_cases_db is None or str(problem_id) not in test_cases_db:
        return None
    cases = test_cases_db[str(problem_id)]
    if not cases:
        return None
    code = extract_code(response)
    if code is None:
        return False
    for inputs, expected in cases:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                input=str(inputs),
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return False
        if result.stdout.strip() != str(expected).strip():
            return False
    return True


def _test_pairs(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if isinstance(value, dict):
        inputs = value.get("inputs", value.get("input", []))
        outputs = value.get("outputs", value.get("output", []))
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        return [(str(inp), str(out)) for inp, out in zip(inputs, outputs)]
    pairs = []
    if isinstance(value, list):
        for case in value:
            if isinstance(case, dict):
                inputs = case.get("input", case.get("stdin"))
                expected = case.get(
                    "output", case.get("stdout", case.get("expected_output"))
                )
                if inputs is not None and expected is not None:
                    pairs.append((str(inputs), str(expected)))
            elif isinstance(case, (list, tuple)) and len(case) == 2:
                pairs.append((str(case[0]), str(case[1])))
    return pairs


def load_livecodebench_tests(local_path=None):
    if local_path is not None:
        with Path(local_path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("LiveCodeBench tests must map problem IDs to test cases")
        return {str(pid): _test_pairs(cases) for pid, cases in data.items()}
    from .datasets import _livecodebench_rows

    out = {}
    for row in _livecodebench_rows():
        pairs = []
        for field in ("input_output", "public_test_cases", "private_test_cases"):
            pairs.extend(_test_pairs(row.get(field)))
        if pairs:
            out[str(row["question_id"])] = pairs
    return out


def grade_solvable_item(response, reference_answers):
    if not reference_answers or not response:
        return False
    refs = (
        reference_answers
        if isinstance(reference_answers, list)
        else [reference_answers]
    )
    refs = [str(ref) for ref in refs if ref is not None and str(ref).strip()]
    letter_refs = [
        ref.strip().upper()
        for ref in refs
        if len(ref.strip()) == 1 and ref.strip().upper() in "ABCDE"
    ]
    if letter_refs and extract_letter(response) in letter_refs:
        return True
    extracted = extract_final_answer(response)
    normalized_got = normalize_text(extracted)
    if not normalized_got:
        return False
    for ref in refs:
        normalized_ref = normalize_text(ref)
        if not normalized_ref:
            continue
        if normalized_ref == normalized_got:
            return True
        expected_number = normalize_number(ref)
        got_number = normalize_number(extracted)
        if expected_number is not None and got_number is not None:
            if abs(expected_number - got_number) < 1e-4 * max(
                1.0, abs(expected_number)
            ):
                return True
        if len(normalized_ref) >= 3 and normalized_ref in normalized_got:
            return True
        if len(normalized_got) >= 3 and normalized_got in normalized_ref:
            return True
    return False


def grade(problem, response, test_cases=None):
    metadata = problem.get("metadata", {})
    if metadata.get("unanswerable", problem.get("unanswerable", False)):
        return False
    dataset = problem.get("dataset", "")
    if dataset == "aime":
        return grade_aime(response, metadata)
    if dataset == "gpqa_diamond":
        return grade_gpqa(response, metadata)
    if dataset == "hle":
        return grade_hle(response, metadata)
    if dataset == "livecodebench":
        return grade_livecodebench_entry(response, problem["id"], test_cases)
    if dataset in {"gpqa_abstain", "mmlu_math_abstain"}:
        return grade_solvable_item(response, metadata.get("reference_answers", []))
    raise ValueError(f"Unknown dataset: {dataset}")
