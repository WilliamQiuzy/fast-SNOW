"""Answer post-processing for VLM4D evaluation.

This module provides standardized answer extraction and formatting
for multiple-choice questions, ensuring consistency across all evaluations.

Phase 8 requirement: Unified answer extraction with deterministic rules.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def extract_multiple_choice_answer(
    response: str,
    choices: Dict[str, str],
    valid_choices: Optional[List[str]] = None,
) -> str:
    """Extract multiple choice answer (A, B, C, D) from VLM response.

    This implements the standardized extraction strategy from Phase 8 protocol:
    1. Direct match: Response starts with A/B/C/D
    2. Keyword match: "Answer: A" patterns
    3. Parenthesis match: "(A)" patterns
    4. First letter: First occurrence of A/B/C/D
    5. Failure: Return empty string

    Args:
        response: VLM's text response
        choices: Dict mapping choice keys to choice text (e.g., {"A": "0", "B": "1"})
        valid_choices: Optional list of valid choices (defaults to A-D)

    Returns:
        Extracted choice key in uppercase (A, B, C, D) or empty string if not found

    Examples:
        >>> extract_multiple_choice_answer("B", {"A": "0", "B": "1", "C": "2", "D": "3"})
        'B'

        >>> extract_multiple_choice_answer("The answer is B", {"A": "0", "B": "1"})
        'B'

        >>> extract_multiple_choice_answer("(C)", {"A": "yes", "B": "no", "C": "maybe"})
        'C'

        >>> extract_multiple_choice_answer("invalid response", {"A": "0", "B": "1"})
        ''
    """
    if valid_choices is None:
        valid_choices = list(choices.keys())

    valid_choices_upper = [c.upper() for c in valid_choices]
    response_clean = response.strip()
    response_upper = response_clean.upper()

    # Strategy 1: Direct match - response starts with ONLY a choice key
    # Must be single letter at start (optionally followed by space, period, or end)
    for key in valid_choices:
        key_upper = key.upper()
        if len(response_clean) == 1 and response_upper == key_upper:
            # Single letter response
            return key_upper
        elif response_clean.startswith(key_upper + " ") or \
             response_clean.startswith(key_upper + ".") or \
             response_clean.startswith(key_upper + ")") or \
             response_clean.startswith(key_upper + "]"):
            # Letter followed by space, period, or closing bracket
            return key_upper

    # Strategy 2: Keyword patterns - "Answer: A", "answer is B", etc.
    keyword_patterns = [
        r"answer[:\s]+([A-D])",           # "answer: A" or "answer A"
        r"answer\s+is[:\s]+([A-D])",     # "answer is A"
        r"answer\s+would\s+be[:\s]+([A-D])",  # "answer would be A"
        r"option[:\s]+([A-D])",          # "option: A"
        r"choice[:\s]+([A-D])",          # "choice: A"
        r"select[:\s]+([A-D])",          # "select: A"
        r"pick[:\s]+([A-D])",            # "pick: A"
    ]

    for pattern in keyword_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            candidate = match.group(1).upper()
            if candidate in valid_choices_upper:
                return candidate

    # Strategy 3: Parenthesis patterns - "(A)", "[B]", etc.
    paren_patterns = [
        r"\(([A-D])\)",   # (A)
        r"\[([A-D])\]",   # [A]
        r"\{([A-D])\}",   # {A}
    ]

    for pattern in paren_patterns:
        match = re.search(pattern, response)
        if match:
            candidate = match.group(1).upper()
            if candidate in valid_choices_upper:
                return candidate

    # Strategy 4: Choice text match - check if response contains the choice text
    # Only use this if no letter was found yet
    response_lower = response.lower()
    for key, text in choices.items():
        if text.lower() in response_lower and len(text) > 2:  # Avoid short spurious matches
            return key.upper()

    # Strategy 5: Standalone letter token (A-D)
    # Avoid matching letters inside words (e.g., "invalid")
    token_match = re.search(r"\b([A-D])\b", response_upper)
    if token_match:
        candidate = token_match.group(1).upper()
        if candidate in valid_choices_upper:
            return candidate

    # Strategy 6: Failure - return empty string
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer to uppercase single letter.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer (A, B, C, D) or empty string

    Examples:
        >>> normalize_answer("a")
        'A'

        >>> normalize_answer("  B  ")
        'B'

        >>> normalize_answer("")
        ''
    """
    normalized = answer.strip().upper()

    # Only accept single letters A-D
    if len(normalized) == 1 and normalized in ['A', 'B', 'C', 'D']:
        return normalized

    return ""


def check_answer_correctness(
    prediction: str,
    ground_truth: str,
    case_sensitive: bool = False,
) -> bool:
    """Check if prediction matches ground truth.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        case_sensitive: Whether to do case-sensitive comparison

    Returns:
        True if correct, False otherwise

    Examples:
        >>> check_answer_correctness("A", "A")
        True

        >>> check_answer_correctness("a", "A", case_sensitive=False)
        True

        >>> check_answer_correctness("a", "A", case_sensitive=True)
        False

        >>> check_answer_correctness("B", "A")
        False
    """
    if case_sensitive:
        return prediction == ground_truth
    else:
        return prediction.upper() == ground_truth.upper()


def extract_answer_with_confidence(
    response: str,
    choices: Dict[str, str],
) -> Tuple[str, float]:
    """Extract answer with confidence score.

    Confidence is based on which strategy matched:
    - Strategy 1-2 (direct/keyword): 1.0 (high confidence)
    - Strategy 3 (parenthesis): 0.9
    - Strategy 4 (text match): 0.7
    - Strategy 5 (first letter): 0.5 (low confidence)
    - No match: 0.0

    Args:
        response: VLM response
        choices: Choice dict

    Returns:
        Tuple of (extracted_answer, confidence_score)

    Examples:
        >>> extract_answer_with_confidence("Answer: B", {"A": "0", "B": "1"})
        ('B', 1.0)

        >>> extract_answer_with_confidence("The B seems correct", {"A": "0", "B": "1"})
        ('B', 0.5)
    """
    response_clean = response.strip()
    response_upper = response_clean.upper()
    valid_choices = list(choices.keys())
    valid_choices_upper = [c.upper() for c in valid_choices]

    # Strategy 1: Direct match
    for key in valid_choices:
        if response_upper.startswith(key.upper()):
            return key.upper(), 1.0

    # Strategy 2: Keyword patterns
    keyword_patterns = [
        r"answer[:\s]+([A-D])",
        r"answer\s+is[:\s]+([A-D])",
        r"answer\s+would\s+be[:\s]+([A-D])",
        r"option[:\s]+([A-D])",
        r"choice[:\s]+([A-D])",
    ]

    for pattern in keyword_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            candidate = match.group(1).upper()
            if candidate in valid_choices_upper:
                return candidate, 1.0

    # Strategy 3: Parenthesis
    paren_patterns = [r"\(([A-D])\)", r"\[([A-D])\]"]
    for pattern in paren_patterns:
        match = re.search(pattern, response)
        if match:
            candidate = match.group(1).upper()
            if candidate in valid_choices_upper:
                return candidate, 0.9

    # Strategy 4: Choice text match
    response_lower = response.lower()
    for key, text in choices.items():
        if text.lower() in response_lower and len(text) > 2:
            return key.upper(), 0.7

    # Strategy 5: Standalone letter token (A-D)
    token_match = re.search(r"\b([A-D])\b", response_upper)
    if token_match:
        candidate = token_match.group(1).upper()
        if candidate in valid_choices_upper:
            return candidate, 0.5

    # No match
    return "", 0.0


def validate_choices(choices: Dict[str, str]) -> bool:
    """Validate that choices dict has correct format.

    Args:
        choices: Choice dict to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_choices({"A": "option1", "B": "option2"})
        True

        >>> validate_choices({})
        False

        >>> validate_choices({"X": "invalid"})
        False
    """
    if not choices:
        return False

    # Check all keys are valid (A, B, C, or D)
    valid_keys = {'A', 'B', 'C', 'D'}
    for key in choices.keys():
        if key.upper() not in valid_keys:
            return False

    # Check all values are non-empty strings
    for value in choices.values():
        if not isinstance(value, str) or not value.strip():
            return False

    return True


def format_question_with_choices(
    question: str,
    choices: Dict[str, str],
    include_instruction: bool = True,
) -> str:
    """Format question with choices for VLM input.

    Phase 8 standard format:
    ```
    {question}
    A. {choice_a}
    B. {choice_b}
    C. {choice_c}
    D. {choice_d}

    Answer with just the letter choice (A, B, C, or D):
    ```

    Args:
        question: Question text
        choices: Choice dict
        include_instruction: Whether to include answer instruction

    Returns:
        Formatted question string

    Examples:
        >>> q = "How many cars?"
        >>> c = {"A": "0", "B": "1", "C": "2", "D": "3"}
        >>> print(format_question_with_choices(q, c, include_instruction=False))
        How many cars?
        A. 0
        B. 1
        C. 2
        D. 3
    """
    lines = [question]

    # Add choices in sorted order
    for key in sorted(choices.keys()):
        lines.append(f"{key}. {choices[key]}")

    if include_instruction:
        lines.append("")
        lines.append("Answer with just the letter choice (A, B, C, or D):")

    return "\n".join(lines)
