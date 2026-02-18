#!/usr/bin/env python3
"""Test suite for Phase 8 evaluation format and answer extraction.

This script validates:
1. Answer extraction works correctly for various response formats
2. Metrics calculation is accurate
3. Output format matches Phase 8 protocol

Run:
    python scripts/test_eval_format.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_snow.reasoning.eval.answer_postprocess import (
    extract_multiple_choice_answer,
    normalize_answer,
    check_answer_correctness,
    validate_choices,
    format_question_with_choices,
)
from fast_snow.reasoning.eval.metrics import Phase8Metrics, EvaluationResult, CategoryMetrics


def test_answer_extraction():
    """Test answer extraction with various response formats."""
    print("=" * 60)
    print("TEST 1: Answer Extraction")
    print("=" * 60)

    choices = {"A": "0", "B": "1", "C": "2", "D": "3"}

    test_cases = [
        # (response, expected_answer, description)
        ("A", "A", "Direct letter"),
        ("B", "B", "Direct letter B"),
        ("The answer is A", "A", "Answer is A"),
        ("I think the answer is B", "B", "I think answer is B"),
        ("Answer: C", "C", "Answer: C pattern"),
        ("(D)", "D", "Parenthesis (D)"),
        ("[B]", "B", "Bracket [B]"),
        ("Option A seems correct", "A", "Option A pattern"),
        ("Let me analyze... A", "A", "Trailing letter"),
        ("C is the correct choice", "C", "C is correct"),
        ("unknown or missing", "", "No valid answer"),
        ("", "", "Empty response"),
    ]

    all_passed = True

    for i, (response, expected, description) in enumerate(test_cases, 1):
        result = extract_multiple_choice_answer(response, choices)

        if result == expected:
            print(f"✅ Test {i}: {description}")
            print(f"   Response: '{response}' → '{result}'")
        else:
            print(f"❌ Test {i}: {description}")
            print(f"   Response: '{response}'")
            print(f"   Expected: '{expected}', Got: '{result}'")
            all_passed = False

    return all_passed


def test_answer_normalization():
    """Test answer normalization."""
    print("\n" + "=" * 60)
    print("TEST 2: Answer Normalization")
    print("=" * 60)

    test_cases = [
        ("a", "A"),
        ("  B  ", "B"),
        ("C", "C"),
        ("", ""),
        ("AB", ""),  # Invalid - too long
        ("1", ""),  # Invalid - not A-D
    ]

    all_passed = True

    for i, (input_val, expected) in enumerate(test_cases, 1):
        result = normalize_answer(input_val)

        if result == expected:
            print(f"✅ Test {i}: '{input_val}' → '{result}'")
        else:
            print(f"❌ Test {i}: '{input_val}'")
            print(f"   Expected: '{expected}', Got: '{result}'")
            all_passed = False

    return all_passed


def test_correctness_checking():
    """Test answer correctness checking."""
    print("\n" + "=" * 60)
    print("TEST 3: Correctness Checking")
    print("=" * 60)

    test_cases = [
        ("A", "A", True, True),
        ("a", "A", False, True),  # Case insensitive
        ("a", "A", True, False),  # Case sensitive
        ("B", "A", False, False),
        ("", "A", False, False),
    ]

    all_passed = True

    for i, (pred, truth, case_sens, expected) in enumerate(test_cases, 1):
        result = check_answer_correctness(pred, truth, case_sensitive=case_sens)

        if result == expected:
            print(f"✅ Test {i}: pred='{pred}', truth='{truth}', "
                  f"case_sens={case_sens} → {result}")
        else:
            print(f"❌ Test {i}: pred='{pred}', truth='{truth}'")
            print(f"   Expected: {expected}, Got: {result}")
            all_passed = False

    return all_passed


def test_choices_validation():
    """Test choices dict validation."""
    print("\n" + "=" * 60)
    print("TEST 4: Choices Validation")
    print("=" * 60)

    test_cases = [
        ({"A": "opt1", "B": "opt2"}, True, "Valid 2 choices"),
        ({"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"}, True, "Valid 4 choices"),
        ({}, False, "Empty choices"),
        ({"X": "invalid"}, False, "Invalid key"),
        ({"A": ""}, False, "Empty value"),
        ({"A": "opt1", "B": None}, False, "None value"),
    ]

    all_passed = True

    for i, (choices, expected, description) in enumerate(test_cases, 1):
        result = validate_choices(choices)

        if result == expected:
            print(f"✅ Test {i}: {description} → {result}")
        else:
            print(f"❌ Test {i}: {description}")
            print(f"   Expected: {expected}, Got: {result}")
            all_passed = False

    return all_passed


def test_question_formatting():
    """Test question formatting."""
    print("\n" + "=" * 60)
    print("TEST 5: Question Formatting")
    print("=" * 60)

    question = "How many cars?"
    choices = {"A": "0", "B": "1", "C": "2", "D": "3"}

    # With instruction
    formatted = format_question_with_choices(question, choices, include_instruction=True)

    expected_lines = [
        "How many cars?",
        "A. 0",
        "B. 1",
        "C. 2",
        "D. 3",
        "",
        "Answer with just the letter choice (A, B, C, or D):",
    ]

    expected = "\n".join(expected_lines)

    if formatted == expected:
        print("✅ Question formatting (with instruction) correct")
        print("\nFormatted output:")
        print("-" * 40)
        print(formatted)
        print("-" * 40)
        return True
    else:
        print("❌ Question formatting incorrect")
        print("\nExpected:")
        print(expected)
        print("\nGot:")
        print(formatted)
        return False


def test_metrics_calculation():
    """Test metrics calculation."""
    print("\n" + "=" * 60)
    print("TEST 6: Metrics Calculation")
    print("=" * 60)

    metrics = Phase8Metrics()

    # Add test results
    test_data = [
        ("001", "ego-centric", "Q1", "A", "A", True),
        ("002", "ego-centric", "Q2", "B", "A", False),
        ("003", "exo-centric", "Q3", "C", "C", True),
        ("004", "exo-centric", "Q4", "D", "C", False),
        ("005", "directional", "Q5", "A", "A", True),
    ]

    for sample_id, cat, question, pred, truth, is_correct in test_data:
        result = EvaluationResult(
            sample_id=sample_id,
            category=cat,
            question=question,
            prediction=pred,
            ground_truth=truth,
            is_correct=is_correct,
        )
        metrics.add_result(result)

    metrics.compute_metrics()

    # Check overall metrics
    expected_total = 5
    expected_correct = 3
    expected_accuracy = 0.6

    all_passed = True

    if metrics.total_samples == expected_total:
        print(f"✅ Total samples: {metrics.total_samples}")
    else:
        print(f"❌ Total samples: expected {expected_total}, got {metrics.total_samples}")
        all_passed = False

    if metrics.correct_samples == expected_correct:
        print(f"✅ Correct samples: {metrics.correct_samples}")
    else:
        print(f"❌ Correct samples: expected {expected_correct}, got {metrics.correct_samples}")
        all_passed = False

    if abs(metrics.overall_accuracy - expected_accuracy) < 0.001:
        print(f"✅ Overall accuracy: {metrics.overall_accuracy:.2%}")
    else:
        print(f"❌ Overall accuracy: expected {expected_accuracy:.2%}, "
              f"got {metrics.overall_accuracy:.2%}")
        all_passed = False

    # Check category metrics
    print(f"\nCategory metrics:")
    for cat_name, cat_metrics in sorted(metrics.categories.items()):
        print(f"  {cat_name}: {cat_metrics}")

    return all_passed


def test_output_format():
    """Test output format matches Phase 8 protocol."""
    print("\n" + "=" * 60)
    print("TEST 7: Output Format")
    print("=" * 60)

    metrics = Phase8Metrics()
    metrics.config = {"model": "test", "temperature": 0.0}
    metrics.timestamp = "2025-01-01T00:00:00"

    # Add sample results
    for i in range(4):
        result = EvaluationResult(
            sample_id=f"test_{i:03d}",
            category="ego-centric",
            question=f"Question {i}",
            prediction="A",
            ground_truth="A",
            is_correct=True,
        )
        metrics.add_result(result)

    metrics.compute_metrics()

    # Check dict format
    output_dict = metrics.to_dict(include_results=False)

    required_keys = {"config", "environment", "timestamp", "metrics"}
    if set(output_dict.keys()) == required_keys:
        print(f"✅ Output dict has required keys: {list(output_dict.keys())}")
    else:
        print(f"❌ Output dict keys incorrect")
        print(f"   Expected: {required_keys}")
        print(f"   Got: {set(output_dict.keys())}")
        return False

    # Check metrics section
    metrics_section = output_dict["metrics"]
    required_metrics_keys = {
        "overall_accuracy",
        "total_samples",
        "correct_samples",
        "category_metrics",
    }

    if set(metrics_section.keys()) == required_metrics_keys:
        print(f"✅ Metrics section has required keys")
    else:
        print(f"❌ Metrics section keys incorrect")
        print(f"   Expected: {required_metrics_keys}")
        print(f"   Got: {set(metrics_section.keys())}")
        return False

    print(f"\n✅ Output format validation passed")
    print(f"\nSample output structure:")
    import json
    print(json.dumps(output_dict, indent=2)[:500] + "...")

    return True


def main():
    """Run all tests."""
    print("\n" + "🧪" * 30)
    print("Phase 8 Evaluation Format Test Suite")
    print("🧪" * 30 + "\n")

    results = []

    results.append(("Answer Extraction", test_answer_extraction()))
    results.append(("Answer Normalization", test_answer_normalization()))
    results.append(("Correctness Checking", test_correctness_checking()))
    results.append(("Choices Validation", test_choices_validation()))
    results.append(("Question Formatting", test_question_formatting()))
    results.append(("Metrics Calculation", test_metrics_calculation()))
    results.append(("Output Format", test_output_format()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
