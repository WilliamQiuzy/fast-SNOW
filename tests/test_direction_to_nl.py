"""Standalone test for 3D direction vector → natural language conversion.

Coordinate system: X=right, Y=down, Z=forward (away from camera).
Tests every single-axis, dual-axis, and triple-axis combination to ensure
no direction is missed or mislabelled.
"""

import math
import numpy as np

# ── Module under test ──────────────────────────────────────────────────────

AXIS_POS = {0: "rightward", 1: "downward", 2: "forward"}
AXIS_NEG = {0: "leftward", 1: "upward", 2: "backward"}

# Threshold: axis component must exceed this to be considered significant.
_AXIS_THRESH = 0.25


def direction_to_words(disp: np.ndarray) -> str:
    """Convert a 3D displacement vector to natural language direction words.

    Returns up to 2 direction words joined by ' and ', ordered by
    dominance priority: Z (depth) > X (lateral) > Y (vertical).
    Returns '' if the vector is near-zero.
    """
    dist = float(np.linalg.norm(disp))
    if dist < 1e-6:
        return ""

    normed = disp / dist

    # Collect significant axes, prioritise Z > X > Y
    parts: list[tuple[float, str]] = []
    for ax in (2, 0, 1):
        v = normed[ax]
        if abs(v) < _AXIS_THRESH:
            continue
        word = AXIS_POS[ax] if v > 0 else AXIS_NEG[ax]
        parts.append((abs(v), word))

    if not parts:
        # Fallback: all components below threshold → pick the largest
        ax = int(np.argmax(np.abs(normed)))
        word = AXIS_POS[ax] if normed[ax] > 0 else AXIS_NEG[ax]
        parts.append((abs(normed[ax]), word))

    # Sort by magnitude (strongest first), keep top 2
    parts.sort(key=lambda x: -x[0])
    words = [w for _, w in parts[:2]]
    return " and ".join(words)


# ── Test helpers ───────────────────────────────────────────────────────────

def _v(x, y, z):
    return np.array([x, y, z], dtype=float)


_PASS = 0
_FAIL = 0


def check(label, vec, *expected_keywords):
    """Assert that direction_to_words(vec) contains ALL expected keywords."""
    global _PASS, _FAIL
    result = direction_to_words(vec)
    missing = [kw for kw in expected_keywords if kw not in result]
    if missing:
        _FAIL += 1
        print(f"  FAIL  {label}")
        print(f"        vec={vec.tolist()}  result=\"{result}\"")
        print(f"        missing: {missing}")
    else:
        _PASS += 1
        print(f"  OK    {label:40s} → \"{result}\"")


# ── Tests ──────────────────────────────────────────────────────────────────

def test_zero_vector():
    print("\n── Zero / near-zero vector ──")
    assert direction_to_words(_v(0, 0, 0)) == "", "zero vector should return ''"
    assert direction_to_words(_v(1e-9, -1e-9, 1e-9)) == "", "near-zero should return ''"
    print("  OK    zero vectors handled correctly")


def test_single_axis():
    """Pure single-axis directions — 6 cardinal directions."""
    print("\n── Single axis (6 cardinal directions) ──")
    check("X+  (pure right)",    _v(1, 0, 0),    "rightward")
    check("X-  (pure left)",     _v(-1, 0, 0),   "leftward")
    check("Y+  (pure down)",     _v(0, 1, 0),    "downward")
    check("Y-  (pure up)",       _v(0, -1, 0),   "upward")
    check("Z+  (pure forward)",  _v(0, 0, 1),    "forward")
    check("Z-  (pure backward)", _v(0, 0, -1),   "backward")


def test_dual_axis():
    """Dual-axis combinations — 12 cases. Both axes should appear."""
    print("\n── Dual axis (12 combinations) ──")
    s = math.sqrt(0.5)  # ~0.707, both components clearly > threshold

    check("X+ Y+ (right-down)",    _v(s, s, 0),    "rightward", "downward")
    check("X+ Y- (right-up)",      _v(s, -s, 0),   "rightward", "upward")
    check("X- Y+ (left-down)",     _v(-s, s, 0),    "leftward", "downward")
    check("X- Y- (left-up)",       _v(-s, -s, 0),   "leftward", "upward")

    check("X+ Z+ (right-forward)", _v(s, 0, s),    "rightward", "forward")
    check("X+ Z- (right-back)",    _v(s, 0, -s),   "rightward", "backward")
    check("X- Z+ (left-forward)",  _v(-s, 0, s),    "leftward", "forward")
    check("X- Z- (left-back)",     _v(-s, 0, -s),   "leftward", "backward")

    check("Y+ Z+ (down-forward)",  _v(0, s, s),    "downward", "forward")
    check("Y+ Z- (down-back)",     _v(0, s, -s),   "downward", "backward")
    check("Y- Z+ (up-forward)",    _v(0, -s, s),    "upward", "forward")
    check("Y- Z- (up-back)",       _v(0, -s, -s),   "upward", "backward")


def test_triple_axis():
    """Triple-axis: all 8 octants. Top-2 dominant axes should appear."""
    print("\n── Triple axis (8 octants, top-2 shown) ──")
    # When all 3 axes are equal (|0.577| each), all exceed threshold.
    # Top-2 by magnitude should appear. Since they're equal, priority: Z > X > Y.
    s = 1.0 / math.sqrt(3)  # ~0.577

    check("+++  (right,down,fwd)",  _v(s, s, s),     "forward", "rightward")
    check("++-  (right,down,back)", _v(s, s, -s),    "backward", "rightward")
    check("+-+  (right,up,fwd)",    _v(s, -s, s),    "forward", "rightward")
    check("+--  (right,up,back)",   _v(s, -s, -s),   "backward", "rightward")
    check("-++  (left,down,fwd)",   _v(-s, s, s),    "forward", "leftward")
    check("-+-  (left,down,back)",  _v(-s, s, -s),   "backward", "leftward")
    check("--+  (left,up,fwd)",     _v(-s, -s, s),   "forward", "leftward")
    check("---  (left,up,back)",    _v(-s, -s, -s),  "backward", "leftward")


def test_asymmetric_triple():
    """Triple-axis with unequal magnitudes — dominant pair should surface."""
    print("\n── Asymmetric triple axis ──")
    # Z dominant, X secondary, Y minor
    check("Z>>X>Y  (fwd+slight right)",
          _v(0.3, 0.1, 0.9), "forward", "rightward")
    # X dominant, Z secondary, Y minor
    check("X>>Z>Y  (right+slight fwd)",
          _v(0.9, 0.1, 0.3), "rightward", "forward")
    # Y dominant, X secondary, Z minor
    check("Y>>X>Z  (down+slight right)",
          _v(0.3, 0.9, 0.1), "downward", "rightward")
    # Y dominant, Z secondary, X minor
    check("Y>>Z>X  (up+slight fwd)",
          _v(0.1, -0.9, 0.3), "upward", "forward")


def test_one_axis_below_threshold():
    """When one axis is barely significant and another is dominant."""
    print("\n── Edge: axis near threshold (0.25) ──")
    # X=0.24 (below threshold), Z=0.97 → only Z should appear
    check("X just below thresh",
          _v(0.24, 0.0, 0.97), "forward")
    result = direction_to_words(_v(0.24, 0.0, 0.97))
    assert "rightward" not in result, f"X=0.24 should be below threshold, got '{result}'"
    print(f"  OK    X=0.24 correctly excluded from \"{result}\"")

    # X=0.26 (above threshold), Z=0.97 → both should appear
    check("X just above thresh",
          _v(0.26, 0.0, 0.97), "forward", "rightward")


def test_real_camera_motion():
    """Test with the actual camera motion from our 4DSG."""
    print("\n── Real data: camera motion dir=[0.961,-0.031,0.275] ──")
    check("camera dir",
          _v(0.961, -0.031, 0.275), "rightward")
    # Z=0.275 is above threshold, so should also appear
    check("camera dir (Z component)",
          _v(0.961, -0.031, 0.275), "forward")
    # Y=-0.031 is far below threshold, should NOT appear
    result = direction_to_words(_v(0.961, -0.031, 0.275))
    assert "upward" not in result and "downward" not in result, \
        f"Y=-0.031 should be excluded, got '{result}'"
    print(f"  OK    Y=-0.031 correctly excluded from \"{result}\"")


def test_real_object_motions():
    """Test with actual object motion directions from our 4DSG."""
    print("\n── Real data: object motion directions ──")
    # Object 1: dir=[0.44,-0.092,0.893] → forward + right
    check("Obj1 dir=[0.44,-0.09,0.89]",
          _v(0.44, -0.092, 0.893), "forward", "rightward")

    # Object 4: dir=[0.467,-0.098,-0.879] → backward + right
    check("Obj4 dir=[0.47,-0.10,-0.88]",
          _v(0.467, -0.098, -0.879), "backward", "rightward")

    # Object 7: dir=[-0.489,-0.311,0.815] → forward + left (Y=-0.311 above thresh)
    check("Obj7 dir=[-0.49,-0.31,0.82]",
          _v(-0.489, -0.311, 0.815), "forward", "leftward")
    # Note: Y=-0.311 exceeds threshold but only top-2 are kept.
    # Z=0.815 > X=0.489 > Y=0.311, so top-2 = forward, leftward. upward is dropped.


def test_fallback_all_below_threshold():
    """When ALL axes are below threshold after normalisation (shouldn't happen
    with unit vector, but test the fallback anyway)."""
    print("\n── Edge: fallback when no axis exceeds threshold ──")
    # With a normalised vector, at least one axis must be >= 1/sqrt(3) ≈ 0.577.
    # So the fallback should only trigger for very unusual unnormalised inputs.
    # Construct a case: [0.2, 0.2, 0.2] normalised = [0.577, 0.577, 0.577] — all above.
    # Actually test: [0.01, 0.0, 0.0] — normalised to [1, 0, 0], well above.
    # The fallback is really for safety. Let's just verify it doesn't crash.
    check("tiny X-only",  _v(0.001, 0, 0), "rightward")
    check("tiny Y-only",  _v(0, -0.001, 0), "upward")
    check("tiny Z-only",  _v(0, 0, -0.001), "backward")


# ── Run all tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("direction_to_words() — comprehensive test suite")
    print("Coord system: X=right, Y=down, Z=forward. Threshold=0.25")
    print("=" * 70)

    test_zero_vector()
    test_single_axis()
    test_dual_axis()
    test_triple_axis()
    test_asymmetric_triple()
    test_one_axis_below_threshold()
    test_real_camera_motion()
    test_real_object_motions()
    test_fallback_all_below_threshold()

    print("\n" + "=" * 70)
    print(f"RESULTS:  {_PASS} passed,  {_FAIL} failed")
    print("=" * 70)

    if _FAIL > 0:
        exit(1)
    else:
        print("ALL TESTS PASSED")
