"""Tests for _compute_motion_descriptor — phase-aware motion description.

Creates synthetic trajectories with known motion patterns and verifies
the descriptor output matches expectations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.pipeline.rose_pipeline import (
    ROSEPipeline,
    _FrameObservation,
)
from rose.reasoning.tokens.step_encoding import STEPToken
from rose.reasoning.tokens.geometry_tokens import CentroidToken, ShapeToken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(positions: np.ndarray, fps: float = 10.0) -> List[_FrameObservation]:
    """Create a list of _FrameObservation from an Nx3 position array."""
    obs = []
    for i, pos in enumerate(positions):
        centroid = CentroidToken(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        shape = ShapeToken(
            x_mu=0.05, x_sigma=0.01, x_min=0, x_max=0.1,
            y_mu=0.05, y_sigma=0.01, y_min=0, y_max=0.1,
            z_mu=0.05, z_sigma=0.01, z_min=0, z_max=0.1,
        )
        step = STEPToken(patch_tokens=[], centroid=centroid, shape=shape, temporal=None)
        obs.append(_FrameObservation(
            frame_idx=i,
            timestamp_s=i / fps,
            step=step,
        ))
    return obs


def _make_pipeline() -> ROSEPipeline:
    """Create a minimal pipeline instance for calling _compute_motion_descriptor."""
    return ROSEPipeline(ROSEConfig())


_PASS = 0
_FAIL = 0


def check(label: str, obs: List[_FrameObservation], *expected_keywords: str,
          forbidden: Tuple[str, ...] = ()):
    """Assert descriptor contains ALL expected keywords and NONE of the forbidden."""
    global _PASS, _FAIL
    pipe = _make_pipeline()
    result = pipe._compute_motion_descriptor(obs)

    missing = [kw for kw in expected_keywords if kw not in result]
    present_forbidden = [kw for kw in forbidden if kw in result]
    ok = not missing and not present_forbidden

    if ok:
        _PASS += 1
        print(f"  OK    {label:55s} → \"{result}\"")
    else:
        _FAIL += 1
        print(f"  FAIL  {label}")
        print(f"        result = \"{result}\"")
        if missing:
            print(f"        missing:   {missing}")
        if present_forbidden:
            print(f"        forbidden: {present_forbidden}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_stationary():
    """Object that doesn't move."""
    print("\n── Stationary ──")
    pos = np.tile([0.5, 0.0, 1.0], (30, 1))
    # Add tiny jitter (noise)
    pos += np.random.RandomState(42).randn(30, 3) * 0.001
    check("no movement (30 frames)", _make_obs(pos), "stationary")

    # Single observation
    check("single obs", _make_obs(pos[:1]), "stationary")

    # Two frames, same position
    check("two frames, same pos", _make_obs(pos[:2]), "stationary")


def test_linear_rightward():
    """Constant velocity rightward (X+)."""
    print("\n── Linear rightward ──")
    n = 50
    t = np.linspace(0, 1, n)
    pos = np.zeros((n, 3))
    pos[:, 0] = t * 0.5  # X: 0 → 0.5 over 5 seconds (at 10fps)
    pos[:, 2] = 1.0  # constant Z
    check("rightward 0.5m over 5s", _make_obs(pos),
          "moving", "rightward", forbidden=("oscillat", "then"))


def test_linear_forward():
    """Constant velocity forward (Z+)."""
    print("\n── Linear forward ──")
    n = 50
    t = np.linspace(0, 1, n)
    pos = np.zeros((n, 3))
    pos[:, 2] = 1.0 + t * 2.0  # Z: 1 → 3
    check("forward 2m over 5s", _make_obs(pos),
          "moving", "forward", forbidden=("oscillat", "then"))


def test_linear_diagonal():
    """Constant velocity rightward + forward."""
    print("\n── Linear diagonal (right + forward) ──")
    n = 50
    t = np.linspace(0, 1, n)
    pos = np.zeros((n, 3))
    pos[:, 0] = t * 0.8  # X: rightward
    pos[:, 2] = 1.0 + t * 1.5  # Z: forward
    check("rightward+forward", _make_obs(pos),
          "moving", "rightward", "forward", forbidden=("oscillat", "then"))


def test_back_and_forth_x():
    """Goes right then returns left — one reversal."""
    print("\n── Back and forth (right then left) ──")
    n = 50
    t = np.linspace(0, 5, n)
    pos = np.zeros((n, 3))
    pos[:, 2] = 1.0
    # Right for first half, left for second half
    mid = n // 2
    pos[:mid, 0] = np.linspace(0, 0.5, mid)
    pos[mid:, 0] = np.linspace(0.5, 0.0, n - mid)
    check("right then left (returns to start)", _make_obs(pos, fps=10),
          "then", "right", "left",
          forbidden=("stationary",))


def test_back_and_forth_z():
    """Goes forward then backward — one reversal."""
    print("\n── Back and forth (forward then backward) ──")
    n = 50
    t = np.linspace(0, 5, n)
    pos = np.zeros((n, 3))
    mid = n // 2
    pos[:mid, 2] = np.linspace(1.0, 2.0, mid)
    pos[mid:, 2] = np.linspace(2.0, 1.0, n - mid)
    check("forward then backward", _make_obs(pos, fps=10),
          "then", "forward", "backward",
          forbidden=("stationary",))


def test_three_phases():
    """Right, then left, then right again — 2 reversals."""
    print("\n── Three phases (right→left→right) ──")
    n = 60
    pos = np.zeros((n, 3))
    pos[:, 2] = 1.0
    # Phase 1: right (0-2s)
    pos[:20, 0] = np.linspace(0, 0.5, 20)
    # Phase 2: left (2-4s)
    pos[20:40, 0] = np.linspace(0.5, -0.3, 20)
    # Phase 3: right (4-6s)
    pos[40:, 0] = np.linspace(-0.3, 0.3, 20)
    check("right→left→right", _make_obs(pos, fps=10),
          "then", "right", "left")


def test_oscillating_x():
    """Repeated left-right oscillation (>=3 reversals)."""
    print("\n── Oscillating left-right ──")
    n = 100
    t = np.linspace(0, 10, n)
    pos = np.zeros((n, 3))
    pos[:, 2] = 1.0
    pos[:, 0] = 0.3 * np.sin(2 * np.pi * t / 2.5)  # ~4 full cycles
    check("sine oscillation X (4 cycles)", _make_obs(pos, fps=10),
          "oscillating", "left-right", "amplitude",
          forbidden=("stationary",))


def test_oscillating_z():
    """Repeated forward-backward oscillation."""
    print("\n── Oscillating forward-backward ──")
    n = 100
    t = np.linspace(0, 10, n)
    pos = np.zeros((n, 3))
    pos[:, 2] = 1.0 + 0.4 * np.sin(2 * np.pi * t / 3.0)
    check("sine oscillation Z", _make_obs(pos, fps=10),
          "oscillating", "forward-backward", "amplitude",
          forbidden=("stationary",))


def test_stationary_then_moving():
    """Stays still for 3s, then moves rightward for 2s."""
    print("\n── Stationary then moving ──")
    n = 50
    pos = np.zeros((n, 3))
    pos[:, 2] = 1.0
    # First 30 frames (3s): stationary
    # Last 20 frames (2s): rightward
    pos[30:, 0] = np.linspace(0, 0.8, 20)
    check("still 3s then rightward 2s", _make_obs(pos, fps=10),
          "then", "right",
          forbidden=("oscillat",))


def test_moving_then_stationary():
    """Moves forward for 2s, then stops for 3s."""
    print("\n── Moving then stationary ──")
    n = 50
    pos = np.zeros((n, 3))
    # First 20 frames: forward
    pos[:20, 2] = np.linspace(1.0, 2.5, 20)
    # Last 30 frames: stationary at z=2.5
    pos[20:, 2] = 2.5
    check("forward 2s then stationary 3s", _make_obs(pos, fps=10),
          "then", "forward", "stationary",
          forbidden=("oscillat",))


def test_real_synth001_obj0():
    """Object 0 from synth_001 — slight rightward+forward drift."""
    print("\n── Real data: synth_001 object 0 ──")
    # From the actual 4dsg.json
    points = [
        [0.35, -0.03, 1.17], [0.35, -0.03, 1.18], [0.35, -0.04, 1.17],
        [0.35, -0.03, 1.16], [0.35, -0.03, 1.17], [0.36, -0.03, 1.18],
        [0.36, -0.03, 1.18], [0.36, -0.04, 1.20], [0.37, -0.04, 1.21],
        [0.37, -0.04, 1.21],
    ]
    pos = np.array(points)
    obs = _make_obs(pos, fps=10)
    # Override timestamps to match real data (4.12-5.0s)
    ts = [4.12, 4.21, 4.33, 4.42, 4.5, 4.62, 4.71, 4.83, 4.92, 5.0]
    real_obs = []
    for i, o in enumerate(obs):
        real_obs.append(_FrameObservation(
            frame_idx=o.frame_idx, timestamp_s=ts[i], step=o.step,
        ))
    # Net displacement is very small (~0.02 right, ~0.04 forward)
    # Should be stationary or slight movement
    pipe = _make_pipeline()
    result = pipe._compute_motion_descriptor(real_obs)
    print(f"  INFO  synth_001 obj 0: \"{result}\"")
    # Just verify it doesn't crash and produces something reasonable
    assert isinstance(result, str) and len(result) > 0
    _pass_inc()


def test_short_trajectory():
    """Very short trajectory (3 frames)."""
    print("\n── Short trajectory (3 frames) ──")
    pos = np.array([[0, 0, 1.0], [0.2, 0, 1.0], [0.4, 0, 1.0]])
    check("3 frames rightward", _make_obs(pos, fps=10),
          "moving", "rightward", forbidden=("oscillat", "then"))


def test_noisy_stationary():
    """Stationary with significant jitter — should still be stationary."""
    print("\n── Noisy stationary ──")
    n = 50
    rng = np.random.RandomState(123)
    pos = np.tile([0, 0, 1.0], (n, 1))
    pos += rng.randn(n, 3) * 0.005  # ~5mm noise
    check("stationary with 5mm noise", _make_obs(pos, fps=10), "stationary")


def _pass_inc():
    global _PASS
    _PASS += 1


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("_compute_motion_descriptor() — phase-aware motion test suite")
    print("Coord system: X=right, Y=down, Z=forward")
    print("=" * 70)

    test_stationary()
    test_linear_rightward()
    test_linear_forward()
    test_linear_diagonal()
    test_back_and_forth_x()
    test_back_and_forth_z()
    test_three_phases()
    test_oscillating_x()
    test_oscillating_z()
    test_stationary_then_moving()
    test_moving_then_stationary()
    test_real_synth001_obj0()
    test_short_trajectory()
    test_noisy_stationary()

    print("\n" + "=" * 70)
    print(f"RESULTS:  {_PASS} passed,  {_FAIL} failed")
    print("=" * 70)

    if _FAIL > 0:
        exit(1)
    else:
        print("ALL TESTS PASSED")
