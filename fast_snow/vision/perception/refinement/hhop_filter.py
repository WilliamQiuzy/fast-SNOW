"""H-hop refinement for implausible geometry filtering.

This module implements the geometric validation described in SNOW paper
Section 3.2 and Table 1, which detects and filters implausible geometries:

1. Extent validation: Objects with unrealistic size (e.g., 50m car roof)
2. Aspect ratio validation: Elongated Gaussians indicating segmentation errors
3. Velocity validation: Implausible motion between frames (e.g., pedestrian moving 32m in 2s)
4. Size consistency: Sudden size changes indicating tracking errors

The H-hop mechanism allows multiple rounds of validation and refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from fast_snow.reasoning.tokens.step_encoding import STEPToken


@dataclass(frozen=True)
class HHopConfig:
    """Configuration for H-hop geometric validation."""

    # Maximum extent along any axis (meters)
    max_extent: float = 50.0

    # Maximum standard deviation along any axis (meters)
    max_sigma: float = 10.0

    # Maximum aspect ratio (longest / shortest extent)
    max_aspect_ratio: float = 20.0

    # Maximum velocity between frames (m/s)
    # Assuming ~1Hz frame rate, this is also max displacement per frame
    max_velocity: float = 30.0

    # Maximum size change ratio between consecutive observations
    max_size_change_ratio: float = 3.0

    # Minimum points for valid object (if available)
    min_points: int = 5

    # Enable specific checks
    check_extent: bool = True
    check_sigma: bool = True
    check_aspect_ratio: bool = True
    check_velocity: bool = True
    check_size_consistency: bool = True


@dataclass
class ValidationResult:
    """Result of geometric validation for a single STEP token."""

    is_valid: bool
    reasons: List[str] = field(default_factory=list)

    # Detailed metrics
    max_extent: Optional[float] = None
    max_sigma: Optional[float] = None
    aspect_ratio: Optional[float] = None
    velocity: Optional[float] = None
    size_change_ratio: Optional[float] = None


def compute_extents(step: STEPToken) -> np.ndarray:
    """Compute extent (max - min) along each axis.

    Returns:
        Array of shape (3,) with [extent_x, extent_y, extent_z]
    """
    s = step.shape
    return np.array([
        s.x_max - s.x_min,
        s.y_max - s.y_min,
        s.z_max - s.z_min,
    ], dtype=float)


def compute_sigmas(step: STEPToken) -> np.ndarray:
    """Compute standard deviations along each axis.

    Returns:
        Array of shape (3,) with [sigma_x, sigma_y, sigma_z]
    """
    s = step.shape
    return np.array([s.x_sigma, s.y_sigma, s.z_sigma], dtype=float)


def compute_centroid(step: STEPToken) -> np.ndarray:
    """Compute centroid position.

    Returns:
        Array of shape (3,) with [x, y, z]
    """
    c = step.centroid
    return np.array([c.x, c.y, c.z], dtype=float)


def compute_volume(step: STEPToken) -> float:
    """Compute approximate volume from extents.

    Returns:
        Volume in cubic meters
    """
    extents = compute_extents(step)
    return float(np.prod(extents))


def validate_extent(step: STEPToken, config: HHopConfig) -> Tuple[bool, Optional[str]]:
    """Validate that object extents are within reasonable bounds.

    Detects: "50m car roof" type errors
    """
    extents = compute_extents(step)
    max_ext = float(np.max(extents))

    if max_ext > config.max_extent:
        return False, f"extent {max_ext:.2f}m exceeds max {config.max_extent}m"

    return True, None


def validate_sigma(step: STEPToken, config: HHopConfig) -> Tuple[bool, Optional[str]]:
    """Validate that point cloud spread is reasonable."""
    sigmas = compute_sigmas(step)
    max_sig = float(np.max(sigmas))

    if max_sig > config.max_sigma:
        return False, f"sigma {max_sig:.2f}m exceeds max {config.max_sigma}m"

    return True, None


def validate_aspect_ratio(step: STEPToken, config: HHopConfig) -> Tuple[bool, Optional[str]]:
    """Validate that object is not unreasonably elongated.

    Detects: Elongated Gaussians from segmentation errors
    """
    extents = compute_extents(step)
    extents_sorted = np.sort(extents)

    # Avoid division by zero
    min_extent = extents_sorted[0]
    max_extent = extents_sorted[-1]

    if min_extent < 0.01:  # Less than 1cm
        # Very thin object, might be valid (e.g., pole) or error
        if max_extent > 5.0:  # But if it's also very long
            return False, f"extremely thin elongated object: {max_extent:.2f}m x {min_extent:.4f}m"
        return True, None

    aspect_ratio = max_extent / min_extent

    if aspect_ratio > config.max_aspect_ratio:
        return False, f"aspect ratio {aspect_ratio:.2f} exceeds max {config.max_aspect_ratio}"

    return True, None


def validate_velocity(
    current_step: STEPToken,
    previous_step: STEPToken,
    dt: float,
    config: HHopConfig,
) -> Tuple[bool, Optional[str], float]:
    """Validate that motion between frames is physically plausible.

    Detects: "Pedestrian moved 32m in 2s" type errors

    Args:
        current_step: Current frame STEP token
        previous_step: Previous frame STEP token
        dt: Time delta between frames (seconds)
        config: Validation config

    Returns:
        Tuple of (is_valid, reason, velocity)
    """
    c1 = compute_centroid(previous_step)
    c2 = compute_centroid(current_step)

    displacement = float(np.linalg.norm(c2 - c1))

    if dt > 0:
        velocity = displacement / dt
    else:
        velocity = displacement  # Assume 1 second if dt not provided

    if velocity > config.max_velocity:
        return False, f"velocity {velocity:.2f}m/s exceeds max {config.max_velocity}m/s", velocity

    return True, None, velocity


def validate_size_consistency(
    current_step: STEPToken,
    previous_step: STEPToken,
    config: HHopConfig,
) -> Tuple[bool, Optional[str], float]:
    """Validate that object size doesn't change drastically between frames.

    Detects: Tracking errors where different objects are associated

    Returns:
        Tuple of (is_valid, reason, size_change_ratio)
    """
    vol1 = compute_volume(previous_step)
    vol2 = compute_volume(current_step)

    # Avoid division by zero
    if vol1 < 1e-6 and vol2 < 1e-6:
        return True, None, 1.0

    if vol1 < 1e-6:
        vol1 = 1e-6
    if vol2 < 1e-6:
        vol2 = 1e-6

    ratio = max(vol1 / vol2, vol2 / vol1)

    if ratio > config.max_size_change_ratio:
        return False, f"size change ratio {ratio:.2f} exceeds max {config.max_size_change_ratio}", ratio

    return True, None, ratio


def validate_single_step(
    step: STEPToken,
    config: HHopConfig,
) -> ValidationResult:
    """Validate a single STEP token without temporal context.

    Args:
        step: STEP token to validate
        config: Validation configuration

    Returns:
        ValidationResult with validity and reasons
    """
    reasons: List[str] = []

    # Compute metrics
    extents = compute_extents(step)
    sigmas = compute_sigmas(step)
    extents_sorted = np.sort(extents)
    min_extent = max(extents_sorted[0], 1e-4)
    aspect_ratio = extents_sorted[-1] / min_extent

    result = ValidationResult(
        is_valid=True,
        max_extent=float(np.max(extents)),
        max_sigma=float(np.max(sigmas)),
        aspect_ratio=float(aspect_ratio),
    )

    # Run checks
    if config.check_extent:
        valid, reason = validate_extent(step, config)
        if not valid:
            reasons.append(reason)

    if config.check_sigma:
        valid, reason = validate_sigma(step, config)
        if not valid:
            reasons.append(reason)

    if config.check_aspect_ratio:
        valid, reason = validate_aspect_ratio(step, config)
        if not valid:
            reasons.append(reason)

    if reasons:
        result = ValidationResult(
            is_valid=False,
            reasons=reasons,
            max_extent=result.max_extent,
            max_sigma=result.max_sigma,
            aspect_ratio=result.aspect_ratio,
        )

    return result


def validate_step_sequence(
    steps: List[STEPToken],
    config: HHopConfig,
    dt: float = 1.0,
) -> List[ValidationResult]:
    """Validate a sequence of STEP tokens with temporal context.

    This enables velocity and size consistency checks.

    Args:
        steps: List of STEP tokens in temporal order
        config: Validation configuration
        dt: Time delta between frames (seconds)

    Returns:
        List of ValidationResult for each step
    """
    results: List[ValidationResult] = []

    for i, step in enumerate(steps):
        # Start with single-step validation
        result = validate_single_step(step, config)
        reasons = list(result.reasons)

        velocity = None
        size_ratio = None

        # Add temporal checks if we have previous step
        if i > 0:
            prev_step = steps[i - 1]

            if config.check_velocity:
                valid, reason, vel = validate_velocity(step, prev_step, dt, config)
                velocity = vel
                if not valid:
                    reasons.append(reason)

            if config.check_size_consistency:
                valid, reason, ratio = validate_size_consistency(step, prev_step, config)
                size_ratio = ratio
                if not valid:
                    reasons.append(reason)

        # Update result
        results.append(ValidationResult(
            is_valid=len(reasons) == 0,
            reasons=reasons,
            max_extent=result.max_extent,
            max_sigma=result.max_sigma,
            aspect_ratio=result.aspect_ratio,
            velocity=velocity,
            size_change_ratio=size_ratio,
        ))

    return results


def detect_implausible(
    steps: Dict[int, STEPToken],
    config: HHopConfig,
) -> List[int]:
    """Return node ids with implausible geometry.

    This is the main entry point for single-frame validation.

    Args:
        steps: Dict mapping node_id -> STEPToken
        config: Validation configuration

    Returns:
        List of node_ids that failed validation
    """
    bad_ids: List[int] = []

    for node_id, step in steps.items():
        result = validate_single_step(step, config)
        if not result.is_valid:
            bad_ids.append(node_id)

    return bad_ids


def filter_implausible(
    steps: Dict[int, STEPToken],
    config: HHopConfig,
) -> Dict[int, STEPToken]:
    """Filter out implausible STEP tokens.

    Args:
        steps: Dict mapping node_id -> STEPToken
        config: Validation configuration

    Returns:
        Filtered dict with only valid tokens
    """
    bad_ids = set(detect_implausible(steps, config))
    return {node_id: step for node_id, step in steps.items() if node_id not in bad_ids}


def hhop_validate(
    steps: Dict[int, STEPToken],
    config: HHopConfig,
    h_hop: int = 1,
) -> Tuple[Dict[int, STEPToken], Dict[int, ValidationResult]]:
    """Run H-hop validation on STEP tokens.

    Multiple rounds of validation can catch different types of errors.

    Args:
        steps: Dict mapping node_id -> STEPToken
        config: Validation configuration
        h_hop: Number of validation hops

    Returns:
        Tuple of:
            - Filtered valid tokens
            - Validation results for all tokens
    """
    current_steps = dict(steps)
    all_results: Dict[int, ValidationResult] = {}

    for hop in range(h_hop):
        for node_id, step in list(current_steps.items()):
            if node_id in all_results and not all_results[node_id].is_valid:
                continue

            result = validate_single_step(step, config)
            all_results[node_id] = result

            if not result.is_valid:
                del current_steps[node_id]

    return current_steps, all_results


def validate_track(
    track_steps: List[STEPToken],
    config: HHopConfig,
    dt: float = 1.0,
) -> Tuple[bool, List[ValidationResult]]:
    """Validate an entire object track.

    Args:
        track_steps: List of STEP tokens for a tracked object
        config: Validation configuration
        dt: Time delta between frames

    Returns:
        Tuple of (overall_valid, per_step_results)
    """
    results = validate_step_sequence(track_steps, config, dt)
    overall_valid = all(r.is_valid for r in results)
    return overall_valid, results
