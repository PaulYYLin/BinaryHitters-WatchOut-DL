"""
Fall Detection System using Rule-Based Methods and MediaPipe.

This package provides a modular fall detection system with the following components:
- Rule-based fall detectors
- Live camera integration
- Visualization utilities
"""

from .camera.live_detector import LiveCameraFallDetector
from .detectors.fall_detector import FallDetectorRuleBased
from .utils.constants import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_FALL_DETECTOR_CONFIG,
    DEFAULT_POSE_LANDMARKER_CONFIG,
    DetectionMethods,
)

__version__ = "1.0.0"

__all__ = [
    "FallDetectorRuleBased",
    "LiveCameraFallDetector",
    "DEFAULT_FALL_DETECTOR_CONFIG",
    "DEFAULT_POSE_LANDMARKER_CONFIG",
    "DEFAULT_CAMERA_CONFIG",
    "DetectionMethods",
]
