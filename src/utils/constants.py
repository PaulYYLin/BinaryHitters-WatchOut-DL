"""
Constants and configurations for fall detection system.
"""


# MediaPipe Pose Landmark Indices
class PoseLandmarks:
    """MediaPipe Pose landmark indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Skeleton Connections for Visualization
SKELETON_CONNECTIONS = [
    # Arms
    (PoseLandmarks.LEFT_SHOULDER, PoseLandmarks.RIGHT_SHOULDER),
    (PoseLandmarks.LEFT_SHOULDER, PoseLandmarks.LEFT_ELBOW),
    (PoseLandmarks.LEFT_ELBOW, PoseLandmarks.LEFT_WRIST),
    (PoseLandmarks.RIGHT_SHOULDER, PoseLandmarks.RIGHT_ELBOW),
    (PoseLandmarks.RIGHT_ELBOW, PoseLandmarks.RIGHT_WRIST),
    # Torso
    (PoseLandmarks.LEFT_SHOULDER, PoseLandmarks.LEFT_HIP),
    (PoseLandmarks.RIGHT_SHOULDER, PoseLandmarks.RIGHT_HIP),
    (PoseLandmarks.LEFT_HIP, PoseLandmarks.RIGHT_HIP),
    # Legs
    (PoseLandmarks.LEFT_HIP, PoseLandmarks.LEFT_KNEE),
    (PoseLandmarks.LEFT_KNEE, PoseLandmarks.LEFT_ANKLE),
    (PoseLandmarks.RIGHT_HIP, PoseLandmarks.RIGHT_KNEE),
    (PoseLandmarks.RIGHT_KNEE, PoseLandmarks.RIGHT_ANKLE),
]


# Default Configuration for Fall Detection
DEFAULT_FALL_DETECTOR_CONFIG = {
    "fall_angle_threshold": 45.0,  # degrees
    "height_drop_threshold": 0.2,  # ratio (20%)
    "velocity_threshold": 0.08,  # normalized units
    "temporal_window": 5,  # frames
    "min_visibility": 0.5,  # visibility score (0-1)
}


# Default Configuration for MediaPipe Pose Landmarker
DEFAULT_POSE_LANDMARKER_CONFIG = {
    "num_poses": 1,
    "min_pose_detection_confidence": 0.5,
    "min_pose_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}


# Camera Configuration
DEFAULT_CAMERA_CONFIG = {"camera_id": 0, "default_fps": 30.0, "max_fps": 120.0}


# Visualization Colors (BGR format for OpenCV)
class Colors:
    """Color constants for visualization (BGR format)."""

    # Status colors
    FALL_TEXT = (0, 0, 255)  # Red
    FALL_BG = (0, 0, 128)  # Dark red
    NORMAL_TEXT = (0, 255, 0)  # Green
    NORMAL_BG = (0, 128, 0)  # Dark green

    # Landmark colors
    LANDMARK = (0, 0, 255)  # Red
    CONNECTION = (0, 255, 0)  # Green

    # UI colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    OVERLAY_BG = (0, 0, 0)


# Detection Methods
class DetectionMethods:
    """Available fall detection methods."""

    ANGLE = "angle"
    HEIGHT = "height"
    VELOCITY = "velocity"
    LANDMARK = "landmark"
    MULTI_CRITERIA = "multi_criteria"


# Model Paths
DEFAULT_MODEL_PATH = "mediapipe-rule-based/src/utils/pose_landmarker_lite.task"
