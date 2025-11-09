import logging

import numpy as np

logger = logging.getLogger(__name__)


class FallDetectorRuleBased:
    """
    Rule-based fall detection using MediaPipe pose landmarks.

    This class implements multiple rule-based methods to detect falls from
    MediaPipe pose landmarks without using machine learning models.

    Detection Methods:
    1. Angle-based: Detect falls based on body angle relative to ground
    2. Height-based: Detect falls based on vertical position changes
    3. Velocity-based: Detect falls based on rapid vertical movement
    4. Landmark-based: Detect falls based on key landmark positions
    5. Multi-criteria: Combine multiple rules for robust detection

    Landmark indices (MediaPipe Pose):
    - 0: nose
    - 11, 12: left/right shoulder
    - 23, 24: left/right hip
    - 25, 26: left/right knee
    - 27, 28: left/right ankle
    """

    def __init__(
        self,
        fall_angle_threshold: float = 45.0,
        height_drop_threshold: float = 0.2,
        velocity_threshold: float = 0.08,
        temporal_window: int = 5,
        min_visibility: float = 0.5,
    ):
        """
        Initialize fall detector with rule-based thresholds.

        Args:
            fall_angle_threshold: Maximum angle (degrees) from vertical before considering fall
            height_drop_threshold: Minimum height drop ratio to trigger fall detection
            velocity_threshold: Minimum vertical velocity to consider as falling
            temporal_window: Number of frames to consider for temporal analysis
            min_visibility: Minimum visibility score for landmarks to be valid
        """
        self.fall_angle_threshold = fall_angle_threshold
        self.height_drop_threshold = height_drop_threshold
        self.velocity_threshold = velocity_threshold
        self.temporal_window = temporal_window
        self.min_visibility = min_visibility

        # Landmark indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28

        # State tracking
        self.state_history: list[str] = []
        self.height_history: list[float] = []

        logger.info("FallDetectorRuleBased initialized")
        logger.info(f"  Fall angle threshold: {fall_angle_threshold}°")
        logger.info(f"  Height drop threshold: {height_drop_threshold}")
        logger.info(f"  Velocity threshold: {velocity_threshold}")

    def _get_midpoint(
        self, landmarks: np.ndarray, idx1: int, idx2: int
    ) -> np.ndarray | None:
        """
        Calculate midpoint between two landmarks.

        Args:
            landmarks: (33, 4) array of landmarks [x, y, z, visibility]
            idx1: Index of first landmark
            idx2: Index of second landmark

        Returns:
            Midpoint coordinates [x, y, z] or None if landmarks not visible
        """
        if (
            landmarks[idx1, 3] < self.min_visibility
            or landmarks[idx2, 3] < self.min_visibility
        ):
            return None

        midpoint = (landmarks[idx1, :3] + landmarks[idx2, :3]) / 2.0
        return midpoint

    def _calculate_body_angle(self, landmarks: np.ndarray) -> float | None:
        """
        Calculate angle of body relative to vertical axis.

        Uses the line from hip center to shoulder center.

        Args:
            landmarks: (33, 4) array of landmarks [x, y, z, visibility]

        Returns:
            Angle in degrees (0° = vertical, 90° = horizontal), or None if invalid
        """
        # Get shoulder and hip midpoints
        shoulder_mid = self._get_midpoint(
            landmarks, self.LEFT_SHOULDER, self.RIGHT_SHOULDER
        )
        hip_mid = self._get_midpoint(landmarks, self.LEFT_HIP, self.RIGHT_HIP)

        if shoulder_mid is None or hip_mid is None:
            return None

        # Calculate vector from hip to shoulder
        torso_vector = shoulder_mid - hip_mid

        # Calculate angle with vertical axis (y-axis points down in image coordinates)
        # Vertical vector in image coordinates: [0, -1, 0]
        vertical = np.array([0, -1, 0])

        # Calculate angle using dot product
        cos_angle = np.dot(torso_vector, vertical) / (
            np.linalg.norm(torso_vector) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle)

        return float(angle_degrees)

    def _calculate_body_height(self, landmarks: np.ndarray) -> float | None:
        """
        Calculate vertical extent of body (normalized height).

        Uses distance from nose to ankle midpoint in y-axis.

        Args:
            landmarks: (33, 4) array of landmarks [x, y, z, visibility]

        Returns:
            Normalized height value, or None if invalid
        """
        # Get key landmarks
        if landmarks[self.NOSE, 3] < self.min_visibility:
            return None

        ankle_mid = self._get_midpoint(landmarks, self.LEFT_ANKLE, self.RIGHT_ANKLE)
        if ankle_mid is None:
            return None

        # Calculate vertical distance (y-axis)
        # In image coordinates, y increases downward
        height = abs(ankle_mid[1] - landmarks[self.NOSE, 1])

        return float(height)

    def _calculate_center_of_mass_height(self, landmarks: np.ndarray) -> float | None:
        """
        Calculate approximate center of mass height.

        Uses hip center as approximation of center of mass.

        Args:
            landmarks: (33, 4) array of landmarks [x, y, z, visibility]

        Returns:
            Y-coordinate of hip center, or None if invalid
        """
        hip_mid = self._get_midpoint(landmarks, self.LEFT_HIP, self.RIGHT_HIP)
        if hip_mid is None:
            return None

        return float(hip_mid[1])

    def _is_horizontal_pose(self, landmarks: np.ndarray) -> bool:
        """
        Check if person is in horizontal position.

        Args:
            landmarks: (33, 4) array of landmarks [x, y, z, visibility]

        Returns:
            True if body is horizontal, False otherwise
        """
        shoulder_mid = self._get_midpoint(
            landmarks, self.LEFT_SHOULDER, self.RIGHT_SHOULDER
        )
        hip_mid = self._get_midpoint(landmarks, self.LEFT_HIP, self.RIGHT_HIP)

        if shoulder_mid is None or hip_mid is None:
            return False

        # Check if shoulder and hip are at similar height (horizontal)
        y_diff = abs(shoulder_mid[1] - hip_mid[1])
        x_diff = abs(shoulder_mid[0] - hip_mid[0])

        # If horizontal spread is much larger than vertical spread
        return bool(x_diff > y_diff * 1.5)

    def detect_fall_angle_based(self, landmarks: np.ndarray) -> tuple[bool, dict]:
        """
        Detect fall based on body angle.

        Args:
            landmarks: (33, 4) array of landmarks [x, y, z, visibility]

        Returns:
            Tuple of (is_fall, info_dict)
        """
        angle = self._calculate_body_angle(landmarks)

        if angle is None:
            return False, {"method": "angle", "angle": None, "valid": False}

        is_fall = angle >= self.fall_angle_threshold

        return is_fall, {
            "method": "angle",
            "angle": angle,
            "threshold": self.fall_angle_threshold,
            "is_fall": is_fall,
            "valid": True,
        }

    def detect_fall_height_based(self, landmarks: np.ndarray) -> tuple[bool, dict]:
        """
        Detect fall based on body height reduction.

        Compares current height with running average of recent heights.

        Args:
            landmarks: (33, 4) array of landmarks [x, y, z, visibility]

        Returns:
            Tuple of (is_fall, info_dict)
        """
        current_height = self._calculate_body_height(landmarks)

        if current_height is None:
            return False, {"method": "height", "height": None, "valid": False}

        # Update height history
        self.height_history.append(current_height)
        if len(self.height_history) > self.temporal_window:
            self.height_history.pop(0)

        # Need sufficient history
        if len(self.height_history) < self.temporal_window:
            return False, {
                "method": "height",
                "height": current_height,
                "valid": False,
                "reason": "insufficient_history",
            }

        # Calculate average height from history
        avg_height = np.mean(self.height_history[:-1])  # Exclude current frame
        height_ratio = current_height / (avg_height + 1e-6)

        # Check if significant height drop
        is_fall = height_ratio < (1.0 - self.height_drop_threshold)

        return is_fall, {
            "method": "height",
            "current_height": current_height,
            "avg_height": avg_height,
            "height_ratio": height_ratio,
            "threshold": self.height_drop_threshold,
            "is_fall": is_fall,
            "valid": True,
        }

    def detect_fall_velocity_based(
        self,
        current_landmarks: np.ndarray,
        previous_landmarks: np.ndarray | None = None,
    ) -> tuple[bool, dict]:
        """
        Detect fall based on rapid downward velocity.

        Args:
            current_landmarks: (33, 4) current frame landmarks
            previous_landmarks: (33, 4) previous frame landmarks (optional)

        Returns:
            Tuple of (is_fall, info_dict)
        """
        if previous_landmarks is None:
            return False, {
                "method": "velocity",
                "velocity": None,
                "valid": False,
                "reason": "no_previous_frame",
            }

        # Calculate center of mass velocity
        current_com = self._calculate_center_of_mass_height(current_landmarks)
        previous_com = self._calculate_center_of_mass_height(previous_landmarks)

        if current_com is None or previous_com is None:
            return False, {"method": "velocity", "velocity": None, "valid": False}

        # Calculate velocity (downward is positive in image coordinates)
        velocity = current_com - previous_com

        # Check if rapid downward movement
        is_fall = velocity > self.velocity_threshold

        return is_fall, {
            "method": "velocity",
            "velocity": velocity,
            "threshold": self.velocity_threshold,
            "is_fall": is_fall,
            "valid": True,
        }

    def detect_fall_landmark_based(self, landmarks: np.ndarray) -> tuple[bool, dict]:
        """
        Detect fall based on specific landmark positions.

        Checks if key body parts (shoulders, hips) are close to ground level.

        Args:
            landmarks: (33, 4) array of landmarks [x, y, z, visibility]

        Returns:
            Tuple of (is_fall, info_dict)
        """
        shoulder_mid = self._get_midpoint(
            landmarks, self.LEFT_SHOULDER, self.RIGHT_SHOULDER
        )
        hip_mid = self._get_midpoint(landmarks, self.LEFT_HIP, self.RIGHT_HIP)
        ankle_mid = self._get_midpoint(landmarks, self.LEFT_ANKLE, self.RIGHT_ANKLE)

        if shoulder_mid is None or hip_mid is None or ankle_mid is None:
            return False, {"method": "landmark", "valid": False}

        # Check if shoulders are close to ankle level (person is down)
        shoulder_to_ankle_dist = abs(shoulder_mid[1] - ankle_mid[1])

        # Check horizontal pose
        is_horizontal = self._is_horizontal_pose(landmarks)

        # Fall if shoulders are very close to ground level and pose is horizontal
        is_fall = (shoulder_to_ankle_dist < 0.4) and is_horizontal

        return is_fall, {
            "method": "landmark",
            "shoulder_ankle_distance": shoulder_to_ankle_dist,
            "is_horizontal": is_horizontal,
            "is_fall": is_fall,
            "valid": True,
        }

    def detect_fall_multi_criteria(
        self,
        landmarks: np.ndarray,
        previous_landmarks: np.ndarray | None = None,
        vote_threshold: int = 1,
    ) -> tuple[bool, dict]:
        """
        Detect fall using multiple criteria with voting mechanism.

        Combines angle, height, velocity, and landmark-based detection.

        Args:
            landmarks: (33, 4) current frame landmarks
            previous_landmarks: (33, 4) previous frame landmarks (optional)
            vote_threshold: Minimum number of methods that must agree

        Returns:
            Tuple of (is_fall, info_dict)
        """
        results = {}
        votes = 0

        # Method 1: Angle-based
        angle_fall, angle_info = self.detect_fall_angle_based(landmarks)
        results["angle"] = angle_info
        if angle_fall and angle_info["valid"]:
            votes += 1

        # Method 2: Height-based
        height_fall, height_info = self.detect_fall_height_based(landmarks)
        results["height"] = height_info
        if height_fall and height_info["valid"]:
            votes += 1

        # Method 3: Velocity-based (if previous frame available)
        if previous_landmarks is not None:
            velocity_fall, velocity_info = self.detect_fall_velocity_based(
                landmarks, previous_landmarks
            )
            results["velocity"] = velocity_info
            if velocity_fall and velocity_info["valid"]:
                votes += 1

        # Method 4: Landmark-based
        landmark_fall, landmark_info = self.detect_fall_landmark_based(landmarks)
        results["landmark"] = landmark_info
        if landmark_fall and landmark_info["valid"]:
            votes += 1

        # Make decision based on votes
        is_fall = votes >= vote_threshold

        return is_fall, {
            "method": "multi_criteria",
            "votes": votes,
            "vote_threshold": vote_threshold,
            "is_fall": is_fall,
            "details": results,
            "valid": True,
        }

    def detect_fall(
        self,
        landmarks: np.ndarray,
        previous_landmarks: np.ndarray | None = None,
        method: str = "multi_criteria",
    ) -> tuple[bool, dict]:
        """
        Detect fall using specified method.

        Args:
            landmarks: (33, 4) current frame landmarks [x, y, z, visibility]
            previous_landmarks: (33, 4) previous frame landmarks (optional)
            method: Detection method to use:
                - 'angle': Angle-based detection
                - 'height': Height-based detection
                - 'velocity': Velocity-based detection
                - 'landmark': Landmark position-based detection
                - 'multi_criteria': Combine multiple methods (recommended)

        Returns:
            Tuple of (is_fall, info_dict)
        """
        if method == "angle":
            return self.detect_fall_angle_based(landmarks)
        elif method == "height":
            return self.detect_fall_height_based(landmarks)
        elif method == "velocity":
            return self.detect_fall_velocity_based(landmarks, previous_landmarks)
        elif method == "landmark":
            return self.detect_fall_landmark_based(landmarks)
        elif method == "multi_criteria":
            return self.detect_fall_multi_criteria(landmarks, previous_landmarks)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def process_video_sequence(
        self, landmarks_sequence: list[np.ndarray], method: str = "multi_criteria"
    ) -> list[dict]:
        """
        Process a sequence of frames and detect falls.

        Args:
            landmarks_sequence: List of (33, 4) landmark arrays
            method: Detection method to use

        Returns:
            List of detection results for each frame
        """
        results = []

        for i, landmarks in enumerate(landmarks_sequence):
            previous_landmarks = landmarks_sequence[i - 1] if i > 0 else None

            is_fall, info = self.detect_fall(
                landmarks, previous_landmarks, method=method
            )

            results.append({"frame_idx": i, "is_fall": is_fall, "info": info})

        return results

    def reset_state(self):
        """Reset internal state tracking."""
        self.state_history.clear()
        self.height_history.clear()
        logger.info("State reset")
