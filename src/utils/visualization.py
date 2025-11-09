import cv2
import numpy as np


def draw_privacy_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    visibility_threshold: float = 0.5,
    line_color: tuple[int, int, int] = (0, 128, 0),
    landmark_color: tuple[int, int, int] = (0, 0, 128),
    line_thickness: int = 2,
    landmark_radius: int = 4,
) -> np.ndarray:
    """
    Draw stick figure skeleton on light background for privacy mode.
    Only shows pose landmarks without any background image.

    Args:
        frame: BGR image frame (used only for dimensions)
        landmarks: (33, 4) array of landmarks [x, y, z, visibility]
        visibility_threshold: Minimum visibility to draw
        line_color: BGR color for connection lines
        landmark_color: BGR color for landmark points
        line_thickness: Thickness of connection lines
        landmark_radius: Radius of landmark circles

    Returns:
        Light frame with stick figure skeleton drawn
    """
    # Create light background with same dimensions as input frame
    height, width = frame.shape[:2]
    black_frame = np.full((height, width, 3), 240, dtype=np.uint8)

    if landmarks is None:
        return black_frame  # Note: variable name retained for backwards compatibility

    # Define skeleton connections (MediaPipe Pose)
    connections = [
        (11, 12),  # Shoulders
        (11, 13),
        (13, 15),  # Left arm
        (12, 14),
        (14, 16),  # Right arm
        (11, 23),
        (12, 24),
        (23, 24),  # Torso
        (23, 25),
        (25, 27),  # Left leg
        (24, 26),
        (26, 28),  # Right leg
    ]

    # Draw connections
    for start_idx, end_idx in connections:
        if (
            landmarks[start_idx, 3] > visibility_threshold
            and landmarks[end_idx, 3] > visibility_threshold
        ):
            start_point = (
                int(landmarks[start_idx, 0] * width),
                int(landmarks[start_idx, 1] * height),
            )
            end_point = (
                int(landmarks[end_idx, 0] * width),
                int(landmarks[end_idx, 1] * height),
            )
            cv2.line(black_frame, start_point, end_point, line_color, line_thickness)

    # Draw landmarks
    for landmark in landmarks:
        if landmark[3] > visibility_threshold:
            x = int(landmark[0] * width)
            y = int(landmark[1] * height)
            cv2.circle(black_frame, (x, y), landmark_radius, landmark_color, -1)

    return black_frame


def draw_skeleton_connections(
    frame: np.ndarray,
    landmarks: np.ndarray,
    visibility_threshold: float = 0.5,
    line_color: tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw skeleton connections on frame.

    Args:
        frame: BGR image frame
        landmarks: (33, 4) array of landmarks [x, y, z, visibility]
        visibility_threshold: Minimum visibility to draw connection
        line_color: RGB color for connection lines
        line_thickness: Thickness of connection lines

    Returns:
        Frame with skeleton connections drawn
    """
    if landmarks is None:
        return frame

    height, width = frame.shape[:2]

    # Define skeleton connections (MediaPipe Pose)
    connections = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),  # Arms
        (11, 23),
        (12, 24),
        (23, 24),  # Torso
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),  # Legs
    ]

    # Draw connections
    for start_idx, end_idx in connections:
        if (
            landmarks[start_idx, 3] > visibility_threshold
            and landmarks[end_idx, 3] > visibility_threshold
        ):
            start_point = (
                int(landmarks[start_idx, 0] * width),
                int(landmarks[start_idx, 1] * height),
            )
            end_point = (
                int(landmarks[end_idx, 0] * width),
                int(landmarks[end_idx, 1] * height),
            )
            cv2.line(frame, start_point, end_point, line_color, line_thickness)

    return frame


def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    visibility_threshold: float = 0.5,
    landmark_color: tuple[int, int, int] = (0, 0, 255),
    landmark_radius: int = 4,
) -> np.ndarray:
    """
    Draw pose landmarks on frame.

    Args:
        frame: BGR image frame
        landmarks: (33, 4) array of landmarks [x, y, z, visibility]
        visibility_threshold: Minimum visibility to draw landmark
        landmark_color: RGB color for landmark points
        landmark_radius: Radius of landmark circles

    Returns:
        Frame with landmarks drawn
    """
    if landmarks is None:
        return frame

    height, width = frame.shape[:2]

    # Draw landmarks
    for landmark in landmarks:
        if landmark[3] > visibility_threshold:
            x = int(landmark[0] * width)
            y = int(landmark[1] * height)
            cv2.circle(frame, (x, y), landmark_radius, landmark_color, -1)

    return frame


def draw_pose_skeleton(
    frame: np.ndarray, landmarks: np.ndarray, visibility_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw complete pose skeleton (connections + landmarks) on frame.

    Args:
        frame: BGR image frame
        landmarks: (33, 4) array of landmarks [x, y, z, visibility]
        visibility_threshold: Minimum visibility to draw

    Returns:
        Frame with skeleton drawn
    """
    frame = draw_skeleton_connections(frame, landmarks, visibility_threshold)
    frame = draw_landmarks(frame, landmarks, visibility_threshold)
    return frame


def draw_status_banner(
    frame: np.ndarray,
    is_fall: bool,
    banner_height: int = 80,
    fall_color: tuple[int, int, int] = (0, 0, 255),
    normal_color: tuple[int, int, int] = (0, 255, 0),
    fall_bg_color: tuple[int, int, int] = (0, 0, 128),
    normal_bg_color: tuple[int, int, int] = (0, 128, 0),
) -> np.ndarray:
    """
    Draw fall detection status banner on frame.

    Args:
        frame: BGR image frame
        is_fall: Whether fall is detected
        banner_height: Height of status banner in pixels
        fall_color: Text color for fall status
        normal_color: Text color for normal status
        fall_bg_color: Background color for fall status
        normal_bg_color: Background color for normal status

    Returns:
        Frame with status banner drawn
    """
    height, width = frame.shape[:2]

    # Determine status color and text
    if is_fall:
        status_text = "FALL DETECTED!"
        text_color = fall_color
        bg_color = fall_bg_color
    else:
        status_text = "Normal"
        text_color = normal_color
        bg_color = normal_bg_color

    # Draw status banner
    cv2.rectangle(frame, (0, 0), (width, banner_height), bg_color, -1)
    cv2.putText(
        frame,
        status_text,
        (10, banner_height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        text_color,
        4,
    )

    return frame


def draw_info_overlay(
    frame: np.ndarray,
    fall_info: dict,
    position: tuple[int, int] = (10, 100),
    font_scale: float = 0.6,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_alpha: float = 0.7,
) -> np.ndarray:
    """
    Draw detection information overlay on frame.

    Args:
        frame: BGR image frame
        fall_info: Dictionary containing detection information
        position: Starting position (x, y) for text overlay
        font_scale: Font scale for text
        text_color: RGB color for text
        bg_alpha: Background transparency (0-1)

    Returns:
        Frame with info overlay drawn
    """
    if not fall_info or not fall_info.get("valid"):
        return frame

    x, y = position
    line_height = 25

    # Prepare info text lines
    info_lines = []

    if "method" in fall_info:
        info_lines.append(f"Method: {fall_info['method']}")

    if fall_info.get("method") == "multi_criteria":
        details = fall_info.get("details", {})
        info_lines.append(
            f"Votes: {fall_info.get('votes', 0)}/{fall_info.get('vote_threshold', 1)}"
        )

        # Add specific method results
        if "angle" in details and details["angle"].get("valid"):
            angle = details["angle"].get("angle", 0)
            info_lines.append(f"  Angle: {angle:.1f}°")

        if "velocity" in details and details["velocity"].get("valid"):
            velocity = details["velocity"].get("velocity", 0)
            info_lines.append(f"  Velocity: {velocity:.3f}")

    # Draw semi-transparent background
    overlay = frame.copy()
    bg_height = len(info_lines) * line_height + 10
    cv2.rectangle(overlay, (x - 5, y - 20), (x + 300, y + bg_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0)

    # Draw text lines
    for i, line in enumerate(info_lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return frame
