import logging
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..detectors.fall_detector import FallDetectorRuleBased
from ..utils.visualization import draw_privacy_skeleton

logger = logging.getLogger(__name__)


class LiveCameraFallDetector:
    """
    Real-time fall detection using live camera feed.

    Combines MediaPipe pose detection with rule-based fall detection
    to provide real-time fall detection from webcam.

    Optimized for edge devices with:
    - Headless mode (no GUI)
    - Ring buffer integration
    - Event manager integration
    - Frame skip for efficient buffering
    """

    def __init__(
        self,
        model_path: str = "./pre-trained-models/pose_landmarker_lite.task",
        camera_id: int = 0,
        fall_detector_config: dict | None = None,
        ring_buffer=None,
        event_manager=None,
        settings=None,
        headless: bool = True,
        privacy_mode: bool = False,
    ):
        """
        Initialize live camera fall detector.

        Args:
            model_path: Path to MediaPipe pose model (.task file)
            camera_id: Camera device ID (0 for default webcam)
            fall_detector_config: Configuration dict for FallDetectorRuleBased
            ring_buffer: LightweightRingBuffer instance (optional)
            event_manager: FallEventManager instance (optional)
            settings: Settings instance with configuration (optional)
            headless: Run without GUI display (True for edge devices)
            privacy_mode: Display only stick figure without background (False by default)
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.headless = headless
        self.privacy_mode = privacy_mode
        self.ring_buffer = ring_buffer
        self.event_manager = event_manager
        self.settings = settings

        # Initialize fall detector
        if fall_detector_config is None:
            fall_detector_config = {
                "fall_angle_threshold": 45.0,
                "height_drop_threshold": 0.2,
                "velocity_threshold": 0.08,
                "temporal_window": 5,
                "min_visibility": 0.5,
            }
        self.fall_detector = FallDetectorRuleBased(**fall_detector_config)

        # Initialize MediaPipe Pose Landmarker for VIDEO mode
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._pose_detection_callback,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # State tracking
        self.latest_landmarks: npt.NDArray[np.float64] | None = None
        self.previous_landmarks: npt.NDArray[np.float64] | None = None
        self.frame_count = 0
        self.fall_detected = False
        self.fall_info: dict[str, Any] = {}
        self.display_fall_status = False  # For visual display (persists longer)
        self.last_fall_time = 0.0  # Timestamp of last fall detection
        self.fall_display_duration = 3.0  # Show red status for 3 seconds

        # Frame skip counter for buffer optimization
        self.frame_skip_counter = 0
        self.frame_skip = settings.FRAME_SKIP if settings else 2

        logger.info("LiveCameraFallDetector initialized")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Camera ID: {camera_id}")
        logger.info(f"  Headless mode: {headless}")
        logger.info(f"  Privacy mode: {privacy_mode}")
        logger.info(f"  Ring buffer: {'enabled' if ring_buffer else 'disabled'}")
        logger.info(f"  Event manager: {'enabled' if event_manager else 'disabled'}")

    def _pose_detection_callback(
        self, result, output_image: mp.Image, timestamp_ms: int
    ):
        """
        Callback function for MediaPipe pose detection results.

        Args:
            result: PoseLandmarker detection result
            output_image: Processed image
            timestamp_ms: Timestamp in milliseconds
        """
        # Store previous landmarks
        if self.latest_landmarks is not None:
            self.previous_landmarks = self.latest_landmarks.copy()

        # Extract landmarks if pose detected
        if result.pose_landmarks:
            pose_landmarks = result.pose_landmarks[0]

            # Convert to numpy array
            self.latest_landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks]
            )

            # Perform fall detection
            self.fall_detected, self.fall_info = self.fall_detector.detect_fall(
                self.latest_landmarks, self.previous_landmarks, method="multi_criteria"
            )
        else:
            self.latest_landmarks = None
            self.fall_detected = False
            self.fall_info = {}

    def _draw_landmarks_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw pose landmarks on frame.

        Args:
            frame: BGR image frame

        Returns:
            Frame with landmarks drawn
        """
        if self.latest_landmarks is None:
            return frame

        height, width = frame.shape[:2]

        # Draw skeleton connections
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
                self.latest_landmarks[start_idx, 3] > 0.5
                and self.latest_landmarks[end_idx, 3] > 0.5
            ):
                start_point = (
                    int(self.latest_landmarks[start_idx, 0] * width),
                    int(self.latest_landmarks[start_idx, 1] * height),
                )
                end_point = (
                    int(self.latest_landmarks[end_idx, 0] * width),
                    int(self.latest_landmarks[end_idx, 1] * height),
                )
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Draw landmarks
        for landmark in self.latest_landmarks:
            if landmark[3] > 0.5:  # visibility threshold
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        return frame

    def _draw_fall_status(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw fall detection status on frame.

        Args:
            frame: BGR image frame

        Returns:
            Frame with status overlay
        """
        height, width = frame.shape[:2]

        # Determine status color and text based on display status
        if self.display_fall_status:
            status_text = "FALL DETECTED!"
            status_color = (0, 0, 255)  # Red
            bg_color = (0, 0, 128)  # Dark red background
        else:
            status_text = "Normal"
            status_color = (0, 255, 0)  # Green
            bg_color = (0, 128, 0)  # Dark green background

        # Draw status banner
        cv2.rectangle(frame, (0, 0), (width, 80), bg_color, -1)
        cv2.putText(
            frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 4
        )

        return frame

    def run(self, window_name: str = "Fall Detection - Live Camera"):
        """
        Run live camera fall detection.

        In headless mode (edge devices): runs without GUI display
        In normal mode: shows visualization window

        Args:
            window_name: Name of the display window (ignored in headless mode)

        Controls (non-headless only):
            - 'q': Quit
        """
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return

        # Set camera resolution if settings available
        if self.settings:
            width, height = self.settings.CAMERA_RESOLUTION
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get camera properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0  # Default fallback FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Camera opened: {width}x{height} @ {fps}fps")
        if not self.headless:
            logger.info("Press 'q' to quit")
        else:
            logger.info("Running in headless mode (Ctrl+C to quit)")

        timestamp_ms = 0
        frame_duration_ms = int(1000 / fps)

        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Process frame (async callback will handle detection)
                self.landmarker.detect_async(mp_image, timestamp_ms)
                timestamp_ms += frame_duration_ms

                # Add frame to ring buffer (with frame skip for efficiency)
                if self.ring_buffer is not None:
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter >= self.frame_skip:
                        # In privacy mode, save stick figure frame instead of original
                        if self.privacy_mode and self.latest_landmarks is not None:
                            privacy_frame = draw_privacy_skeleton(
                                frame, self.latest_landmarks
                            )
                            self.ring_buffer.add_frame(privacy_frame)
                        else:
                            self.ring_buffer.add_frame(frame)
                        self.frame_skip_counter = 0

                # Trigger fall event if detected
                if self.fall_detected and self.event_manager is not None:
                    logger.warning("Fall detected! Triggering event...")
                    # Include landmarks in fall_info for experiment mode
                    fall_info_with_landmarks = self.fall_info.copy()
                    if self.latest_landmarks is not None:
                        fall_info_with_landmarks["landmarks"] = self.latest_landmarks
                    self.event_manager.trigger_fall(fall_info_with_landmarks)
                    # Set display status and timestamp
                    self.display_fall_status = True
                    self.last_fall_time = time.time()
                    # Reset fall detected flag to avoid repeated triggers
                    # (cooldown is handled by event manager)
                    self.fall_detected = False

                # Update display status based on time elapsed
                if self.display_fall_status:
                    if time.time() - self.last_fall_time > self.fall_display_duration:
                        self.display_fall_status = False

                self.frame_count += 1

                # Log stats periodically
                if self.frame_count % 300 == 0:  # Every ~10 seconds @ 30fps
                    logger.info(f"Processed {self.frame_count} frames")
                    if self.ring_buffer:
                        info = self.ring_buffer.get_buffer_info()
                        logger.info(
                            f"Ring buffer: {info['num_frames']} frames, "
                            f"{info['total_size_mb']:.1f}MB"
                        )

                # Display frame (only if not headless)
                if not self.headless:
                    # In privacy mode, show only stick figure on black background
                    if self.privacy_mode:
                        display_frame = draw_privacy_skeleton(
                            frame, self.latest_landmarks
                        )
                    else:
                        # Normal mode: draw landmarks on original frame
                        display_frame = self._draw_landmarks_on_frame(frame)

                    # Draw fall detection status
                    display_frame = self._draw_fall_status(display_frame)
                    # Show window
                    cv2.imshow(window_name, display_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit requested by user")
                        break
                else:
                    # Small delay to prevent CPU spinning
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Cleanup
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            self.landmarker.close()
            logger.info("Camera released and cleaned up")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
