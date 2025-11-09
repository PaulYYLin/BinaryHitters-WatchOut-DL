"""
Configuration management for fall detection system.
Loads settings from config.yaml and environment variables.
"""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Settings:
    """
    Centralized configuration management.
    All settings can be overridden via environment variables.
    """

    def __init__(self):
        # Load YAML configuration
        config_path = Path(os.getenv("CONFIG_PATH", "./config.yaml"))
        config = self._load_yaml_config(config_path)

        # Camera settings (from YAML)
        camera_config = config.get("camera", {})
        self.CAMERA_ID: int = camera_config.get("camera_id", 0)
        resolution = camera_config.get("resolution", {})
        self.CAMERA_RESOLUTION: tuple[int, int] = (
            resolution.get("width", 640),
            resolution.get("height", 480),
        )
        self.CAPTURE_FPS: int = camera_config.get("capture_fps", 30)

        # Display settings (from YAML)
        display_config = config.get("display", {})
        self.HEADLESS_MODE: bool = display_config.get("headless_mode", True)
        self.PRIVACY_MODE: bool = display_config.get("privacy_mode", False)
        self.FALL_DISPLAY_DURATION: float = display_config.get(
            "fall_display_duration", 3.0
        )

        # Fall detection settings (from YAML)
        fall_config = config.get("fall_detection", {})
        self.FALL_ANGLE_THRESHOLD: float = fall_config.get("fall_angle_threshold", 45.0)
        self.HEIGHT_DROP_THRESHOLD: float = fall_config.get(
            "height_drop_threshold", 0.2
        )
        self.VELOCITY_THRESHOLD: float = fall_config.get("velocity_threshold", 0.08)
        self.TEMPORAL_WINDOW: int = fall_config.get("temporal_window", 5)
        self.MIN_VISIBILITY: float = fall_config.get("min_visibility", 0.5)
        self.VOTE_THRESHOLD: int = fall_config.get("vote_threshold", 1)

        # Ring buffer settings (from YAML)
        buffer_config = config.get("buffer", {})
        self.BUFFER_DURATION: int = buffer_config.get("duration", 30)
        self.BUFFER_FPS: int = buffer_config.get("fps", 15)
        self.FRAME_SKIP: int = buffer_config.get("frame_skip", 2)
        self.JPEG_QUALITY: int = buffer_config.get("jpeg_quality", 70)

        # Video encoding settings (from YAML)
        video_config = config.get("video", {})
        self.CLIP_DURATION: int = video_config.get("clip_duration", 15)
        self.VIDEO_CODEC: str = video_config.get("codec", "mp4v")
        self.VIDEO_BITRATE: int = video_config.get("bitrate", 1000000)

        # Event management (from YAML)
        events_config = config.get("events", {})
        self.COOLDOWN_PERIOD: int = events_config.get("cooldown_period", 15)
        self.MAX_WORKERS: int = events_config.get("max_workers", 2)

        # Experiment mode settings (from YAML)
        experiment_config = config.get("experiment", {})
        self.EXP_MODE: bool = experiment_config.get("exp_mode", False)
        self.EXP_OUTPUT_DIR: Path = Path(
            experiment_config.get("output_dir", "experiment_data")
        )
        self.EXP_SAVE_LANDMARKS: bool = experiment_config.get("save_landmarks", True)
        self.EXP_SAVE_DETECTION_DETAILS: bool = experiment_config.get(
            "save_detection_details", True
        )
        self.EXP_SAVE_VIDEO: bool = experiment_config.get("save_video", True)

        # API settings (from environment variables)
        self.API_BASE_URL: str = os.getenv("API_BASE_URL", "")
        self.API_SUCCESS_ENDPOINT: str = (
            f"{self.API_BASE_URL}{os.getenv('API_UPLOAD_ENDPOINT', '')}"
            if self.API_BASE_URL
            else ""
        )
        self.API_FAILURE_ENDPOINT: str = os.getenv("API_FAILURE_ENDPOINT", "")
        self.API_CHECKIN_ENDPOINT: str = (
            f"{self.API_BASE_URL}{os.getenv('API_CHECKIN_ENDPOINT', '')}"
            if self.API_BASE_URL
            else ""
        )
        self.DEVICE_UID: str = os.getenv("DEVICE_UID", "")
        self.API_KEY: str = os.getenv("API_KEY", "")
        self.API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))
        self.API_RETRY_ATTEMPTS: int = int(os.getenv("API_RETRY_ATTEMPTS", "3"))
        self.API_RETRY_DELAYS: tuple[int, ...] = (1, 2, 4)
        self.CHECKIN_INTERVAL: int = int(os.getenv("CHECKIN_INTERVAL", "60"))

        # Paths (from environment variables)
        self.TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", "/tmp/fall_events"))
        self.LOG_DIR: Path = Path(os.getenv("LOG_DIR", "./logs"))
        self.MODEL_PATH: Path = Path(
            os.getenv("MODEL_PATH", "./src/utils/pose_landmarker_lite.task")
        )
        self.CONFIG_PATH: Path = config_path

        # Validate critical settings
        self._validate()

    def _load_yaml_config(self, config_path: Path) -> dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml

        Returns:
            Dictionary with configuration values
        """
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config or {}
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}

    def _validate(self):
        """Validate critical configuration settings."""
        # Check API endpoints
        if not self.API_BASE_URL:
            logger.warning("API_BASE_URL not set - API features will be disabled")

        if not self.API_SUCCESS_ENDPOINT:
            logger.warning(
                "API_SUCCESS_ENDPOINT not set - fall events will not be uploaded"
            )

        if not self.API_FAILURE_ENDPOINT:
            logger.warning(
                "API_FAILURE_ENDPOINT not set - failure notifications will not be sent"
            )

        if not self.API_CHECKIN_ENDPOINT:
            logger.warning(
                "API_CHECKIN_ENDPOINT not set - device check-ins will not be performed"
            )

        if not self.DEVICE_UID:
            logger.warning(
                "DEVICE_UID not set - device identification will be disabled"
            )

        # Check model path
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {self.MODEL_PATH}. "
                "Please ensure pose_landmarker_lite.task is in the correct location."
            )

        # Create directories if they don't exist
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        if self.EXP_MODE:
            self.EXP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Experiment mode enabled. Data will be saved to {self.EXP_OUTPUT_DIR}"
            )

        # Validate numeric ranges
        if not 0 <= self.JPEG_QUALITY <= 100:
            logger.warning(f"Invalid JPEG_QUALITY: {self.JPEG_QUALITY}, using 70")
            self.JPEG_QUALITY = 70

        if self.BUFFER_FPS > self.CAPTURE_FPS:
            logger.warning(
                f"BUFFER_FPS ({self.BUFFER_FPS}) > CAPTURE_FPS ({self.CAPTURE_FPS}), "
                f"setting BUFFER_FPS = CAPTURE_FPS"
            )
            self.BUFFER_FPS = self.CAPTURE_FPS

        logger.info("Configuration validated successfully")

    def get_max_buffer_frames(self) -> int:
        """Calculate maximum number of frames in ring buffer."""
        return self.BUFFER_DURATION * self.BUFFER_FPS

    def get_clip_frames(self) -> int:
        """Calculate number of frames in a video clip."""
        return self.CLIP_DURATION * self.BUFFER_FPS

    def log_config(self):
        """Log current configuration (for debugging)."""
        logger.info("=" * 60)
        logger.info("Fall Detection System Configuration")
        logger.info("=" * 60)
        logger.info(f"Camera ID: {self.CAMERA_ID}")
        logger.info(
            f"Camera Resolution: {self.CAMERA_RESOLUTION[0]}x{self.CAMERA_RESOLUTION[1]}"
        )
        logger.info(f"Capture FPS: {self.CAPTURE_FPS}")
        logger.info(
            f"Buffer FPS: {self.BUFFER_FPS} (saving every {self.FRAME_SKIP} frames)"
        )
        logger.info(
            f"Buffer Duration: {self.BUFFER_DURATION}s ({self.get_max_buffer_frames()} frames)"
        )
        logger.info(
            f"Clip Duration: {self.CLIP_DURATION}s ({self.get_clip_frames()} frames)"
        )
        logger.info(f"JPEG Quality: {self.JPEG_QUALITY}")
        logger.info(f"Video Codec: {self.VIDEO_CODEC}")
        logger.info(f"Cooldown Period: {self.COOLDOWN_PERIOD}s")
        logger.info(f"Headless Mode: {self.HEADLESS_MODE}")
        logger.info(f"Privacy Mode: {self.PRIVACY_MODE}")
        logger.info(f"Fall Angle Threshold: {self.FALL_ANGLE_THRESHOLD}°")
        logger.info(f"Height Drop Threshold: {self.HEIGHT_DROP_THRESHOLD}")
        logger.info(f"Velocity Threshold: {self.VELOCITY_THRESHOLD}")
        logger.info(f"API Base URL: {self.API_BASE_URL or 'NOT SET'}")
        logger.info(f"API Success Endpoint: {self.API_SUCCESS_ENDPOINT or 'NOT SET'}")
        logger.info(f"API Failure Endpoint: {self.API_FAILURE_ENDPOINT or 'NOT SET'}")
        logger.info(f"API Checkin Endpoint: {self.API_CHECKIN_ENDPOINT or 'NOT SET'}")
        logger.info(f"Device UID: {self.DEVICE_UID or 'NOT SET'}")
        logger.info(f"Checkin Interval: {self.CHECKIN_INTERVAL}s")
        logger.info(f"Model Path: {self.MODEL_PATH}")
        logger.info("=" * 60)


# Singleton instance
_settings_instance = None


def get_settings() -> Settings:
    """
    Get singleton Settings instance.

    Returns:
        Settings instance with current configuration
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
