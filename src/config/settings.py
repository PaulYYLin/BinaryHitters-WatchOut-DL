"""
Configuration management for fall detection system.
Loads settings from environment variables with sensible defaults for edge devices.
"""

import logging
import os
from pathlib import Path

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
        # Camera settings
        self.CAMERA_ID: int = int(os.getenv("CAMERA_ID", "0"))
        self.CAMERA_RESOLUTION: tuple[int, int] = self._parse_resolution(
            os.getenv("CAMERA_RESOLUTION", "640x480")
        )
        self.CAPTURE_FPS: int = int(os.getenv("CAPTURE_FPS", "30"))

        # Ring buffer settings (edge device optimized)
        self.BUFFER_DURATION: int = int(os.getenv("BUFFER_DURATION", "30"))  # seconds
        self.BUFFER_FPS: int = int(
            os.getenv("BUFFER_FPS", "15")
        )  # reduced FPS for buffer
        self.FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "2"))  # save every Nth frame
        self.JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "70"))  # 0-100

        # Video encoding settings
        self.CLIP_DURATION: int = int(os.getenv("CLIP_DURATION", "15"))  # seconds
        self.VIDEO_CODEC: str = os.getenv(
            "VIDEO_CODEC", "mp4v"
        )  # mp4v is faster than H264 on CPU
        self.VIDEO_BITRATE: int = int(os.getenv("VIDEO_BITRATE", "1000000"))  # 1 Mbps

        # API settings
        self.API_SUCCESS_ENDPOINT: str = os.getenv("API_SUCCESS_ENDPOINT", "")
        self.API_FAILURE_ENDPOINT: str = os.getenv("API_FAILURE_ENDPOINT", "")
        self.API_KEY: str = os.getenv("API_KEY", "")
        self.API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))  # seconds
        self.API_RETRY_ATTEMPTS: int = int(os.getenv("API_RETRY_ATTEMPTS", "3"))
        self.API_RETRY_DELAYS: tuple[int, ...] = (1, 2, 4)  # exponential backoff

        # Event management
        self.COOLDOWN_PERIOD: int = int(os.getenv("COOLDOWN_PERIOD", "15"))  # seconds

        # Performance settings
        self.MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))  # thread pool size
        self.HEADLESS_MODE: bool = os.getenv("HEADLESS_MODE", "true").lower() == "true"

        # Paths
        self.TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", "/tmp/fall_events"))
        self.LOG_DIR: Path = Path(os.getenv("LOG_DIR", "./logs"))
        self.MODEL_PATH: Path = Path(
            os.getenv("MODEL_PATH", "./src/utils/pose_landmarker_lite.task")
        )

        # Validate critical settings
        self._validate()

    def _parse_resolution(self, resolution_str: str) -> tuple[int, int]:
        """
        Parse resolution string like '640x480' into tuple (640, 480).

        Args:
            resolution_str: Resolution in format 'WIDTHxHEIGHT'

        Returns:
            Tuple of (width, height)
        """
        try:
            width, height = resolution_str.lower().split("x")
            return (int(width), int(height))
        except ValueError:
            logger.warning(
                f"Invalid resolution format: {resolution_str}, using default 640x480"
            )
            return (640, 480)

    def _validate(self):
        """Validate critical configuration settings."""
        # Check API endpoints
        if not self.API_SUCCESS_ENDPOINT:
            logger.warning(
                "API_SUCCESS_ENDPOINT not set - fall events will not be uploaded"
            )

        if not self.API_FAILURE_ENDPOINT:
            logger.warning(
                "API_FAILURE_ENDPOINT not set - failure notifications will not be sent"
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
        logger.info(f"API Success Endpoint: {self.API_SUCCESS_ENDPOINT or 'NOT SET'}")
        logger.info(f"API Failure Endpoint: {self.API_FAILURE_ENDPOINT or 'NOT SET'}")
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
