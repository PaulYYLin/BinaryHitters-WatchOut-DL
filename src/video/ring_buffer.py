"""
Lightweight ring buffer for edge devices.
Uses JPEG compression to minimize memory usage.
"""

import logging
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LightweightRingBuffer:
    """
    Memory-efficient ring buffer for video frames.

    Stores frames as compressed JPEG data to reduce memory footprint
    by ~95% compared to raw BGR frames. Designed for edge devices with
    limited RAM.

    Example memory usage for 640x480 frames:
    - Raw BGR: ~900KB per frame * 450 frames = ~400MB
    - JPEG (quality 70): ~40KB per frame * 450 frames = ~18MB

    Attributes:
        max_frames: Maximum number of frames to store
        buffer: Deque containing compressed frames with timestamps
        quality: JPEG compression quality (0-100)
        frame_count: Total frames added (for statistics)
    """

    def __init__(self, duration: int = 30, fps: int = 15, quality: int = 70):
        """
        Initialize ring buffer.

        Args:
            duration: Duration of video to buffer in seconds
            fps: Frames per second to store
            quality: JPEG compression quality (0-100), 70 is good balance
        """
        self.max_frames = duration * fps
        self.buffer: deque = deque(maxlen=self.max_frames)
        self.quality = quality
        self.frame_count = 0
        self.width: int | None = None
        self.height: int | None = None

        logger.info(
            f"Initialized RingBuffer: {duration}s @ {fps}fps = {self.max_frames} frames, "
            f"JPEG quality {quality}"
        )

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add frame to buffer with JPEG compression.

        Compresses frame to JPEG format in memory to save space.
        Automatically removes oldest frame when buffer is full.

        Args:
            frame: BGR image frame from OpenCV (numpy array)

        Returns:
            True if frame was added successfully, False otherwise
        """
        try:
            # Store frame dimensions (needed for decoding later)
            if self.width is None or self.height is None:
                self.height, self.width = frame.shape[:2]

            # Compress frame to JPEG in memory
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            success, encoded = cv2.imencode(".jpg", frame, encode_params)

            if not success:
                logger.error("Failed to encode frame to JPEG")
                return False

            # Store compressed data with timestamp
            self.buffer.append(
                {
                    "data": encoded.tobytes(),  # Compressed JPEG bytes
                    "timestamp": time.time(),
                    "size": len(encoded),  # For statistics
                }
            )

            self.frame_count += 1

            # Log statistics every 100 frames
            if self.frame_count % 100 == 0:
                avg_size = np.mean([f["size"] for f in self.buffer]) / 1024  # KB
                total_size = sum(f["size"] for f in self.buffer) / (1024 * 1024)  # MB
                logger.debug(
                    f"Buffer stats: {len(self.buffer)}/{self.max_frames} frames, "
                    f"avg {avg_size:.1f}KB/frame, total {total_size:.1f}MB"
                )

            return True

        except Exception as e:
            logger.error(f"Error adding frame to buffer: {e}")
            return False

    def get_clip(self, center_time: float, duration: int = 15) -> list[dict]:
        """
        Extract video clip centered around a specific time.

        Retrieves frames from buffer within time range:
        [center_time - duration/2, center_time + duration/2]

        Args:
            center_time: Timestamp to center the clip around (from time.time())
            duration: Duration of clip in seconds (default 15s)

        Returns:
            List of frame dictionaries with compressed data and timestamps
            Empty list if no frames found in time range
        """
        half_duration = duration / 2
        start_time = center_time - half_duration
        end_time = center_time + half_duration

        # Extract frames within time range
        clip_frames = [
            frame
            for frame in self.buffer
            if start_time <= frame["timestamp"] <= end_time
        ]

        logger.info(
            f"Extracted {len(clip_frames)} frames for clip "
            f"(requested {duration}s around {center_time:.2f})"
        )

        if len(clip_frames) == 0:
            logger.warning(
                f"No frames found in time range [{start_time:.2f}, {end_time:.2f}]. "
                f"Buffer contains {len(self.buffer)} frames from "
                f"{self.buffer[0]['timestamp']:.2f} to {self.buffer[-1]['timestamp']:.2f}"
                if self.buffer
                else "Buffer is empty"
            )

        return clip_frames

    def decode_frame(self, frame_data: dict) -> np.ndarray | None:
        """
        Decode compressed JPEG frame back to BGR image.

        Args:
            frame_data: Dictionary containing 'data' key with JPEG bytes

        Returns:
            BGR image as numpy array, or None if decoding fails
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data["data"], np.uint8)

            # Decode JPEG
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("Failed to decode JPEG frame")
                return None

            return img

        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return None

    def get_buffer_info(self) -> dict:
        """
        Get buffer statistics and information.

        Returns:
            Dictionary with buffer statistics
        """
        if not self.buffer:
            return {
                "num_frames": 0,
                "max_frames": self.max_frames,
                "total_size_mb": 0,
                "avg_size_kb": 0,
                "oldest_timestamp": None,
                "newest_timestamp": None,
                "duration_seconds": 0,
            }

        sizes = [f["size"] for f in self.buffer]
        timestamps = [f["timestamp"] for f in self.buffer]

        return {
            "num_frames": len(self.buffer),
            "max_frames": self.max_frames,
            "total_size_mb": sum(sizes) / (1024 * 1024),
            "avg_size_kb": np.mean(sizes) / 1024,
            "oldest_timestamp": min(timestamps),
            "newest_timestamp": max(timestamps),
            "duration_seconds": max(timestamps) - min(timestamps),
        }

    def clear(self):
        """Clear all frames from buffer."""
        self.buffer.clear()
        self.frame_count = 0
        logger.info("Ring buffer cleared")

    def __len__(self) -> int:
        """Return number of frames currently in buffer."""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation of buffer."""
        info = self.get_buffer_info()
        return (
            f"LightweightRingBuffer("
            f"frames={info['num_frames']}/{info['max_frames']}, "
            f"size={info['total_size_mb']:.1f}MB, "
            f"quality={self.quality})"
        )
