"""
Async video encoder for edge devices.
Encodes video clips in background to avoid blocking main detection loop.
"""

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AsyncVideoEncoder:
    """
    Non-blocking video encoder for edge devices.

    Encodes compressed JPEG frames from ring buffer into MP4 video files.
    Uses asyncio to run encoding in background thread pool, preventing
    blocking of the main camera/detection loop.

    Uses mp4v codec which is faster than H264 on CPU-only devices,
    making it ideal for edge deployment.
    """

    def __init__(
        self,
        codec: str = "mp4v",
        fps: int = 15,
        bitrate: int = 1000000,
        resolution: tuple[int, int] = (640, 480),
    ):
        """
        Initialize video encoder.

        Args:
            codec: FourCC codec name ('mp4v', 'H264', 'MJPG', etc.)
            fps: Frames per second for output video
            bitrate: Target bitrate in bits/second (1000000 = 1 Mbps)
            resolution: Output resolution (width, height)
        """
        self.codec = codec
        self.fps = fps
        self.bitrate = bitrate
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*codec)

        logger.info(
            f"Initialized VideoEncoder: {codec} @ {fps}fps, "
            f"{resolution[0]}x{resolution[1]}, {bitrate/1e6:.1f}Mbps"
        )

    async def encode_clip(
        self, frames: list[dict], output_path: Path, decode_func
    ) -> bool:
        """
        Encode frames to video file asynchronously.

        Runs encoding in executor thread pool to avoid blocking.
        Decodes JPEG frames and writes to MP4 file.

        Args:
            frames: List of frame dictionaries with compressed JPEG data
            output_path: Path where to save the video file
            decode_func: Function to decode compressed frames (from ring buffer)

        Returns:
            True if encoding successful, False otherwise
        """
        if not frames:
            logger.warning("No frames to encode")
            return False

        try:
            # Run blocking encoding operation in thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, self._encode_blocking, frames, output_path, decode_func
            )

            if success:
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(
                    f"Encoded {len(frames)} frames to {output_path.name} "
                    f"({file_size:.2f}MB)"
                )
            else:
                logger.error(f"Failed to encode video to {output_path}")

            return success

        except Exception as e:
            logger.error(f"Error during async encoding: {e}")
            return False

    def _encode_blocking(
        self, frames: list[dict], output_path: Path, decode_func
    ) -> bool:
        """
        Blocking encoding operation (runs in thread pool).

        Args:
            frames: List of frame dictionaries
            output_path: Output file path
            decode_func: Function to decode frames

        Returns:
            True if successful, False otherwise
        """
        writer = None
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize video writer
            writer = cv2.VideoWriter(
                str(output_path), self.fourcc, self.fps, self.resolution
            )

            if not writer.isOpened():
                logger.error("Failed to open VideoWriter")
                return False

            # Decode and write each frame
            encoded_count = 0
            for frame_data in frames:
                # Decode JPEG frame
                frame = decode_func(frame_data)

                if frame is None:
                    logger.warning("Skipping frame: decode failed")
                    continue

                # Resize if necessary
                if frame.shape[:2][::-1] != self.resolution:
                    frame = cv2.resize(frame, self.resolution)

                # Write frame
                writer.write(frame)
                encoded_count += 1

            if encoded_count == 0:
                logger.error("No frames were successfully encoded")
                return False

            logger.debug(f"Encoded {encoded_count}/{len(frames)} frames successfully")
            return True

        except Exception as e:
            logger.error(f"Error in blocking encode: {e}")
            return False

        finally:
            # Always release writer
            if writer is not None:
                writer.release()

    async def encode_clip_from_buffer(
        self, ring_buffer, center_time: float, duration: int, output_path: Path
    ) -> bool:
        """
        Convenience method: extract clip from buffer and encode.

        Args:
            ring_buffer: LightweightRingBuffer instance
            center_time: Timestamp to center clip around
            duration: Duration of clip in seconds
            output_path: Where to save video file

        Returns:
            True if successful, False otherwise
        """
        # Extract frames from buffer
        frames = ring_buffer.get_clip(center_time, duration)

        if not frames:
            logger.warning(
                f"No frames found in buffer for time {center_time} +/- {duration/2}s"
            )
            return False

        # Encode using buffer's decode method
        return await self.encode_clip(frames, output_path, ring_buffer.decode_frame)

    def get_estimated_file_size(self, duration: int) -> float:
        """
        Estimate output file size in MB.

        Args:
            duration: Video duration in seconds

        Returns:
            Estimated file size in megabytes
        """
        # Rough estimation: bitrate * duration / 8 (bits to bytes)
        size_bytes = (self.bitrate * duration) / 8
        size_mb = size_bytes / (1024 * 1024)
        return size_mb

    async def test_encoding(self, output_path: Path) -> bool:
        """
        Test encoding with dummy frames (for debugging).

        Args:
            output_path: Where to save test video

        Returns:
            True if test successful
        """
        logger.info("Running encoder test with dummy frames...")

        # Create dummy frames (checkerboard pattern)
        dummy_frames = []
        for i in range(self.fps * 2):  # 2 seconds of video
            frame = np.zeros((*self.resolution[::-1], 3), dtype=np.uint8)
            # Checkerboard pattern
            frame[::20, ::20] = [255, 255, 255]
            # Add frame number text
            cv2.putText(
                frame,
                f"Frame {i}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Encode to JPEG (simulate ring buffer)
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            dummy_frames.append({"data": encoded.tobytes(), "timestamp": i / self.fps})

        # Decode function for dummy frames
        def decode_dummy(frame_data):
            nparr = np.frombuffer(frame_data["data"], np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Encode
        success = await self.encode_clip(dummy_frames, output_path, decode_dummy)

        if success:
            logger.info(f"Test encoding successful: {output_path}")
        else:
            logger.error("Test encoding failed")

        return success
