"""
Async video encoder for edge devices.
Encodes video clips in background to avoid blocking main detection loop.
"""

import asyncio
import logging
import subprocess
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

    Uses H.264 codec with web-optimized settings for S3 streaming compatibility.
    Falls back to mp4v if H.264 is not available.
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
            codec: FourCC codec name ('mp4v', 'H264', 'avc1', etc.)
            fps: Frames per second for output video
            bitrate: Target bitrate in bits/second (1000000 = 1 Mbps)
            resolution: Output resolution (width, height)
        """
        # Prefer H.264/AVC1 for web compatibility
        if codec.lower() in ["h264", "avc1", "x264"]:
            # Try avc1 first (best web compatibility)
            self.codec = "avc1"
            self.fourcc = cv2.VideoWriter_fourcc(*"avc1")
        else:
            self.codec = codec
            self.fourcc = cv2.VideoWriter_fourcc(*codec)

        self.fps = fps
        self.bitrate = bitrate
        self.resolution = resolution
        self.use_ffmpeg_postprocess = codec.lower() in ["h264", "avc1", "x264"]

        logger.info(
            f"Initialized VideoEncoder: {self.codec} @ {fps}fps, "
            f"{resolution[0]}x{resolution[1]}, {bitrate/1e6:.1f}Mbps"
        )
        if self.use_ffmpeg_postprocess:
            logger.info(
                "Will use ffmpeg post-processing for web optimization (faststart)"
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
        temp_path = None
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # If using ffmpeg post-processing, write to temp file first
            if self.use_ffmpeg_postprocess:
                temp_path = (
                    output_path.parent / f"{output_path.stem}_temp{output_path.suffix}"
                )
                write_path = temp_path
            else:
                write_path = output_path

            # Initialize video writer
            writer = cv2.VideoWriter(
                str(write_path), self.fourcc, self.fps, self.resolution
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

            # Release writer before post-processing
            writer.release()
            writer = None

            # Post-process with ffmpeg for web optimization
            if self.use_ffmpeg_postprocess:
                if not self._optimize_for_web(temp_path, output_path):
                    logger.warning(
                        "ffmpeg optimization failed, using unoptimized video"
                    )
                    # Fall back to temp file
                    if temp_path.exists():
                        temp_path.rename(output_path)
                else:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Error in blocking encode: {e}")
            return False

        finally:
            # Always release writer
            if writer is not None:
                writer.release()
            # Clean up temp file if it exists
            if temp_path and temp_path.exists() and output_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    def _optimize_for_web(self, input_path: Path, output_path: Path) -> bool:
        """
        Optimize video for web streaming using ffmpeg.

        Moves moov atom to the beginning (faststart) for S3 streaming.

        Args:
            input_path: Input video file
            output_path: Output optimized video file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if ffmpeg is available
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, timeout=5
            )
            if result.returncode != 0:
                logger.warning("ffmpeg not available, skipping optimization")
                return False

            logger.debug("Optimizing video with ffmpeg for web streaming...")

            # Use ffmpeg to re-encode with faststart
            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",  # H.264 codec
                "-preset",
                "fast",  # Fast encoding
                "-crf",
                "23",  # Quality (lower = better, 18-28 range)
                "-movflags",
                "+faststart",  # Enable streaming
                "-y",  # Overwrite output
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode != 0:
                stderr_text = result.stderr.decode("utf-8") if result.stderr else ""
                logger.warning(f"ffmpeg failed: {stderr_text[:200]}")
                return False

            logger.debug("Video optimized for web streaming successfully")
            return True

        except FileNotFoundError:
            logger.warning("ffmpeg not found, install it for better web compatibility")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg optimization timeout")
            return False
        except Exception as e:
            logger.warning(f"ffmpeg optimization error: {e}")
            return False

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
