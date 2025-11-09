"""
Main entry point for Fall Detection System (Edge Device Optimized).

This script runs the live camera fall detection system with:
- MediaPipe pose estimation
- Rule-based fall detection
- Video recording with ring buffer
- Async API integration
- Optimized for edge devices
"""

import asyncio
import logging
import signal
import sys

from src import DEFAULT_FALL_DETECTOR_CONFIG, LiveCameraFallDetector
from src.api import AsyncAPIClient
from src.config import get_settings
from src.events import FallEventManager
from src.video import AsyncVideoEncoder, LightweightRingBuffer


# Configure logging
def setup_logging(settings):
    """
    Setup logging configuration.

    Args:
        settings: Settings instance
    """
    log_file = settings.LOG_DIR / "fall_detection.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


class FallDetectionSystem:
    """
    Main fall detection system orchestrator.

    Manages all components:
    - Camera detector
    - Ring buffer
    - Video encoder
    - API client
    - Event manager
    """

    def __init__(self):
        """Initialize system components."""
        # Load configuration
        self.settings = get_settings()
        setup_logging(self.settings)

        logger.info("=" * 80)
        logger.info("Fall Detection System - Edge Device Optimized")
        logger.info("=" * 80)

        # Log configuration
        self.settings.log_config()

        # Check if model exists
        if not self.settings.MODEL_PATH.exists():
            logger.error(f"Model file not found: {self.settings.MODEL_PATH}")
            logger.info(
                "Download from: "
                "https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models"
            )
            sys.exit(1)

        # Initialize components
        self._init_components()

        # Background tasks
        self.background_tasks = []
        self.running = False

    def _init_components(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")

        # 1. Ring buffer for video storage
        self.ring_buffer = LightweightRingBuffer(
            duration=self.settings.BUFFER_DURATION,
            fps=self.settings.BUFFER_FPS,
            quality=self.settings.JPEG_QUALITY,
        )

        # 2. Video encoder for creating clips
        self.video_encoder = AsyncVideoEncoder(
            codec=self.settings.VIDEO_CODEC,
            fps=self.settings.BUFFER_FPS,
            bitrate=self.settings.VIDEO_BITRATE,
            resolution=self.settings.CAMERA_RESOLUTION,
        )

        # 3. API client for uploading events
        self.api_client = AsyncAPIClient(
            success_endpoint=self.settings.API_SUCCESS_ENDPOINT,
            failure_endpoint=self.settings.API_FAILURE_ENDPOINT,
            api_key=self.settings.API_KEY,
            timeout=self.settings.API_TIMEOUT,
            retry_attempts=self.settings.API_RETRY_ATTEMPTS,
            retry_delays=self.settings.API_RETRY_DELAYS,
        )

        # 4. Event manager for coordinating fall events
        self.event_manager = FallEventManager(
            ring_buffer=self.ring_buffer,
            video_encoder=self.video_encoder,
            api_client=self.api_client,
            settings=self.settings,
        )

        # 5. Live camera detector
        self.detector = LiveCameraFallDetector(
            model_path=str(self.settings.MODEL_PATH),
            camera_id=self.settings.CAMERA_ID,
            fall_detector_config=DEFAULT_FALL_DETECTOR_CONFIG,
            ring_buffer=self.ring_buffer,
            event_manager=self.event_manager,
            settings=self.settings,
            headless=self.settings.HEADLESS_MODE,
        )

        logger.info("All components initialized successfully")

    async def run(self):
        """
        Run the fall detection system.

        Starts:
        1. Event processor (background task)
        2. Camera detector (main loop in executor)
        """
        self.running = True

        try:
            # Start event processor in background
            logger.info("Starting event processor...")
            event_task = asyncio.create_task(self.event_manager.process_events())
            self.background_tasks.append(event_task)

            # Run camera detector in executor (blocking operation)
            logger.info("Starting camera detector...")
            logger.info("=" * 80)
            if self.settings.HEADLESS_MODE:
                logger.info("System running in HEADLESS mode (Ctrl+C to stop)")
            else:
                logger.info("Press 'q' in the video window to quit")
            logger.info("=" * 80)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.detector.run)

        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """
        Graceful shutdown of all components.
        """
        logger.info("=" * 80)
        logger.info("Shutting down system...")
        logger.info("=" * 80)

        # Stop event manager
        if self.event_manager:
            logger.info("Stopping event manager...")
            await self.event_manager.stop()
            self.event_manager.log_statistics()

        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close API client
        if self.api_client:
            logger.info("Closing API client...")
            await self.api_client.close()

        # Log final statistics
        if self.ring_buffer:
            logger.info(f"Final ring buffer state: {self.ring_buffer}")

        logger.info("=" * 80)
        logger.info("Shutdown complete")
        logger.info("=" * 80)


async def main():
    """
    Main entry point with async support.
    """
    # Create system
    system = FallDetectionSystem()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler(sig):
        logger.info(f"Received signal {sig}")
        # Create a task to shutdown gracefully
        asyncio.create_task(system.shutdown())
        loop.stop()

    # Register signal handlers (Unix only)
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):

            def make_handler(s: signal.Signals = sig) -> None:
                signal_handler(s)

            loop.add_signal_handler(sig, make_handler)

    # Run system
    try:
        await system.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
