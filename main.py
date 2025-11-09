"""
Alternative main entry point with proper async handling for macOS GUI.

This version runs the detector synchronously in main thread (for OpenCV),
while using a separate thread for async event processing.
"""

import asyncio
import logging
import signal
import sys
import threading
import time

from src import DEFAULT_FALL_DETECTOR_CONFIG, LiveCameraFallDetector
from src.api import AsyncAPIClient
from src.config import get_settings
from src.events import FallEventManager
from src.video import AsyncVideoEncoder, LightweightRingBuffer


def setup_logging(settings):
    """Setup logging configuration."""
    log_file = settings.LOG_DIR / "fall_detection.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


class AsyncEventProcessor:
    """Runs event processing in a separate thread with its own event loop."""

    def __init__(self, event_manager):
        self.event_manager = event_manager
        self.loop = None
        self.thread = None
        self.running = False

    def start(self):
        """Start the async processor in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=False)
        self.thread.start()
        logger.info("Async event processor thread started")

    def _run_event_loop(self):
        """Run the event loop in thread."""
        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # Run the event processor
            self.loop.run_until_complete(self.event_manager.process_events())
        except Exception as e:
            logger.error(f"Event processor error: {e}", exc_info=True)
        finally:
            self.loop.close()

    def stop(self):
        """Stop the event processor."""
        self.running = False
        if self.loop:
            # Stop the event manager
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.event_manager.stop())
            )

        if self.thread:
            self.thread.join(timeout=30)
            logger.info("Async event processor thread stopped")


class HybridFallDetectionSystem:
    """
    Hybrid fall detection system that runs OpenCV in main thread
    and async processing in background thread.
    """

    def __init__(self):
        """Initialize system components."""
        # Load configuration
        self.settings = get_settings()
        setup_logging(self.settings)

        logger.info("=" * 80)
        logger.info("Fall Detection System - Hybrid Mode for macOS")
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

        # Background processors
        self.async_processor = None
        self.checkin_thread = None
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
            checkin_endpoint=self.settings.API_CHECKIN_ENDPOINT,
            device_uid=self.settings.DEVICE_UID,
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
        # Override default fall detector config with settings
        fall_detector_config = DEFAULT_FALL_DETECTOR_CONFIG.copy()
        fall_detector_config["fall_angle_threshold"] = (
            self.settings.FALL_ANGLE_THRESHOLD
        )
        fall_detector_config["height_drop_threshold"] = (
            self.settings.HEIGHT_DROP_THRESHOLD
        )
        fall_detector_config["velocity_threshold"] = self.settings.VELOCITY_THRESHOLD

        self.detector = LiveCameraFallDetector(
            model_path=str(self.settings.MODEL_PATH),
            camera_id=self.settings.CAMERA_ID,
            fall_detector_config=fall_detector_config,
            ring_buffer=self.ring_buffer,
            event_manager=self.event_manager,
            settings=self.settings,
            headless=self.settings.HEADLESS_MODE,
            privacy_mode=self.settings.PRIVACY_MODE,
        )

        logger.info("All components initialized successfully")

    def _checkin_thread_worker(self):
        """Thread worker for periodic device check-in."""
        if not self.settings.API_CHECKIN_ENDPOINT or not self.settings.DEVICE_UID:
            logger.info("Device check-in not configured, skipping periodic check-ins")
            return

        logger.info(
            f"Starting periodic device check-in (interval: {self.settings.CHECKIN_INTERVAL}s)"
        )

        while self.running:
            try:
                # Perform check-in using synchronous method
                success = self.api_client.device_checkin_sync()

                if success:
                    logger.info("Device check-in completed successfully")
                else:
                    logger.warning("Device check-in failed")

            except Exception as e:
                logger.error(f"Error during device check-in: {e}", exc_info=True)

            # Wait for next check-in interval
            if self.running:
                time.sleep(self.settings.CHECKIN_INTERVAL)

        logger.info("Periodic device check-in stopped")

    def run(self):
        """
        Run the fall detection system synchronously in main thread.
        This is required for OpenCV on macOS.
        """
        self.running = True

        try:
            # Start async event processor in separate thread
            logger.info("Starting async event processor...")
            self.async_processor = AsyncEventProcessor(self.event_manager)
            self.async_processor.start()

            # Wait a moment for processor to start
            time.sleep(0.5)

            # Start periodic device check-in in a separate thread
            logger.info("Starting periodic device check-in...")
            self.checkin_thread = threading.Thread(
                target=self._checkin_thread_worker, daemon=True
            )
            self.checkin_thread.start()

            # Run camera detector IN MAIN THREAD (required for OpenCV on macOS)
            logger.info("Starting camera detector...")
            logger.info("=" * 80)
            if self.settings.HEADLESS_MODE:
                logger.info("System running in HEADLESS mode (Ctrl+C to stop)")
            else:
                logger.info("Press 'q' in the video window to quit")
            logger.info("=" * 80)

            # Run detector synchronously in main thread
            self.detector.run()

        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown of all components."""
        logger.info("=" * 80)
        logger.info("Shutting down system...")
        logger.info("=" * 80)

        # Signal threads to stop
        self.running = False

        # Stop async processor
        if self.async_processor:
            logger.info("Stopping async event processor...")
            self.async_processor.stop()
            self.event_manager.log_statistics()

        # Wait for checkin thread to finish
        if self.checkin_thread and self.checkin_thread.is_alive():
            logger.info("Waiting for check-in thread to stop...")
            self.checkin_thread.join(timeout=5)

        # Close API client synchronously
        if self.api_client and self.api_client._session:
            logger.info("Closing API client...")
            # Create a simple event loop just for cleanup
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.api_client.close())
            loop.close()

        # Log final statistics
        if self.ring_buffer:
            logger.info(f"Final ring buffer state: {self.ring_buffer}")

        logger.info("=" * 80)
        logger.info("Shutdown complete")
        logger.info("=" * 80)


def main():
    """Main entry point for hybrid mode."""
    system = HybridFallDetectionSystem()

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run system
    try:
        system.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
