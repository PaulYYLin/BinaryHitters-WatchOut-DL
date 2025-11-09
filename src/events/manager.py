"""
Fall event manager with cooldown and async processing.
Coordinates video encoding and API uploads without blocking main detection loop.
"""

import asyncio
import logging
import queue
import time
from datetime import datetime

from ..utils.experiment_logger import ExperimentDataLogger

logger = logging.getLogger(__name__)


class FallEventManager:
    """
    Manages fall detection events with cooldown and background processing.

    Prevents event spam by enforcing cooldown period between events.
    Processes events asynchronously to avoid blocking the main camera loop.

    Architecture:
    - Main thread calls trigger_fall() when fall detected (non-blocking)
    - Background task processes events from queue
    - Each event: extract clip → encode video → upload to API → cleanup

    This design ensures real-time fall detection is never blocked by
    I/O operations (video encoding, network uploads).
    """

    def __init__(self, ring_buffer, video_encoder, api_client, settings):
        """
        Initialize event manager.

        Args:
            ring_buffer: LightweightRingBuffer instance
            video_encoder: AsyncVideoEncoder instance
            api_client: AsyncAPIClient instance
            settings: Settings instance with configuration
        """
        self.ring_buffer = ring_buffer
        self.video_encoder = video_encoder
        self.api_client = api_client
        self.settings = settings

        # Event queue (thread-safe for cross-thread communication)
        # Using standard queue.Queue instead of asyncio.Queue for thread safety
        self.queue: queue.Queue = queue.Queue()

        # Cooldown tracking
        self.last_event_time: float = 0
        self.cooldown_period: int = settings.COOLDOWN_PERIOD

        # Statistics
        self.total_events_triggered = 0
        self.total_events_processed = 0
        self.total_events_uploaded = 0
        self.total_events_failed = 0

        # Running flag
        self.running = False

        # Experiment mode
        self.exp_mode = settings.EXP_MODE
        self.exp_logger = None
        if self.exp_mode:
            self.exp_logger = ExperimentDataLogger(settings.EXP_OUTPUT_DIR)
            logger.info(f"Experiment mode enabled: {settings.EXP_OUTPUT_DIR}")

        logger.info(f"Initialized FallEventManager: cooldown={self.cooldown_period}s")

    def trigger_fall(self, fall_info: dict | None = None) -> bool:
        """
        Trigger fall event (called from main thread).

        Non-blocking operation that adds event to queue if cooldown has passed.
        This is safe to call from the main camera/detection loop.

        Args:
            fall_info: Optional dictionary with fall detection details

        Returns:
            True if event was queued, False if still in cooldown
        """
        current_time = time.time()

        # Check cooldown
        time_since_last = current_time - self.last_event_time
        if time_since_last < self.cooldown_period:
            remaining = self.cooldown_period - time_since_last
            logger.info(
                f"Fall detected but in cooldown period " f"(wait {remaining:.1f}s more)"
            )
            return False

        # Update last event time
        self.last_event_time = current_time

        # Generate event ID for experiment mode
        event_id = None
        if self.exp_mode and self.exp_logger:
            event_id = self.exp_logger.generate_event_id()

        # Create event data
        event = {
            "timestamp": current_time,
            "datetime": datetime.fromtimestamp(current_time).isoformat(),
            "fall_info": fall_info or {},
            "event_id": event_id,
        }

        # Add to queue (thread-safe, non-blocking)
        try:
            self.queue.put_nowait(event)
            self.total_events_triggered += 1
            logger.info("=" * 60)
            logger.info("🚨 FALL DETECTED! Event queued for processing")
            logger.info(f"   Event #{self.total_events_triggered}")
            logger.info(f"   Time: {event['datetime']}")
            logger.info(f"   Queue size: {self.queue.qsize()}")
            logger.info("   Video will be uploaded in ~7 seconds...")
            logger.info("=" * 60)
            return True

        except queue.Full:
            logger.error("Event queue is full, cannot queue fall event")
            return False

    async def process_events(self):
        """
        Background task: process events from queue.

        Continuously runs in background, processing queued fall events.
        For each event:
        1. Save experiment data (if enabled)
        2. Wait for post-fall video to be recorded (7 seconds)
        3. Extract video clip from ring buffer (7s before + 7s after)
        4. Encode clip to MP4
        5. Upload to API
        6. Cleanup temporary files

        This should be run as an asyncio task:
            task = asyncio.create_task(manager.process_events())
        """
        self.running = True
        logger.info("✅ Event processor started and running in background")
        logger.info("   Uploads will happen immediately when falls are detected")

        try:
            while self.running:
                # Wait for event from thread-safe queue (non-blocking with timeout)
                try:
                    # Use get with timeout to allow graceful shutdown
                    event = self.queue.get(timeout=1.0)
                except queue.Empty:
                    # No events, yield control to event loop and continue
                    await asyncio.sleep(0.01)
                    continue

                # Log that we're processing immediately
                logger.info(
                    "📥 Event processor received event, starting processing NOW..."
                )

                # Process event asynchronously
                await self._process_single_event(event)

                # Mark task as done for queue.join() in shutdown
                self.queue.task_done()

        except asyncio.CancelledError:
            logger.info("Event processor cancelled")
            raise

        finally:
            logger.info("Event processor stopped")

    async def _process_single_event(self, event: dict):
        """
        Process a single fall event.

        Args:
            event: Event dictionary with timestamp and fall info
        """
        timestamp = event["timestamp"]
        event_id = event.get("event_id")

        logger.info("=" * 60)
        logger.info(f"🔄 Processing fall event from {event['datetime']}")
        logger.info(f"   Event ID: {event_id or 'N/A'}")
        logger.info("=" * 60)

        self.total_events_processed += 1

        # Generate output filename
        if event_id:
            video_filename = f"{event_id}.mp4"
        else:
            video_filename = f"fall_{int(timestamp)}.mp4"
        video_path = self.settings.TEMP_DIR / video_filename

        try:
            # Step 1: Save experiment data (if enabled)
            fall_info = event.get("fall_info", {})
            landmarks = fall_info.get("landmarks")

            if self.exp_mode and self.exp_logger and event_id:
                # Save JSON data with landmarks
                self.exp_logger.save_fall_event(
                    event_id=event_id,
                    landmarks=landmarks,
                    fall_info=fall_info,
                    timestamp=timestamp,
                    metadata={"device_id": self.settings.CAMERA_ID},
                )

            # Step 2: Wait for post-fall video to be recorded
            # Need to wait half of clip_duration to ensure we have enough frames after the fall
            post_fall_wait_time = self.settings.CLIP_DURATION / 2
            logger.info(
                f"⏱️  Step 1/3: Waiting {post_fall_wait_time:.1f}s for post-fall video..."
            )
            await asyncio.sleep(post_fall_wait_time)

            # Step 3: Encode video clip
            logger.info(
                f"🎬 Step 2/3: Encoding {self.settings.CLIP_DURATION}s video clip..."
            )

            # Use experiment video path if in exp_mode, otherwise use temp path
            if self.exp_mode and self.exp_logger and event_id:
                final_video_path = self.exp_logger.get_video_path(event_id)
            else:
                final_video_path = video_path

            encode_success = await self.video_encoder.encode_clip_from_buffer(
                self.ring_buffer,
                center_time=timestamp,
                duration=self.settings.CLIP_DURATION,
                output_path=final_video_path,
            )

            if not encode_success:
                logger.error("Failed to encode video clip")
                await self._handle_failure(event, "Video encoding failed")
                return

            # Step 4: Prepare metadata for API (exclude landmarks to avoid serialization issues)
            metadata = {
                "timestamp": timestamp,
                "datetime": event["datetime"],
                "device_id": self.settings.CAMERA_ID,
                "clip_duration": self.settings.CLIP_DURATION,
                "event_id": event_id,
            }

            # Add fall_info to metadata but exclude landmarks (already saved separately)
            for key, value in fall_info.items():
                if (
                    key != "landmarks"
                ):  # Skip landmarks to avoid numpy array serialization issues
                    metadata[key] = value

            # Step 5: Upload to API (use temp copy if in exp_mode to preserve original)
            upload_path = final_video_path
            if self.exp_mode and not self.settings.EXP_SAVE_VIDEO:
                # If exp_mode but not saving video, still need temp file for upload
                upload_path = video_path
                # Copy to temp location for upload
                import shutil

                shutil.copy2(final_video_path, upload_path)

            logger.info(f"📤 Step 3/3: Uploading {video_filename} to API...")
            logger.info(
                f"   Video size: {upload_path.stat().st_size / 1024 / 1024:.2f} MB"
            )
            logger.info("   This upload happens immediately in the background...")

            upload_success = await self.api_client.send_fall_event(
                upload_path, metadata
            )

            if upload_success:
                self.total_events_uploaded += 1
                logger.info(
                    f"✅ Fall event uploaded successfully! "
                    f"(Total uploaded: {self.total_events_uploaded}/{self.total_events_processed})"
                )
            else:
                self.total_events_failed += 1
                logger.error(f"❌ Failed to upload fall event {video_filename}")

        except Exception as e:
            logger.error(f"Error processing fall event: {e}")
            await self._handle_failure(event, f"Processing error: {str(e)}")

        finally:
            # Step 6: Cleanup - delete temporary file (but not exp_mode videos)
            if not self.exp_mode and video_path.exists():
                try:
                    video_path.unlink()
                    logger.debug(f"Deleted temporary file: {video_filename}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")
            elif self.exp_mode and self.exp_logger and event_id:
                # In exp_mode, log that video is saved
                video_saved = self.exp_logger.get_video_path(event_id)
                if video_saved.exists():
                    logger.info(f"Experiment video saved: {video_saved}")

    async def _handle_failure(self, event: dict, error_message: str):
        """
        Handle event processing failure.

        Sends failure notification to API if endpoint is configured.

        Args:
            event: Original event data
            error_message: Description of failure
        """
        self.total_events_failed += 1

        # Prepare metadata (exclude landmarks to avoid serialization issues)
        fall_info = event.get("fall_info", {})
        metadata = {
            "timestamp": event["timestamp"],
            "datetime": event["datetime"],
            "device_id": self.settings.CAMERA_ID,
        }

        # Add fall_info but exclude landmarks
        for key, value in fall_info.items():
            if key != "landmarks":
                metadata[key] = value

        # Send failure notification (text only)
        await self.api_client.send_failure_notification(metadata, error_message)

    async def stop(self):
        """
        Stop event processor gracefully.

        Waits for queue to empty before stopping.
        """
        logger.info("Stopping event processor...")
        self.running = False

        # Wait for queue to empty (with timeout)
        remaining = self.queue.qsize()
        if remaining > 0:
            logger.info(f"Waiting for {remaining} events to process...")
            timeout = remaining * 30  # 30s per event estimate

            # Wait for queue.join() in a non-blocking way
            start_time = time.time()
            while not self.queue.empty() and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.5)

            if not self.queue.empty():
                logger.warning(
                    f"Timeout waiting for queue to empty, "
                    f"{self.queue.qsize()} events remain"
                )

    def get_statistics(self) -> dict:
        """
        Get event processing statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_triggered": self.total_events_triggered,
            "total_processed": self.total_events_processed,
            "total_uploaded": self.total_events_uploaded,
            "total_failed": self.total_events_failed,
            "queue_size": self.queue.qsize(),
            "success_rate": (
                self.total_events_uploaded / self.total_events_processed * 100
                if self.total_events_processed > 0
                else 0
            ),
            "last_event_time": self.last_event_time,
            "time_since_last": time.time() - self.last_event_time,
            "in_cooldown": (time.time() - self.last_event_time) < self.cooldown_period,
        }

    def log_statistics(self):
        """Log current statistics."""
        stats = self.get_statistics()
        logger.info("=" * 60)
        logger.info("Fall Event Statistics")
        logger.info("=" * 60)
        logger.info(f"Events Triggered: {stats['total_triggered']}")
        logger.info(f"Events Processed: {stats['total_processed']}")
        logger.info(f"Events Uploaded: {stats['total_uploaded']}")
        logger.info(f"Events Failed: {stats['total_failed']}")
        logger.info(f"Success Rate: {stats['success_rate']:.1f}%")
        logger.info(f"Queue Size: {stats['queue_size']}")
        logger.info(f"In Cooldown: {stats['in_cooldown']}")
        logger.info("=" * 60)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"FallEventManager("
            f"processed={stats['total_processed']}, "
            f"uploaded={stats['total_uploaded']}, "
            f"queue={stats['queue_size']})"
        )
