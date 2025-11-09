"""
Fall event manager with cooldown and async processing.
Coordinates video encoding and API uploads without blocking main detection loop.
"""

import asyncio
import logging
import time
from datetime import datetime

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

        # Event queue (thread-safe)
        self.queue: asyncio.Queue = asyncio.Queue()

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

        # Create event data
        event = {
            "timestamp": current_time,
            "datetime": datetime.fromtimestamp(current_time).isoformat(),
            "fall_info": fall_info or {},
        }

        # Add to queue (non-blocking)
        try:
            self.queue.put_nowait(event)
            self.total_events_triggered += 1
            logger.info(
                f"Fall event triggered and queued "
                f"(total: {self.total_events_triggered})"
            )
            return True

        except asyncio.QueueFull:
            logger.error("Event queue is full, cannot queue fall event")
            return False

    async def process_events(self):
        """
        Background task: process events from queue.

        Continuously runs in background, processing queued fall events.
        For each event:
        1. Extract video clip from ring buffer
        2. Encode clip to MP4
        3. Upload to API
        4. Cleanup temporary files

        This should be run as an asyncio task:
            task = asyncio.create_task(manager.process_events())
        """
        self.running = True
        logger.info("Event processor started")

        try:
            while self.running:
                # Wait for event (with timeout to allow graceful shutdown)
                try:
                    event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Process event
                await self._process_single_event(event)

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
        logger.info(f"Processing fall event from {event['datetime']}")

        self.total_events_processed += 1

        # Generate output filename
        video_filename = f"fall_{int(timestamp)}.mp4"
        video_path = self.settings.TEMP_DIR / video_filename

        try:
            # Step 1: Encode video clip
            logger.info(f"Encoding {self.settings.CLIP_DURATION}s video clip...")
            encode_success = await self.video_encoder.encode_clip_from_buffer(
                self.ring_buffer,
                center_time=timestamp,
                duration=self.settings.CLIP_DURATION,
                output_path=video_path,
            )

            if not encode_success:
                logger.error("Failed to encode video clip")
                await self._handle_failure(event, "Video encoding failed")
                return

            # Step 2: Prepare metadata
            metadata = {
                "timestamp": timestamp,
                "datetime": event["datetime"],
                "device_id": self.settings.CAMERA_ID,
                "clip_duration": self.settings.CLIP_DURATION,
                **event.get("fall_info", {}),
            }

            # Step 3: Upload to API
            logger.info(f"Uploading {video_filename} to API...")
            upload_success = await self.api_client.send_fall_event(video_path, metadata)

            if upload_success:
                self.total_events_uploaded += 1
                logger.info(
                    f"Fall event processed successfully "
                    f"(uploaded: {self.total_events_uploaded}/{self.total_events_processed})"
                )
            else:
                self.total_events_failed += 1
                logger.error("Failed to upload fall event")

        except Exception as e:
            logger.error(f"Error processing fall event: {e}")
            await self._handle_failure(event, f"Processing error: {str(e)}")

        finally:
            # Step 4: Cleanup - always delete temporary file
            if video_path.exists():
                try:
                    video_path.unlink()
                    logger.debug(f"Deleted temporary file: {video_filename}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

    async def _handle_failure(self, event: dict, error_message: str):
        """
        Handle event processing failure.

        Sends failure notification to API if endpoint is configured.

        Args:
            event: Original event data
            error_message: Description of failure
        """
        self.total_events_failed += 1

        metadata = {
            "timestamp": event["timestamp"],
            "datetime": event["datetime"],
            "device_id": self.settings.CAMERA_ID,
            **event.get("fall_info", {}),
        }

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
            try:
                await asyncio.wait_for(self.queue.join(), timeout=timeout)
            except TimeoutError:
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
