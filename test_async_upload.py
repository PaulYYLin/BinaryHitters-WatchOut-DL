"""
Test script to verify that event processing and uploads happen in background.
This simulates the fall detection system without requiring actual camera or fall detection.
"""

import asyncio
import logging
import time
from pathlib import Path

from src.config import get_settings
from src.events import FallEventManager
from src.video import AsyncVideoEncoder, LightweightRingBuffer
from src.api import AsyncAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """
    Test that event processing happens concurrently with main loop.
    """
    logger.info("=" * 80)
    logger.info("Testing Async Upload - Event Processing in Background")
    logger.info("=" * 80)

    # Initialize components
    settings = get_settings()

    ring_buffer = LightweightRingBuffer(
        duration=30, fps=15, quality=70
    )

    video_encoder = AsyncVideoEncoder(
        codec=settings.VIDEO_CODEC,
        fps=15,
        resolution=(640, 480)
    )

    api_client = AsyncAPIClient(
        success_endpoint=settings.API_SUCCESS_ENDPOINT,
        failure_endpoint=settings.API_FAILURE_ENDPOINT,
        device_uid=settings.DEVICE_UID
    )

    event_manager = FallEventManager(
        ring_buffer=ring_buffer,
        video_encoder=video_encoder,
        api_client=api_client,
        settings=settings
    )

    # Start event processor in background
    logger.info("\n🚀 Starting event processor...")
    event_task = asyncio.create_task(event_manager.process_events())

    # Wait a bit to ensure event processor is running
    await asyncio.sleep(0.5)

    # Simulate main detection loop
    logger.info("\n🎥 Simulating main camera detection loop...")
    logger.info("   (This represents the camera running)")

    for i in range(5):
        logger.info(f"\n--- Main Loop Iteration {i+1}/5 ---")

        if i == 1:
            # Simulate fall detection
            logger.info("💥 Simulating FALL DETECTED in main loop...")
            success = event_manager.trigger_fall({
                "test": True,
                "iteration": i
            })

            if success:
                logger.info("✓ Fall event queued successfully")
                logger.info("⏰ Main loop continues immediately (not blocked)")
            else:
                logger.warning("✗ Failed to queue fall event")

        # Main loop continues doing work
        logger.info("🔄 Main loop doing other work...")
        await asyncio.sleep(2)

    logger.info("\n📺 Main loop finished (like closing camera window)")

    # Stop event manager
    logger.info("\n🛑 Stopping event manager...")
    await event_manager.stop()

    # Cancel event task
    if not event_task.done():
        event_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass

    # Close API client
    await api_client.close()

    # Show statistics
    logger.info("\n" + "=" * 80)
    logger.info("Test Complete - Statistics:")
    logger.info("=" * 80)
    event_manager.log_statistics()

    logger.info("\n✅ If you see event processing happening while main loop continues,")
    logger.info("   then uploads are working in background correctly!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted")
