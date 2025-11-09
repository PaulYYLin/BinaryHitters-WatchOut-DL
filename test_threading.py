"""
Quick test to verify threading setup works.
"""
import asyncio
import logging
import queue
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_thread_safe_queue():
    """Test that thread-safe queue works with async processing."""

    # Create thread-safe queue
    event_queue = queue.Queue()

    # Flag for running
    running = True
    processed_count = 0

    async def async_processor():
        """Async task that processes events from queue."""
        nonlocal processed_count
        logger.info("✅ Async processor started")

        while running:
            try:
                event = event_queue.get(timeout=0.5)
                logger.info(f"📥 Processing event: {event}")
                await asyncio.sleep(0.5)  # Simulate async work
                processed_count += 1
                logger.info(f"✅ Event processed (total: {processed_count})")
                event_queue.task_done()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

    def sync_producer():
        """Sync thread that produces events."""
        logger.info("🎥 Producer thread started")
        for i in range(3):
            time.sleep(1)
            logger.info(f"🚨 Producing event {i+1}")
            event_queue.put(f"Event {i+1}")
        logger.info("🏁 Producer finished")

    # Start async processor
    processor_task = asyncio.create_task(async_processor())

    # Start sync producer in thread
    producer_thread = threading.Thread(target=sync_producer, daemon=False)
    producer_thread.start()

    # Wait for producer to finish
    while producer_thread.is_alive():
        await asyncio.sleep(0.1)

    # Wait a bit for queue to empty
    await asyncio.sleep(2)

    # Stop
    running = False
    processor_task.cancel()
    try:
        await processor_task
    except asyncio.CancelledError:
        pass

    logger.info(f"\n📊 Test Results:")
    logger.info(f"   Events produced: 3")
    logger.info(f"   Events processed: {processed_count}")
    logger.info(f"   Queue size: {event_queue.qsize()}")

    if processed_count == 3:
        logger.info("\n🎉 TEST PASSED: Threading works correctly!")
        return True
    else:
        logger.error("\n❌ TEST FAILED: Some events were not processed")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_thread_safe_queue())
    exit(0 if result else 1)
