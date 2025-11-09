"""
Test if OpenCV windows work in a thread on macOS.
"""
import cv2
import numpy as np
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_opencv_in_thread():
    """Test OpenCV window in a separate thread."""
    logger.info("Testing OpenCV in thread...")

    success = False

    def opencv_worker():
        nonlocal success
        try:
            # Create a simple test image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Test Window", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Try to show window
            cv2.imshow("Test", img)
            cv2.waitKey(100)  # Wait 100ms
            cv2.destroyAllWindows()

            logger.info("✅ OpenCV window worked in thread!")
            success = True

        except Exception as e:
            logger.error(f"❌ OpenCV window failed in thread: {e}")
            success = False

    # Run in thread
    thread = threading.Thread(target=opencv_worker)
    thread.start()
    thread.join(timeout=5)

    return success


if __name__ == "__main__":
    # Note: On macOS, OpenCV windows may require main thread
    # But cv2.VideoCapture and processing can run in threads
    logger.info("=" * 60)
    logger.info("OpenCV Thread Test")
    logger.info("=" * 60)

    # Test 1: OpenCV in thread (may fail on macOS with GUI)
    result = test_opencv_in_thread()

    if result:
        logger.info("\n✅ OpenCV windows work in threads on this system")
    else:
        logger.warning("\n⚠️  OpenCV windows require main thread (typical on macOS)")
        logger.info("This is OK - the system will handle it correctly")

    exit(0)  # Always exit 0, this is informational
