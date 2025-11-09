"""
Async API client for fall detection notifications.
Handles both success (video upload) and failure (text-only) endpoints.
"""

import asyncio
import logging
import time
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


class AsyncAPIClient:
    """
    Non-blocking API client with retry logic.

    Supports two types of notifications:
    1. Success: Upload video file + metadata (when fall detected and video encoded)
    2. Failure: Send text notification only (when video upload fails)

    Both endpoints support retry with exponential backoff for reliability
    on edge devices with potentially unstable network connections.
    """

    def __init__(
        self,
        success_endpoint: str,
        failure_endpoint: str,
        api_key: str | None = None,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delays: tuple[int, ...] = (1, 2, 4),
    ):
        """
        Initialize API client.

        Args:
            success_endpoint: URL for successful fall event uploads (with video)
            failure_endpoint: URL for failure notifications (text only)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delays: Delay in seconds between retries (exponential backoff)
        """
        self.success_endpoint = success_endpoint
        self.failure_endpoint = failure_endpoint
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_attempts = retry_attempts
        self.retry_delays = retry_delays

        # Session will be created when needed (in async context)
        self._session: aiohttp.ClientSession | None = None

        logger.info(
            f"Initialized API Client: " f"timeout={timeout}s, retries={retry_attempts}"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session.

        Returns:
            Active ClientSession instance
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("API client session closed")

    def _get_headers(self) -> dict[str, str]:
        """
        Get HTTP headers including authentication if available.

        Returns:
            Dictionary of HTTP headers
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def send_fall_event(self, video_path: Path, metadata: dict) -> bool:
        """
        Send fall event with video to success endpoint.

        Uploads video file with metadata as multipart/form-data.
        Retries on failure with exponential backoff.

        Args:
            video_path: Path to encoded video file
            metadata: Dictionary with fall event metadata (timestamp, location, etc.)

        Returns:
            True if upload successful (or successful after retry), False otherwise
        """
        if not self.success_endpoint:
            logger.warning("Success endpoint not configured, skipping upload")
            return False

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        session = await self._get_session()

        # Try with retries
        for attempt in range(self.retry_attempts):
            try:
                logger.info(
                    f"Uploading fall event (attempt {attempt + 1}/{self.retry_attempts}): "
                    f"{video_path.name}"
                )

                # Prepare multipart form data
                data = aiohttp.FormData()

                # Add video file
                data.add_field(
                    "video",
                    open(video_path, "rb"),
                    filename=video_path.name,
                    content_type="video/mp4",
                )

                # Add metadata as JSON
                for key, value in metadata.items():
                    data.add_field(key, str(value))

                # Send POST request
                async with session.post(
                    self.success_endpoint, data=data, headers=self._get_headers()
                ) as response:

                    if response.status == 200:
                        logger.info(
                            f"Successfully uploaded fall event: {video_path.name}"
                        )
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Upload failed with status {response.status}: {error_text}"
                        )

            except TimeoutError:
                logger.warning(f"Upload timeout (attempt {attempt + 1})")

            except aiohttp.ClientError as e:
                logger.warning(
                    f"Network error during upload (attempt {attempt + 1}): {e}"
                )

            except Exception as e:
                logger.error(
                    f"Unexpected error during upload (attempt {attempt + 1}): {e}"
                )

            # Wait before retry (except on last attempt)
            if attempt < self.retry_attempts - 1:
                delay = (
                    self.retry_delays[attempt]
                    if attempt < len(self.retry_delays)
                    else self.retry_delays[-1]
                )
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        # All retries failed - send failure notification
        logger.error(
            f"Failed to upload video after {self.retry_attempts} attempts, "
            f"sending failure notification"
        )
        await self.send_failure_notification(
            metadata, "Video upload failed after all retries"
        )

        return False

    async def send_failure_notification(
        self, metadata: dict, error_message: str
    ) -> bool:
        """
        Send text-only failure notification.

        Lightweight JSON POST when video upload fails or encoding fails.
        Also uses retry logic for reliability.

        Args:
            metadata: Fall event metadata
            error_message: Description of what went wrong

        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.failure_endpoint:
            logger.warning("Failure endpoint not configured, skipping notification")
            return False

        session = await self._get_session()

        # Prepare JSON payload
        payload = {
            "event_type": "fall_detection_failure",
            "error": error_message,
            "timestamp": time.time(),
            **metadata,  # Include original metadata
        }

        # Try with retries
        for attempt in range(self.retry_attempts):
            try:
                logger.info(
                    f"Sending failure notification (attempt {attempt + 1}/{self.retry_attempts})"
                )

                async with session.post(
                    self.failure_endpoint, json=payload, headers=self._get_headers()
                ) as response:

                    if response.status == 200:
                        logger.info("Failure notification sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failure notification failed with status {response.status}: "
                            f"{error_text}"
                        )

            except TimeoutError:
                logger.warning(f"Failure notification timeout (attempt {attempt + 1})")

            except aiohttp.ClientError as e:
                logger.warning(
                    f"Network error during failure notification (attempt {attempt + 1}): {e}"
                )

            except Exception as e:
                logger.error(
                    f"Unexpected error during failure notification (attempt {attempt + 1}): {e}"
                )

            # Wait before retry
            if attempt < self.retry_attempts - 1:
                delay = (
                    self.retry_delays[attempt]
                    if attempt < len(self.retry_delays)
                    else self.retry_delays[-1]
                )
                await asyncio.sleep(delay)

        logger.error(
            f"Failed to send failure notification after {self.retry_attempts} attempts"
        )
        return False

    async def health_check(self) -> bool:
        """
        Check if API endpoints are reachable.

        Returns:
            True if at least one endpoint is reachable, False otherwise
        """
        if not self.success_endpoint and not self.failure_endpoint:
            logger.warning("No API endpoints configured")
            return False

        session = await self._get_session()
        results = []

        # Check success endpoint
        if self.success_endpoint:
            try:
                async with session.get(
                    self.success_endpoint, headers=self._get_headers()
                ) as response:
                    reachable = response.status < 500
                    results.append(reachable)
                    logger.info(
                        f"Success endpoint: {'reachable' if reachable else 'unreachable'} "
                        f"(status {response.status})"
                    )
            except Exception as e:
                logger.warning(f"Success endpoint unreachable: {e}")
                results.append(False)

        # Check failure endpoint
        if self.failure_endpoint:
            try:
                async with session.get(
                    self.failure_endpoint, headers=self._get_headers()
                ) as response:
                    reachable = response.status < 500
                    results.append(reachable)
                    logger.info(
                        f"Failure endpoint: {'reachable' if reachable else 'unreachable'} "
                        f"(status {response.status})"
                    )
            except Exception as e:
                logger.warning(f"Failure endpoint unreachable: {e}")
                results.append(False)

        return any(results)

    def __repr__(self) -> str:
        """String representation of API client."""
        return (
            f"AsyncAPIClient("
            f"success={bool(self.success_endpoint)}, "
            f"failure={bool(self.failure_endpoint)}, "
            f"retries={self.retry_attempts})"
        )
