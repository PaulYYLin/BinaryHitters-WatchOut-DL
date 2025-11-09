"""
Async API client for fall detection notifications.
Handles both success (video upload) and failure (text-only) endpoints.
"""

import asyncio
import logging
import time
import urllib.error
import urllib.request
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
        checkin_endpoint: str | None = None,
        device_uid: str | None = None,
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
            checkin_endpoint: URL for device check-in endpoint
            device_uid: Unique identifier for this device
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delays: Delay in seconds between retries (exponential backoff)
        """
        self.success_endpoint = success_endpoint
        self.failure_endpoint = failure_endpoint
        self.checkin_endpoint = checkin_endpoint
        self.device_uid = device_uid
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

                # Add device_uid (required by API)
                if self.device_uid:
                    data.add_field("device_uid", self.device_uid)

                # Add metadata as JSON
                for key, value in metadata.items():
                    data.add_field(key, str(value))

                # Send POST request with progress logging
                logger.info(f"Starting upload of {video_path.name}...")
                async with session.post(
                    self.success_endpoint, data=data, headers=self._get_headers()
                ) as response:

                    if response.status == 200:
                        logger.info(
                            f"✓ Successfully uploaded fall event: {video_path.name}"
                        )
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"✗ Upload failed with status {response.status}: {error_text}"
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

    def device_checkin_sync(self) -> bool:
        """
        Send device check-in request to the server (synchronous version).

        This method is designed to be called from a thread without async/await.
        Uses urllib for synchronous HTTP requests.

        Returns:
            True if check-in successful, False otherwise
        """
        if not self.checkin_endpoint:
            logger.warning("Check-in endpoint not configured, skipping check-in")
            return False

        if not self.device_uid:
            logger.warning("Device UID not configured, skipping check-in")
            return False

        # Construct the full URL with device_uid
        checkin_url = f"{self.checkin_endpoint}/{self.device_uid}"

        try:
            logger.info(f"Sending device check-in to {checkin_url}")

            # Create request with headers
            req = urllib.request.Request(checkin_url, method="GET")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")

            # Send request with timeout
            with urllib.request.urlopen(req, timeout=self.timeout.total) as response:
                status = response.status
                logger.info(f"Received response with status: {status}")

                if status == 200:
                    response_text = response.read().decode("utf-8")
                    logger.info(
                        f"Device check-in successful for {self.device_uid}, "
                        f"response: {response_text[:100]}"
                    )
                    return True
                else:
                    error_text = response.read().decode("utf-8")
                    logger.warning(
                        f"Check-in failed with status {status}: {error_text}"
                    )
                    return False

        except urllib.error.HTTPError as e:
            error_text = e.read().decode("utf-8") if e.fp else str(e)
            logger.warning(f"Check-in HTTP error {e.code}: {error_text}")
            return False

        except urllib.error.URLError as e:
            logger.warning(f"Check-in URL error: {e.reason}")
            return False

        except TimeoutError:
            logger.warning("Check-in timeout")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during check-in: {e}")
            return False

    async def device_checkin(self) -> bool:
        """
        Send device check-in request to the server.

        This method should be called periodically (e.g., every minute)
        to notify the server that the device is online and operational.

        Returns:
            True if check-in successful, False otherwise
        """
        if not self.checkin_endpoint:
            logger.warning("Check-in endpoint not configured, skipping check-in")
            return False

        if not self.device_uid:
            logger.warning("Device UID not configured, skipping check-in")
            return False

        session = await self._get_session()

        # Construct the full URL with device_uid
        checkin_url = f"{self.checkin_endpoint}/{self.device_uid}"

        try:
            logger.info(f"Sending device check-in to {checkin_url}")

            async with session.get(
                checkin_url, headers=self._get_headers()
            ) as response:
                logger.info(f"Received response with status: {response.status}")

                if response.status == 200:
                    response_text = await response.text()
                    logger.info(
                        f"Device check-in successful for {self.device_uid}, "
                        f"response: {response_text[:100]}"
                    )
                    return True
                else:
                    error_text = await response.text()
                    logger.warning(
                        f"Check-in failed with status {response.status}: {error_text}"
                    )
                    return False

        except TimeoutError:
            logger.warning("Check-in timeout")
            return False

        except aiohttp.ClientError as e:
            logger.warning(f"Network error during check-in: {e}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during check-in: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of API client."""
        return (
            f"AsyncAPIClient("
            f"success={bool(self.success_endpoint)}, "
            f"failure={bool(self.failure_endpoint)}, "
            f"checkin={bool(self.checkin_endpoint)}, "
            f"retries={self.retry_attempts})"
        )
