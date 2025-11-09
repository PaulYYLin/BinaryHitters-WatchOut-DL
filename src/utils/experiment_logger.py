"""
Experiment data logger for fall detection analysis.

Saves detailed fall detection data including:
- Landmark positions
- Detection method results
- Trigger conditions
- Video clips (ring buffer)
- Event metadata
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class ExperimentDataLogger:
    """
    Logger for experiment data collection and analysis.

    Saves fall detection events with detailed information for
    offline analysis and algorithm improvement.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize experiment data logger.

        Args:
            output_dir: Directory to save experiment data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.video_dir = self.output_dir / "videos"
        self.data_dir.mkdir(exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)

        logger.info(f"ExperimentDataLogger initialized: {output_dir}")

    def generate_event_id(self) -> str:
        """
        Generate unique event ID.

        Returns:
            Unique event ID string
        """
        # Format: YYYYMMDD_HHMMSS_UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"

    def save_fall_event(
        self,
        event_id: str,
        landmarks: npt.NDArray[np.float64] | None,
        fall_info: dict[str, Any],
        timestamp: float,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save fall event data to JSON file.

        Args:
            event_id: Unique event identifier
            landmarks: (33, 4) array of pose landmarks [x, y, z, visibility]
            fall_info: Fall detection results dictionary
            timestamp: Event timestamp
            metadata: Additional metadata (optional)

        Returns:
            Path to saved JSON file
        """
        # Prepare fall_info (exclude landmarks from fall_info to avoid duplication)
        fall_info_clean = {k: v for k, v in fall_info.items() if k != "landmarks"}

        # Prepare data structure
        data = {
            "event_id": event_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "fall_info": fall_info_clean,
            "metadata": metadata or {},
        }

        # Add landmarks if available
        if landmarks is not None:
            data["landmarks"] = {
                "shape": list(landmarks.shape),
                "data": landmarks.tolist(),
            }

        # Extract triggered conditions
        triggered_conditions = self._extract_triggered_conditions(fall_info_clean)
        data["triggered_conditions"] = triggered_conditions

        # Save to file
        json_path = self.data_dir / f"{event_id}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved experiment data: {json_path}")
        return json_path

    def _extract_triggered_conditions(
        self, fall_info: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract which conditions triggered the fall detection.

        Args:
            fall_info: Fall detection results dictionary

        Returns:
            Dictionary with triggered conditions
        """
        triggered: dict[str, Any] = {
            "methods_triggered": [],
            "votes": 0,
            "details": {},
        }

        # Check if multi-criteria method was used
        if fall_info.get("method") == "multi_criteria":
            triggered["votes"] = fall_info.get("votes", 0)
            details = fall_info.get("details", {})

            # Check each detection method
            for method_name, method_info in details.items():
                if method_info.get("is_fall", False) and method_info.get(
                    "valid", False
                ):
                    triggered["methods_triggered"].append(method_name)
                    triggered["details"][method_name] = {
                        "triggered": True,
                        **method_info,
                    }
                else:
                    triggered["details"][method_name] = {
                        "triggered": False,
                        **method_info,
                    }
        else:
            # Single method detection
            method = fall_info.get("method", "unknown")
            is_fall = fall_info.get("is_fall", False)
            if is_fall:
                triggered["methods_triggered"].append(method)
            triggered["details"][method] = fall_info

        return triggered

    def get_video_path(self, event_id: str) -> Path:
        """
        Get path for video file.

        Args:
            event_id: Event identifier

        Returns:
            Path for video file
        """
        return self.video_dir / f"{event_id}.mp4"

    def save_summary(self, events: list[dict[str, Any]]) -> Path:
        """
        Save summary of all experiment events.

        Args:
            events: List of event dictionaries

        Returns:
            Path to summary file
        """
        summary = {
            "total_events": len(events),
            "generated_at": datetime.now().isoformat(),
            "events": events,
        }

        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved experiment summary: {summary_path}")
        return summary_path

    def load_event_data(self, event_id: str) -> dict[str, Any] | None:
        """
        Load event data from file.

        Args:
            event_id: Event identifier

        Returns:
            Event data dictionary or None if not found
        """
        json_path = self.data_dir / f"{event_id}.json"

        if not json_path.exists():
            logger.warning(f"Event data not found: {event_id}")
            return None

        with open(json_path) as f:
            data: dict[str, Any] = json.load(f)

        # Convert landmarks back to numpy array
        if "landmarks" in data:
            landmarks_info = data["landmarks"]
            data["landmarks"] = np.array(landmarks_info["data"])

        return data

    def list_events(self) -> list[str]:
        """
        List all saved event IDs.

        Returns:
            List of event IDs
        """
        json_files = list(self.data_dir.glob("*.json"))
        event_ids = [f.stem for f in json_files]
        return sorted(event_ids)

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about saved experiment data.

        Returns:
            Dictionary with statistics
        """
        event_ids = self.list_events()

        # Calculate storage usage
        data_size = sum(f.stat().st_size for f in self.data_dir.glob("*.json"))
        video_size = sum(f.stat().st_size for f in self.video_dir.glob("*.mp4"))

        data_size_mb = data_size / (1024 * 1024)
        video_size_mb = video_size / (1024 * 1024)
        total_size_mb = data_size_mb + video_size_mb

        stats: dict[str, Any] = {
            "total_events": len(event_ids),
            "data_dir": str(self.data_dir),
            "video_dir": str(self.video_dir),
            "events": event_ids,
            "data_size_mb": data_size_mb,
            "video_size_mb": video_size_mb,
            "total_size_mb": total_size_mb,
        }

        return stats

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"ExperimentDataLogger("
            f"events={stats['total_events']}, "
            f"size={stats['total_size_mb']:.1f}MB)"
        )
