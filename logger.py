"""
Logging utilities for the by-law infraction detection pipeline.
Logs all results organized by by-law for easy analysis and segregation.
Predictions are grouped by their final label (after user corrections).
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Configuration constants
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "predictions_log.json")


def ensure_log_directory():
    """Create the logs directory if it doesn't exist."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def _load_or_initialize():
    """Load existing log file or create a fresh one."""
    ensure_log_directory()

    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, recreate it
            pass

    # Create or recreate the file
    log_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_predictions": 0
        },
        "predictions_by_bylaw": {}
    }

    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=2)

    return log_data


def _ensure_bylaw_in_log(log_data, bylaw_id):
    """Ensure a bylaw section exists in the log structure."""
    if bylaw_id not in log_data["predictions_by_bylaw"]:
        log_data["predictions_by_bylaw"][bylaw_id] = {
            "entries": [],
            "total_count": 0,
            "violations_count": 0
        }
    return log_data


def log_prediction(
    video_file,
    frames_extracted,
    frames_flagged_by_yolo,
    model_prediction,
    user_feedback,
    final_label,
    bylaws_list=None
):
    """
    Log a prediction result, organized by final bylaw (after user corrections).

    Args:
        video_file (str): Path to the input video file
        frames_extracted (int): Total frames extracted from video
        frames_flagged_by_yolo (int): Frames flagged by YOLO triage
        model_prediction (dict): CLIP prediction output
        user_feedback (dict): User feedback and corrections
        final_label (dict): Final label after user feedback
        bylaws_list (list): List of bylaw definitions (not used in new structure)
    """
    # Load existing or create new log file
    log_data = _load_or_initialize()

    # Get the final bylaw (after user correction)
    final_bylaw_id = final_label["bylaw"]

    # Ensure the bylaw section exists
    log_data = _ensure_bylaw_in_log(log_data, final_bylaw_id)

    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "video_file": str(video_file),
        "frames_extracted": frames_extracted,
        "frames_flagged_by_yolo": frames_flagged_by_yolo,
        "model_prediction": model_prediction,
        "user_feedback": user_feedback,
        "final_label": final_label
    }

    # Add entry to the appropriate bylaw section
    log_data["predictions_by_bylaw"][final_bylaw_id]["entries"].append(log_entry)

    # Update counts
    log_data["predictions_by_bylaw"][final_bylaw_id]["total_count"] += 1
    if final_label["violated"]:
        log_data["predictions_by_bylaw"][final_bylaw_id]["violations_count"] += 1

    # Update metadata
    log_data["metadata"]["last_updated"] = datetime.now().isoformat()
    log_data["metadata"]["total_predictions"] = sum(
        section["total_count"] for section in log_data["predictions_by_bylaw"].values()
    )

    # Write back to file
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"✓ Logged to {LOG_FILE}")


def log_error(video_file, error_message):
    """
    Log an error during pipeline execution.

    Args:
        video_file (str): Path to the input video file
        error_message (str): Error description
    """
    # Load existing or create new log file
    log_data = _load_or_initialize()

    # Store errors under a special "ERRORS" section
    if "ERRORS" not in log_data["predictions_by_bylaw"]:
        log_data["predictions_by_bylaw"]["ERRORS"] = {
            "entries": [],
            "total_count": 0,
            "violations_count": 0
        }

    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "video_file": str(video_file),
        "error": error_message,
        "status": "failed"
    }

    log_data["predictions_by_bylaw"]["ERRORS"]["entries"].append(error_entry)
    log_data["predictions_by_bylaw"]["ERRORS"]["total_count"] += 1

    # Update metadata
    log_data["metadata"]["last_updated"] = datetime.now().isoformat()

    # Write back to file
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"✗ Error logged to {LOG_FILE}")
