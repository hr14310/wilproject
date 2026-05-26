"""
Core pipeline for by-law infraction detection.
Handles frame extraction, YOLO triage, and InternVL4 classification.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import json
import re
import torch

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Frame extraction settings
FRAMES_PER_SECOND = 2.0  # Extract frames at 2 FPS (configurable)

# YOLO triage settings
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for YOLO detections
YOLO_MODEL_NAME = "yolov8n"      # YOLOv8 nano for general detection


# InternVL4 model settings
INTERNVL_MODEL_ID = "openai/clip-vit-base-patch32"  # CLIP for classification

# Debug settings
DEBUG_MODE = False  # Set to True to save flagged frames to disk
DEBUG_FRAMES_DIR = "debug_frames"

# Classes that trigger general YOLO flagging
TRIGGER_CLASSES = {
    "person", "car", "truck", "motorcycle",  # unsafe driving / general presence
    "boat",                                   # drowning context
    "hard hat",                               #construction
    "gun", "knife"                           #Weapon detection
}

# ============================================================================
# GLOBAL MODEL CACHE
# ============================================================================

_yolo_model = None
_internvl_model = None
_internvl_processor = None


def get_yolo_model():
    """Load general YOLO model once and cache it."""
    global _yolo_model
    if _yolo_model is None:
        print(f"Loading YOLO model ({YOLO_MODEL_NAME})...")
        _yolo_model = YOLO(f"{YOLO_MODEL_NAME}.pt")
    return _yolo_model



def get_internvl_model():
    """Load CLIP model for classification."""
    global _internvl_model, _internvl_processor
    if _internvl_model is None:
        print(f"Loading CLIP model ({INTERNVL_MODEL_ID})...")
        from transformers import CLIPProcessor, CLIPModel
        import warnings
        warnings.filterwarnings('ignore')

        _internvl_processor = CLIPProcessor.from_pretrained(INTERNVL_MODEL_ID)
        _internvl_model = CLIPModel.from_pretrained(INTERNVL_MODEL_ID)
        _internvl_model.eval()

    return _internvl_model, _internvl_processor


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def apply_white_balance(image):
    """
    Apply automatic white balance correction using gray world assumption.

    Args:
        image (np.ndarray): Input image (BGR format from OpenCV)

    Returns:
        np.ndarray: White-balanced image
    """
    if image is None or image.size == 0:
        return image

    result = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        mean = image[:, :, channel].mean()
        if mean > 0:
            result[:, :, channel] = np.clip(image[:, :, channel].astype(np.float32) / mean * 128, 0, 255)
        else:
            result[:, :, channel] = image[:, :, channel].astype(np.float32)

    return np.uint8(result)


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    for improved low-light robustness.

    Args:
        image (np.ndarray): Input image (BGR format)

    Returns:
        np.ndarray: CLAHE-enhanced image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def apply_dashcam_correction(image):
    """
    Correct fisheye/wide-angle lens distortion common in dashcam footage.
    Also reduces windshield glare via dehazing.
    """
    h, w = image.shape[:2]

    K = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([-0.35, 0.1, 0, 0], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0.5)
    undistorted = cv2.undistort(image, K, dist_coeffs, None, new_K)

    lab = cv2.cvtColor(undistorted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.float32) * 0.88 + 8, 0, 255).astype(np.uint8)
    undistorted = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return undistorted


def preprocess_frame(frame, is_dashcam=False):
    """
    Apply full preprocessing pipeline to a frame.
    Set is_dashcam=True for forward-facing camera footage.
    """
    if frame is None or frame.size == 0:
        return None

    if is_dashcam:
        frame = apply_dashcam_correction(frame)

    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    frame = apply_white_balance(frame)
    frame = apply_clahe(frame)
    frame = frame.astype(np.float32) / 255.0

    return frame


# ============================================================================
# FRAME EXTRACTION
# ============================================================================

def extract_frames(video_path, is_dashcam=False):
    """
    Extract frames from video at FRAMES_PER_SECOND rate.

    Args:
        video_path (str): Path to the input video file
        is_dashcam (bool): Apply dashcam lens correction if True

    Returns:
        tuple: (list of preprocessed frames, list of frame indices, total frames extracted)
        Raises: ValueError if video cannot be read
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_interval = max(1, int(fps / FRAMES_PER_SECOND))

    frames = []
    frame_indices = []
    frame_count = 0

    print(f"Extracting frames from {video_path} at {FRAMES_PER_SECOND} FPS...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            preprocessed = preprocess_frame(frame, is_dashcam=is_dashcam)
            if preprocessed is not None:
                frames.append(preprocessed)
                frame_indices.append(frame_count)

        frame_count += 1

    cap.release()
    print(f"✓ Extracted {len(frames)} frames (original video had {frame_count} frames)")

    return frames, frame_indices, frame_count




# ============================================================================
# YOLO TRIAGE
# ============================================================================

def flag_frames_with_yolo(frames, frame_indices):
    """
    Use YOLOv8 to flag frames containing trigger classes.

    Args:
        frames (list): List of preprocessed frames (normalized to [0, 1])
        frame_indices (list): Original frame indices for tracking

    Returns:
        tuple: (flagged_frames, flagged_indices, count of flagged frames)
    """
    if not frames:
        return [], [], 0

    yolo = get_yolo_model()

    flagged_frames = []
    flagged_indices = []

    print(f"\nRunning YOLO triage on {len(frames)} frames...")

    for frame_idx, (frame, orig_idx) in enumerate(zip(frames, frame_indices)):
        frame_uint8 = (frame * 255).astype(np.uint8)

        results = yolo(frame_uint8, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

        is_flagged = False
        if results and results[0].boxes is not None:
            detected_classes = results[0].boxes.cls.cpu().numpy()
            class_names = results[0].names

            for cls_idx in detected_classes:
                cls_name = class_names[int(cls_idx)]
                if cls_name.lower() in TRIGGER_CLASSES:
                    is_flagged = True
                    break

        if is_flagged:
            flagged_frames.append(frame)
            flagged_indices.append(orig_idx)

        if (frame_idx + 1) % max(1, len(frames) // 10) == 0:
            print(f"  Progress: {frame_idx + 1}/{len(frames)}")

    print(f"✓ YOLO flagged {len(flagged_frames)} frames")

    return flagged_frames, flagged_indices, len(flagged_frames)

# ============================================================================
# INTERNVL4 / CLIP INFERENCE
# ============================================================================

def construct_bylaw_prompt():
    """Construct a detailed prompt for InternVL4 about by-laws."""
    from bylaws import get_all_bylaws

    bylaws = get_all_bylaws()
    prompt = """You are a by-law enforcement assistant analyzing images for potential violations.

Below is a list of by-laws and their descriptions:

"""
    for bylaw in bylaws:
        prompt += f"- {bylaw['id']}: {bylaw['name']}\n"
        prompt += f"  Description: {bylaw['description']}\n"
        prompt += f"  Visual cues: {', '.join(bylaw['visual_cues'])}\n\n"

    prompt += """Analyze the provided image(s) and determine:
1. Which (if any) of the above by-laws is present in the image
2. Whether that by-law is being VIOLATED or merely present but not violated

Return ONLY a JSON object (no other text) with this exact structure:
{
  "bylaw": "<bylaw_id or NONE>",
  "violated": true or false,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of the detection>"
}

If no by-laws are being violated, return:
{
  "bylaw": "NONE",
  "violated": false,
  "confidence": 1.0,
  "reasoning": "No violations detected"
}

IMPORTANT: Return ONLY the JSON object, no additional text."""

    return prompt


def classify_with_internvl(flagged_frames, flagged_indices):
    """
    Use CLIP to classify flagged frames for by-law violations.

    Args:
        flagged_frames (list): Preprocessed frames from YOLO triage
        flagged_indices (list): Original frame indices

    Returns:
        dict: Classification result with bylaw, violated, confidence, reasoning
    """
    if not flagged_frames:
        return {
            "bylaw": "NONE",
            "violated": False,
            "confidence": 1.0,
            "reasoning": "No frames were flagged by initial triage."
        }

    try:
        from bylaws import get_all_bylaws

        model, processor = get_internvl_model()
        print(f"\nRunning CLIP classification on {len(flagged_frames)} flagged frames...")

        flagged_frames_uint8 = [(f * 255).astype(np.uint8) for f in flagged_frames]
        frame_pil = Image.fromarray(cv2.cvtColor(flagged_frames_uint8[0], cv2.COLOR_BGR2RGB))

        bylaws = get_all_bylaws()
        texts = []
        bylaw_ids = []

        for bylaw in bylaws:
            description = bylaw['description']
            visual_cues = ", ".join(bylaw.get('visual_cues', []))
            if visual_cues:
                text = f"A violation of {bylaw['name']}: {description} Visible signs: {visual_cues}"
            else:
                text = f"A violation of {bylaw['name']}: {description}"
            texts.append(text)
            bylaw_ids.append(bylaw['id'])

        texts.append("No by-law violation visible in this image")
        bylaw_ids.append("NONE")

        inputs = processor(
            text=texts,
            images=frame_pil,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image

        probs = torch.softmax(logits_per_image, dim=1)[0]
        best_idx = torch.argmax(probs).item()
        best_bylaw = bylaw_ids[best_idx]
        best_confidence = probs[best_idx].item()
        violated = best_bylaw != "NONE"

        if best_bylaw == "NONE":
            reasoning = "No violations detected in the image."
        else:
            bylaw_info = next((b for b in bylaws if b['id'] == best_bylaw), None)
            reasoning = f"Detected potential violation of {bylaw_info['name']}" if bylaw_info else "Violation detected"

        return {
            "bylaw": best_bylaw,
            "violated": violated,
            "confidence": float(best_confidence),
            "reasoning": reasoning
        }

    except Exception as e:
        import traceback
        print(f"Warning: CLIP classification error: {str(e)}")
        traceback.print_exc()
        return {
            "bylaw": "UNCERTAIN",
            "violated": False,
            "confidence": 0.5,
            "reasoning": "Classification failed. Please review manually."
        }


def parse_internvl_response(response_text):
    """
    Parse InternVL4's response to extract structured prediction.

    Args:
        response_text (str): Raw text response from InternVL4

    Returns:
        dict: Parsed prediction or fallback if parsing fails
    """
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            prediction = json.loads(json_str)
            required_fields = ["bylaw", "violated", "confidence", "reasoning"]
            if all(field in prediction for field in required_fields):
                if not isinstance(prediction["confidence"], (int, float)):
                    prediction["confidence"] = float(prediction["confidence"])
                return prediction
    except (json.JSONDecodeError, AttributeError):
        pass

    print("⚠ InternVL4 returned malformed output, using fallback")
    return {
        "bylaw": "UNCERTAIN",
        "violated": False,
        "confidence": 0.5,
        "reasoning": "Unable to parse model response. Please review manually."
    }


# ============================================================================
# DEBUG MODE
# ============================================================================

def save_debug_frames(flagged_frames, flagged_indices):
    """
    Save flagged frames to disk for debugging (only if DEBUG_MODE is True).

    Args:
        flagged_frames (list): Frames to save (normalized float BGR)
        flagged_indices (list): Original frame indices
    """
    if not DEBUG_MODE:
        return

    debug_dir = Path(DEBUG_FRAMES_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(flagged_frames)} flagged frames to {DEBUG_FRAMES_DIR}/...")

    for idx, (frame, frame_idx) in enumerate(zip(flagged_frames, flagged_indices)):
        frame_uint8 = (frame * 255).astype(np.uint8)
        filename = debug_dir / f"flagged_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(filename), frame_uint8)

    print(f"✓ Saved debug frames to {DEBUG_FRAMES_DIR}/")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(video_path, is_dashcam=False):
    """
    Run the complete by-law detection pipeline.

    Args:
        video_path (str): Path to the input video
        is_dashcam (bool): Apply dashcam lens correction if True

    Returns:
        dict: Pipeline results including frame counts and predictions
        Raises: ValueError or Exception if pipeline fails
    """
    try:
        # Stage 1: Extract frames
        frames, frame_indices, total_frames = extract_frames(video_path, is_dashcam=is_dashcam)

        # Stage 2: 3-pass YOLO triage (general + weapon full-frame + weapon tiled)
        flagged_frames, flagged_indices, num_flagged = flag_frames_with_yolo(frames, frame_indices)

        # Stage 3: CLIP classification
        prediction = classify_with_internvl(flagged_frames, flagged_indices)

        # Debug: Save flagged frames
        save_debug_frames(flagged_frames, flagged_indices)

        return {
            "success": True,
            "total_frames": total_frames,
            "frames_extracted": len(frames),
            "frames_flagged": num_flagged,
            "prediction": prediction,
            "flagged_frames": flagged_frames,
            "flagged_indices": flagged_indices
        }

    except Exception as e:
        print(f"✗ Pipeline error: {str(e)}")
        raise