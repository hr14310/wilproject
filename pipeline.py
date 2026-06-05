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
YOLO_MODEL_NAME = "yolov8n"  # YOLOv8 nano (fastest)

# InternVL4 model settings
INTERNVL_MODEL_ID = "openai/clip-vit-base-patch32"  # Change CLIP model for classification
# Debug settings
DEBUG_MODE = False  # Set to True to save flagged frames to disk
DEBUG_FRAMES_DIR = "debug_frames"

# Classes that trigger YOLO flagging (proxy for by-law violations)
TRIGGER_CLASSES = {
    "person", "dog", "cat", "bicycle", "car", "bottle",
    "sports ball", "knife", "baseball bat"
}

# ============================================================================
# GLOBAL MODEL CACHE
# ============================================================================

_yolo_model = None
_internvl_model = None
_internvl_processor = None


def get_yolo_model():
    """Load YOLO model once and cache it."""
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
    Assumes the average color in the image should be gray.

    Args:
        image (np.ndarray): Input image (BGR format from OpenCV)

    Returns:
        np.ndarray: White-balanced image
    """
    if image is None or image.size == 0:
        return image

    # Calculate the mean of each channel
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
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to BGR
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def preprocess_frame(frame):
    """
    Apply full preprocessing pipeline to a frame.

    Args:
        frame (np.ndarray): Raw frame from video

    Returns:
        np.ndarray: Preprocessed frame (640x640, normalized to [0,1])
    """
    if frame is None or frame.size == 0:
        return None

    # Step 1: Resize to 640x640
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Step 2: Apply white balance correction
    frame = apply_white_balance(frame)

    # Step 3: Apply CLAHE
    frame = apply_clahe(frame)

    # Step 4: Normalize pixel values to [0, 1]
    frame = frame.astype(np.float32) / 255.0

    return frame


# ============================================================================
# FRAME EXTRACTION
# ============================================================================

def extract_frames(video_path):
    """
    Extract frames from video at FRAMES_PER_SECOND rate.

    Args:
        video_path (str): Path to the input video file

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
        fps = 30  # Default fallback

    frame_interval = max(1, int(fps / FRAMES_PER_SECOND))

    frames = []
    frame_indices = []
    frame_count = 0

    print(f"Extracting frames from {video_path} at {FRAMES_PER_SECOND} FPS...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract every frame_interval-th frame
        if frame_count % frame_interval == 0:
            preprocessed = preprocess_frame(frame)
            if preprocessed is not None:
                frames.append(preprocessed)
                frame_indices.append(frame_count)

        frame_count += 1

    cap.release()

    print(f"Extracted {len(frames)} frames (original video had {frame_count} frames)")

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
        # Convert normalized frame back to [0, 255] for YOLO
        frame_uint8 = (frame * 255).astype(np.uint8)

        # Run YOLO inference
        results = yolo(frame_uint8, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

        # Check if any trigger classes were detected
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

        # Progress indicator
        if (frame_idx + 1) % max(1, len(frames) // 10) == 0:
            print(f"  Progress: {frame_idx + 1}/{len(frames)}")

    print(f"YOLO flagged {len(flagged_frames)} frames")

    return flagged_frames, flagged_indices, len(flagged_frames)


# ============================================================================
# INTERNVL4 INFERENCE
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
        flagged_frames (list): List of preprocessed frames from YOLO triage
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

        # Convert flagged frames back to uint8
        flagged_frames_uint8 = [(f * 255).astype(np.uint8) for f in flagged_frames]

        # Get all bylaws
        bylaws = get_all_bylaws()

        # Create classification texts for each bylaw with visual cues for better recognition
        texts = []
        bylaw_ids = []
        for bylaw in bylaws:
            # Build a more descriptive text that includes visual cues
            description = bylaw['description']
            visual_cues = ", ".join(bylaw.get('visual_cues', []))

            if visual_cues:
                # Include visual cues to help CLIP recognize the violation
                text = f"A violation of {bylaw['name']}: {description} Visible signs: {visual_cues}"
            else:
                text = f"A violation of {bylaw['name']}: {description}"

            texts.append(text)
            bylaw_ids.append(bylaw['id'])

        # Also add "no violation" option
        '''
        texts.append("No by-law violation visible in this image")
        bylaw_ids.append("NONE")
        
        texts.append("Person cooking food on a stove with steam or smoke from a pot or pan")
        bylaw_ids.append("NONE")
        texts.append("Normal kitchen cooking with smoke or steam rising from cookware")
        bylaw_ids.append("NONE")
        texts.append("Chef or person preparing food on stovetop indoors")
        bylaw_ids.append("NONE")
        '''
        # Get model device
        device = next(model.parameters()).device

        # Classify ALL flagged frames and keep the strongest violation
        best_violation_confidence = 0.0
        best_violation_bylaw = "NONE"
        best_frame_idx = 0
        votes = []

        for frame_num, frame_uint8 in enumerate(flagged_frames_uint8):
            # Convert to PIL image
            frame_pil = Image.fromarray(cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB))

            # Process image and texts with CLIP
            inputs = processor(
                text=texts,
                images=frame_pil,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get CLIP logits
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image

            # Get probabilities
            probs = torch.softmax(logits_per_image, dim=1)[0]

            # Find best match for this frame
            best_idx = torch.argmax(probs).item()
            frame_bylaw = bylaw_ids[best_idx]
            frame_confidence = probs[best_idx].item()

            print(f"  Frame {frame_num + 1}/{len(flagged_frames_uint8)}: {frame_bylaw} ({frame_confidence:.2f})")
            votes.append((frame_bylaw, frame_confidence))
            # Track the strongest violation (not NONE) across all frames
            if frame_bylaw != "NONE" and frame_confidence > best_violation_confidence:
                best_violation_confidence = frame_confidence
                best_violation_bylaw = frame_bylaw
                best_frame_idx = frame_num

        # Decide final prediction: use the best violation if found, otherwise NONE
        # Decide final prediction using majority vote + confidence threshold
        MAJORITY_THRESHOLD = 0.6   # 60% of frames must agree it's a violation
        CONFIDENCE_THRESHOLD = 0.85  # Winning frame confidence must exceed this

        violation_votes = [(b, c) for b, c in votes if b != "NONE"]
        total_frames = len(votes)
        violation_count = len(violation_votes)

        majority_reached = total_frames > 0 and (violation_count / total_frames) >= MAJORITY_THRESHOLD

        if majority_reached and violation_votes:
            best_bylaw = max(violation_votes, key=lambda x: x[1])[0]
            best_confidence = max(violation_votes, key=lambda x: x[1])[1]
            violated = best_confidence >= CONFIDENCE_THRESHOLD

            # If confidence too low to be meaningful, treat as no violation
            if not violated:
                best_bylaw = "NONE"
                best_confidence = 1.0
                reasoning = "No violations detected with sufficient confidence."
            else:
                bylaw_info = next((b for b in bylaws if b['id'] == best_bylaw), None)
                reasoning = f"Detected potential violation of {bylaw_info['name']}" if bylaw_info else "Violation detected"
        else:
            best_bylaw = "NONE"
            best_confidence = 1.0
            violated = False
            reasoning = "No violations detected in the image."

        prediction = {
            "bylaw": best_bylaw,
            "violated": violated,
            "confidence": float(best_confidence),
            "reasoning": reasoning
        }

        return prediction

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
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            prediction = json.loads(json_str)

            # Validate required fields
            required_fields = ["bylaw", "violated", "confidence", "reasoning"]
            if all(field in prediction for field in required_fields):
                # Ensure confidence is a float
                if not isinstance(prediction["confidence"], (int, float)):
                    prediction["confidence"] = float(prediction["confidence"])
                return prediction
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback if parsing fails
    print("InternVL4 returned malformed output, using fallback")
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
        # Convert normalized frame back to uint8 (frame is already in BGR from preprocessing)
        frame_uint8 = (frame * 255).astype(np.uint8)

        filename = debug_dir / f"flagged_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(filename), frame_uint8)

    print(f"Saved debug frames to {DEBUG_FRAMES_DIR}/")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(video_path):
    """
    Run the complete by-law detection pipeline.

    Args:
        video_path (str): Path to the input video

    Returns:
        dict: Pipeline results including frame counts and predictions
        Raises: ValueError or Exception if pipeline fails
    """
    try:
        # Stage 1: Extract frames
        frames, frame_indices, total_frames = extract_frames(video_path)

        # Stage 2: YOLO triage
        flagged_frames, flagged_indices, num_flagged = flag_frames_with_yolo(frames, frame_indices)

        # Fallback: If YOLO flagged nothing, sample frames for CLIP anyway
        # This handles scenarios like fire/smoke where no YOLO-detectable objects exist
        if num_flagged == 0 and len(frames) > 0:
            print("YOLO flagged 0 frames - sampling frames for CLIP fallback...")
            sample_count = min(10, len(frames))
            step = max(1, len(frames) // sample_count)
            flagged_frames = [frames[i] for i in range(0, len(frames), step)][:sample_count]
            flagged_indices = [frame_indices[i] for i in range(0, len(frame_indices), step)][:sample_count]
            print(f"Sampled {len(flagged_frames)} frames for CLIP analysis")

        # Stage 3: InternVL4 classification
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
        print(f"Pipeline error: {str(e)}")
        raise
