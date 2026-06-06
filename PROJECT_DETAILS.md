# By-Law Infraction Detection System - Comprehensive Project Documentation

## Executive Summary

The By-Law Infraction Detection System is a desktop application that automatically detects potential by-law violations in video footage using a two-stage computer vision pipeline. It combines fast object detection (YOLO) with deep semantic analysis (CLIP) and human-in-the-loop validation to achieve high accuracy while maintaining transparency throughout the detection process.

**Key Capabilities:**
- Process videos up to several minutes in length
- Extract and analyze frames at configurable rates (default: 2 FPS)
- Display real-time frame triage results before classification
- Provide detailed detection metadata (object types, confidence scores)
- Organize results by by-law for team-based analysis
- Support extensible by-law definitions without code changes

---

## Project Architecture

### System Overview

```
INPUT (Video File)
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 1: Frame Extraction & Preprocessing      │
├─────────────────────────────────────────────────┤
│ • Extract frames at 2 FPS                       │
│ • Resize to 640×640 pixels                      │
│ • Apply white balance correction                │
│ • Apply CLAHE for contrast enhancement          │
│ • Normalize to [0, 1] float32                   │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 2: YOLO Fast Triage                       │
├─────────────────────────────────────────────────┤
│ • Run YOLOv8 Nano on each frame                 │
│ • Detect trigger classes (person, dog, etc.)   │
│ • Confidence threshold: 0.5                     │
│ • Flag relevant frames                          │
│ • Collect detection metadata                    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 3: User Reviews Flagged Frames           │
├─────────────────────────────────────────────────┤
│ • Display carousel of flagged frames            │
│ • Show detected objects with confidence scores  │
│ • Allow navigation (Previous/Next)              │
│ • User approves or cancels analysis             │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 4: CLIP Deep Classification               │
├─────────────────────────────────────────────────┤
│ • Use CLIP vision-language model                │
│ • Compare image against bylaw descriptions      │
│ • Include visual cues in text prompts           │
│ • Return best-match bylaw + confidence          │
│ • Truncate prompts to 77 tokens max             │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 5: Human Validation                       │
├─────────────────────────────────────────────────┤
│ • Display AI prediction                         │
│ • Ask: "Is bylaw correct?"                      │
│ • Ask: "Is violation status correct?"           │
│ • Allow user to correct either field            │
│ • Merge corrections with prediction             │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 6: Results Logging                        │
├─────────────────────────────────────────────────┤
│ • Store complete prediction record              │
│ • Log organized by final bylaw (after correction)│
│ • Track model vs. user decision                 │
│ • Update bylaw-specific statistics              │
│ • Write to predictions_log.json                 │
└─────────────────────────────────────────────────┘
    ↓
OUTPUT (predictions_log.json organized by bylaw)
```

### Component Interaction

**app.py** (Main Application)
- `VideoSelectionWindow` class: Enhanced GUI with video preview, metadata display, and playback controls
- `FramePreviewWindow` class: Carousel UI for reviewing flagged frames with detection metadata
- `run_analysis()` function: Orchestrates the complete pipeline
- `main()` function: Continuous video selection loop

**pipeline.py** (Video Processing & Classification)
- `extract_frames()`: Extracts frames at configurable FPS with preprocessing
- `preprocess_frame()`: Applies resize → white balance → CLAHE → normalization
- `flag_frames_with_yolo()`: YOLO inference and frame flagging with detection collection
- `classify_with_internvl()`: CLIP classification with semantic analysis
- `run_pipeline()`: Coordinates all stages and returns results

**feedback.py** (User Validation)
- `display_prediction()`: Console output of AI prediction
- `collect_user_feedback()`: Interactive two-question validation
- `merge_feedback_with_prediction()`: Combines AI and user corrections

**logger.py** (Results Management)
- `_load_or_initialize()`: Creates/loads predictions_log.json
- `log_prediction()`: Appends prediction to JSON organized by final bylaw
- `log_error()`: Logs pipeline errors to ERRORS section

**bylaws.py** (Configuration)
- `BYLAWS` list: Extensible definition of all by-laws
- `get_bylaw_by_id()`: Lookup by-law by ID
- `get_all_bylaws()`: Returns all by-law definitions

---

## By-Laws Configuration

### Current By-Laws

#### BL-001: Leash Law
- **Description:** Dogs must be on a leash at all times in public spaces.
- **Visual Cues:** 
  - Dog without leash
  - Dog roaming freely
  - Unleashed dog
  - Dog off-leash near people
- **CLIP Prompt:** "A violation of Leash Law: Dogs must be on a leash at all times in public spaces. Visible signs: dog without leash, dog roaming freely, unleashed dog, dog off-leash near people"

#### BL-002: No Trespassing / Fence Jumping
- **Description:** Persons must not climb, jump over, or cross fences into restricted areas.
- **Visual Cues:**
  - Person climbing fence
  - Person jumping over fence
  - Person scaling barrier
  - Person crossing fence line
- **CLIP Prompt:** "A violation of No Trespassing / Fence Jumping: Persons must not climb, jump over, or cross fences into restricted areas. Visible signs: person climbing fence, person jumping over fence, person scaling barrier, person crossing fence line"

#### BL-003: Littering Prohibition
- **Description:** Persons must not discard waste in public spaces.
- **Visual Cues:**
  - Person throwing trash on ground
  - Person dropping garbage
  - Throwing bottle or can
  - Leaving trash behind
- **CLIP Prompt:** "A violation of Littering Prohibition: Persons must not discard waste in public spaces. Visible signs: person throwing trash on ground, person dropping garbage, throwing bottle or can, leaving trash behind"

#### BL-004: Public Violence / Affray
- **Description:** Physical violence, fighting, or threatening behaviour in public is prohibited.
- **Visual Cues:**
  - Person striking another
  - Physical altercation between persons
  - Aggressive threatening gestures
  - Person attacking another
- **CLIP Prompt:** "A violation of Public Violence / Affray: Physical violence, fighting, or threatening behaviour in public is prohibited. Visible signs: person striking another, physical altercation between persons, aggressive threatening gestures, person attacking another"

### Adding New By-Laws

To add a new by-law (e.g., BL-005):

1. Edit `bylaws.py` and append to `BYLAWS` list:
```python
{
    "id": "BL-005",
    "name": "Your By-Law Name",
    "description": "Detailed description of what constitutes a violation.",
    "visual_cues": ["visual cue 1", "visual cue 2", "visual cue 3", "visual cue 4"]
}
```

2. The system automatically:
   - Includes it in CLIP classification prompts
   - Adds it to the logging structure in predictions_log.json
   - Tracks violations under the new bylaw ID
   - No code changes required

---

## Technical Implementation Details

### Stage 1: Frame Extraction & Preprocessing

**Frame Extraction:**
- Uses OpenCV's `cv2.VideoCapture` to read video files
- Extracts frames at `FRAMES_PER_SECOND` rate (default: 2.0 FPS)
- Calculates frame interval: `frame_interval = fps / FRAMES_PER_SECOND`
- Skips frames to achieve target rate
- Returns preprocessed frames + original frame indices

**Preprocessing Pipeline (for each frame):**

1. **Resize to 640×640 pixels**
   - Standardizes input size for YOLO consistency
   - Method: Linear interpolation (`cv2.INTER_LINEAR`)

2. **White Balance Correction (Gray World Assumption)**
   - Calculates mean of each RGB channel
   - Scales channels so average = 128
   - Formula: `pixel_new = pixel_old * (128 / channel_mean)`
   - Handles varying lighting conditions

3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Applied to L channel in LAB color space
   - Clip limit: 2.0 (prevents over-amplification)
   - Tile grid size: 8×8
   - Benefits: Enhances details in low-light scenarios without noise amplification

4. **Normalization to [0, 1]**
   - Converts from uint8 [0-255] to float32 [0-1]
   - Formula: `normalized = uint8_frame / 255.0`
   - Required by YOLO and CLIP models

**Output:** Frames in shape (640, 640, 3) with dtype float32, values in [0, 1]

### Stage 2: YOLO Triage

**Model Details:**
- Architecture: YOLOv8 Nano (smallest, fastest variant)
- Weights: `yolov8n.pt` (~6.3 MB)
- Trained on: COCO dataset (80 object classes)
- Device: CPU (auto-detects GPU if available)

**Detection Process:**
```python
# For each preprocessed frame:
frame_uint8 = (frame * 255).astype(np.uint8)  # Convert back to uint8
results = yolo(frame_uint8, conf=0.5)         # Run inference

# Extract detections:
detected_classes = results[0].boxes.cls       # Class indices [0, 15, 2, ...]
confidences = results[0].boxes.conf           # Confidence scores [0.95, 0.87, ...]
class_names = results[0].names                # Name mapping {0: 'person', 15: 'dog', ...}
```

**Trigger Classes (9 total):**
```python
TRIGGER_CLASSES = {
    "person",        # Humans (for violence, trespassing, littering)
    "dog",           # Dogs (for leash law)
    "cat",           # Cats (for leash law - edge case)
    "bicycle",       # Vehicles/property
    "car",           # Vehicles (may indicate restricted area)
    "bottle",        # Littering (waste)
    "sports ball",   # Activity objects
    "knife",         # Weapons (violence indicator)
    "baseball bat"   # Weapons/tools
}
```

**Frame Flagging Logic:**
- If ANY detected object's class is in `TRIGGER_CLASSES` → frame flagged
- Collects ALL trigger class detections + confidence scores
- Stores metadata: `{"classes": ["person", "dog"], "confidences": [0.92, 0.88]}`

**Output:** Flagged frames + detection metadata passed to user review

### Stage 3: User Frame Review

**Frame Preview Carousel:**
- Displays flagged frames one at a time
- Shows: Frame index, detected objects, confidence scores
- Navigation: Previous/Next buttons
- Actions: Proceed to Classification or Cancel

**Example Display:**
```
Frame 3 of 8 Flagged
Detected: person (0.92), dog (0.88)

[← Previous]  [Next →]
[✓ Proceed to Classification]  [✗ Cancel Analysis]
```

### Stage 4: CLIP Classification

**Model Details:**
- Architecture: CLIP (Contrastive Language-Image Pre-training)
- Model ID: `openai/clip-vit-base-patch32`
- Processor: CLIPProcessor from Hugging Face transformers
- Token limit: 77 tokens (enforced via truncation)

**Classification Process:**

1. **Convert frames to PIL Images:**
   - Convert float32 [0, 1] frames to uint8 [0, 255]
   - Apply cv2.cvtColor BGR→RGB conversion
   - Create PIL Image objects

2. **Build Text Prompts:**
   - For each bylaw: `"A violation of {name}: {description} Visible signs: {visual_cues}"`
   - Add "No by-law violation visible in this image" as negative class
   - Total: N bylaws + 1 negative = 5 text prompts

3. **Image-Text Similarity Matching:**
   ```
   CLIP Processor:
   ├─ Resize image to 224×224
   ├─ Normalize with ImageNet mean/std
   ├─ Tokenize text (max 77 tokens)
   └─ Create batch inputs
   
   CLIP Model:
   ├─ Encode image → vision embedding
   ├─ Encode texts → text embeddings
   ├─ Compute logits (cosine similarity)
   └─ Apply softmax → probabilities [0.0-1.0]
   ```

4. **Select Best Match:**
   - Find highest probability class
   - Return: bylaw ID, confidence score, reasoning
   - Example: `{"bylaw": "BL-004", "violated": True, "confidence": 0.95}`

**Text Prompt Example (Leash Law):**
```
"A violation of Leash Law: Dogs must be on a leash at all times 
in public spaces. Visible signs: dog without leash, dog roaming 
freely, unleashed dog, dog off-leash near people"
```

**Output:** Single prediction dictionary with best-match bylaw

### Stage 5: Human Validation

**Two-Question Feedback System:**

Question 1: "Is the identified by-law correct?" (Y/N)
- If NO: Present dropdown of all bylaws, user selects correct one

Question 2: "Is the violation status correct?" (Y/N)
- If NO: User confirms whether violation occurred (Y/N)

**Feedback Dictionary:**
```python
{
    "bylaw_correct": True/False,
    "violation_correct": True/False,
    "corrected_bylaw": "BL-002" or None,
    "corrected_violated": True/False or None
}
```

**Merging Logic:**
- If user corrected bylaw → use `corrected_bylaw`
- Else use model's `prediction["bylaw"]`
- Same for violation status
- Result: `final_label` with user-validated values

### Stage 6: JSON Logging

**Log File Structure:** `logs/predictions_log.json`

**Format:**
```json
{
  "metadata": {
    "created": "2026-04-28T16:55:00",
    "last_updated": "2026-04-28T17:04:47.136497",
    "total_predictions": 5
  },
  "predictions_by_bylaw": {
    "BL-001": {
      "entries": [...],
      "total_count": 0,
      "violations_count": 0
    },
    "BL-002": {
      "entries": [
        {
          "timestamp": "2026-04-28T16:55:29.997498",
          "video_file": "path/to/video.mp4",
          "frames_extracted": 6,
          "frames_flagged_by_yolo": 6,
          "model_prediction": {
            "bylaw": "BL-002",
            "violated": true,
            "confidence": 0.77,
            "reasoning": "Detected potential violation..."
          },
          "user_feedback": {
            "bylaw_correct": true,
            "violation_correct": true,
            "corrected_bylaw": null,
            "corrected_violated": null
          },
          "final_label": {
            "bylaw": "BL-002",
            "violated": true
          }
        }
      ],
      "total_count": 3,
      "violations_count": 3
    },
    "BL-003": { ... },
    "BL-004": { ... },
    "ERRORS": { ... }  // Pipeline failures logged here
  }
}
```

**Organization Logic:**
- Entries grouped by `final_label["bylaw"]` (user's final decision, not model's)
- If model predicts BL-002 but user corrects to BL-001 → stored under BL-001
- `total_count`: Number of predictions for this bylaw
- `violations_count`: Number of confirmed violations (violated=true)

**Entry Contents:**
- **timestamp:** ISO format (UTC)
- **video_file:** Full path to source video
- **frames_extracted:** Total frames extracted from video
- **frames_flagged_by_yolo:** Frames that passed YOLO triage
- **model_prediction:** AI output (before user correction)
- **user_feedback:** User's answers to validation questions
- **final_label:** Final label after merging corrections

---

## User Interface Features

### Video Selection Window (Enhanced)

**Layout:** 900×700 pixels

**Components:**

1. **Browse Button + Drag-Drop Area**
   - Click to open file browser
   - Or drag video directly onto app
   - Supports: MP4, AVI, MOV, MKV, WebM, FLV, WMV

2. **Left Panel: Metadata Display**
   - **Resolution:** 1920×1080 (or actual video resolution)
   - **Duration:** 4m 32s (calculated from frame count / FPS)
   - **FPS:** 30 (frames per second)
   - **File:** Filename only

3. **Left Panel: Thumbnail Preview**
   - 160×120 pixel preview of first frame
   - Updates on video selection

4. **Right Panel: Video Preview Canvas**
   - 480×300 pixel playback area
   - Displays current frame during playback
   - Black background

5. **Playback Controls**
   - **Play/Pause Button:** Toggle playback
   - **Seek Slider:** Navigate through frames
   - **Frame Info:** Current frame / total frames
   - Uses threading to prevent UI freeze

6. **Start Analysis Button**
   - Large green button
   - Disabled until video selected
   - Starts the pipeline

### Frame Preview Window (Carousel)

**Layout:** 700×650 pixels

**Components:**

1. **Title:** "Flagged Frames Preview (X total)"

2. **Frame Display Canvas**
   - 480×300 pixels
   - Shows current flagged frame

3. **Frame Information**
   - "Frame X of Y Flagged" (e.g., "Frame 3 of 8 Flagged")

4. **Detection Metadata**
   - "Detected: person (0.92), dog (0.88)"
   - Shows all trigger classes found by YOLO
   - Includes confidence scores

5. **Navigation Buttons**
   - **← Previous:** View previous flagged frame
   - **Next →:** View next flagged frame
   - Only navigate through flagged frames, not all frames

6. **Action Buttons**
   - **✓ Proceed to Classification:** Continue to CLIP analysis
   - **✗ Cancel Analysis:** Abort and return to video selection

---

## Installation & Setup

### Requirements
- Python 3.8+
- 4 GB RAM minimum
- 2 GB free disk space
- GPU optional but recommended for speed

### Dependencies
```
ultralytics>=8.0.0          # YOLOv8
opencv-python>=4.8.0        # Video processing
Pillow>=10.0.0             # Image handling
torch>=2.0.0               # PyTorch
torchvision>=0.15.0        # Vision utilities
transformers==4.40.0       # CLIP model
timm>=0.9.0                # Model utilities
sentencepiece>=0.2.0       # Tokenization
protobuf>=3.20.0           # Protocol buffers
einops>=0.7.0              # Tensor operations
accelerate>=0.24.0         # Distributed inference
numpy>=1.24.0              # Numerical computing
```

### Installation Steps
```bash
# Clone repository
git clone https://github.com/hr14310/wilproject.git
cd wilproject

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### First Run
- Application will download YOLOv8 model (~6.3 MB) on first use
- Will download CLIP model (~340 MB) on first use
- Total initial setup: ~15-20 minutes depending on internet speed

---

## Usage Workflow

### GUI Mode (Default)
```
1. python app.py
2. Window opens: Video Selection UI
3. Select video (browse or drag-drop)
4. Preview video + metadata displayed
5. Click "Start Analysis"
6. Terminal shows: Frame extraction → YOLO triage → CLIP classification
7. Frame preview window appears: Review flagged frames
8. Click "Proceed to Classification"
9. Terminal shows: AI prediction
10. Answer two validation questions
11. Results logged to predictions_log.json
12. Choose: "Process another video?" (Yes/No)
13. Loop back to step 3 or exit
```

### Command Line Mode (Single Video)
```bash
python app.py /path/to/video.mp4
```
- Skips GUI
- Processes single video
- Outputs results to predictions_log.json
- Exits after completion

---

## Configuration

### Tunable Parameters (pipeline.py)

**Frame Extraction:**
```python
FRAMES_PER_SECOND = 2.0  # Default: 2 FPS
# Lower = fewer frames (faster) but might miss violations
# Higher = more frames (slower) but better coverage
```

**YOLO Triage:**
```python
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Default: 0.5 (0.0-1.0)
# Lower = more false positives (more frames flagged)
# Higher = more false negatives (might miss violations)

YOLO_MODEL_NAME = "yolov8n"  # Options: yolov8n, yolov8s, yolov8m, yolov8l
# n=nano (fastest, smallest), l=large (slowest, most accurate)
```

**Debug Mode:**
```python
DEBUG_MODE = False  # Set to True to save flagged frames to disk
DEBUG_FRAMES_DIR = "debug_frames"  # Directory for debug frames
```

---

## Performance Metrics

### Typical Processing Times (per video)

**Test Video:** 702 frames @ 30 FPS = 23.4 seconds
**Extracted:** 51 frames @ 2 FPS
**CPU: Intel i7, 16GB RAM**

| Stage | Time | Notes |
|-------|------|-------|
| Frame Extraction | 2-3s | Depends on video codec/resolution |
| YOLO Triage (51 frames) | 15-20s | Per-frame processing |
| Frame Preview UI | <1s | Just displays |
| CLIP Classification (6 frames) | 5-10s | Model inference |
| User Feedback | Variable | Interactive |
| Logging | <1s | JSON write |
| **Total** | **25-35s** | Per video |

**GPU Acceleration (NVIDIA CUDA):**
- CLIP classification: ~2-3x faster
- YOLO: ~1.5-2x faster
- Overall: ~2x speedup expected

---

## Error Handling

### Common Issues & Solutions

**"Cannot open video file"**
- Cause: Codec not supported or file corrupted
- Solution: Try converting to MP4 with H.264 codec

**"CLIP model inference failed"**
- Cause: Out of memory (OOM)
- Solution: Close other applications, reduce FRAMES_PER_SECOND

**"image 'pyimage11' doesn't exist"**
- Cause: PhotoImage reference lost (Tkinter issue)
- Solution: Restart application, select a different video first

**"Token length exceeds 77"**
- Cause: By-law description + visual cues too long
- Solution: Shorten visual_cues or description in bylaws.py

**Double execution error (rare)**
- Cause: Mainloop quit/destroy state issue
- Solution: Already fixed in current version

---

## Data Organization & Analysis

### Using predictions_log.json for Analysis

**Query by By-Law:**
```python
import json
with open('logs/predictions_log.json') as f:
    log_data = json.load(f)

# Get all leash law violations:
bl_001 = log_data['predictions_by_bylaw']['BL-001']
print(f"BL-001 entries: {bl_001['total_count']}")
print(f"Violations: {bl_001['violations_count']}")
```

**Statistics:**
- Total predictions per bylaw
- Violation rate per bylaw
- Model accuracy (predictions matching user corrections)
- Processing metadata (frame counts, timestamps)

### Team Segregation
- Each bylaw section contains all relevant predictions
- Team can filter by bylaw for focused analysis
- Easy to generate reports by bylaw category

---

## Extending the System

### Adding New By-Laws (No Code Changes)
1. Edit bylaws.py
2. Add new entry to BYLAWS list
3. System automatically integrates

### Customizing YOLO Trigger Classes
Edit `pipeline.py`:
```python
TRIGGER_CLASSES = {
    "person", "dog", "cat",  # Add/remove classes
    # ... other classes
}
```

### Improving Prompts
Edit bylaw descriptions and visual_cues in `bylaws.py`:
```python
"visual_cues": [
    "more specific cue 1",
    "more specific cue 2",
    # Add more detailed cues for better CLIP matching
]
```

---

## Technical Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Video Processing | OpenCV | 4.8.0+ | Frame extraction, color conversion |
| Object Detection | YOLOv8 | Nano | Fast triage for relevant frames |
| Image-Text Matching | CLIP | ViT-B/32 | Semantic by-law classification |
| Deep Learning | PyTorch | 2.0.0+ | Model inference |
| GUI Framework | Tkinter | 3.8+ | Desktop application UI |
| Image Processing | Pillow | 10.0.0+ | PIL image handling |
| Data Storage | JSON | Native | Human-readable, structured logging |
| Language | Python | 3.8+ | Cross-platform compatibility |

---

## Project Files

```
WIL_final/
├── app.py                 # Main application (GUI + orchestration)
├── pipeline.py            # Video processing & classification
├── feedback.py            # User validation system
├── logger.py              # JSON logging & organization
├── bylaws.py              # By-law configuration
├── requirements.txt       # Python dependencies
├── README.md              # Quick start guide
├── PROJECT_DETAILS.md     # This comprehensive documentation
├── yolov8n.pt             # YOLO model weights (~6.3 MB)
├── logs/
│   └── predictions_log.json  # Main output file (organized by bylaw)
└── .gitignore             # Git ignore patterns
```

---

## Future Enhancements

**Potential Improvements:**
1. **Bounding Box Visualization** — Draw rectangles on frames showing detected objects
2. **Spatial Relationship Detection** — Analyze proximity between objects (e.g., dog near person for leash detection)
3. **Multi-Frame Context** — Analyze sequences of frames for motion-based detection
4. **Real-Time Streaming** — Process live camera feeds instead of pre-recorded videos
5. **Performance Optimization** — GPU batch processing for multiple frames
6. **Advanced Metrics** — Generate detailed reports with graphs and statistics
7. **Model Fine-Tuning** — Custom CLIP training on by-law-specific examples
8. **Mobile Version** — Deploy as web app for on-site use
9. **Automated Corrections** — Use feedback history to improve CLIP prompts
10. **Multi-Language Support** — Support bylaws in different languages

---

## Maintenance & Support

### Troubleshooting Workflow
1. Check console output for error messages
2. Verify video file is valid (try another video)
3. Check system resources (RAM, disk space)
4. Review configuration constants in pipeline.py
5. Check predictions_log.json for recent entries

### Logging Best Practices
- Review predictions_log.json regularly
- Check for patterns in user corrections
- Monitor violation rates per bylaw
- Use metadata to improve by-law descriptions if needed

### Performance Tuning
- Adjust FRAMES_PER_SECOND based on video resolution
- Lower YOLO_CONFIDENCE_THRESHOLD for edge cases
- Close other applications for faster processing
- Use GPU if available for significant speedup

---

## Summary

The By-Law Infraction Detection System provides a robust, extensible solution for automated violation detection with human oversight. Its two-stage pipeline (YOLO + CLIP) balances speed with accuracy, while the human validation layer ensures high-quality results. The extensible by-law system allows easy addition of new violations without code changes, making it suitable for diverse regulatory environments.

**Key Strengths:**
- Fast, real-time processing
- High transparency (frame preview with detections)
- Human-in-the-loop validation for accuracy
- Extensible without code changes
- Comprehensive logging and analytics
- Clean, organized JSON output by bylaw

**Current Capabilities:**
- Process videos of any length
- Detect 4 core by-laws
- Support 9 YOLO trigger classes
- Validate predictions with user feedback
- Organize results for team analysis

**Ideal For:**
- Municipal enforcement agencies
- Security teams
- Video surveillance analysis
- Compliance monitoring
- Research and development
