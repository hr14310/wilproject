# By-Law Infraction Detection System

A desktop application for automatically detecting potential by-law violations in video footage using a two-stage computer vision pipeline with human-in-the-loop validation.

## Overview

This system processes video files to identify by-law violations through:
1. **Frame extraction and preprocessing** — Intelligent frame sampling and image enhancement
2. **Fast YOLO triage** — Quick object detection to flag relevant frames
3. **CLIP classification** — Deep semantic analysis of flagged frames
4. **Human validation** — User confirmation and correction of predictions
5. **Structured logging** — Results organized by by-law for team analysis

## How It Works

### Pipeline Stages

**Stage 1: Frame Extraction & Preprocessing**
- Extracts frames at configurable rate (default: 2 FPS)
- Resizes frames to 640×640 for consistent processing
- Applies white balance correction (gray world assumption)
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-light robustness
- Normalizes pixel values to [0, 1] range

**Stage 2: YOLO Triage**
- Uses YOLOv8 Nano for fast object detection
- Flags frames containing "trigger classes":
  - Person, dog, cat, bicycle, car, bottle, sports ball, knife, baseball bat
- Confidence threshold: 0.5 (configurable in `pipeline.py`)
- Significantly reduces frames for detailed analysis

**Stage 3: CLIP Classification**
- Uses CLIP vision-language model for semantic understanding
- Compares flagged frames against detailed bylaw descriptions and visual cues
- Scores confidence for each bylaw violation
- Returns top-match bylaw with confidence score

**Stage 4: Human Validation**
- Displays AI prediction with bylaw name, violation status, and confidence
- Asks two validation questions:
  1. "Is the identified by-law correct?"
  2. "Is the violation status correct?"
- Allows user to correct either the bylaw or violation status
- Ensures accuracy through human oversight

**Stage 5: Logging**
- Stores complete prediction record organized by final bylaw
- Tracks both model predictions and user corrections
- Maintains violation counts per bylaw for analytics
- Enables team segregation and analysis by by-law category

## Installation

### Requirements
- Python 3.8+
- 4 GB RAM minimum
- 2 GB free disk space
- GPU recommended (NVIDIA CUDA-compatible)

### Setup

```bash
# Clone/download the repository
cd WIL_final

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Usage

### GUI Mode (Default)

```bash
python app.py
```

1. A window opens for video file selection
2. Browse or drag-drop a video file
3. Click "Start Analysis"
4. Monitor progress in the terminal
5. Answer validation questions in the terminal (Y/N prompts)
6. Results are automatically logged
7. Choose to process another video or exit

### Command Line Mode

```bash
python app.py /path/to/video.mp4
```

Runs analysis on a single video without the GUI loop.

## By-Laws

The system monitors 4 configurable by-laws:

- **BL-001**: Leash Law — Dogs must be on a leash at all times in public spaces
- **BL-002**: No Trespassing / Fence Jumping — Persons must not climb, jump over, or cross fences into restricted areas
- **BL-003**: Littering Prohibition — Persons must not discard waste in public spaces
- **BL-004**: Public Violence / Affray — Physical violence, fighting, or threatening behaviour in public is prohibited

### Adding New By-Laws

Edit `bylaws.py` and append to the `BYLAWS` list:

```python
{
    "id": "BL-005",
    "name": "Your By-Law Name",
    "description": "Detailed description of what constitutes a violation.",
    "visual_cues": ["visual cue 1", "visual cue 2", "visual cue 3"]
}
```

The system automatically:
- Includes new bylaws in CLIP classification
- Adds new bylaw sections to the logging structure
- Tracks violations for the new bylaw

## Configuration

Edit constants in `pipeline.py`:

```python
# Frame extraction
FRAMES_PER_SECOND = 2.0           # Extract frame rate (default: 2 FPS)

# YOLO triage
YOLO_CONFIDENCE_THRESHOLD = 0.5   # Detection threshold (0.0-1.0)
YOLO_MODEL_NAME = "yolov8n"       # Model variant (nano, small, medium, etc.)

# Debug mode
DEBUG_MODE = False                # Save flagged frames to disk
DEBUG_FRAMES_DIR = "debug_frames"  # Output directory for debug frames
```

## Output Format

Results are logged to `logs/predictions_log.json` organized by bylaw:

```json
{
  "metadata": {
    "created": "2026-04-28T16:55:00",
    "last_updated": "2026-04-28T17:08:00",
    "total_predictions": 5
  },
  "predictions_by_bylaw": {
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
            "confidence": 0.97,
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
    }
  }
}
```

### Log Structure Notes

- Entries are grouped by **final_label** bylaw (the user's final decision)
- If model predicts BL-002 but user corrects to BL-001, entry appears under BL-001
- `violations_count` tracks confirmed violations per bylaw
- Errors are logged under "ERRORS" section with `status: "failed"`

## Project Structure

```
app.py              — Main application entry point (GUI + pipeline orchestration)
pipeline.py         — Video processing and classification pipeline
feedback.py         — User feedback collection and validation
logger.py           — Results logging with bylaw organization
bylaws.py           — By-law definitions (extensible configuration)
requirements.txt    — Python dependencies
README.md           — This file

logs/
  predictions_log.json  — Complete prediction log organized by bylaw
  
yolov8n.pt          — YOLOv8 Nano model weights (auto-downloaded on first run)
```

## Dependencies

- **ultralytics** — YOLOv8 object detection
- **torch & torchvision** — PyTorch deep learning framework
- **transformers** — CLIP vision-language model
- **opencv-python** — Image processing
- **Pillow** — Image handling
- **numpy** — Numerical computing

See `requirements.txt` for exact versions.

## Performance Tips

- **GPU Acceleration**: CLIP inference is GPU-accelerated. Use NVIDIA GPUs for significant speedup
- **Frame Rate**: Increase `FRAMES_PER_SECOND` for more detailed analysis (slower)
- **YOLO Threshold**: Lower `YOLO_CONFIDENCE_THRESHOLD` to catch more objects (may increase false positives)
- **Batch Processing**: Process multiple videos by selecting "Process another video?" in the loop

## Troubleshooting

**Token Length Error**: Prompts exceed CLIP's 77-token limit
- Solution: Shorten bylaw descriptions or reduce visual cues in `bylaws.py`

**GPU Out of Memory**: CUDA memory exhausted
- Solution: Use a smaller YOLO variant or reduce frame batch size

**Slow Processing**: Pipeline takes too long
- Solution: Enable GPU acceleration or reduce `FRAMES_PER_SECOND`

## License & Attribution

This system uses:
- **YOLOv8** (Ultralytics, AGPL-3.0)
- **CLIP** (OpenAI, MIT)
- **PyTorch** (Meta, BSD)

## Support

For issues or feature requests, refer to the code documentation in each module.
