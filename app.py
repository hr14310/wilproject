"""
By-Law Infraction Detection Application
Desktop application for detecting by-law violations in video footage.
Enhanced with video preview, frame inspection, and improved UI.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time

from pipeline import run_pipeline
from feedback import display_prediction, collect_user_feedback, merge_feedback_with_prediction
from logger import log_prediction, log_error
from bylaws import get_all_bylaws


class VideoSelectionWindow:
    """Enhanced GUI window for video file selection with preview and playback."""

    def __init__(self, root):
        """Initialize video selection window."""
        self.root = root
        self.root.title("By-Law Infraction Detection - Select & Preview Video")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        self.selected_video = None
        self.video_properties = None
        self.playback_thread = None
        self.is_playing = False
        self.playback_frame_index = 0
        self.stop_playback = False
        self.photoimages = {}  # Cache for PhotoImage objects

        self._create_widgets()
        self._setup_drag_drop()

    def _create_widgets(self):
        """Create UI widgets for video selection and preview."""
        # Title
        title = tk.Label(
            self.root,
            text="By-Law Infraction Detection",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        title.pack(pady=10)

        # Selection area (Browse + Drag drop)
        selection_frame = tk.Frame(self.root, bg="#f0f0f0")
        selection_frame.pack(fill=tk.X, padx=20, pady=5)

        browse_btn = tk.Button(
            selection_frame,
            text="📂 Browse for Video",
            command=self._browse_file,
            padx=15,
            pady=8,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10),
            cursor="hand2"
        )
        browse_btn.pack(side=tk.LEFT)

        self.drop_label = tk.Label(
            selection_frame,
            text="or Drag Video Here",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="gray"
        )
        self.drop_label.pack(side=tk.LEFT, padx=10)

        # Main content area (split: left = thumbnail + metadata, right = preview)
        content_frame = tk.Frame(self.root, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left: Thumbnail and metadata
        left_frame = tk.Frame(content_frame, bg="white", relief=tk.RIDGE, bd=1)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        # Thumbnail canvas
        self.thumbnail_canvas = tk.Canvas(
            left_frame,
            width=160,
            height=120,
            bg="#e0e0e0",
            highlightthickness=0
        )
        self.thumbnail_canvas.pack(pady=10, padx=10)

        # Metadata labels
        self.metadata_labels = {}
        metadata_frame = tk.Frame(left_frame, bg="white")
        metadata_frame.pack(fill=tk.X, padx=10, pady=5)

        for label_name in ["Resolution", "Duration", "FPS", "File"]:
            frame = tk.Frame(metadata_frame, bg="white")
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=f"{label_name}:", font=("Arial", 8, "bold"), bg="white", width=10, anchor="w").pack(side=tk.LEFT)
            self.metadata_labels[label_name] = tk.Label(
                frame, text="—", font=("Arial", 8), bg="white", fg="gray", wraplength=120, justify=tk.LEFT
            )
            self.metadata_labels[label_name].pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Right: Video preview/playback
        right_frame = tk.Frame(content_frame, bg="white", relief=tk.RIDGE, bd=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right_frame,
            text="Video Preview",
            font=("Arial", 10, "bold"),
            bg="white"
        ).pack(pady=5)

        # Preview canvas
        self.preview_canvas = tk.Canvas(
            right_frame,
            width=480,
            height=300,
            bg="#000000",
            highlightthickness=0
        )
        self.preview_canvas.pack(pady=10, padx=10)

        # Playback controls
        control_frame = tk.Frame(right_frame, bg="white")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.play_btn = tk.Button(
            control_frame,
            text="▶ Play",
            command=self._toggle_playback,
            padx=10,
            pady=5,
            bg="#FF9800",
            fg="white",
            font=("Arial", 9),
            cursor="hand2",
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=2)

        # Seek slider
        self.seek_var = tk.DoubleVar()
        self.seek_slider = tk.Scale(
            control_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.seek_var,
            command=self._on_seek,
            bg="white",
            highlightthickness=0,
            state=tk.DISABLED
        )
        self.seek_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Frame info
        self.frame_info_label = tk.Label(
            right_frame,
            text="No video selected",
            font=("Arial", 8),
            bg="white",
            fg="gray"
        )
        self.frame_info_label.pack(pady=5)

        # Bottom: File path display and Start Analysis button
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.X, padx=20, pady=10)

        self.path_label = tk.Label(
            bottom_frame,
            text="No file selected",
            font=("Arial", 9),
            bg="#f0f0f0",
            fg="gray",
            wraplength=850
        )
        self.path_label.pack(pady=5)

        analyze_btn = tk.Button(
            bottom_frame,
            text="▶ Start Analysis",
            command=self._start_analysis,
            padx=20,
            pady=10,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            cursor="hand2"
        )
        analyze_btn.pack()

    def _setup_drag_drop(self):
        """Setup drag and drop functionality (gracefully skip if unavailable)."""
        try:
            import tkinterdnd2
            self.root.drop_target_register(tkinterdnd2.DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_label.config(text="or Drag Video Here")
        except (ImportError, AttributeError):
            pass

    def _on_drop(self, event):
        """Handle file drop event."""
        try:
            files = event.data.strip('{}').split()
            if files:
                self._set_video(files[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dropped file: {str(e)}")

    def _browse_file(self):
        """Open file dialog to select video."""
        filetypes = (
            ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv"),
            ("All Files", "*.*")
        )

        filepath = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=filetypes
        )

        if filepath:
            self._set_video(filepath)

    def _set_video(self, path):
        """Set the selected video path and load metadata."""
        if not Path(path).exists():
            messagebox.showerror("Error", f"File not found: {path}")
            return

        # Stop any ongoing playback
        self.stop_playback = True
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1)

        # Clear old PhotoImage references
        self.photoimages.clear()

        try:
            # Extract video properties
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Cannot open video file: {path}")
                cap.release()
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = frame_count / fps if fps > 0 else 0

            self.video_properties = {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_sec": duration_sec,
                "path": path
            }

            # Format duration as MM:SS
            minutes = int(duration_sec) // 60
            seconds = int(duration_sec) % 60

            # Update metadata display
            self.metadata_labels["Resolution"].config(text=f"{width}×{height}")
            self.metadata_labels["Duration"].config(text=f"{minutes}m {seconds}s")
            self.metadata_labels["FPS"].config(text=f"{fps:.1f}")
            self.metadata_labels["File"].config(text=Path(path).name)

            # Extract first frame for thumbnail
            ret, frame = cap.read()
            if ret:
                # Resize for thumbnail
                thumb = cv2.resize(frame, (160, 120))
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                thumb_pil = Image.fromarray(thumb_rgb)
                thumb_photo = ImageTk.PhotoImage(thumb_pil)

                # Store reference and display
                self.photoimages["thumbnail"] = thumb_photo
                self.thumbnail_canvas.delete("all")  # Clear old image
                self.thumbnail_canvas.create_image(80, 60, image=thumb_photo)

                # Display first frame in preview canvas
                preview = cv2.resize(frame, (480, 300))
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                preview_pil = Image.fromarray(preview_rgb)
                preview_photo = ImageTk.PhotoImage(preview_pil)

                self.photoimages["preview"] = preview_photo
                self.preview_canvas.delete("all")  # Clear old image
                self.preview_canvas.create_image(240, 150, image=preview_photo)

            cap.release()

            # Update UI state
            self.selected_video = path
            self.is_playing = False
            self.playback_frame_index = 0
            self.stop_playback = False

            self.path_label.config(text=f"✓ Selected: {Path(path).name}", fg="#4CAF50")
            self.play_btn.config(state=tk.NORMAL)
            self.seek_slider.config(state=tk.NORMAL, to=frame_count - 1 if frame_count > 0 else 100)
            self.seek_slider.set(0)
            self.frame_info_label.config(text=f"Frame 0 / {frame_count}", fg="#333")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")

    def _toggle_playback(self):
        """Toggle video playback."""
        if not self.selected_video:
            messagebox.showwarning("Warning", "Please select a video first.")
            return

        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pause")
            self.stop_playback = False

            # Start playback thread
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()

    def _playback_worker(self):
        """Worker thread for video playback."""
        cap = cv2.VideoCapture(self.selected_video)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.playback_frame_index)
        fps = self.video_properties["fps"]
        frame_delay = 1 / fps if fps > 0 else 0.033

        while self.is_playing and not self.stop_playback:
            ret, frame = cap.read()
            if not ret:
                self.is_playing = False
                self.play_btn.config(text="▶ Play")
                break

            # Display frame
            preview = cv2.resize(frame, (480, 300))
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            preview_pil = Image.fromarray(preview_rgb)
            preview_photo = ImageTk.PhotoImage(preview_pil)

            self.photoimages["preview"] = preview_photo
            self.preview_canvas.create_image(240, 150, image=preview_photo)

            # Update frame info and slider
            self.playback_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_count = self.video_properties["frame_count"]
            self.seek_slider.set(self.playback_frame_index)
            self.frame_info_label.config(text=f"Frame {self.playback_frame_index} / {frame_count}", fg="#333")

            time.sleep(frame_delay)

        cap.release()

    def _on_seek(self, value):
        """Handle seek slider movement."""
        if not self.selected_video or self.is_playing:
            return

        try:
            self.playback_frame_index = int(float(value))
            cap = cv2.VideoCapture(self.selected_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.playback_frame_index)
            ret, frame = cap.read()

            if ret:
                preview = cv2.resize(frame, (480, 300))
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                preview_pil = Image.fromarray(preview_rgb)
                preview_photo = ImageTk.PhotoImage(preview_pil)

                self.photoimages["preview"] = preview_photo
                self.preview_canvas.create_image(240, 150, image=preview_photo)

            frame_count = self.video_properties["frame_count"]
            self.frame_info_label.config(text=f"Frame {self.playback_frame_index} / {frame_count}", fg="#333")
            cap.release()
        except Exception:
            pass

    def _start_analysis(self):
        """Start the analysis process."""
        if not self.selected_video:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return

        self.stop_playback = True
        self.is_playing = False
        self.root.quit()


class FramePreviewWindow:
    """Window for previewing flagged frames before CLIP classification."""

    def __init__(self, flagged_frames, flagged_detections):
        """Initialize frame preview window."""
        self.flagged_frames = flagged_frames
        self.flagged_detections = flagged_detections
        self.current_frame_idx = 0
        self.proceed = False
        self.root = None

    def show(self):
        """Display the frame preview window and wait for user decision."""
        if not self.flagged_frames:
            # No frames to preview
            return True

        self.root = tk.Toplevel()
        self.root.title("Flagged Frames Review")
        self.root.geometry("700x650")
        self.root.configure(bg="#f0f0f0")

        # Title
        tk.Label(
            self.root,
            text=f"Flagged Frames Preview ({len(self.flagged_frames)} total)",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        ).pack(pady=10)

        # Frame display canvas
        self.canvas = tk.Canvas(
            self.root,
            width=480,
            height=300,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas.pack(pady=10)

        # Frame info
        self.info_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 10),
            bg="#f0f0f0"
        )
        self.info_label.pack(pady=5)

        # Detection info
        self.detection_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 9),
            bg="#f0f0f0",
            fg="gray",
            wraplength=500,
            justify=tk.LEFT
        )
        self.detection_label.pack(pady=5)

        # Navigation buttons
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="← Previous",
            command=self._prev_frame,
            padx=10,
            pady=5,
            bg="#FF9800",
            fg="white",
            font=("Arial", 9)
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Next →",
            command=self._next_frame,
            padx=10,
            pady=5,
            bg="#FF9800",
            fg="white",
            font=("Arial", 9)
        ).pack(side=tk.LEFT, padx=5)

        # Action buttons (Proceed and Cancel)
        action_frame = tk.Frame(self.root, bg="#f0f0f0")
        action_frame.pack(pady=15, fill=tk.X, padx=20)

        tk.Button(
            action_frame,
            text="✓ Proceed to Classification",
            command=self._proceed,
            padx=15,
            pady=10,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            cursor="hand2",
            width=25
        ).pack(side=tk.LEFT, padx=5, expand=True)

        tk.Button(
            action_frame,
            text="✗ Cancel Analysis",
            command=self._cancel,
            padx=15,
            pady=10,
            bg="#f44336",
            fg="white",
            font=("Arial", 10),
            cursor="hand2",
            width=20
        ).pack(side=tk.LEFT, padx=5, expand=True)

        self._display_frame()
        self.root.wait_window()

        return self.proceed

    def _display_frame(self):
        """Display current frame and metadata."""
        if not self.flagged_frames:
            return

        frame = self.flagged_frames[self.current_frame_idx]

        # Convert frame from float32 [0, 1] to uint8 [0, 255] if needed
        if frame.dtype == np.float32:
            frame = (frame * 255).astype(np.uint8)

        # Resize and display frame
        preview = cv2.resize(frame, (480, 300))

        # Convert BGR to RGB for PIL (frames are in BGR format)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        preview_pil = Image.fromarray(preview_rgb)
        preview_photo = ImageTk.PhotoImage(preview_pil)

        self.canvas.create_image(240, 150, image=preview_photo)
        self.canvas.image = preview_photo  # Keep a reference

        # Update frame info
        self.info_label.config(
            text=f"Frame {self.current_frame_idx + 1} of {len(self.flagged_frames)} Flagged"
        )

        # Update detection info
        detections = self.flagged_detections[self.current_frame_idx]
        if detections and detections.get("classes"):
            detection_text = "Detected: " + ", ".join(
                f"{cls} ({conf:.2f})"
                for cls, conf in zip(detections["classes"], detections["confidences"])
            )
        else:
            detection_text = "No object detections"

        self.detection_label.config(text=detection_text)

    def _prev_frame(self):
        """Show previous frame."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self._display_frame()

    def _next_frame(self):
        """Show next frame."""
        if self.current_frame_idx < len(self.flagged_frames) - 1:
            self.current_frame_idx += 1
            self._display_frame()

    def _proceed(self):
        """Proceed to classification."""
        self.proceed = True
        self.root.destroy()

    def _cancel(self):
        """Cancel analysis."""
        self.proceed = False
        self.root.destroy()


def run_analysis(video_path):
    """Run the complete analysis pipeline for a single video."""
    print("=" * 60)
    print("BY-LAW INFRACTION DETECTION SYSTEM")
    print("=" * 60)
    print(f"Video: {video_path}\n")

    try:
        # Run the detection pipeline (one time only)
        print("Processing video...")
        result = run_pipeline(video_path)

        if not result["success"]:
            raise Exception("Pipeline execution failed")

        # Extract results
        frames_extracted = result["frames_extracted"]
        frames_flagged = result["frames_flagged"]
        prediction = result["prediction"]
        flagged_frames = result.get("flagged_frames", [])
        flagged_detections = result.get("flagged_detections", [])

        # Show frame preview if frames were flagged
        if flagged_frames:
            preview_window = FramePreviewWindow(flagged_frames, flagged_detections)
            if not preview_window.show():
                print("\n✗ Analysis cancelled by user.")
                return False

        # Display prediction to user
        display_prediction(prediction)

        # Collect user feedback
        feedback = collect_user_feedback()

        # Merge feedback with prediction
        final_label = merge_feedback_with_prediction(prediction, feedback)

        # Log the result with bylaws list for index initialization
        log_prediction(
            video_file=video_path,
            frames_extracted=frames_extracted,
            frames_flagged_by_yolo=frames_flagged,
            model_prediction=prediction,
            user_feedback=feedback,
            final_label=final_label,
            bylaws_list=get_all_bylaws()
        )

        print("\n" + "=" * 60)
        print("FINAL LABEL")
        print("=" * 60)
        print(f"By-law       : {final_label['bylaw']}")
        print(f"Violated     : {'Yes' if final_label['violated'] else 'No'}")
        print("=" * 60)
        print("\n✓ Pipeline completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Pipeline error: {str(e)}")
        log_error(video_path, str(e))
        return False


def main():
    """Main entry point."""
    # Check if video path was provided via command line
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not Path(video_path).exists():
            print(f"✗ Error: Video file not found: {video_path}")
            sys.exit(1)
        success = run_analysis(video_path)
        sys.exit(0 if success else 1)
    else:
        # Launch GUI for continuous video selection
        while True:
            root = tk.Tk()
            app = VideoSelectionWindow(root)
            root.mainloop()

            # After window closes, check if a video was selected
            if app.selected_video:
                success = run_analysis(app.selected_video)
                if not success:
                    messagebox.showerror("Error", "Analysis failed. Please try another video.")

                # Ask if user wants to process another video
                response = messagebox.askyesno("Continue", "Process another video?")
                if not response:
                    break
            else:
                break

            # Clean up the window
            try:
                root.destroy()
            except:
                pass

        print("✓ Application closed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
