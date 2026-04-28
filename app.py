"""
By-Law Infraction Detection Application
Desktop application for detecting by-law violations in video footage.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import sys
from pathlib import Path

from pipeline import run_pipeline
from feedback import display_prediction, collect_user_feedback, merge_feedback_with_prediction
from logger import log_prediction, log_error
from bylaws import get_all_bylaws


class VideoSelectionWindow:
    """Simple GUI window for video file selection."""

    def __init__(self, root):
        """Initialize video selection window."""
        self.root = root
        self.root.title("By-Law Infraction Detection - Select Video")
        self.root.geometry("500x300")
        self.root.configure(bg="#f0f0f0")

        self.selected_video = None
        self._create_widgets()
        self._setup_drag_drop()

    def _create_widgets(self):
        """Create UI widgets."""
        # Title
        title = tk.Label(
            self.root,
            text="By-Law Infraction Detection",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        )
        title.pack(pady=20)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="Select a video file to analyze",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="gray"
        )
        instructions.pack()

        # Drop zone / Info area
        self.drop_frame = tk.Frame(self.root, bg="white", relief=tk.RIDGE, bd=2)
        self.drop_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.drop_label = tk.Label(
            self.drop_frame,
            text="📁 Click 'Browse' to Select Video",
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#333"
        )
        self.drop_label.pack(expand=True)

        self.info_label = tk.Label(
            self.drop_frame,
            text="Supports: MP4, AVI, MOV, MKV, WebM, FLV, WMV",
            font=("Arial", 9),
            bg="white",
            fg="gray"
        )
        self.info_label.pack()

        # File path display
        self.path_label = tk.Label(
            self.root,
            text="No file selected",
            font=("Arial", 9),
            bg="#f0f0f0",
            fg="gray",
            wraplength=450
        )
        self.path_label.pack(pady=5)

        # Browse button
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        browse_btn = tk.Button(
            button_frame,
            text="📂 Browse for Video",
            command=self._browse_file,
            padx=15,
            pady=8,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10),
            cursor="hand2"
        )
        browse_btn.pack(side=tk.LEFT, padx=5)

        analyze_btn = tk.Button(
            button_frame,
            text="▶ Start Analysis",
            command=self._start_analysis,
            padx=15,
            pady=8,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10),
            cursor="hand2"
        )
        analyze_btn.pack(side=tk.LEFT, padx=5)

    def _setup_drag_drop(self):
        """Setup drag and drop functionality (gracefully skip if unavailable)."""
        try:
            import tkinterdnd2
            self.root.drop_target_register(tkinterdnd2.DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_label.config(text="📁 Drag Video Here or Click 'Browse'")
            self.info_label.config(text="")
        except (ImportError, AttributeError):
            # Drag-and-drop not available, use file browser only
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
        """Set the selected video path."""
        if not Path(path).exists():
            messagebox.showerror("Error", f"File not found: {path}")
            return

        self.selected_video = path
        filename = Path(path).name
        self.path_label.config(text=f"✓ Selected: {filename}", fg="#4CAF50")
        self.drop_label.config(text=f"✓ Ready for analysis")

    def _start_analysis(self):
        """Start the analysis process."""
        if not self.selected_video:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return

        # Just quit the mainloop - run_analysis will be called in main()
        self.root.quit()


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
            root.destroy()  # Clean up after mainloop exits

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

        print("✓ Application closed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
