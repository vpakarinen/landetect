from PIL import Image, ImageTk
from PIL import ImageGrab

import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import datetime
import logging
import json
import cv2
import os

from image_processor import detect_landmarks_on_image
from media_processor import process_video_frame
from tkinter import filedialog
from ui import UI

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class LandmarkDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.screenshots_dir = "screenshots"
        self.landmarks_dir = "landmarks"
        self.logs_dir = "logs"
        if not os.path.exists(self.screenshots_dir):
            os.makedirs(self.screenshots_dir)
        if not os.path.exists(self.landmarks_dir):
            os.makedirs(self.landmarks_dir)
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        self.last_screenshot_time = 0
        self.last_export_time = 0
        self.throttle_delay = 1000

        self.setup_logging()

        self.vid = None
        self.image_path = None
        self.all_landmarks = []
        self.frame_count = 0
        self.playing = True
        self.delay = 15
        self.last_frame_time = 0

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh_image = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.face_mesh_video = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.ui = UI(window, self)
        self.update()
        self.window.mainloop()

    def load_image(self):
        """Load an image from a file."""
        try:
            logging.info("Loading image...")
            if hasattr(self, 'vid') and self.vid:
                self.vid.release()
                self.vid = None
                self.ui.canvas.delete("all")

            self.image_path = filedialog.askopenfilename(initialdir=".", title="Select an image",
                                                       filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
            if self.image_path:
                logging.info(f"Selected image: {self.image_path}")
                self.image = Image.open(self.image_path)
                self.image = self.image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(self.image)
                self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.ui.canvas.image = self.photo
                self.detect_landmarks_on_image()
                logging.info(f"Successfully loaded image: {self.image_path}")
            else:
                logging.info("Image selection cancelled")
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            logging.info("Disabling play/pause button in load_image")
            self.ui.btn_play_pause.config(state=tk.DISABLED)

    def load_video(self):
        """Load a video from a file."""
        try:
            logging.info("Loading video...")
            if hasattr(self, 'image_path'):
                delattr(self, 'image_path')
                self.ui.canvas.delete("all")

            video_path = filedialog.askopenfilename(initialdir=".", title="Select a video",
                                                   filetypes=(("Video files", "*.mp4;*.avi;*.mov"), ("all files", "*.*")))
            if video_path:
                logging.info(f"Selected video path: {video_path}")
                if hasattr(self, 'vid') and self.vid:
                    logging.info("Releasing previous video")
                    self.vid.release()
                
                self.vid = cv2.VideoCapture(video_path)
                if not self.vid:
                    raise Exception("Failed to create VideoCapture object")
                
                if self.vid.isOpened():
                    # Get video properties
                    width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(self.vid.get(cv2.CAP_PROP_FPS))
                    total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    logging.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {total_frames}")
                    
                    self.frame_count = 0
                    self.playing = True  # Start in playing state
                    self.ui.btn_play_pause.config(state=tk.NORMAL, text="Pause")
                    logging.info("Successfully opened video and enabled play/pause button")
                else:
                    error_msg = f"Error opening video file: {video_path}"
                    logging.error(error_msg)
                    tk.messagebox.showerror("Error", error_msg)
                    self.vid = None
                    self.ui.btn_play_pause.config(state=tk.DISABLED)
            else:
                logging.info("Video selection cancelled")
                self.ui.btn_play_pause.config(state=tk.DISABLED)
        except Exception as e:
            error_msg = f"Error loading video: {str(e)}"
            logging.error(error_msg)
            tk.messagebox.showerror("Error", error_msg)
            if hasattr(self, 'vid') and self.vid:
                self.vid.release()
            self.vid = None
            self.ui.btn_play_pause.config(state=tk.DISABLED)

    def detect_landmarks_on_image(self):
        """Detect landmarks on a loaded image."""
        if hasattr(self, 'image_path') and (not hasattr(self, 'vid') or self.vid is None):
            detect_landmarks_on_image(self, self.image_path, self.face_mesh_image, self.mp_drawing, self.mp_face_mesh, self.ui)

    def clear_canvas(self):
        """Clear the canvas."""
        self.ui.canvas.delete("all")

    def export_to_json(self):
        current_time = int(datetime.datetime.now().timestamp() * 1000)
        if current_time - self.last_export_time < self.throttle_delay:
            return
        self.last_export_time = current_time

        if hasattr(self, 'all_landmarks') and self.all_landmarks:
            now = datetime.datetime.now()
            filename = f"landmark_data_{now.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.landmarks_dir, filename)
            logging.info(f"Exporting landmarks to {filename}")
            landmarks_copy = {"landmarks": self.all_landmarks.copy(), "timestamp": now.isoformat()}
            with open(filepath, 'w') as f:
                json.dump(landmarks_copy, f, indent=2)
            print(f"Landmark data exported to {filepath}")
        else:
            print("No landmark data available to export.")

    def take_screenshot(self):
        """Take a screenshot of the current canvas content."""
        canvas_content = self.ui.canvas.find_all()
        if not canvas_content:
            logging.warning("Canvas is empty. Can't take screenshot.")
            return

        current_time = int(datetime.datetime.now().timestamp() * 1000)
        if current_time - self.last_screenshot_time < self.throttle_delay:
            return
        self.last_screenshot_time = current_time

        try:
            x = self.window.winfo_rootx() + self.ui.canvas.winfo_x()
            y = self.window.winfo_rooty() + self.ui.canvas.winfo_y()
            x1 = x + self.ui.canvas.winfo_width()
            y1 = y + self.ui.canvas.winfo_height()
            # Capture the specified area
            screenshot = ImageGrab.grab().crop((x, y, x1, y1))

            now = datetime.datetime.now()
            filename = f"screenshot_{now.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.screenshots_dir, filename)
            screenshot.save(filepath)
            logging.info(f"Screenshot saved to {filepath}")
        except Exception as e:
            logging.error(f"Error taking screenshot: {e}")

    def update(self):
        """Update the frame for video display."""
        try:
            current_time = int(datetime.datetime.now().timestamp() * 1000)
            
            # Check if enough time has passed since last frame
            if current_time - self.last_frame_time >= self.delay:
                if hasattr(self, 'vid') and self.vid and self.vid.isOpened() and self.playing:
                    try:
                        ret, frame = self.vid.read()
                        if ret:
                            # Process frame and update UI
                            frame = process_video_frame(self, frame)
                            if frame is not None:
                                # Convert frame to PhotoImage and display
                                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                image = image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                                photo = ImageTk.PhotoImage(image)
                                self.ui.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                                self.ui.canvas.image = photo
                                self.frame_count += 1
                        else:
                            # End of video reached
                            logging.info(f"End of video reached. Processed {self.frame_count} frames.")
                            self.vid.release()
                            self.vid = None
                            self.ui.canvas.delete("all")
                            self.ui.btn_play_pause.config(state=tk.DISABLED)
                            self.export_to_json()
                            self.frame_count = 0
                            return
                    except Exception as e:
                        logging.error(f"Error processing video frame: {e}")
                        if self.vid:
                            self.vid.release()
                            self.vid = None
                        self.ui.canvas.delete("all")
                        self.ui.btn_play_pause.config(state=tk.DISABLED)
                        tk.messagebox.showerror("Error", f"Error processing video frame: {str(e)}")
                        return
                
                self.last_frame_time = current_time
        except Exception as e:
            logging.error(f"Error in update loop: {e}")
        finally:
            # Schedule next update
            self.window.after(max(1, self.delay // 2), self.update)

    def setup_logging(self):
        """Setup logging configuration with timestamp-based log files."""
        now = datetime.datetime.now()
        log_filename = os.path.join(self.logs_dir, f"landetect_{now.strftime('%Y%m%d_%H%M%S')}.log")

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def toggle_play_pause(self):
        """Toggle video playback state."""
        if not hasattr(self, 'vid') or not self.vid or not self.vid.isOpened():
            logging.warning("Cannot toggle play/pause: No video loaded or video is not open")
            return
            
        self.playing = not self.playing
        logging.info(f"Video playback {'resumed' if self.playing else 'paused'}")
        
        if self.playing:
            self.ui.btn_play_pause.config(text="Pause")
        else:
            self.ui.btn_play_pause.config(text="Play")

if __name__ == "__main__":
    try:
        window = tk.Tk()
        window.geometry("640x600")
        app = LandmarkDetectorApp(window, "LanDetect - Landmark Detector")
    except Exception as e:
        logging.error(f"Error starting application: {e}")
        import traceback
        logging.error(traceback.format_exc())
