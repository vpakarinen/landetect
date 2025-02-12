import mediapipe as mp
import tkinter as tk
import datetime
import logging
import json
import cv2
import os
from PIL import ImageGrab
from media_processor import update
from tkinter import filedialog
from PIL import Image, ImageTk
from ui import UI

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class LandmarkDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.screenshots_dir = "screenshots"
        self.landmarks_dir = "landmarks"
        if not os.path.exists(self.screenshots_dir):
            os.makedirs(self.screenshots_dir)
        if not os.path.exists(self.landmarks_dir):
            os.makedirs(self.landmarks_dir)

        # Throttling for buttons
        self.last_screenshot_time = 0
        self.last_export_time = 0
        self.throttle_delay = 1000

        self.vid = None
        self.image_path = None
        self.all_landmarks = []
        self.frame_count = 0

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh_image = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2
        )
        self.face_mesh_video = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.ui = UI(window, self)

        self.delay = 15
        self.update()
        self.window.mainloop()

    def load_image(self):
        """Load an image from a file."""
        try:
            if hasattr(self, 'vid') and self.vid:
                self.vid.release()
                self.vid = None
                self.ui.canvas.delete("all")

            self.image_path = filedialog.askopenfilename(initialdir=".", title="Select an image",
                                                       filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
            if self.image_path:
                self.image = Image.open(self.image_path)
                self.image = self.image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(self.image)
                self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.ui.canvas.image = self.photo
                self.detect_landmarks_on_image()
        except Exception as e:
            logging.error(f"Error loading image: {e}")

    def load_video(self):
        """Load a video from a file."""
        try:
            if hasattr(self, 'image_path'):
                delattr(self, 'image_path')
                self.ui.canvas.delete("all")

            video_path = filedialog.askopenfilename(initialdir=".", title="Select a video",
                                                   filetypes=(("Video files", "*.mp4;*.avi;*.mov"), ("all files", "*.*")))
            if video_path:
                 if hasattr(self, 'vid') and self.vid:
                     self.vid.release()
                 self.vid = cv2.VideoCapture(video_path)
                 if self.vid.isOpened():
                     self.frame_count = 0
                     logging.info(f"Successfully opened video: {video_path}")
                 else:
                     logging.error(f"Error opening video file: {video_path}")
                     self.vid = None
            else:
                logging.info("Video selection cancelled")
        except Exception as e:
            logging.error(f"Error loading video: {e}")

    def detect_landmarks_on_image(self):
        """Detect landmarks on a loaded image."""
        if hasattr(self, 'image_path') and (not hasattr(self, 'vid') or self.vid is None):
            try:
                self.all_landmarks = []
                
                image = cv2.imread(self.image_path)
                if image is None:
                    logging.warning("Failed to load image.")
                    return
                logging.info(f"Original image shape: {image.shape}")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logging.info(f"RGB image shape: {image_rgb.shape}")
                
                display_image = image_rgb.copy()
                
                results = self.face_mesh_image.process(image_rgb)
                logging.info(f"Face detection results: {results}")
                logging.info(f"Multi face landmarks present: {results.multi_face_landmarks is not None}")
                if results.multi_face_landmarks:
                    logging.info(f"Number of faces detected: {len(results.multi_face_landmarks)}")
                    logging.info(f"First face landmarks count: {len(results.multi_face_landmarks[0].landmark)}")

                if results.multi_face_landmarks:
                    logging.info(f"Found {len(results.multi_face_landmarks)} faces")
                    for face_landmarks in results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=display_image,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                        )

                        for idx, landmark in enumerate(face_landmarks.landmark):
                            x = landmark.x * image.shape[1]
                            y = landmark.y * image.shape[0]
                            self.all_landmarks.append({"frame": 0, "landmark_id": idx, "x": x, "y": y})
                        logging.info(f"Total landmarks detected: {len(self.all_landmarks)}")

                    self.image = Image.fromarray(display_image)
                    self.image = self.image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(self.image)
                    self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    self.ui.canvas.image = self.photo
                    self.export_to_json()
                    
                    logging.info("Landmarks exported successfully")
                else:
                    logging.warning("No faces detected in the image")
                    logging.info("Trying BGR color space...")

                    results = self.face_mesh_image.process(image)
                    if results.multi_face_landmarks:
                        logging.info("Face detected in BGR color space!")
                        for face_landmarks in results.multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image=display_image,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                            )
                            for idx, landmark in enumerate(face_landmarks.landmark):
                                x = landmark.x * image.shape[1]
                                y = landmark.y * image.shape[0]
                                self.all_landmarks.append({"frame": 0, "landmark_id": idx, "x": x, "y": y})
                        self.image = Image.fromarray(display_image)
                        self.image = self.image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                        self.photo = ImageTk.PhotoImage(self.image)
                        self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                        self.ui.canvas.image = self.photo
                        self.export_to_json()
                    else:
                        logging.warning("No faces detected in BGR color space either.")

            except Exception as e:
                logging.error(f"Error detecting landmarks on image: {e}")
                logging.exception("Full traceback:")

    def clear_canvas(self):
        """Clear the canvas."""
        self.ui.canvas.delete("all")

    def export_to_json(self):
        # Check throttling
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
        # Check throttling
        current_time = int(datetime.datetime.now().timestamp() * 1000)
        if current_time - self.last_screenshot_time < self.throttle_delay:
            return
        self.last_screenshot_time = current_time

        try:
            now = datetime.datetime.now()
            filename = f"screenshot_{now.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.screenshots_dir, filename)
            
            x = self.ui.canvas.winfo_rootx()
            y = self.ui.canvas.winfo_rooty()
            width = self.ui.canvas.winfo_width()
            height = self.ui.canvas.winfo_height()
            
            screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
            screenshot.save(filepath, "PNG", optimize=True)
            print(f"Screenshot saved to {filepath}")
        except Exception as e:
            logging.error(f"Error taking screenshot: {e}")

    def update(self):
        """Update the frame for video display."""
        update(self)

if __name__ == "__main__":
    try:
        window = tk.Tk()
        window.geometry("800x600")
        app = LandmarkDetectorApp(window, "LanDetect - Landmark Detector")
    except Exception as e:
        logging.exception("An error occurred during application startup:")
