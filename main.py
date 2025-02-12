import mediapipe as mp
import tkinter as tk
import datetime
import logging
import json
import cv2
import io

from tkinter import filedialog
from PIL import Image, ImageTk
from ui import UI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LandmarkDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize video related attributes
        self.vid = None
        self.image_path = None
        self.all_landmarks = []
        self.frame_count = 0

        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # UI elements
        self.ui = UI(window, self)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def load_image(self):
        """Load an image from a file."""
        try:
            self.image_path = filedialog.askopenfilename(initialdir=".", title="Select an image",
                                                       filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
            if self.image_path:
                self.image = Image.open(self.image_path)
                self.image = self.image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                logging.info(f"Image size after resizing: {self.image.size}")
                self.photo = ImageTk.PhotoImage(self.image)
                logging.info(f"PhotoImage created: {self.photo}")
                self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.ui.canvas.image = self.photo
                self.detect_landmarks_on_image()
        except Exception as e:
            logging.error(f"Error loading image: {e}")

    def load_video(self):
        """Load a video from a file."""
        try:
            self.video_path = filedialog.askopenfilename(initialdir=".", title="Select a video",
                                                        filetypes=(("Video files", "*.mp4;*.avi;*.mov"), ("all files", "*.*")))
            if self.video_path:
                self.vid = cv2.VideoCapture(self.video_path)
                if not self.vid.isOpened():
                    logging.error("Error opening video file")
                    self.vid = None
                    return
                else:
                    self.all_landmarks = []
                    logging.info(f"Video file opened successfully: {self.video_path}")
        except Exception as e:
            logging.error(f"Error loading video: {e}")

    def detect_landmarks_on_image(self):
        """Detect landmarks on a loaded image."""
        if hasattr(self, 'image_path'):
            try:
                image = cv2.imread(self.image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1),
                        )

                        self.all_landmarks = []
                        for idx, landmark in enumerate(face_landmarks.landmark):
                            logging.info(f"Landmark {idx} detected: x={landmark.x}, y={landmark.y}")
                            x = landmark.x * image.shape[1]
                            y = landmark.y * image.shape[0]
                            self.all_landmarks.append({"frame": 0, "landmark_id": idx, "x": x, "y": y})
                        logging.info(f"Total landmarks detected: {len(self.all_landmarks)}")

                # Convert the image to Tkinter format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.image = Image.fromarray(image)
                self.image = self.image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(self.image)
                self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            except Exception as e:
                logging.error(f"Error detecting landmarks on image: {e}")
            self.export_to_json()

    def clear_canvas(self):
        """Clear the canvas."""
        self.ui.canvas.delete("all")

    def export_to_json(self):
        if hasattr(self, 'all_landmarks') and self.all_landmarks:
            with open('landmark_data.json', 'w') as f:
                json.dump(self.all_landmarks, f)
            print("Landmark data exported to landmark_data.json")
        else:
            print("No landmark data available to export.")

    def take_screenshot(self):
        """Take a screenshot of the current canvas content."""
        try:
            # Generate a unique filename using the current datetime
            now = datetime.datetime.now()
            filename = f"screenshot_{now.strftime('%Y%m%d_%H%M%S')}.png"

            # Get the current canvas content as a postscript image
            ps = self.ui.canvas.postscript(colormode='color')

            # Use PIL to convert the postscript image to a PNG image
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            img.save(filename, "png")
            print(f"Screenshot saved to {filename}")
        except Exception as e:
            logging.error(f"Error taking screenshot: {e}")

    def update(self):
        """Update the frame for video display."""
        try:
            if self.vid and self.vid.isOpened():
                try:
                    ret, frame = self.vid.read()
                    if ret:
                        # Process frame for landmarks
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.face_mesh.process(image_rgb)

                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                # Draw landmarks on the frame
                                self.mp_drawing.draw_landmarks(
                                    image=frame,
                                    landmark_list=face_landmarks,
                                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1),
                                )
                                
                                # Store landmark coordinates
                                for idx, landmark in enumerate(face_landmarks.landmark):
                                    x = landmark.x * frame.shape[1]
                                    y = landmark.y * frame.shape[0]
                                    print(f"Frame {self.frame_count}, Landmark {idx}: x={x:.2f}, y={y:.2f}")
                                    self.all_landmarks.append({"frame": self.frame_count, "landmark_id": idx, "x": x, "y": y})

                        # Convert frame to PIL Image and display
                        self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        self.image = self.image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                        self.photo = ImageTk.PhotoImage(self.image)
                        self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                        self.ui.canvas.image = self.photo 
                        
                        self.frame_count += 1
                    else:
                        logging.info(f"End of video reached. Processed {self.frame_count} frames.")
                        if self.vid:
                            self.vid.release()
                            self.vid = None
                        self.ui.canvas.delete("all")
                        self.export_to_json()
                        self.frame_count = 0

                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
                    if self.vid:
                        self.vid.release()
                        self.vid = None
                    self.ui.canvas.delete("all")

            self.window.after(self.delay, self.update)
        except Exception as e:
            logging.error(f"Error in update loop: {e}")

if __name__ == "__main__":
    try:
        window = tk.Tk()
        window.geometry("800x600")
        app = LandmarkDetectorApp(window, "LanDetect - Landmark Detector")
    except Exception as e:
        logging.exception("An error occurred during application startup:")
