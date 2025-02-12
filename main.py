import mediapipe as mp
import tkinter as tk
import datetime
import logging
import json
import cv2
import io

from tkinter import filedialog
from PIL import Image, ImageTk

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
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # UI elements
        window_width = 800
        window_height = 600
        self.canvas_width = window_width - 20
        self.canvas_height = window_height - 150

        # Canvas for displaying image or video
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Horizontal line above buttons
        separator = tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN)
        separator.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        self.btn_load_image = tk.Button(window, text="Load Image", width=20, command=self.load_image)
        self.btn_load_image.grid(row=2, column=0, padx=10, pady=10)

        self.btn_load_video = tk.Button(window, text="Load Video", width=20, command=self.load_video)
        self.btn_load_video.grid(row=2, column=1, padx=10, pady=10)

        self.btn_output_landmarks = tk.Button(window, text="Output Landmarks", width=20, command=self.output_landmarks)
        self.btn_output_landmarks.grid(row=2, column=2, padx=10, pady=10)

        self.btn_export_to_json = tk.Button(window, text="Export to JSON", width=20, command=self.export_to_json)
        self.btn_export_to_json.grid(row=2, column=3, padx=10, pady=10)

        self.btn_screenshot = tk.Button(window, text="Take Screenshot", width=20, command=self.take_screenshot)
        self.btn_screenshot.grid(row=3, column=0, columnspan=4, padx=10, pady=10)

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
                self.image = self.image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(self.image)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
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
                    return  # Exit the function if video cannot be opened
                else:
                    self.all_landmarks = []
                    logging.info(f"Video file opened successfully: {self.video_path}")
        except Exception as e:
            logging.error(f"Error loading video: {e}")

    def output_landmarks(self):
        """Output landmarks to console."""
        try:
            if hasattr(self, 'image_path') and self.image_path:
                image = cv2.imread(self.image_path)
                if image is None:
                    logging.warning("Failed to load image.")
                    return
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, landmark in enumerate(face_landmarks.landmark):
                            x = landmark.x * image.shape[1]
                            y = landmark.y * image.shape[0]
                            print(f"Landmark {idx}: x={x}, y={y}")
                            self.all_landmarks.append({"frame": 0, "landmark_id": idx, "x": x, "y": y})
            elif hasattr(self, 'vid') and self.vid and self.vid.isOpened():
                pass
            else:
                logging.warning("No image or video source available.")
                return

        except Exception as e:
            logging.error(f"Error outputting landmarks: {e}")

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

                # Convert the image to Tkinter format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.image = Image.fromarray(image)
                self.image = self.image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(self.image)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            except Exception as e:
                logging.error(f"Error detecting landmarks on image: {e}")

    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete("all")

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
            ps = self.canvas.postscript(colormode='color')

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
                        self.image = self.image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
                        self.photo = ImageTk.PhotoImage(self.image)
                        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                        
                        self.frame_count += 1
                    else:
                        logging.info(f"End of video reached. Processed {self.frame_count} frames.")
                        if self.vid:
                            self.vid.release()
                            self.vid = None
                        self.canvas.delete("all")
                        self.frame_count = 0  # Reset frame counter

                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
                    if self.vid:
                        self.vid.release()
                        self.vid = None
                    self.canvas.delete("all")

            self.window.after(self.delay, self.update)
        except Exception as e:
            logging.error(f"Error in update loop: {e}")

if __name__ == "__main__":
    try:
        window = tk.Tk()
        LandmarkDetectorApp(window, "LanDetect - Face and Body Landmark Detection")
    except Exception as e:
        logging.error(f"An error occurred during application startup: {e}")
