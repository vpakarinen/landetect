import mediapipe as mp
import tkinter as tk
import numpy as np
import datetime
import logging
import json
import cv2
import os

from media_processor import process_video_frame, detect_landmarks_on_image

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

        self.vid = None
        self.image_path = None
        self.all_landmarks = []
        self.frame_landmarks = []
        self.frame_count = 0
        self.playing = True
        self.delay = 15
        self.last_frame_time = 0
        self.realtime_capture = False

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

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
        self.realtime_capture = False
        
        # Get hand detection setting from UI
        self.hand_detection_enabled = self.ui.hand_detection_var
        
        self.window.bind('<space>', lambda e: self.toggle_play_pause())
        self.window.bind('<Control-o>', lambda e: self.load_video())
        self.window.bind('<Control-i>', lambda e: self.load_image())
        self.window.bind('<Control-e>', lambda e: self.export_to_json())
        
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
            self.ui.btn_play_pause.config(state=tk.DISABLED)

    def load_video(self):
        """Load and process a video file."""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
            )
            
            if file_path:
                logging.debug("Video file selection initiated")
                self.vid = cv2.VideoCapture(file_path)
                
                if not self.vid.isOpened():
                    raise Exception("Failed to open video file")
                    
                width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(self.vid.get(cv2.CAP_PROP_FPS))
                total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
                
                logging.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {total_frames}")
                
                ret, frame = self.vid.read()
                if ret:
                    from media_processor import process_video_frame
                    frame, _ = process_video_frame(self, frame)
                    if frame is not None:
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image = image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(image)
                        self.ui.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                        self.ui.canvas.image = photo
                
                self.playing = False
                self.frame_count = 0
                self.ui.btn_play_pause.config(state=tk.NORMAL)
                self.ui.btn_play_pause.config(text="Play")
                self.ui.enable_frame_controls(True)
                
                logging.info(f"Successfully loaded video: {file_path}")
            else:
                logging.debug("Video file selection dialog closed with no file selected")
                logging.info("Video selection cancelled")
                
        except Exception as e:
            logging.error(f"Error loading video: {str(e)}")
            tk.messagebox.showerror("Error", f"Error loading video: {str(e)}")

    def previous_frame(self):
        """Go to previous frame in video."""
        if self.vid and self.vid.isOpened():
            current_pos = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
            new_pos = max(0, current_pos - 2)
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = self.vid.read()
            if ret:
                frame, _ = process_video_frame(self, frame)
                if frame is not None:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.image = image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(image=self.image)
                    self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    self.frame_count = new_pos

    def next_frame(self):
        """Go to next frame in video."""
        if self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame, _ = process_video_frame(self, frame)
                if frame is not None:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.image = image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(image=self.image)
                    self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    self.frame_count += 1

    def detect_landmarks_on_image(self):
        """Detect landmarks on a loaded image."""
        try:
            if hasattr(self, 'image_path') and (not hasattr(self, 'vid') or self.vid is None):
                cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
                
                # Initialize processed image and landmarks
                processed_image = cv_image.copy()
                landmarks = []
                
                # Face detection if enabled
                if self.ui.face_detection_var.get():
                    logging.info("Face detection enabled, processing...")
                    processed_image, face_landmarks = detect_landmarks_on_image(self, cv_image)
                    if face_landmarks:
                        landmarks.extend(face_landmarks)
                        logging.info("Face landmarks detected")
                    else:
                        logging.info("No faces detected in the image")
                
                # Hand detection if enabled
                if self.ui.hand_detection_var.get():  
                    logging.info("Hand detection enabled, processing...")  
                    try:
                        height, width = cv_image.shape[:2]
                        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                        hand_results = self.hands.process(image_rgb)
                        
                        if hand_results.multi_hand_landmarks:
                            logging.info(f"Found {len(hand_results.multi_hand_landmarks)} hands")  
                            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                                # Create custom drawing styles for hands
                                hand_landmark_style = self.mp_drawing.DrawingSpec(
                                    color=(20, 80, 255),  # Deep blue for landmarks
                                    thickness=2,          # Thinner points
                                    circle_radius=1       # Smaller circles
                                )
                                hand_connection_style = self.mp_drawing.DrawingSpec(
                                    color=(50, 50, 50),   # Dark gray for connections
                                    thickness=1           # Thinner lines
                                )
                                
                                # Draw hand landmarks with custom style
                                self.mp_drawing.draw_landmarks(
                                    processed_image,
                                    hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    landmark_drawing_spec=hand_landmark_style,
                                    connection_drawing_spec=hand_connection_style
                                )
                                
                                # Add hand landmarks to the data
                                hand_data = {
                                    'handedness': handedness.classification[0].label,
                                    'confidence': handedness.classification[0].score,
                                    'landmarks': []
                                }
                                
                                for idx, landmark in enumerate(hand_landmarks.landmark):
                                    x = int(landmark.x * width)
                                    y = int(landmark.y * height)
                                    z = landmark.z
                                    hand_data['landmarks'].append({
                                        'id': idx,
                                        'x': x,
                                        'y': y,
                                        'z': z
                                    })
                                
                                landmarks.append({'hand_data': hand_data})
                        else:
                            logging.info("No hands detected in the image")
                    except Exception as e:
                        logging.error(f"Error in hand detection: {str(e)}")
                
                if processed_image is not None:
                    processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                    processed_pil = processed_pil.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(processed_pil)
                    self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    self.ui.canvas.image = self.photo
                    self.all_landmarks = landmarks

        except Exception as e:
            logging.error(f"Error detecting landmarks on image: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def clear_canvas(self):
        """Clear the canvas."""
        self.ui.canvas.delete("all")

    def export_to_json(self):
        """Export landmarks to JSON file."""
        try:
            if not self.all_landmarks:
                logging.info("No landmark data available to export.")
                tk.messagebox.showinfo("Export", "No landmark data available to export.")
                return

            now = datetime.datetime.now()
            filename = f"landmark_data_{now.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.landmarks_dir, filename)
            
            landmarks_data = {
                "metadata": {
                    "timestamp": now.isoformat(),
                    "total_frames": self.frame_count,
                    "capture_mode": "real-time" if self.realtime_capture else "single-frame",
                    "mediapipe_version": mp.__version__,
                    "application_version": "1.0.0",
                    "face_mesh_config": {
                        "static_image_mode": False,
                        "max_num_faces": 5,
                        "refine_landmarks": True,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5
                    }
                },
                "frames": self.all_landmarks
            }
            
            with open(filepath, 'w') as f:
                json.dump(landmarks_data, f, indent=2)
            
            logging.info(f"Landmarks exported to {filepath}")
            tk.messagebox.showinfo("Export", f"Landmarks exported to {filename}")
            
            if self.realtime_capture:
                self.all_landmarks = []
                
        except Exception as e:
            error_msg = f"Error exporting landmarks: {str(e)}"
            logging.error(error_msg)
            tk.messagebox.showerror("Error", error_msg)


    def start_realtime_capture(self):
        """Start real-time landmark capture"""
        self.realtime_capture = True
        logging.info("Real-time landmark capture enabled")
        self.all_landmarks = []
        
    def stop_realtime_capture(self):
        """Stop real-time landmark capture"""
        self.realtime_capture = False
        logging.info("Real-time landmark capture disabled")
        
    def update(self):
        """Update the frame for video display."""
        try:
            current_time = int(datetime.datetime.now().timestamp() * 1000)
            
            if current_time - self.last_frame_time >= self.delay:
                if hasattr(self, 'vid') and self.vid and self.vid.isOpened() and self.playing:
                    try:
                        ret, frame = self.vid.read()
                        if ret:
                            frame, landmarks = process_video_frame(self, frame)
                            if frame is not None:
                                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                image = image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
                                photo = ImageTk.PhotoImage(image)
                                self.ui.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                                self.ui.canvas.image = photo
                                
                                if self.realtime_capture:
                                    if landmarks:
                                        self.all_landmarks.extend(landmarks)
                                else:
                                    if landmarks:
                                        self.all_landmarks = landmarks
                                
                                self.frame_count += 1
                        else:
                            logging.info(f"End of video reached. Processed {self.frame_count} frames.")
                            self.vid.release()
                            self.vid = None
                            self.ui.canvas.delete("all")
                            self.ui.btn_play_pause.config(state=tk.DISABLED)
                            
                            if len(self.all_landmarks) > 0:
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
            self.window.after(max(1, self.delay // 2), self.update)


    def toggle_play_pause(self):
        """Toggle video playback state."""
        if not hasattr(self, 'vid') or not self.vid or not self.vid.isOpened():
            logging.warning("Cannot toggle play/pause: No video loaded or video is not open")
            return
            
        self.playing = not self.playing
        
        if self.playing:
            self.ui.btn_play_pause.config(text="Pause")
        else:
            self.ui.btn_play_pause.config(text="Play")

if __name__ == "__main__":
    from logger_setup import setup_logger
    logger = setup_logger(__name__)
    try:
        window = tk.Tk()
        window.geometry("640x600")
        app = LandmarkDetectorApp(window, "LanDetect - Landmark Detector")
    except Exception as e:
        logger.exception("Error starting application")
        import traceback
        logger.error(traceback.format_exc())
