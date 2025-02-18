import logging
import cv2
import os

from logger_setup import setup_logger
from PIL import Image, ImageTk

logger = setup_logger(__name__)

def detect_landmarks_on_image(self, image_path, face_mesh_image, mp_drawing, mp_face_mesh, ui):
    """Detect landmarks on a loaded image."""
    try:
        self.all_landmarks = []
        
        image = cv2.imread(image_path)
        if image is None:
            logging.warning("Failed to load image.")
            return
            
        logging.info(f"Starting landmark detection for image: {os.path.basename(image_path)}")
        
        height, width = image.shape[:2]
        scale_factor = 1.0
        orig_image = image.copy()
        orig_height, orig_width = orig_image.shape[:2]
        logging.info(f"Original image dimensions: {orig_width}x{orig_height}")
        
        if width < 640 or height < 480:
            scale_factor = max(640 / width, 480 / height)
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            logging.info(f"Upscaling image by factor {scale_factor:.2f} for better detection")
            height, width = image.shape[:2]
        
        if width < 1024 or height < 768:
                additional_scale = 1.5
                if scale_factor * additional_scale > 3.0:
                    additional_scale = 3.0 / scale_factor
                logging.info(f"Additional upscaling by factor {additional_scale:.2f} for improved detection in low resolution")
                image = cv2.resize(image, None, fx=additional_scale, fy=additional_scale, interpolation=cv2.INTER_CUBIC)
                height, width = image.shape[:2]
        
        if height > width:
            min_face_size = int(min(height, width) * 0.03)
        else:
            if scale_factor > 1.0:
                min_face_size = int(min(height, width) * 0.04)
            else:
                min_face_size = int(min(height, width) * 0.06)
        
        logging.info(f"Working image dimensions: {width}x{height}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.info(f"RGB image shape: {image_rgb.shape}")
        
        display_image = image_rgb.copy()
        
        scales = [0.75, 1.0, 1.25, 1.5]
        best_results = None
        max_faces = 0
        
        logging.info(f"Attempting detection at scales: {scales}")
        
        best_scale = 1.0
        for scale in scales:
            logging.info(f"Trying scale: {scale}")
            scaled_image = cv2.resize(image_rgb, (0, 0), fx=scale, fy=scale)
            results = face_mesh_image.process(scaled_image)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > max_faces:
                best_results = results
                max_faces = len(results.multi_face_landmarks)
                best_scale = scale
                logging.info(f"New best result at scale {scale}: found {max_faces} faces")
        
        results = best_results if best_results else face_mesh_image.process(image_rgb)
        overall_scale = scale_factor * best_scale
        
        logging.info(f"Face detection results: {results}")
        logging.info(f"Multi face landmarks present: {results.multi_face_landmarks is not None}")
        
        if results.multi_face_landmarks:
            logging.info(f"Number of faces detected: {len(results.multi_face_landmarks)}")
            logging.info(f"First face landmarks count: {len(results.multi_face_landmarks[0].landmark)}")
            logging.info(f"Found {len(results.multi_face_landmarks)} faces")
            
            faces_with_size = []
            for face_landmarks in results.multi_face_landmarks:
                x_coords = [landmark.x * width for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * height for landmark in face_landmarks.landmark]
                face_width = max(x_coords) - min(x_coords)
                face_height = max(y_coords) - min(y_coords)
                face_size = face_width * face_height
                faces_with_size.append((face_size, face_landmarks))
            
            faces_with_size.sort(reverse=True)
            effective_min_face_size = min_face_size
            if max_faces > 1:
                effective_min_face_size = int(min_face_size * 0.03)
            valid_faces = [(size, landmarks) for size, landmarks in faces_with_size if size >= effective_min_face_size * effective_min_face_size]
            logging.info(f"Valid faces after size filtering: {len(valid_faces)}")
            
            landmark_spec = mp_drawing.DrawingSpec(
                color=(255, 0, 0),
                thickness=2,
                circle_radius=1
            )
            connection_spec = mp_drawing.DrawingSpec(
                color=(0, 0, 255),
                thickness=2
            )
            
            overlay = display_image.copy()
            
            for face_idx, (size, face_landmarks) in enumerate(valid_faces):
                logging.info(f"Processing face {face_idx + 1}, relative size: {size:.4f}")
                mp_drawing.draw_landmarks(
                    image=overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec,
                )
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = (landmark.x * width) / overall_scale
                    y = (landmark.y * height) / overall_scale
                    self.all_landmarks.append({"face_index": face_idx, "landmark_id": idx, "x": round(x, 2), "y": round(y, 2)})
                logging.info(f"Face {face_idx + 1}: Processed {len(face_landmarks.landmark)} landmarks")
            
            alpha = 0.6
            display_image = cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0)
            display_image = cv2.resize(display_image, (orig_width, orig_height), interpolation=cv2.INTER_AREA)
            logging.info("Restored image to original dimensions")
            
            self.image = Image.fromarray(display_image)
            self.image = self.image.resize((ui.canvas_width, ui.canvas_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)
            ui.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            ui.canvas.image = self.photo
            
            self.export_to_json()
            logging.info("Landmarks exported successfully")
            logging.info(f"Total landmarks processed: {len(self.all_landmarks)}")
        else:
            logging.warning("No faces detected in the image")
            logging.info("Trying BGR color space...")
            
            results = face_mesh_image.process(image)
            if results.multi_face_landmarks:
                logging.info("Face detected in BGR color space!")
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=display_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = landmark.x * width
                        y = landmark.y * height
                        self.all_landmarks.append({"landmark_id": idx, "x": x, "y": y})
                
                self.image = Image.fromarray(display_image)
                self.image = self.image.resize((ui.canvas_width, ui.canvas_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(self.image)
                ui.canvas.create_image(0, 0, image=self.photo, anchor="nw")
                ui.canvas.image = self.photo
                self.export_to_json()
            else:
                logging.warning("No faces detected in BGR color space either.")
    except Exception as e:
        logging.error(f"Error detecting landmarks: {e}")
        import traceback
        logging.error(traceback.format_exc())
