import logging
import cv2
import os

from PIL import Image, ImageTk

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
        min_face_size = int(min(height, width) * 0.1)
        
        logging.info(f"Original image shape: {image.shape}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.info(f"RGB image shape: {image_rgb.shape}")
        
        display_image = image_rgb.copy()
        
        scales = [1.0, 0.75, 1.25]
        best_results = None
        max_faces = 0
        
        logging.info(f"Attempting detection at scales: {scales}")
        
        for scale in scales:
            logging.info(f"Trying scale: {scale}")
            scaled_image = cv2.resize(image_rgb, (0, 0), fx=scale, fy=scale)
            results = face_mesh_image.process(scaled_image)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > max_faces:
                best_results = results
                max_faces = len(results.multi_face_landmarks)
                logging.info(f"New best result at scale {scale}: found {max_faces} faces")
        
        results = best_results if best_results else face_mesh_image.process(image_rgb)
        
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
            valid_faces = [(size, landmarks) for size, landmarks in faces_with_size if size * width * height >= min_face_size * min_face_size]
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
                    x = landmark.x * width
                    y = landmark.y * height
                    self.all_landmarks.append({"face_index": face_idx, "landmark_id": idx, "x": x, "y": y})
                logging.info(f"Face {face_idx + 1}: Processed {len(face_landmarks.landmark)} landmarks")
            
            alpha = 0.6
            display_image = cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0)

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
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
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
