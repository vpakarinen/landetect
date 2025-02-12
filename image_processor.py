from PIL import Image, ImageTk
import logging
import cv2

def detect_landmarks_on_image(self, image_path, face_mesh_image, mp_drawing, mp_face_mesh, ui):
    """Detect landmarks on a loaded image."""
    try:
        self.all_landmarks = []
        
        image = cv2.imread(image_path)
        if image is None:
            logging.warning("Failed to load image.")
            return
        logging.info(f"Original image shape: {image.shape}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.info(f"RGB image shape: {image_rgb.shape}")
        
        display_image = image_rgb.copy()
        
        results = face_mesh_image.process(image_rgb)
        logging.info(f"Face detection results: {results}")
        logging.info(f"Multi face landmarks present: {results.multi_face_landmarks is not None}")
        
        if results.multi_face_landmarks:
            logging.info(f"Number of faces detected: {len(results.multi_face_landmarks)}")
            logging.info(f"First face landmarks count: {len(results.multi_face_landmarks[0].landmark)}")
            logging.info(f"Found {len(results.multi_face_landmarks)} faces")
            
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=display_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                )

                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = landmark.x * image.shape[1]
                    y = landmark.y * image.shape[0]
                    self.all_landmarks.append({"landmark_id": idx, "x": x, "y": y})
                logging.info(f"Total landmarks detected: {len(self.all_landmarks)}")

            self.image = Image.fromarray(display_image)
            self.image = self.image.resize((ui.canvas_width, ui.canvas_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)
            ui.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            ui.canvas.image = self.photo

            self.export_to_json()
            logging.info("Landmarks exported successfully")
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
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                    )
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = landmark.x * image.shape[1]
                        y = landmark.y * image.shape[0]
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
