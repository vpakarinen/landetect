import tkinter as tk
import cv2
from PIL import Image, ImageTk
import logging

def update(self):
    try:
        if hasattr(self, 'vid') and self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh_video.process(image)

                if results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                    )
                    
                    self.all_landmarks = []
                    for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                        logging.info(f"Landmark {idx}: x={landmark.x}, y={landmark.y}")
                        x = landmark.x * frame.shape[1]
                        y = landmark.y * frame.shape[0]
                        self.all_landmarks.append({"frame": self.frame_count, "landmark_id": idx, "x": x, "y": y})

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.image = image.resize((self.ui.canvas_width, self.ui.canvas_height), Image.Resampling.LANCZOS)
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
