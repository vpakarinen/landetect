import tkinter as tk
import logging
import cv2

from PIL import Image, ImageTk

def process_video_frame(app, frame):
    """Process a single video frame and return the processed frame and landmarks."""
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_landmarks = []
        
        try:
            results = app.face_mesh_video.process(image)
            
            if results and results.multi_face_landmarks:
                landmark_spec = app.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=2,
                    circle_radius=1
                )
                connection_spec = app.mp_drawing.DrawingSpec(
                    color=(0, 0, 255),
                    thickness=2
                )
                
                overlay = frame.copy()
                app.mp_drawing.draw_landmarks(
                    image=overlay,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=app.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec,
                )
                
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    face_data = {
                        "frame": app.frame_count,
                        "face_index": face_idx,
                        "landmarks": []
                    }
                    
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = round(landmark.x * frame.shape[1], 2)
                        y = round(landmark.y * frame.shape[0], 2)
                        z = round(landmark.z, 3)
                        
                        face_data["landmarks"].append({
                            "id": idx,
                            "position": {
                                "x": x,
                                "y": y,
                                "z": z
                            }
                        })
                    
                    frame_landmarks.append(face_data)
            
        except Exception as e:
            logging.error(f"Error processing landmarks: {e}")
        
        return frame, frame_landmarks
            
    except Exception as e:
        logging.error(f"Error in process_video_frame: {e}")
        return None, []

def update(app):
    """Update the video frame and process landmarks."""
    try:
        if not hasattr(app, 'vid') or not app.vid or not app.vid.isOpened():
            return
            
        if not app.playing:
            return
            
        ret, frame = app.vid.read()
        if ret:
            frame, frame_landmarks = process_video_frame(app, frame)
            
            if frame is not None:
                try:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    app.image = image.resize((app.ui.canvas_width, app.ui.canvas_height), Image.Resampling.LANCZOS)
                    app.photo = ImageTk.PhotoImage(app.image)
                    app.ui.canvas.create_image(0, 0, image=app.photo, anchor=tk.NW)
                    app.ui.canvas.image = app.photo
                    
                    app.frame_count += 1
                    
                    if app.realtime_capture:
                        app.all_landmarks.extend(frame_landmarks)
                    
                except Exception as e:
                    logging.error(f"Error displaying frame: {e}")
                    
        else:
            logging.info(f"End of video reached. Processed {app.frame_count} frames.")
            if app.vid:
                app.vid.release()
                app.vid = None
            app.ui.canvas.delete("all")
            app.ui.btn_play_pause.config(state=tk.DISABLED)
            app.export_to_json()
            app.frame_count = 0
            
    except Exception as e:
        logging.error(f"Critical error in update: {e}")
        if app.vid:
            app.vid.release()
            app.vid = None
        app.ui.canvas.delete("all")
        app.ui.btn_play_pause.config(state=tk.DISABLED)
        
    app.window.after(app.delay, app.update)
