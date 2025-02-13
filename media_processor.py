import tkinter as tk
import logging
import cv2
from PIL import Image, ImageTk

def process_video_frame(app, frame):
    """Process a single video frame and return the processed frame."""
    try:
        # Convert frame from BGR to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
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
                
                app.all_landmarks = []
                for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                    x = landmark.x * frame.shape[1]
                    y = landmark.y * frame.shape[0]
                    app.all_landmarks.append({"frame": app.frame_count, "landmark_id": idx, "x": x, "y": y})
            
        except Exception as e:
            logging.error(f"Error processing landmarks: {e}")
            # Continue with frame display even if landmark processing fails
        
        return frame
            
    except Exception as e:
        logging.error(f"Error in process_video_frame: {e}")
        return None

def update(app):
    """Update the video frame and process landmarks."""
    try:
        if not hasattr(app, 'vid') or not app.vid or not app.vid.isOpened():
            return
            
        if not app.playing:
            return  # Don't process new frames when paused
            
        ret, frame = app.vid.read()
        if ret:
            frame = process_video_frame(app, frame)
            
            if frame is not None:
                try:
                    # Convert frame back to RGB for display
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    app.image = image.resize((app.ui.canvas_width, app.ui.canvas_height), Image.Resampling.LANCZOS)
                    app.photo = ImageTk.PhotoImage(app.image)
                    app.ui.canvas.create_image(0, 0, image=app.photo, anchor=tk.NW)
                    app.ui.canvas.image = app.photo
                    
                    app.frame_count += 1
                    
                except Exception as e:
                    logging.error(f"Error displaying frame: {e}")
                    
        else:
            logging.info(f"End of video reached. Processed {app.frame_count} frames.")
            if app.vid:
                app.vid.release()
                app.vid = None
            app.ui.canvas.delete("all")
            app.ui.btn_play_pause.config(state=tk.DISABLED)  # Disable play/pause button at end of video
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
