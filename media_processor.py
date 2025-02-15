import tkinter as tk
import logging
import cv2

from concurrent.futures import ThreadPoolExecutor
from logger_setup import setup_logger
from PIL import Image, ImageTk

executor = ThreadPoolExecutor(max_workers=5)
logger = setup_logger(__name__)
frame_cache = {}

def _process_video_frame_internal(app, frame):
    """Internal helper to process a single video frame and return the processed frame and landmarks."""
    try:
        original_h, original_w = frame.shape[:2]
        scale_factor = 1.0
        if original_w < 640 or original_h < 480:
            scale_factor = max(640 / original_w, 480 / original_h)
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_landmarks = []
        
        try:
            results = app.face_mesh_video.process(rgb_image)
            
            if results and results.multi_face_landmarks:
                lips_spec = app.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=1,
                    circle_radius=0
                )
                eyes_spec = app.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=1,
                    circle_radius=0
                )
                face_spec = app.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=0
                )
                mesh_spec = app.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=1,
                    circle_radius=0
                )
                
                overlay = frame.copy()
                
                for face_landmarks in results.multi_face_landmarks:
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=lips_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mesh_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eyes_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eyes_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eyes_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eyes_spec
                    )
                    
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=face_spec
                    )
                
                alpha = 0.4
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    face_data = {
                        "frame": app.frame_count,
                        "face_index": face_idx,
                        "landmarks": []
                    }
                    
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = landmark.x * frame.shape[1]
                        y = landmark.y * frame.shape[0]
                        if scale_factor > 1.0:
                            x = round(x / scale_factor, 2)
                            y = round(y / scale_factor, 2)
                        else:
                            x = round(x, 2)
                            y = round(y, 2)
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
        
        if scale_factor > 1.0:
            frame = cv2.resize(frame, (original_w, original_h), interpolation=cv2.INTER_AREA)
            
        return frame, frame_landmarks
            
    except Exception as e:
        logging.error(f"Error in _process_video_frame_internal: {e}")
        return None, []

def process_video_frame(app, frame):
    """Process a single video frame using caching and multi-threading. Returns the processed frame and landmarks."""
    key = (id(app.vid), app.frame_count)
    global frame_cache, executor
    if key in frame_cache:
        future = frame_cache[key]
    else:
        future = executor.submit(_process_video_frame_internal, app, frame)
        frame_cache[key] = future
    return future.result()

def detect_landmarks_on_image(app, frame):
    """Detect landmarks on a single image."""
    try:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_landmarks = []
        
        try:
            results = app.face_mesh_image.process(rgb_image)
            
            if results and results.multi_face_landmarks:
                lips_spec = app.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=1,
                    circle_radius=0
                )
                eyes_spec = app.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=1,
                    circle_radius=0
                )
                face_spec = app.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=0
                )
                mesh_spec = app.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=1,
                    circle_radius=0
                )
                
                overlay = frame.copy()
                
                for face_landmarks in results.multi_face_landmarks:
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=lips_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mesh_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eyes_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eyes_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eyes_spec
                    )
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eyes_spec
                    )
                    
                    app.mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=app.mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=face_spec
                    )
                
                alpha = 0.4
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    face_data = {
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
            logging.error(f"Error processing landmarks on image: {e}")
        
        return frame, frame_landmarks
            
    except Exception as e:
        logging.error(f"Error in detect_landmarks_on_image: {e}")
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
