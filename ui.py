from tkinter import ttk
import tkinter as tk

from screenshot import take_screenshot
from logger_setup import setup_logger

logger = setup_logger(__name__)

class UI:
    def __init__(self, window, app):
        self.window = window
        self.app = app
        self.window_width = 800
        self.window_height = 700
        self.canvas_width = self.window_width - 20
        self.canvas_height = self.window_height - 200

        window.minsize(self.window_width, self.window_height)
        
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height, bg="gray")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.canvas.bind('<Left>', lambda e: self.prev_frame()) 
        self.canvas.bind('<Right>', lambda e: self.next_frame())
        self.canvas.focus_set()

        separator = tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN)
        separator.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        control_frame = tk.LabelFrame(window, text="Controls")
        control_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=5)

        self.btn_load_image = tk.Button(control_frame, text="Load Image", width=15, 
                                        relief="groove", bd=2, command=self.app.load_image)
        self.create_tooltip(self.btn_load_image, "Load image file (Ctrl+I)")
        self.btn_load_image.grid(row=0, column=0, padx=5, pady=5)

        self.btn_load_video = tk.Button(control_frame, text="Load Video", width=15, 
                                        relief="groove", bd=2, command=self.app.load_video)
        self.create_tooltip(self.btn_load_video, "Load video file (Ctrl+O)")
        self.btn_load_video.grid(row=0, column=1, padx=5, pady=5)

        self.btn_export_to_json = tk.Button(control_frame, text="Export to JSON", width=15, 
                                            relief="groove", bd=2, command=self.app.export_to_json)
        self.create_tooltip(self.btn_export_to_json, "Export landmarks to JSON (Ctrl+E)")
        self.btn_export_to_json.grid(row=0, column=2, padx=5, pady=5)

        self.btn_play_pause = tk.Button(control_frame, text="Pause", width=15, 
                                        relief="groove", bd=2, command=self.app.toggle_play_pause, state=tk.DISABLED)
        self.create_tooltip(self.btn_play_pause, "Play/Pause video (Space)")
        self.btn_play_pause.grid(row=0, column=3, padx=5, pady=5)

        frame_control_frame = tk.Frame(control_frame)
        frame_control_frame.grid(row=1, column=0, columnspan=4, pady=5)

        self.btn_prev_frame = tk.Button(frame_control_frame, text="←", width=4, 
                                        relief="groove", bd=2, command=self.prev_frame, state=tk.DISABLED)
        self.create_tooltip(self.btn_prev_frame, "Previous frame (Left Arrow)")
        self.btn_prev_frame.pack(side=tk.LEFT, padx=2)

        self.btn_screenshot = tk.Button(frame_control_frame, text="Screenshot", width=15, 
                                        relief="groove", bd=2, command=lambda: take_screenshot(self.app))
        self.btn_screenshot.pack(side=tk.LEFT, padx=5)

        self.btn_next_frame = tk.Button(frame_control_frame, text="→", width=4, 
                                        relief="groove", bd=2, command=self.next_frame, state=tk.DISABLED)
        self.create_tooltip(self.btn_next_frame, "Next frame (Right Arrow)")
        self.btn_next_frame.pack(side=tk.LEFT, padx=2)

        options_frame = tk.LabelFrame(window, text="Options")
        options_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=(5, 10), sticky="ew")

        self.realtime_capture_var = tk.BooleanVar(value=False)
        self.realtime_capture_cb = tk.Checkbutton(
            options_frame, 
            text="Capture landmarks in real-time",
            variable=self.realtime_capture_var,
            command=self.toggle_realtime_capture
        )
        self.realtime_capture_cb.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        window.grid_columnconfigure(0, weight=1)
        for i in range(4):
            window.grid_rowconfigure(i, weight=1 if i == 0 else 0)

    def prev_frame(self):
        """Go to previous frame and pause video"""
        if self.app.playing:
            self.app.toggle_play_pause()
        self.app.previous_frame()

    def next_frame(self):
        """Go to next frame and pause video"""
        if self.app.playing:
            self.app.toggle_play_pause()
        self.app.next_frame()

    def enable_frame_controls(self, enable=True):
        """Enable or disable frame navigation controls"""
        state = tk.NORMAL if enable else tk.DISABLED
        self.btn_prev_frame.config(state=state)
        self.btn_next_frame.config(state=state)

    def create_tooltip(self, widget, text):
        """Create a tooltip for a given widget."""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, justify=tk.LEFT,
                              background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            def hide_tooltip():
                tooltip.destroy()
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            tooltip.bind('<Leave>', lambda e: hide_tooltip())
        widget.bind('<Enter>', show_tooltip)

    def toggle_realtime_capture(self):
        is_enabled = self.realtime_capture_var.get()
        if is_enabled:
            self.app.start_realtime_capture()
            if not self.app.playing:
                self.app.toggle_play_pause()
            self.btn_play_pause.config(state=tk.DISABLED, text="Pause")
        else:
            self.app.stop_realtime_capture()
            self.btn_play_pause.config(state=tk.NORMAL, text="Play")
        self.window.after(50, lambda: self.canvas.focus_set())
    
    def force_start_capture(self):
        logger.info("Force starting real-time capture.")
        self.app.start_realtime_capture()
        if not self.app.playing:
            self.app.toggle_play_pause()
        self.btn_play_pause.config(state=tk.DISABLED, text="Pause")
        
    def video_ended(self):
        """Callback invoked when video playback ends."""
        self.app.stop_realtime_capture()
        if self.app.playing:
            self.app.toggle_play_pause()
        self.app.playing = False
        if hasattr(self.app, 'cleanup'):
            try:
                self.app.cleanup()
            except Exception:
                pass
        else:
            for method in ['stop_video', 'cleanup_video', 'terminate_video']:
                if hasattr(self.app, method):
                    try:
                        getattr(self.app, method)()
                    except Exception:
                        pass
        self.realtime_capture_var.set(False)
        self.btn_play_pause.config(state=tk.NORMAL, text="Play")
        self.realtime_capture_cb.config(state=tk.DISABLED)

        logger.info("Video ended; video stopped, playback reset, background processes terminated, and real-time capture disabled.")

def filter_duplicate_meshes(meshes, base_threshold=50):
    """Filter out duplicate meshes that are too close in proximity."""
    if not meshes:
        return []
    filtered = [meshes[0]]
    for mesh in meshes[1:]:
        threshold = base_threshold
        if 'angle' in mesh and abs(mesh['angle']) > 30:
            threshold = int(base_threshold * 1.5)
        duplicate = False
        for f in filtered:
            dx = mesh['center'][0] - f['center'][0]
            dy = mesh['center'][1] - f['center'][1]
            if (dx**2 + dy**2)**0.5 < threshold:
                duplicate = True
                break
        if not duplicate:
            filtered.append(mesh)
    return filtered

def draw_meshes(meshes, image):
    """Draw meshes on the given image after filtering duplicates."""
    filtered_meshes = filter_duplicate_meshes(meshes)
    from logger_setup import setup_logger
    logger = setup_logger(__name__)
    for mesh in filtered_meshes:
        logger.info(f"Drawing mesh at {mesh['center']}")
    return image
