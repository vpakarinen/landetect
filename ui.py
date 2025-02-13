import tkinter as tk
from tkinter import ttk

class UI:
    def __init__(self, window, app):
        self.window = window
        self.app = app
        self.window_width = 640
        self.window_height = 600
        self.canvas_width = self.window_width - 20
        self.canvas_height = self.window_height - 150

        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        separator = tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN)
        separator.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        control_frame = ttk.Frame(window)
        control_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=5)

        self.btn_load_image = tk.Button(control_frame, text="Load Image", width=20, command=self.app.load_image)
        self.btn_load_image.grid(row=0, column=0, padx=10, pady=5)

        self.btn_load_video = tk.Button(control_frame, text="Load Video", width=20, command=self.app.load_video)
        self.btn_load_video.grid(row=0, column=1, padx=10, pady=5)

        self.btn_export_to_json = tk.Button(control_frame, text="Export to JSON", width=20, command=self.app.export_to_json)
        self.btn_export_to_json.grid(row=0, column=2, padx=10, pady=5)

        self.btn_play_pause = tk.Button(control_frame, text="Pause", width=20, command=self.app.toggle_play_pause, state=tk.DISABLED)
        self.btn_play_pause.grid(row=0, column=3, padx=10, pady=5)

        options_frame = ttk.LabelFrame(window, text="Options")
        options_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=5, sticky="ew")

        self.realtime_capture_var = tk.BooleanVar(value=False)
        self.realtime_capture_cb = ttk.Checkbutton(
            options_frame, 
            text="Capture landmarks in real-time",
            variable=self.realtime_capture_var,
            command=self.toggle_realtime_capture
        )
        self.realtime_capture_cb.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.btn_screenshot = tk.Button(window, text="Take Screenshot", width=20, command=self.app.take_screenshot)
        self.btn_screenshot.grid(row=4, column=1, padx=10, pady=10)

        window.grid_rowconfigure(4, weight=1)

    def toggle_realtime_capture(self):
        """Handle real-time capture toggle"""
        is_enabled = self.realtime_capture_var.get()
        if is_enabled:
            self.app.start_realtime_capture()
        else:
            self.app.stop_realtime_capture()
