from tkinter import ttk
import tkinter as tk

class UI:
    def __init__(self, window, app):
        self.window = window
        self.app = app
        self.window_width = 800
        self.window_height = 700
        self.canvas_width = self.window_width - 20
        self.canvas_height = self.window_height - 200

        window.minsize(self.window_width, self.window_height)
        
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.canvas.bind('<Left>', lambda e: self.prev_frame()) 
        self.canvas.bind('<Right>', lambda e: self.next_frame())
        self.canvas.focus_set()

        separator = tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN)
        separator.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        control_frame = ttk.LabelFrame(window, text="Controls")
        control_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=5)

        self.btn_load_image = tk.Button(control_frame, text="Load Image", width=20, command=self.app.load_image)
        self.create_tooltip(self.btn_load_image, "Load image file (Ctrl+I)")
        self.btn_load_image.grid(row=0, column=0, padx=5, pady=5)

        self.btn_load_video = tk.Button(control_frame, text="Load Video", width=20, command=self.app.load_video)
        self.create_tooltip(self.btn_load_video, "Load video file (Ctrl+O)")
        self.btn_load_video.grid(row=0, column=1, padx=5, pady=5)

        self.btn_export_to_json = tk.Button(control_frame, text="Export to JSON", width=20, command=self.app.export_to_json)
        self.create_tooltip(self.btn_export_to_json, "Export landmarks to JSON (Ctrl+E)")
        self.btn_export_to_json.grid(row=0, column=2, padx=5, pady=5)

        self.btn_play_pause = tk.Button(control_frame, text="Pause", width=20, command=self.app.toggle_play_pause, state=tk.DISABLED)
        self.create_tooltip(self.btn_play_pause, "Play/Pause video (Space)")
        self.btn_play_pause.grid(row=0, column=3, padx=5, pady=5)

        # Frame navigation controls
        frame_control_frame = ttk.Frame(control_frame)
        frame_control_frame.grid(row=1, column=0, columnspan=4, pady=5)

        self.btn_prev_frame = tk.Button(frame_control_frame, text="←", width=3, command=self.prev_frame, state=tk.DISABLED)
        self.create_tooltip(self.btn_prev_frame, "Previous frame (Left Arrow)")
        self.btn_prev_frame.pack(side=tk.LEFT, padx=2)

        self.btn_screenshot = tk.Button(frame_control_frame, text="Take Screenshot", width=20, command=self.app.take_screenshot)
        self.btn_screenshot.pack(side=tk.LEFT, padx=5)

        self.btn_next_frame = tk.Button(frame_control_frame, text="→", width=3, command=self.next_frame, state=tk.DISABLED)
        self.create_tooltip(self.btn_next_frame, "Next frame (Right Arrow)")
        self.btn_next_frame.pack(side=tk.LEFT, padx=2)

        options_frame = ttk.LabelFrame(window, text="Options")
        options_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=(5, 10), sticky="ew")

        self.realtime_capture_var = tk.BooleanVar(value=False)
        self.realtime_capture_cb = ttk.Checkbutton(
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
            
            label = ttk.Label(tooltip, text=text, justify=tk.LEFT,
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
        else:
            self.app.stop_realtime_capture()
