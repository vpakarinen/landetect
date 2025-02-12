import tkinter as tk

class UI:
    def __init__(self, window, app):
        self.window = window
        self.app = app
        self.window_width = 800
        self.window_height = 600
        self.canvas_width = self.window_width - 20
        self.canvas_height = self.window_height - 150

        # Canvas for displaying image or video
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Horizontal line above buttons
        separator = tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN)
        separator.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        self.btn_load_image = tk.Button(window, text="Load Image", width=20, command=self.app.load_image)
        self.btn_load_image.grid(row=2, column=0, padx=10, pady=10)

        self.btn_load_video = tk.Button(window, text="Load Video", width=20, command=self.app.load_video)
        self.btn_load_video.grid(row=2, column=1, padx=10, pady=10)

        self.btn_export_to_json = tk.Button(window, text="Export to JSON", width=20, command=self.app.export_to_json)
        self.btn_export_to_json.grid(row=2, column=2, padx=10, pady=10)

        self.btn_screenshot = tk.Button(window, text="Take Screenshot", width=20, command=self.app.take_screenshot)
        self.btn_screenshot.grid(row=3, column=1, padx=10, pady=10)

        # Adjust row configure to reduce padding at the bottom
        window.grid_rowconfigure(3, weight=1)
