import datetime
import logging
import os

from PIL import ImageGrab
import tkinter as tk

def take_screenshot(app):
    """Capture a screenshot of the app's canvas and save it to the screenshots directory."""
    canvas_content = app.ui.canvas.find_all()
    if not canvas_content:
        logging.warning("Can't take screenshot when canvas is empty.")
        tk.messagebox.showwarning("Warning", "Can't take screenshot when canvas is empty.")
        return

    current_time = int(datetime.datetime.now().timestamp() * 1000)
    if current_time - app.last_screenshot_time < app.throttle_delay:
        return
    app.last_screenshot_time = current_time

    try:
        x = app.window.winfo_rootx() + app.ui.canvas.winfo_x()
        y = app.window.winfo_rooty() + app.ui.canvas.winfo_y()
        x1 = x + app.ui.canvas.winfo_width()
        y1 = y + app.ui.canvas.winfo_height()

        screenshot = ImageGrab.grab().crop((x, y, x1, y1))

        now = datetime.datetime.now()
        filename = f"screenshot_{now.strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(app.screenshots_dir, filename)
        screenshot.save(filepath)
        logging.info(f"Screenshot saved to {filepath}")
    except Exception as e:
        logging.error(f"Error taking screenshot: {e}")
