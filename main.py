import os
import logging
import threading
import shutil
import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from config import Config
from detection import EnhancedDetectionEngine
from database import DetectionDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A decorator to wrap UI callback methods with error handling.
def safe_run(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logging.exception(f"Exception in {func.__name__}: {e}")
            self.display_status(f"An error occurred in {func.__name__}.", error=True)
    return wrapper

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Foresight AI - Missing Person Detection")
        self.geometry("1000x750")
        
        # Initialize configuration and state variables
        self.config_obj = Config()
        self.video_path = ""
        self.reference_image_paths = []
        self.task_folder = None  # Current task folder
        self.detector = None      # Instance of EnhancedDetectionEngine

        # Navigation management: a stack of (function, args, kwargs)
        self.screen_stack = []
        self.current_screen_info = (self.setup_main_ui, (), {})

        # Top frame with Back button (always visible, fixed at top left)
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.pack(side="top", fill="x")
        self.back_button = ctk.CTkButton(self.top_frame, text="Back", command=self.go_back)
        self.back_button.pack(side="left", anchor="nw", padx=10, pady=10)
        self.update_back_button_visibility()

        # Main container for dynamic content (placed below top frame)
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True)
        
        # Load the main UI screen
        self.setup_main_ui()
    
    def update_back_button_visibility(self):
        """Enable or disable the back button based on navigation stack."""
        if self.screen_stack:
            self.back_button.configure(state="normal")
        else:
            self.back_button.configure(state="disabled")
    
    def navigate_to(self, func, *args, **kwargs):
        """Push the current screen and navigate to a new one."""
        self.screen_stack.append(self.current_screen_info)
        self.current_screen_info = (func, args, kwargs)
        self.clear_main_container()
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Navigation error in {func.__name__}: {e}")
            self.display_status(f"Navigation error in {func.__name__}.", error=True)
        self.update_back_button_visibility()
    
    def go_back(self):
        """Go back to the previous screen if available."""
        if self.screen_stack:
            self.clear_main_container()
            self.current_screen_info = self.screen_stack.pop()
            func, args, kwargs = self.current_screen_info
            try:
                func(*args, **kwargs)
            except Exception as e:
                logging.exception(f"Navigation error in {func.__name__}: {e}")
                self.display_status(f"Navigation error in {func.__name__}.", error=True)
            self.update_back_button_visibility()
    
    def clear_main_container(self):
        """Remove all widgets from the main container."""
        for widget in self.main_container.winfo_children():
            widget.destroy()
    
    def display_status(self, message, error=False):
        """Display a temporary status message at the bottom of the main container."""
        status_label = ctk.CTkLabel(self.main_container, text=message, text_color="red" if error else "green")
        status_label.pack(side="bottom", pady=5)
        self.after(3000, status_label.destroy)
    
    @safe_run
    def refresh_ui(self):
        """Reset the UI to its initial state."""
        self.task_name_var.set("")
        self.video_path_var.set("")
        self.reference_image_paths = []
        self.ref_status_label.configure(text="No reference images added")
        for widget in self.ref_thumbnail_frame.winfo_children():
            widget.destroy()
        self.preview_canvas.configure(image=None)
        self.preview_canvas.image = None
        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")
        self.task_folder = None
        # Reset navigation stack
        self.screen_stack = []
        self.current_screen_info = (self.setup_main_ui, (), {})
        self.display_status("UI refreshed.")
        self.setup_main_ui()
    
    # ---------------------------
    # Main UI Screen (Home Screen)
    # ---------------------------
    @safe_run
    def setup_main_ui(self):
        """Set up the main UI screen with task creation, file selection, and control buttons."""
        self.clear_main_container()
        
        ctk.set_appearance_mode(self.config_obj.get("ui_settings.theme"))
        ctk.set_default_color_theme("dark-blue")
        scroll_frame = ctk.CTkScrollableFrame(self.main_container, width=980, height=720)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        header = ctk.CTkLabel(
            scroll_frame,
            text="Foresight AI",
            font=(self.config_obj.get("ui_settings.font"), 24, "bold"),
            text_color=self.config_obj.get("ui_settings.accent_color")
        )
        header.pack(pady=10)
        
        # Task creation frame
        task_frame = ctk.CTkFrame(scroll_frame)
        task_frame.pack(pady=10, padx=10, fill="x")
        task_label = ctk.CTkLabel(task_frame, text="Task Name:")
        task_label.grid(row=0, column=0, padx=10, pady=5)
        self.task_name_var = ctk.StringVar()
        task_entry = ctk.CTkEntry(task_frame, textvariable=self.task_name_var, width=200)
        task_entry.grid(row=0, column=1, padx=10, pady=5)
        create_task_btn = ctk.CTkButton(task_frame, text="Create New Task", command=self.create_new_task)
        create_task_btn.grid(row=0, column=2, padx=10, pady=5)
        
        # Progress frame
        progress_frame = ctk.CTkFrame(scroll_frame)
        progress_frame.pack(pady=5, padx=10, fill="x")
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, padx=10)
        self.progress_label = ctk.CTkLabel(progress_frame, text="0%")
        self.progress_label.grid(row=0, column=1, padx=10)
        
        # File selection for video
        file_frame = ctk.CTkFrame(scroll_frame)
        file_frame.pack(pady=10, padx=10, fill="x")
        self.video_path_var = ctk.StringVar(value="")
        video_btn = ctk.CTkButton(file_frame, text="Select Video", command=self.select_video)
        video_btn.grid(row=0, column=0, padx=10, pady=5)
        video_label = ctk.CTkLabel(file_frame, textvariable=self.video_path_var, width=400)
        video_label.grid(row=0, column=1, padx=10, pady=5)
        
        # Reference image input
        ref_frame = ctk.CTkFrame(scroll_frame)
        ref_frame.pack(pady=10, padx=10, fill="x")
        add_ref_btn = ctk.CTkButton(ref_frame, text="Add Reference Image", command=self.add_reference_image)
        add_ref_btn.grid(row=0, column=0, padx=10, pady=5)
        self.ref_status_label = ctk.CTkLabel(ref_frame, text="No reference images added", width=400)
        self.ref_status_label.grid(row=0, column=1, padx=10, pady=5)
        
        # Thumbnails for reference images
        self.ref_thumbnail_frame = ctk.CTkFrame(scroll_frame)
        self.ref_thumbnail_frame.pack(pady=5, padx=10, fill="x")
        
        # Video preview area
        preview_frame = ctk.CTkFrame(scroll_frame)
        preview_frame.pack(pady=10)
        preview_label = ctk.CTkLabel(preview_frame, text="Video Preview")
        preview_label.pack()
        self.preview_canvas = ctk.CTkLabel(preview_frame)
        self.preview_canvas.pack()
        
        # Control buttons
        control_frame = ctk.CTkFrame(scroll_frame)
        control_frame.pack(pady=10)
        self.start_button = ctk.CTkButton(control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=10)
        self.pause_button = ctk.CTkButton(control_frame, text="Pause", command=self.toggle_pause, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=10)
        self.stop_button = ctk.CTkButton(control_frame, text="Stop", command=self.stop_detection, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=10)
        self.results_button = ctk.CTkButton(control_frame, text="Results", command=self.open_results, state="disabled")
        self.results_button.grid(row=0, column=3, padx=10)
        self.tasks_button = ctk.CTkButton(control_frame, text="Tasks", command=self.open_tasks_screen)
        self.tasks_button.grid(row=0, column=4, padx=10)
        self.refresh_button = ctk.CTkButton(control_frame, text="Refresh", command=self.refresh_ui)
        self.refresh_button.grid(row=0, column=5, padx=10)
        
        # Set current screen info to main UI
        self.current_screen_info = (self.setup_main_ui, (), {})
        self.update_back_button_visibility()
    
    # ---------------------------
    # Task & File Operations
    # ---------------------------
    @safe_run
    def create_new_task(self):
        """Create a new task folder with required subfolders."""
        task_name = self.task_name_var.get().strip()
        if not task_name:
            self.display_status("Error: Please enter a task name.", error=True)
            return
        base_folder = os.path.join(os.getcwd(), "tasks", task_name)
        subfolders = ["detected_faces", "reports", "temp", "reference_image", "cctv_footage"]
        try:
            for sub in subfolders:
                os.makedirs(os.path.join(base_folder, sub), exist_ok=True)
            self.task_folder = base_folder
            self.display_status(f"Task '{task_name}' created at: {base_folder}")
            logging.info(f"Created new task folder structure at {base_folder}")
        except Exception as e:
            logging.exception(f"Error creating task folder: {e}")
            self.display_status(f"Error creating task folder: {e}", error=True)
    
    @safe_run
    def add_reference_image(self):
        """Add a reference image (max 4) and update thumbnails."""
        if len(self.reference_image_paths) >= 4:
            self.display_status("Error: Maximum of 4 reference images allowed.", error=True)
            return
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if path:
            self.reference_image_paths.append(path)
            self.ref_status_label.configure(text=f"{len(self.reference_image_paths)} reference image(s) added")
            self.display_reference_thumbnails()
    
    @safe_run
    def display_reference_thumbnails(self):
        """Display thumbnails of added reference images."""
        for widget in self.ref_thumbnail_frame.winfo_children():
            widget.destroy()
        for idx, path in enumerate(self.reference_image_paths):
            try:
                img = Image.open(path)
                img.thumbnail((100, 100))
                photo = ImageTk.PhotoImage(img)
                lbl = ctk.CTkLabel(self.ref_thumbnail_frame, image=photo, text="")
                lbl.image = photo  # Keep a reference
                lbl.grid(row=idx // 4, column=idx % 4, padx=5, pady=5)
            except Exception as e:
                logging.exception(f"Error loading thumbnail for {path}: {e}")
    
    @safe_run
    def select_video(self):
        """Select a video file and update the display."""
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if path:
            self.video_path_var.set(path)
    
    def update_progress(self, value):
        """Update the progress bar and label."""
        try:
            self.progress_bar.set(value)
            self.progress_label.configure(text=f"{value*100:.1f}%")
        except Exception as e:
            logging.exception(f"Error updating progress: {e}")
    
    @safe_run
    def update_video_preview(self, frame):
        """Update the video preview with the given frame."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            image = image.resize((640, 480))
            self.photo = ImageTk.PhotoImage(image)
            self.preview_canvas.configure(image=self.photo)
            self.preview_canvas.image = self.photo
        except Exception as e:
            logging.exception(f"Error updating video preview: {e}")
    
    @safe_run
    def start_detection(self):
        """Validate inputs, copy files to task folders, and start video processing."""
        video_path = self.video_path_var.get()
        if not os.path.exists(video_path):
            self.display_status("Error: Please select a valid video file.", error=True)
            return
        if not self.reference_image_paths:
            self.display_status("Error: Please add at least one reference image.", error=True)
            return
        if not self.task_folder:
            self.display_status("Error: Please create a new task first.", error=True)
            return
        
        # Update config paths to task-specific folders
        self.config_obj.data["paths"]["output_folder"] = os.path.join(self.task_folder, "detected_faces")
        self.config_obj.data["paths"]["reports"] = os.path.join(self.task_folder, "reports")
        self.config_obj.data["paths"]["temp"] = os.path.join(self.task_folder, "temp")
        
        # Copy reference images into task folder
        reference_dest_folder = os.path.join(self.task_folder, "reference_image")
        os.makedirs(reference_dest_folder, exist_ok=True)
        for image_path in self.reference_image_paths:
            try:
                shutil.copy(image_path, reference_dest_folder)
            except Exception as e:
                logging.exception(f"Error copying reference image {image_path}: {e}")
        
        # Copy selected video into task's cctv_footage folder
        cctv_dest_folder = os.path.join(self.task_folder, "cctv_footage")
        os.makedirs(cctv_dest_folder, exist_ok=True)
        try:
            shutil.copy(video_path, cctv_dest_folder)
        except Exception as e:
            logging.exception(f"Error copying video file {video_path}: {e}")
        
        # Initialize and start the detection engine
        self.detector = EnhancedDetectionEngine(
            self.config_obj,
            progress_callback=self.update_progress,
            video_frame_callback=self.update_video_preview
        )
        if not self.detector.load_reference_images(self.reference_image_paths):
            self.display_status("Error: Failed to load reference images.", error=True)
            return
        
        self.pause_button.configure(state="normal", text="Pause")
        self.stop_button.configure(state="normal")
        self.start_button.configure(state="disabled")
        self.results_button.configure(state="disabled")
        
        detection_thread = threading.Thread(target=self.run_detection, args=(video_path,))
        detection_thread.start()
    
    def run_detection(self, video_path):
        """Run video processing and update the UI when done."""
        try:
            count = self.detector.process_video(video_path)
            if count == 0:
                self.display_status("Detection complete: Person not found.")
            else:
                self.display_status(f"Detection complete: Total Matches: {count}")
        except Exception as e:
            logging.exception(f"Error during detection: {e}")
            self.display_status("Error during detection process.", error=True)
        finally:
            self.results_button.configure(state="normal")
            self.start_button.configure(state="normal")
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
    
    @safe_run
    def toggle_pause(self):
        """Toggle pause/resume on video processing."""
        if self.pause_button.cget("text") == "Pause":
            self.detector.pause()
            self.pause_button.configure(text="Resume")
        else:
            self.detector.resume()
            self.pause_button.configure(text="Pause")
    
    @safe_run
    def stop_detection(self):
        """Stop video processing."""
        try:
            self.detector.stop()
        except Exception as e:
            logging.exception(f"Error stopping detection: {e}")
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
    
    # ---------------------------
    # Dynamic Content Screens (Navigation-enabled)
    # ---------------------------
    @safe_run
    def open_results(self):
        """Navigate to the Results screen that lists all content types."""
        self.navigate_to(self._open_results)
    
    @safe_run
    def _open_results(self):
        """Display the Results screen with all original feature buttons."""
        dynamic_frame = ctk.CTkScrollableFrame(self.main_container, width=980, height=720)
        dynamic_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        header = ctk.CTkLabel(
            dynamic_frame,
            text="Results",
            font=(self.config_obj.get("ui_settings.font"), 24, "bold"),
            text_color=self.config_obj.get("ui_settings.accent_color")
        )
        header.pack(pady=10)
        
        # Restored buttons for all features:
        btn_cctv = ctk.CTkButton(dynamic_frame, text="CCTV Footage", 
                                  command=lambda: self.display_folder_contents("cctv_footage"))
        btn_cctv.pack(pady=5, fill="x", padx=20)
        btn_detected = ctk.CTkButton(dynamic_frame, text="Detected Faces", 
                                     command=lambda: self.display_folder_contents("detected_faces"))
        btn_detected.pack(pady=5, fill="x", padx=20)
        btn_reference = ctk.CTkButton(dynamic_frame, text="Reference Image", 
                                      command=lambda: self.display_folder_contents("reference_image"))
        btn_reference.pack(pady=5, fill="x", padx=20)
        btn_report = ctk.CTkButton(dynamic_frame, text="Report", 
                                   command=lambda: self.display_folder_contents("reports"))
        btn_report.pack(pady=5, fill="x", padx=20)
        btn_shot = ctk.CTkButton(dynamic_frame, text="Shot Videos", 
                                 command=lambda: self.display_folder_contents("temp"))
        btn_shot.pack(pady=5, fill="x", padx=20)
    
    @safe_run
    def open_tasks_screen(self):
        """Navigate to the Tasks screen listing all task folders."""
        self.navigate_to(self._open_tasks_screen)
    
    @safe_run
    def _open_tasks_screen(self):
        tasks_folder = os.path.join(os.getcwd(), "tasks")
        dynamic_frame = ctk.CTkScrollableFrame(self.main_container, width=980, height=720)
        dynamic_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        header = ctk.CTkLabel(
            dynamic_frame,
            text="Tasks",
            font=(self.config_obj.get("ui_settings.font"), 24, "bold"),
            text_color=self.config_obj.get("ui_settings.accent_color")
        )
        header.pack(pady=10)
        
        if not os.path.exists(tasks_folder) or not os.listdir(tasks_folder):
            no_task_label = ctk.CTkLabel(dynamic_frame, text="No tasks available.")
            no_task_label.pack(pady=10)
        else:
            for task in os.listdir(tasks_folder):
                task_path = os.path.join(tasks_folder, task)
                task_frame = ctk.CTkFrame(dynamic_frame)
                task_frame.pack(pady=5, padx=10, fill="x")
                
                task_label = ctk.CTkLabel(task_frame, text=task, width=200)
                task_label.pack(side="left", padx=5)
                
                open_btn = ctk.CTkButton(task_frame, text="Open", 
                                         command=lambda tp=task_path: self.open_task_contents(tp))
                open_btn.pack(side="left", padx=5)
                
                del_btn = ctk.CTkButton(task_frame, text="Delete", 
                                        command=lambda tp=task_path: self.delete_task(tp))
                del_btn.pack(side="left", padx=5)
    
    @safe_run
    def open_task_contents(self, task_path):
        """Set the selected task folder and navigate to its results screen."""
        self.task_folder = task_path
        self.navigate_to(self._open_results)
    
    @safe_run
    def delete_task(self, task_path):
        """Delete a task folder and refresh the Tasks screen."""
        try:
            shutil.rmtree(task_path)
            self.display_status("Task deleted successfully.")
        except Exception as e:
            logging.exception(f"Error deleting task {task_path}: {e}")
            self.display_status(f"Error deleting task: {e}", error=True)
        self._open_tasks_screen()
    
    @safe_run
    def display_folder_contents(self, folder_key):
        """Navigate to a screen displaying the contents of a specified folder within the current task."""
        self.navigate_to(self._display_folder_contents, folder_key)
    
    @safe_run
    def _display_folder_contents(self, folder_key):
        """Display folder contents arranged in a grid with 2 columns."""
        if not self.task_folder:
            self.display_status("Error: No task folder set.", error=True)
            return
        
        folder_path = os.path.join(self.task_folder, folder_key)
        dynamic_frame = ctk.CTkScrollableFrame(self.main_container, width=980, height=720)
        dynamic_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        header = ctk.CTkLabel(
            dynamic_frame,
            text=f"Contents of {folder_key}",
            font=(self.config_obj.get("ui_settings.font"), 20, "bold"),
            text_color=self.config_obj.get("ui_settings.accent_color")
        )
        header.pack(pady=10)
        
        if not os.path.exists(folder_path):
            no_label = ctk.CTkLabel(dynamic_frame, text="Folder does not exist.")
            no_label.pack(pady=10)
        else:
            try:
                files = os.listdir(folder_path)
            except Exception as e:
                logging.exception(f"Error listing files in {folder_path}: {e}")
                self.display_status(f"Error accessing folder: {e}", error=True)
                return
            if not files:
                no_label = ctk.CTkLabel(dynamic_frame, text="No files found.")
                no_label.pack(pady=10)
            else:
                # Create a container for grid layout (2 columns)
                grid_container = ctk.CTkFrame(dynamic_frame)
                grid_container.pack(fill="both", expand=True)
                for idx, file in enumerate(files):
                    file_path = os.path.join(folder_path, file)
                    file_frame = ctk.CTkFrame(grid_container, width=400, height=150)
                    row = idx // 2
                    col = idx % 2
                    file_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
                    
                    ext = os.path.splitext(file)[1].lower()
                    if ext in [".jpg", ".jpeg", ".png", ".gif"]:
                        try:
                            img = Image.open(file_path)
                            img.thumbnail((100, 100))
                            photo = ImageTk.PhotoImage(img)
                        except Exception as e:
                            logging.exception(f"Error loading image {file_path}: {e}")
                            photo = None
                    elif ext in [".mp4", ".avi", ".mov"]:
                        img = Image.new("RGB", (100, 100), color="gray")
                        photo = ImageTk.PhotoImage(img)
                    else:
                        photo = None
                    
                    if photo:
                        img_label = ctk.CTkLabel(file_frame, image=photo, text="")
                        img_label.image = photo
                        img_label.pack(side="top", padx=5, pady=5)
                    
                    name_label = ctk.CTkLabel(file_frame, text=file, width=200)
                    name_label.pack(side="top", padx=5, pady=5)
                    
                    btn_frame = ctk.CTkFrame(file_frame)
                    btn_frame.pack(side="top", pady=5)
                    open_btn = ctk.CTkButton(btn_frame, text="Open", command=lambda fp=file_path: os.startfile(fp))
                    open_btn.pack(side="left", padx=5)
                    del_btn = ctk.CTkButton(btn_frame, text="Delete", command=lambda fp=file_path, fk=folder_key: self.delete_file(fp, fk))
                    del_btn.pack(side="left", padx=5)
    
    @safe_run
    def delete_file(self, file_path, folder_key):
        """Delete the specified file and refresh the current folder view."""
        try:
            os.remove(file_path)
            self.display_status(f"{os.path.basename(file_path)} deleted.")
        except Exception as e:
            logging.exception(f"Error deleting file {file_path}: {e}")
            self.display_status(f"Error deleting file: {e}", error=True)
        self.display_folder_contents(folder_key)

if __name__ == "__main__":
    try:
        app = FaceRecognitionApp()
        app.mainloop()
    except Exception as e:
        logging.exception(f"Fatal error in application: {e}")
