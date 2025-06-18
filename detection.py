import cv2
import face_recognition
import numpy as np
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from config import Config

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedDetectionEngine:
    def __init__(self, config, progress_callback=None, video_frame_callback=None):
        """
        Initialize the EnhancedDetectionEngine.
        
        Args:
            config (Config): Configuration object containing settings.
            progress_callback (callable, optional): Callback to report processing progress.
            video_frame_callback (callable, optional): Callback to pass processed frames for GUI preview.
        """
        self.config = config
        self.progress_callback = progress_callback
        self.video_frame_callback = video_frame_callback
        self.frame_skip = config.get("frame_skip", 5)
        self.confidence_threshold = config.get("detection_threshold", 0.5)
        self.mask_std_threshold = config.get("mask_std_threshold", 20)
        self.clip_duration = config.get("clip_duration_seconds", 5)
        self.reference_encodings = []
        self.matches = []
        self.report_data = []
        self.detection_path = []
        self.pause_event = threading.Event()  # For pause/resume control
        self.stop_event = threading.Event()   # For stopping processing
        self.current_clip_writer = None
        self.clip_frames_remaining = 0

    def load_reference_images(self, image_paths):
        """
        Load the provided reference images (one by one) and compute their face encodings.
        
        Args:
            image_paths (List[str]): List of file paths to reference images.
        
        Returns:
            bool: True if at least one encoding is loaded; False otherwise.
        """
        self.reference_encodings = []
        max_images = self.config.get("max_reference_images")
        if max_images is not None and len(image_paths) > max_images:
            logging.warning(f"More than {max_images} reference images provided; only the first {max_images} will be used.")
            image_paths = image_paths[:max_images]
        for path in image_paths:
            if not os.path.exists(path):
                logging.error(f"Reference image not found: {path}")
                continue
            try:
                ref_image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(ref_image)
                if encodings:
                    self.reference_encodings.append(encodings[0])
                else:
                    logging.error(f"No face found in the reference image: {path}")
            except Exception as e:
                logging.error(f"Error loading reference image {path}: {e}")
        if not self.reference_encodings:
            logging.error("No valid reference encodings loaded.")
            return False
        logging.info(f"Loaded {len(self.reference_encodings)} reference image(s) successfully.")
        return True

    def _detect_mask(self, face_image):
        """
        Heuristic for mask detection:
          - Convert the face region to grayscale.
          - Compute its standard deviation; a lower std-dev suggests a mask.
        
        Args:
            face_image (np.ndarray): The image region containing the face.
        
        Returns:
            bool: True if a mask is detected, False otherwise.
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        std = np.std(gray)
        return std < self.mask_std_threshold

    def _start_video_clip(self, frame_shape, fps):
        """
        Start recording a video clip for a detection event.
        
        Args:
            frame_shape (Tuple[int, int, int]): The shape of the current video frame.
            fps (float): The frames per second of the video.
        """
        temp_folder = self.config.get("paths.temp")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        clip_filename = f"clip_{timestamp}.avi"
        clip_path = os.path.join(temp_folder, clip_filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        height, width, _ = frame_shape
        self.current_clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        self.clip_frames_remaining = int(fps * self.clip_duration)
        logging.info(f"Started recording clip: {clip_path} for {self.clip_frames_remaining} frames.")

    def process_video(self, video_path):
        """
        Process a video file for face and mask detection.
        Features include:
          - Frame skipping for efficiency.
          - Pause/Resume functionality.
          - Automatic clip recording upon detection events.
          - In-window preview updates (via callback).
        
        Args:
            video_path (str): Path to the video file.
        
        Returns:
            int: Total number of detections found.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Could not open video file.")
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        current_frame = 0
        match_count = 0

        logging.info(f"Processing video: {video_path}")
        logging.info(f"Total frames: {total_frames}, FPS: {fps}")

        while not self.stop_event.is_set() and current_frame < total_frames:
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # Write frame to clip if recording is active.
            if self.current_clip_writer is not None:
                self.current_clip_writer.write(frame)
                self.clip_frames_remaining -= 1
                if self.clip_frames_remaining <= 0:
                    self.current_clip_writer.release()
                    self.current_clip_writer = None

            # Process every nth frame.
            if current_frame % self.frame_skip != 0:
                current_frame += 1
                continue

            # Detect faces in the current frame.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                match_found = False
                best_confidence = 0
                for ref_encoding in self.reference_encodings:
                    matches = face_recognition.compare_faces([ref_encoding], face_encoding)
                    confidence = face_recognition.face_distance([ref_encoding], face_encoding)[0]
                    if matches[0] and confidence <= self.confidence_threshold:
                        match_found = True
                        best_confidence = max(best_confidence, 1 - confidence)
                if match_found:
                    top, right, bottom, left = face_location
                    face_region = frame[top:bottom, left:right]
                    has_mask = self._detect_mask(face_region)
                    box_color = (0, 255, 0) if has_mask else (0, 0, 255)
                    label_text = f"Mask ({best_confidence*100:.1f}%)" if has_mask else f"No Mask ({best_confidence*100:.1f}%)"
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                    cv2.putText(frame, label_text, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    center_x = int((left + right) / 2)
                    center_y = int((top + bottom) / 2)
                    self.detection_path.append((center_x, center_y))
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    output_folder = self.config.get("paths.output_folder")
                    image_filename = f"match_{timestamp}_{current_frame}.jpg"
                    output_path = os.path.join(output_folder, image_filename)
                    cv2.imwrite(output_path, face_region)
                    detection_info = {
                        'frame': current_frame,
                        'timestamp': str(timedelta(seconds=current_frame / fps)),
                        'image_path': output_path,
                        'confidence': best_confidence,
                        'mask': has_mask
                    }
                    self.matches.append(detection_info)
                    self.report_data.append(detection_info)
                    match_count += 1
                    # Start recording a clip if not already active.
                    if self.current_clip_writer is None:
                        self._start_video_clip(frame.shape, fps)
            if len(self.detection_path) > 1:
                cv2.polylines(frame, [np.array(self.detection_path, dtype=np.int32)], False, (255, 0, 0), 2)
            if self.progress_callback:
                progress = current_frame / total_frames
                self.progress_callback(progress)
            if self.video_frame_callback:
                self.video_frame_callback(frame)
            current_frame += 1

        cap.release()
        self.generate_report()
        logging.info(f"Total matches found: {match_count}")
        return match_count

    def pause(self):
        """Pause video processing."""
        self.pause_event.set()
        logging.info("Processing paused.")

    def resume(self):
        """Resume video processing."""
        self.pause_event.clear()
        logging.info("Processing resumed.")

    def stop(self):
        """Stop video processing."""
        self.stop_event.set()
        logging.info("Processing stopped by user.")

    def generate_report(self):
        """
        Generate a text report of all detections and save it in the reports folder.
        """
        reports_folder = self.config.get("paths.reports")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(reports_folder, f"detection_report_{timestamp}.txt")
        try:
            with open(report_path, "w") as f:
                f.write("Enhanced Face Recognition Detection Report\n")
                f.write("=" * 50 + "\n")
                for idx, data in enumerate(self.report_data, start=1):
                    f.write(f"Detection #{idx}\n")
                    f.write(f"Frame: {data['frame']}\n")
                    f.write(f"Timestamp: {data['timestamp']}\n")
                    f.write(f"Confidence: {data['confidence']*100:.2f}%\n")
                    f.write(f"Mask: {'Yes' if data['mask'] else 'No'}\n")
                    f.write(f"Image Path: {data['image_path']}\n")
                    f.write("-" * 50 + "\n")
            logging.info(f"Detection report generated: {report_path}")
        except Exception as e:
            logging.error(f"Error generating report: {e}")

if __name__ == "__main__":
    # Example usage when running this module directly:
    config = Config()
    engine = EnhancedDetectionEngine(config)
    # Example list of reference images – these can be provided one by one
    reference_images = [
        "ref1.jpg",
        "ref2.jpg",
        "ref3.jpg",
        "ref4.jpg"
    ]
    if engine.load_reference_images(reference_images):
        logging.info("Reference images loaded successfully.")
    else:
        logging.error("Failed to load any reference images.")
    # Uncomment and update the following lines to process a video:
    # video_path = "path_to_video.mp4"
    # engine.process_video(video_path)
