import os
import yaml
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self, config_file="config.yaml"):
        """
        Initialize the configuration object by loading default settings and merging 
        with any external configuration from the YAML file.
        """
        self.data = self._get_default_config()
        self.load_config(config_file)
        self._validate_paths()

    def _get_default_config(self):
        """
        Return the default configuration settings.
        """
        return {
            "frame_skip": 5,  # Process every nth frame
            "detection_threshold": 0.5,  # Confidence threshold for face recognition
            "mask_std_threshold": 20,  # Standard deviation threshold for mask detection
            "clip_duration_seconds": 5,  # Length of video clips when a match is found
            "max_reference_images": 4,  # Maximum number of reference images to use
            "paths": {
                "output_folder": "detected_faces",      # Directory for saving detected faces
                "reports": "reports",                     # Directory for saving reports
                "temp": "temp",                           # Directory for temporary files (e.g., video clips)
                "reference_image": "reference_image",     # Directory for storing reference images
                "cctv_footage": "cctv_footage"              # Directory for storing CCTV footage
            },
            "ui_settings": {
                "theme": "dark",                          # UI theme (dark/light)
                "accent_color": "#00ff88",                # Highlight color for UI elements
                "font": "Arial"                           # Font used in the UI
            }
        }

    def load_config(self, config_file):
        """
        Load external configuration from the YAML file if it exists and merge it 
        with the default configuration.
        """
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    external_config = yaml.safe_load(f) or {}
                self._deep_merge(self.data, external_config)
                logging.info(f"Loaded external configuration from {config_file}.")
            except Exception as e:
                logging.error(f"Error loading config from {config_file}: {e}")

    def _deep_merge(self, base, update):
        """
        Recursively merge the update dictionary into the base dictionary.
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _validate_paths(self):
        """
        Ensure that required directories exist.
        For the folders 'cctv_footage', 'reference_image', and 'reports', auto-creation 
        is skipped unless the path is inside a task folder (i.e. the absolute path contains 'tasks').
        """
        paths = self.data.get("paths", {})
        for key, path in paths.items():
            if key in ["cctv_footage", "reference_image", "reports"]:
                # Auto-create only if the folder is located inside a 'tasks' directory
                if "tasks" in os.path.abspath(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                        logging.info(f"Created or validated path: {path}")
                    except Exception as e:
                        logging.error(f"Error creating directory {path}: {e}")
                else:
                    logging.info(f"Skipped auto-creation for '{key}' outside tasks: {path}")
            else:
                try:
                    os.makedirs(path, exist_ok=True)
                    logging.info(f"Created or validated path: {path}")
                except Exception as e:
                    logging.error(f"Error creating directory {path}: {e}")

    def update_paths(self, output_folder=None, reports=None, temp=None, reference_image=None, cctv_footage=None):
        """
        Update the folder paths in the configuration. Only provided arguments are updated.
        
        Args:
            output_folder (str): New folder for saving detected faces.
            reports (str): New folder for saving reports.
            temp (str): New folder for temporary files.
            reference_image (str): New folder for reference images.
            cctv_footage (str): New folder for CCTV footage.
        """
        if output_folder:
            self.data["paths"]["output_folder"] = output_folder
        if reports:
            self.data["paths"]["reports"] = reports
        if temp:
            self.data["paths"]["temp"] = temp
        if reference_image:
            self.data["paths"]["reference_image"] = reference_image
        if cctv_footage:
            self.data["paths"]["cctv_footage"] = cctv_footage
        self._validate_paths()

    def get(self, key, default=None):
        """
        Retrieve a configuration value using dot notation (e.g., 'paths.temp').
        If the key is not found, return the default value.
        """
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

if __name__ == "__main__":
    # Example usage when run directly
    config = Config()
    logging.info(f"Frame skip: {config.get('frame_skip')}")
    logging.info(f"Temporary folder: {config.get('paths.temp')}")
