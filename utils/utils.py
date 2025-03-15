"""
Utility functions for MedExplain AI Pro.

This module provides common utility functions used throughout the application,
including file handling, string processing, and helper functions.
"""

import os
import json
import logging
import base64
import platform
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image

# Import configuration
from config import (
    DATA_DIR,
    STATIC_DIR,
    CSS_FILE,
    LOGO_FILE,
    MEDICAL_DISCLAIMER
)

# Configure logger
logger = logging.getLogger(__name__)

def load_json_file(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file
        default: Value to return if file doesn't exist or can't be loaded

    Returns:
        Loaded JSON data or default value
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"JSON file not found: {file_path}")
            return default

        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return default

def save_json_file(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save the file to
        indent: Indentation level for JSON formatting

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)

        logger.debug(f"Successfully saved JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False

def load_css() -> None:
    """
    Load and apply custom CSS to the Streamlit application.
    """
    try:
        if CSS_FILE.exists():
            with open(CSS_FILE, 'r') as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                logger.debug(f"Loaded CSS from {CSS_FILE}")
        else:
            # Apply default styles if CSS file doesn't exist
            st.markdown("""
            <style>
                .main-card {
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                }
                .symptom-card {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    padding: 1.2rem;
                    margin-bottom: 1rem;
                    border-left: 4px solid #3498db;
                }
                .severity-high {
                    color: #dc3545;
                    font-weight: bold;
                }
                .severity-medium {
                    color: #fd7e14;
                    font-weight: bold;
                }
                .severity-low {
                    color: #28a745;
                    font-weight: bold;
                }
                .disclaimer {
                    background-color: #f8d7da;
                    border-left: 5px solid #dc3545;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .citation {
                    font-size: 0.8em;
                    color: #6c757d;
                    border-left: 3px solid #3498db;
                    padding-left: 10px;
                    margin: 10px 0;
                }
            </style>
            """, unsafe_allow_html=True)
            logger.warning(f"CSS file not found at {CSS_FILE}. Using default styles.")
    except Exception as e:
        logger.error(f"Error loading CSS: {str(e)}")
        # Continue without custom CSS

def load_image(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Load an image from file.

    Args:
        image_path: Path to the image file

    Returns:
        Loaded PIL Image or None if loading fails
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None

        return Image.open(image_path)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def get_logo_image() -> Optional[str]:
    """
    Get the application logo as a base64 encoded string for HTML embedding.

    Returns:
        Base64 encoded image string or None if loading fails
    """
    try:
        if not LOGO_FILE.exists():
            logger.warning(f"Logo file not found at {LOGO_FILE}")
            return None

        img = Image.open(LOGO_FILE)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error loading logo: {str(e)}")
        return None

def display_disclaimer() -> None:
    """
    Display the medical disclaimer in the Streamlit app.
    """
    st.markdown(f"""
    <div class="disclaimer">
    {MEDICAL_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)

def format_date(date_str: str, input_format: str = "%Y-%m-%d %H:%M:%S",
               output_format: str = "%B %d, %Y at %I:%M %p") -> str:
    """
    Format a date string into a different format.

    Args:
        date_str: Date string to format
        input_format: Format of the input date string
        output_format: Desired output format

    Returns:
        Formatted date string
    """
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except Exception:
        return date_str  # Return original string if conversion fails

def sanitize_string(text: str) -> str:
    """
    Sanitize a string by removing or escaping potentially harmful characters.

    Args:
        text: String to sanitize

    Returns:
        Sanitized string
    """
    if not text:
        return ""

    # Replace HTML special characters
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;"
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text

def create_download_link(data: Any, filename: str, link_text: str, mime: str = "text/plain") -> str:
    """
    Create a download link for data.

    Args:
        data: Data to download
        filename: Name for the downloaded file
        link_text: Text to display for the link
        mime: MIME type of the data

    Returns:
        HTML string with download link
    """
    try:
        if isinstance(data, pd.DataFrame):
            data = data.to_csv(index=False)

        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        logger.error(f"Error creating download link: {str(e)}")
        return "Error creating download link"

def merge_dictionaries(dict1: Dict, dict2: Dict, override: bool = True) -> Dict:
    """
    Merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence if override=True)
        override: Whether to override values from dict1 with values from dict2

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dictionaries(result[key], value, override)
        elif key not in result or override:
            result[key] = value

    return result

def get_system_info() -> Dict[str, str]:
    """
    Get system information.

    Returns:
        Dictionary with system information
    """
    try:
        import streamlit as st_version

        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "streamlit_version": st_version.__version__,
            "app_version": st.session_state.get("app_version", "unknown")
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}

def debounce(wait_time: float) -> Callable:
    """
    Decorator to debounce a function (prevent it from being called too frequently).

    Args:
        wait_time: Time to wait between function calls in seconds

    Returns:
        Debounced function
    """
    def decorator(function):
        last_called = [0.0]

        def debounced(*args, **kwargs):
            current_time = datetime.now().timestamp()
            if current_time - last_called[0] >= wait_time:
                last_called[0] = current_time
                return function(*args, **kwargs)
            return None

        return debounced

    return decorator

def create_directories() -> None:
    """
    Create necessary directories for the application.
    """
    try:
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        STATIC_DIR.mkdir(exist_ok=True, parents=True)
        logger.info("Created necessary directories")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
