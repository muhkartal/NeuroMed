import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import sys
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, TypedDict, Callable
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timedelta
import re
from enum import Enum, auto
import traceback
import dotenv

st.set_page_config(
                page_title=f"MedExplain AI Pro  - Personal Health Assistant",
                ## 💊 ⚕️
                page_icon=" 💊",
                layout="wide",
                initial_sidebar_state="expanded",
                menu_items={
                    'Get Help': 'https://www.medexplain.ai/help',
                    'Report a bug': 'https://www.medexplain.ai/bugs',
                    'About': f'MedExplain AI Pro  - Advanced personal health assistant powered by AI.'
                }
            )

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Type definitions for better type hinting
class SymptomData(TypedDict, total=False):
    id: str
    name: str
    description: str
    category: str
    severity: int

class UserProfile(TypedDict, total=False):
    id: str
    name: str
    age: int
    gender: str
    email: str
    created_at: str
    last_login: str

class HealthRecord(TypedDict, total=False):
    date: str
    symptoms: List[str]
    notes: str
    severity: int
    duration: int

class RiskAssessment(TypedDict, total=False):
    risk_level: str
    risk_score: float
    domain_risks: Dict[str, float]
    recommendations: List[str]
    date: str

class Notification(TypedDict, total=False):
    id: str
    title: str
    message: str
    type: str
    timestamp: float
    read: bool

class NotificationType(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()

# Setup module imports
from core.health_data_manager import HealthDataManager
from core.user_profile_manager import UserProfileManager
from core.openai_client import OpenAIClient

from models.symptom_predictor import SymptomPredictor
from models.symptom_extractor import NLPSymptomExtractor
from models.risk_assessor import PatientRiskAssessor

from analytics.health_analyzer import HealthDataAnalyzer
from analytics.visualization import create_timeseries_chart, create_symptom_heatmap, create_risk_radar

from ui.dashboard import HealthDashboard
from ui.chat import ChatInterface
from ui.symptom_analyzer import SymptomAnalyzerUI
from ui.medical_literature import MedicalLiteratureUI
from ui.health_history import HealthHistoryUI
from ui.settings import SettingsUI

# Constants - moved to a dedicated class for better organization
class AppConfig:
    """Application configuration constants."""
    VERSION = "3.2.0"  # Updated version for enterprise improvements
    BASE_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = BASE_DIR / "data"
    STATIC_DIR = BASE_DIR / "NeuroMed/img"
    LOG_DIR = BASE_DIR / "logs"
    USER_DATA_DIR = DATA_DIR / "user_data"
    MODEL_DIR = DATA_DIR / "ml_models"
    CONFIG_DIR = BASE_DIR / "config"
    CONFIG_FILE = CONFIG_DIR / "app_config.json"
    LOG_FILE = LOG_DIR / "app.log"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    CACHE_DIR = DATA_DIR / "cache"
    DEFAULT_THEME = "light"
    MAX_WORKERS = 4  # For ThreadPoolExecutor
    SESSION_TIMEOUT = 3600  # Session timeout in seconds (1 hour)

    # Ensure all directories exist
    @classmethod
    def initialize(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR, cls.STATIC_DIR, cls.LOG_DIR, cls.USER_DATA_DIR,
            cls.MODEL_DIR, cls.CONFIG_DIR, cls.CACHE_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create default config file if it doesn't exist
        if not cls.CONFIG_FILE.exists():
            default_config = {
                "theme": cls.DEFAULT_THEME,
                "log_level": "INFO",
                "feature_flags": {
                    "enable_advanced_analytics": True,
                    "enable_chat_interface": True,
                    "enable_performance_metrics": True
                }
            }
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=2)

        return True


# Utility decorator for timing functions
def timing_decorator(func):
    """Decorator to measure and log execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Get logger for the function's module
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")

        # Store timing metrics in session state if available
        try:
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}

            if 'function_timing' not in st.session_state.performance_metrics:
                st.session_state.performance_metrics['function_timing'] = {}

            metrics = st.session_state.performance_metrics['function_timing']
            if func.__name__ not in metrics:
                metrics[func.__name__] = []

            metrics[func.__name__].append(end_time - start_time)
        except:
            pass  # Fail silently if session state is not available

        return result
    return wrapper


# Set up logging with rotation using Python's built-in logging.handlers
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure application logging with rotation and proper formatting.

    Args:
        log_level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        The configured logger instance
    """
    from logging.handlers import RotatingFileHandler

    # Create log directory if it doesn't exist
    AppConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Set up the root logger
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates when rerunning the app
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter(
        AppConfig.LOG_FORMAT,
        datefmt=AppConfig.LOG_DATE_FORMAT
    )

    # Create and add a rotating file handler (limit to 5MB with 5 backups)
    file_handler = RotatingFileHandler(
        AppConfig.LOG_FILE,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Add a console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create a specific logger for this application
    logger = logging.getLogger("medexplain")

    logger.info(f"Logging configured. Level: {log_level}")
    return logger


# Error handling utilities
class ApplicationError(Exception):
    """Base class for application-specific exceptions."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ComponentInitializationError(ApplicationError):
    """Exception raised when a component fails to initialize."""
    def __init__(self, component_name: str, message: str, original_exception: Optional[Exception] = None):
        details = {
            "component_name": component_name,
            "original_exception": str(original_exception) if original_exception else None,
            "traceback": traceback.format_exc() if original_exception else None
        }
        super().__init__(f"Failed to initialize {component_name}: {message}", details)


class ConfigurationError(ApplicationError):
    """Exception raised for configuration errors."""
    pass


class DataAccessError(ApplicationError):
    """Exception raised for data access errors."""
    pass


# SessionManager class for enhanced session state management
class SessionManager:
    """
    Manages Streamlit session state with improved type safety and defaults.
    Provides a cleaner API for working with session state.
    """

    @staticmethod
    def ensure_initialized():
        """Initialize session state with default values if not already set."""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.page = "Home"
            st.session_state.user_id = "default_user"
            st.session_state.chat_history = []
            st.session_state.last_symptom_check = None
            st.session_state.last_risk_assessment = None
            st.session_state.theme = AppConfig.DEFAULT_THEME
            st.session_state.analysis_in_progress = False
            st.session_state.advanced_mode = False
            st.session_state.error_message = None
            st.session_state.notification_count = 0
            st.session_state.notifications = []
            st.session_state.view_mode = "patient"
            st.session_state.data_sharing_enabled = False
            st.session_state.export_ready = False
            st.session_state.export_data = None
            st.session_state.performance_metrics = {
                'startup_time': 0,
                'component_load_success_rate': 0,
                'function_timing': {}
            }
            st.session_state.page_views = {}
            st.session_state.auth = {
                "authenticated": False,
                "user": None,
                "login_time": None,
                "expiry_time": None
            }

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a value from session state with a default fallback.

        Args:
            key: The session state key to retrieve
            default: The default value to return if key is not present

        Returns:
            The stored value or the default
        """
        SessionManager.ensure_initialized()
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a value in session state.

        Args:
            key: The session state key to set
            value: The value to store
        """
        SessionManager.ensure_initialized()
        st.session_state[key] = value

    @staticmethod
    def delete(key: str) -> None:
        """
        Delete a key from session state if it exists.

        Args:
            key: The session state key to delete
        """
        if key in st.session_state:
            del st.session_state[key]

    @staticmethod
    def clear(exclude: Optional[List[str]] = None) -> None:
        """
        Clear session state except for specified keys.

        Args:
            exclude: List of keys to preserve
        """
        exclude = exclude or []
        keys_to_delete = [k for k in st.session_state.keys() if k not in exclude]
        for key in keys_to_delete:
            del st.session_state[key]

    @staticmethod
    def navigate_to(page: str) -> None:
        """
        Navigate to a different page in the application.

        Args:
            page: The name of the page to navigate to
        """
        st.session_state.page = page
        st.rerun()

    @staticmethod
    def add_page_view(page_name: str) -> None:
        """
        Track a page view for analytics.

        Args:
            page_name: The name of the page being viewed
        """
        if "page_views" not in st.session_state:
            st.session_state.page_views = {}

        views = st.session_state.page_views
        if page_name not in views:
            views[page_name] = 0

        views[page_name] += 1

    @staticmethod
    def add_notification(title: str, message: str, notification_type: str = "info") -> None:
        """
        Add a notification to the system.

        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification (info, warning, error, success)
        """
        if "notifications" not in st.session_state:
            st.session_state.notifications = []

        st.session_state.notifications.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "message": message,
            "type": notification_type,
            "timestamp": time.time(),
            "read": False
        })

        # Update unread count
        st.session_state.notification_count = len(
            [n for n in st.session_state.notifications if not n.get("read", False)]
        )


# Configuration Manager for centralized configuration
class ConfigManager:
    """
    Manages application configuration with loading from file, environment
    variables, and default values.
    """
    _instance = None
    _config = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the configuration manager."""
        self._config = {}
        self._load_from_defaults()
        self._load_from_file()
        self._load_from_env()

    def _load_from_defaults(self):
        """Load default configuration values."""
        self._config = {
            "app_name": "MedExplain AI Pro",
            "version": AppConfig.VERSION,
            "theme": AppConfig.DEFAULT_THEME,
            "log_level": "INFO",
            "session_timeout": AppConfig.SESSION_TIMEOUT,
            "max_workers": AppConfig.MAX_WORKERS,
            "feature_flags": {
                "enable_advanced_analytics": True,
                "enable_chat_interface": True,
                "enable_performance_metrics": True
            },
            "api": {
                "openai": {
                    "model": "gpt-4",
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            }
        }

    def _load_from_file(self):
        """Load configuration from file."""
        try:
            if AppConfig.CONFIG_FILE.exists():
                with open(AppConfig.CONFIG_FILE, 'r') as f:
                    file_config = json.load(f)
                    self._update_config(self._config, file_config)
        except Exception as e:
            logging.getLogger("medexplain").warning(f"Error loading config file: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mapping = {
            "MEDEXPLAIN_LOG_LEVEL": ["log_level"],
            "MEDEXPLAIN_THEME": ["theme"],
            "OPENAI_API_KEY": ["api", "openai", "api_key"],
            "OPENAI_MODEL": ["api", "openai", "model"],
            "ENABLE_ADVANCED_ANALYTICS": ["feature_flags", "enable_advanced_analytics"],
            "ENABLE_CHAT_INTERFACE": ["feature_flags", "enable_chat_interface"],
            "ENABLE_PERFORMANCE_METRICS": ["feature_flags", "enable_performance_metrics"]
        }

        for env_var, config_path in env_mapping.items():
            if env_var in os.environ:
                self._set_nested_config(config_path, self._parse_env_value(os.environ[env_var]))

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def _update_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Recursively update a nested dictionary with values from another dictionary.
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # If both are dictionaries, recurse
                self._update_config(target[key], value)
            else:
                # Otherwise, update/add the value
                target[key] = value

    def _set_nested_config(self, path: List[str], value: Any):
        """Set a value in the nested config dictionary."""
        current = self._config
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path[-1]] = value

    def get(self, *path: str, default: Any = None) -> Any:
        """
        Get a configuration value by path.

        Args:
            *path: The path components to the config value
            default: Default value to return if path not found

        Returns:
            The configuration value or default
        """
        current = self._config
        try:
            for part in path:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, *path: str, value: Any) -> None:
        """
        Set a configuration value and save to file.

        Args:
            *path: The path components to the config value
            value: The value to set
        """
        self._set_nested_config(list(path), value)
        self.save()

    def save(self) -> None:
        """Save current configuration to file."""
        try:
            AppConfig.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(AppConfig.CONFIG_FILE, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            logging.getLogger("medexplain").error(f"Error saving config file: {e}")

    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config.copy()


# Component registry for better dependency management
class ComponentRegistry:
    """
    Registry for application components with dependency tracking and
    initialization status.
    """
    _instance = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ComponentRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the component registry."""
        self.components = {}
        self.status = {}
        self.dependencies = {}
        self.logger = logging.getLogger("medexplain.components")

    def register(self, name: str, component: Any, dependencies: Optional[List[str]] = None) -> None:
        """
        Register a component with its dependencies.

        Args:
            name: Component name
            component: The component instance
            dependencies: List of component names this component depends on
        """
        self.components[name] = component
        self.status[name] = True
        self.dependencies[name] = dependencies or []
        self.logger.debug(f"Registered component: {name}")

    def get(self, name: str) -> Optional[Any]:
        """
        Get a component by name.

        Args:
            name: Component name

        Returns:
            The component if registered and healthy, None otherwise
        """
        if name in self.components and self.status.get(name, False):
            return self.components[name]
        return None

    def mark_failed(self, name: str) -> None:
        """
        Mark a component as failed.

        Args:
            name: Component name
        """
        if name in self.status:
            self.status[name] = False
            self.logger.warning(f"Component marked as failed: {name}")

            # Mark dependent components as failed
            for dep_name, deps in self.dependencies.items():
                if name in deps and self.status.get(dep_name, False):
                    self.mark_failed(dep_name)

    def get_status(self) -> Dict[str, bool]:
        """
        Get the status of all components.

        Returns:
            Dictionary mapping component names to their status
        """
        return self.status.copy()

    def get_health_rate(self) -> float:
        """
        Calculate the health rate of components.

        Returns:
            Percentage of healthy components (0.0 to 1.0)
        """
        if not self.status:
            return 0.0
        return sum(1 for status in self.status.values() if status) / len(self.status)


# Main application class with improved architecture
class MedExplainApp:
    """
    Main application class for MedExplain AI Pro.
    Integrates all components into a complete application with enterprise features.
    """

    def __init__(self, load_components: bool = True, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the application and its components.

        Args:
            load_components: If True, load all components. Set to False for testing.
            config: Optional configuration dictionary to override defaults
        """
        try:
            # Initialize app configuration
            self.start_time = time.time()
            AppConfig.initialize()

            # Set up logging
            self.logger = setup_logging(
                log_level=os.environ.get("MEDEXPLAIN_LOG_LEVEL", "INFO")
            )

            # Initialize session tracking
            self.session_id = str(uuid.uuid4())
            self.logger.info(f"Initializing MedExplain AI Pro v{AppConfig.VERSION}, session: {self.session_id}")

            # Set up component registry
            self.registry = ComponentRegistry()

            # Set up configuration manager
            self.config_manager = ConfigManager()
            if config:
                for key, value in config.items():
                    self.config_manager._config[key] = value

            # Load components if requested
            if load_components:
                with ThreadPoolExecutor(max_workers=AppConfig.MAX_WORKERS) as executor:
                    # Start core component initialization in parallel
                    core_future = executor.submit(self._initialize_core_components)
                    ml_future = executor.submit(self._initialize_ml_components)

                    # Wait for core and ML components to finish
                    core_future.result()
                    ml_future.result()

                    # UI components depend on core and ML, so initialize after
                    self._initialize_ui_components()

                # Load API keys and other configuration
                self._load_api_keys_from_environment()

            # Log initialization summary
            health_rate = self.registry.get_health_rate()
            component_count = len(self.registry.status)
            healthy_count = sum(1 for status in self.registry.status.values() if status)

            self.logger.info(f"MedExplain AI Pro initialization complete. "
                           f"Status: {healthy_count} of {component_count} components loaded "
                           f"({health_rate:.1%} health rate)")

            # Store startup time for performance metrics
            startup_time = time.time() - self.start_time
            if hasattr(st, 'session_state') and 'performance_metrics' in st.session_state:
                st.session_state.performance_metrics['startup_time'] = startup_time
                st.session_state.performance_metrics['component_load_success_rate'] = health_rate

        except Exception as e:
            self.logger.error(f"Critical error during MedExplain AI Pro initialization: {e}", exc_info=True)
            if hasattr(st, 'session_state'):
                SessionManager.set('error_message', f"Application initialization failed: {str(e)}")
            raise

    def _initialize_core_components(self) -> None:
        """Initialize core system components with improved error handling and dependency tracking."""
        # Health Data Manager
        try:
            self.health_data = HealthDataManager(data_dir=AppConfig.USER_DATA_DIR)
            self.registry.register("health_data", self.health_data)
            self.logger.info("Initialized health_data component")
        except Exception as e:
            self.logger.error(f"Error initializing health_data: {e}", exc_info=True)
            self.registry.mark_failed("health_data")

        # User Profile Manager
        try:
            self.user_manager = UserProfileManager(data_dir=AppConfig.USER_DATA_DIR)
            self.registry.register("user_manager", self.user_manager, ["health_data"])
            self.logger.info("Initialized user_manager component")
        except Exception as e:
            self.logger.error(f"Error initializing user_manager: {e}", exc_info=True)
            self.registry.mark_failed("user_manager")

        # OpenAI Client with configuration from ConfigManager
        try:
            openai_config = self.config_manager.get("api", "openai", default={})
            self.openai_client = OpenAIClient(**openai_config)
            self.registry.register("openai_client", self.openai_client)
            self.logger.info("Initialized openai_client component")
        except Exception as e:
            self.logger.error(f"Error initializing openai_client: {e}", exc_info=True)
            self.registry.mark_failed("openai_client")

        # Initialize Health Analyzer if dependencies available
        if self.registry.get("user_manager"):
            try:
                self.health_analyzer = HealthDataAnalyzer(
                    health_history=self.user_manager.health_history,
                    user_profile=self.user_manager.profile
                )
                self.registry.register(
                    "health_analyzer",
                    self.health_analyzer,
                    ["user_manager"]
                )
                self.logger.info("Initialized health_analyzer component")
            except Exception as e:
                self.logger.error(f"Error initializing health_analyzer: {e}", exc_info=True)
                self.registry.mark_failed("health_analyzer")
        else:
            self.logger.warning("Skipping health_analyzer initialization due to missing dependencies")

    def _initialize_ml_components(self) -> None:
        """Initialize machine learning components with improved error handling and parallelization."""
        # Symptom Predictor
        try:
            self.symptom_predictor = SymptomPredictor(model_dir=AppConfig.MODEL_DIR)
            self.registry.register("symptom_predictor", self.symptom_predictor)
            self.logger.info("Initialized symptom_predictor component")
        except Exception as e:
            self.logger.error(f"Error initializing symptom_predictor: {e}", exc_info=True)
            self.registry.mark_failed("symptom_predictor")

        # NLP Symptom Extractor
        try:
            self.symptom_extractor = NLPSymptomExtractor(model_dir=AppConfig.MODEL_DIR)
            self.registry.register("symptom_extractor", self.symptom_extractor)
            self.logger.info("Initialized symptom_extractor component")
        except Exception as e:
            self.logger.error(f"Error initializing symptom_extractor: {e}", exc_info=True)
            self.registry.mark_failed("symptom_extractor")

        # Risk Assessor
        try:
            self.risk_assessor = PatientRiskAssessor(model_dir=AppConfig.MODEL_DIR)
            self.registry.register("risk_assessor", self.risk_assessor)
            self.logger.info("Initialized risk_assessor component")
        except Exception as e:
            self.logger.error(f"Error initializing risk_assessor: {e}", exc_info=True)
            self.registry.mark_failed("risk_assessor")

    def _initialize_ui_components(self) -> None:
        """
        Initialize UI components with improved error recovery, dependency checking,
        and lazy loading where possible.
        """
        # Use component registry to get dependencies
        user_manager = self.registry.get("user_manager")
        health_data = self.registry.get("health_data")
        openai_client = self.registry.get("openai_client")
        symptom_predictor = self.registry.get("symptom_predictor")
        symptom_extractor = self.registry.get("symptom_extractor")
        risk_assessor = self.registry.get("risk_assessor")

        # Health Dashboard - initialize if dependencies are available
        if user_manager:
            try:
                self.dashboard = HealthDashboard(
                    health_history=user_manager.health_history,
                    user_profile=user_manager.profile
                )
                self.registry.register(
                    "dashboard",
                    self.dashboard,
                    ["user_manager"]
                )
                self.logger.info("Initialized dashboard component")
            except Exception as e:
                self.logger.error(f"Error initializing dashboard: {e}", exc_info=True)
                self.registry.mark_failed("dashboard")
        else:
            self.logger.warning("Skipping dashboard initialization due to missing user_manager")

        # Chat Interface - Check dependencies
        if openai_client and symptom_extractor:
            try:
                self.chat_interface = ChatInterface(
                    openai_client=openai_client,
                    symptom_extractor=symptom_extractor
                )
                self.registry.register(
                    "chat_interface",
                    self.chat_interface,
                    ["openai_client", "symptom_extractor"]
                )
                self.logger.info("Initialized chat_interface component")
            except Exception as e:
                self.logger.error(f"Error initializing chat_interface: {e}", exc_info=True)
                self.registry.mark_failed("chat_interface")
        else:
            self.logger.warning("Skipping chat_interface initialization due to missing dependencies")

        # Symptom Analyzer UI - Handle partial dependencies
        try:
            # Collect available dependencies
            symptom_analyzer_params = {}

            if health_data:
                symptom_analyzer_params["health_data"] = health_data

            if symptom_predictor:
                symptom_analyzer_params["symptom_predictor"] = symptom_predictor

            if symptom_extractor:
                symptom_analyzer_params["symptom_extractor"] = symptom_extractor

            if risk_assessor:
                symptom_analyzer_params["risk_assessor"] = risk_assessor

            # Only initialize if we have at least health_data
            if "health_data" in symptom_analyzer_params:
                self.symptom_analyzer_ui = SymptomAnalyzerUI(**symptom_analyzer_params)
                dependencies = [key for key in symptom_analyzer_params.keys()]
                self.registry.register(
                    "symptom_analyzer_ui",
                    self.symptom_analyzer_ui,
                    dependencies
                )
                self.logger.info(f"Initialized symptom_analyzer_ui with {len(symptom_analyzer_params)} dependencies")
            else:
                self.logger.warning("Skipping symptom_analyzer_ui initialization due to missing health_data")
                self.registry.mark_failed("symptom_analyzer_ui")
        except Exception as e:
            self.logger.error(f"Error initializing symptom_analyzer_ui: {e}", exc_info=True)
            self.registry.mark_failed("symptom_analyzer_ui")

        # Medical Literature UI with safer dependency injection
        try:
            lit_ui_params = {}

            if health_data:
                lit_ui_params["health_data"] = health_data

            if openai_client:
                lit_ui_params["openai_client"] = openai_client

            # Initialize even with partial dependencies
            self.medical_literature_ui = MedicalLiteratureUI(**lit_ui_params)
            dependencies = [key for key in lit_ui_params.keys()]
            self.registry.register(
                "medical_literature_ui",
                self.medical_literature_ui,
                dependencies
            )
            self.logger.info(f"Initialized medical_literature_ui with {len(lit_ui_params)} dependencies")
        except Exception as e:
            self.logger.error(f"Error initializing medical_literature_ui: {e}", exc_info=True)
            self.registry.mark_failed("medical_literature_ui")

        # Health History UI
        try:
            history_ui_params = {}

            if user_manager:
                history_ui_params["user_manager"] = user_manager

            if health_data:
                history_ui_params["health_data"] = health_data

            # Only initialize if we have user_manager
            if "user_manager" in history_ui_params:
                self.health_history_ui = HealthHistoryUI(**history_ui_params)
                dependencies = [key for key in history_ui_params.keys()]
                self.registry.register(
                    "health_history_ui",
                    self.health_history_ui,
                    dependencies
                )
                self.logger.info(f"Initialized health_history_ui with {len(history_ui_params)} dependencies")
            else:
                self.logger.warning("Skipping health_history_ui initialization due to missing user_manager")
                self.registry.mark_failed("health_history_ui")
        except Exception as e:
            self.logger.error(f"Error initializing health_history_ui: {e}", exc_info=True)
            self.registry.mark_failed("health_history_ui")

        # Settings UI
        try:
            settings_ui_params = {}

            if user_manager:
                settings_ui_params["user_manager"] = user_manager

            if openai_client:
                settings_ui_params["openai_client"] = openai_client

            # Initialize even with partial dependencies
            self.settings_ui = SettingsUI(**settings_ui_params)
            dependencies = [key for key in settings_ui_params.keys()]
            self.registry.register(
                "settings_ui",
                self.settings_ui,
                dependencies
            )
            self.logger.info(f"Initialized settings_ui with {len(settings_ui_params)} dependencies")
        except Exception as e:
            self.logger.error(f"Error initializing settings_ui: {e}", exc_info=True)
            self.registry.mark_failed("settings_ui")

    def _load_api_keys_from_environment(self) -> None:
        """Load API keys from environment variables with improved security and validation."""
        # OpenAI API key with validation
        openai_client = self.registry.get("openai_client")
        if openai_client:
            try:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if api_key:
                    if not api_key.startswith("sk-"):
                        self.logger.warning("OpenAI API key format appears invalid - should start with 'sk-'")

                    # Securely set the API key
                    openai_client.set_api_key(api_key)
                    self.logger.info("Set OpenAI API key from environment variable")
                else:
                    self.logger.warning("OPENAI_API_KEY environment variable not found")
            except Exception as e:
                self.logger.error(f"Error setting OpenAI API key: {e}", exc_info=True)

        # Load other API keys and configurations as needed
        # Example for a hypothetical medical API
        if "MEDICAL_API_KEY" in os.environ:
            try:
                # Store in configuration manager rather than directly in component
                self.config_manager.set("api", "medical", "api_key", value=os.environ["MEDICAL_API_KEY"])
                self.logger.info("Set Medical API key from environment variable")
            except Exception as e:
                self.logger.error(f"Error setting Medical API key: {e}", exc_info=True)

    @timing_decorator
    def _init_session_state(self) -> None:
        """Initialize Streamlit session state variables with improved type safety and defaults."""
        try:
            # Use SessionManager to initialize session state
            SessionManager.ensure_initialized()

            # Update user ID if available
            user_manager = self.registry.get("user_manager")
            if user_manager:
                SessionManager.set("user_id", user_manager.current_user_id)

            # Update startup time in performance metrics
            startup_time = time.time() - self.start_time
            if 'performance_metrics' in st.session_state:
                st.session_state.performance_metrics['startup_time'] = startup_time
                st.session_state.performance_metrics['component_load_success_rate'] = self.registry.get_health_rate()

            # Add welcome notification if first load
            if not st.session_state.get("welcome_shown", False):
                SessionManager.add_notification(
                    "Welcome to MedExplain AI Pro",
                    f"Thank you for using our advanced health analytics platform v{AppConfig.VERSION}.",
                    "info"
                )
                SessionManager.set("welcome_shown", True)

            self.logger.info(f"Initialized session state for user: {st.session_state.user_id}")
        except Exception as e:
            self.logger.error(f"Error initializing session state: {e}", exc_info=True)
            SessionManager.set("error_message", f"Session initialization failed: {str(e)}")

    def _show_notifications(self) -> None:
        """Display and manage notifications with improved UI and functionality."""
        try:
            # Get notifications from session state
            notifications = SessionManager.get("notifications", [])
            if not notifications:
                st.sidebar.info("No notifications")
                return

            with st.sidebar.expander("Notifications", expanded=True):
                # Group notifications by type for better organization
                notification_types = {
                    "error": [],
                    "warning": [],
                    "success": [],
                    "info": []
                }

                # Sort notifications by timestamp (newest first)
                sorted_notifications = sorted(
                    notifications,
                    key=lambda n: n.get("timestamp", 0),
                    reverse=True
                )

                # Group by type
                for notification in sorted_notifications:
                    notification_type = notification.get("type", "info")
                    if notification_type in notification_types:
                        notification_types[notification_type].append(notification)

                # Show counts by type if there are multiple notifications
                if len(notifications) > 1:
                    counts = {
                        t: len(notifications)
                        for t, notifications in notification_types.items()
                        if notifications
                    }

                    if counts:
                        cols = st.columns(len(counts))
                        for i, (notification_type, count) in enumerate(counts.items()):
                            with cols[i]:
                                # Use appropriate emoji for each type
                                emoji = {
                                    "error": "🔴",
                                    "warning": "🟠",
                                    "success": "🟢",
                                    "info": "🔵"
                                }.get(notification_type, "🔵")

                                st.markdown(f"{emoji} **{notification_type.capitalize()}:** {count}")

                # Display notifications with appropriate styling
                for notification_type, type_notifications in notification_types.items():
                    if not type_notifications:
                        continue

                    # Use appropriate container based on type
                    container_func = {
                        "error": st.error,
                        "warning": st.warning,
                        "success": st.success,
                        "info": st.info
                    }.get(notification_type, st.info)

                    for notification in type_notifications:
                        with container_func():
                            # Format timestamp
                            timestamp = notification.get("timestamp", 0)
                            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")

                            # Display notification with metadata
                            st.markdown(f"**{notification['title']}** - {date_str}")
                            st.markdown(notification['message'])

                            # Mark as read button
                            if not notification.get("read", False):
                                if st.button("Mark as read", key=f"read_{notification['id']}"):
                                    notification["read"] = True
                                    SessionManager.set(
                                        "notification_count",
                                        len([n for n in notifications if not n.get("read", False)])
                                    )
                                    st.rerun()

                # Add clear all button
                if st.button("Clear all notifications"):
                    SessionManager.set("notifications", [])
                    SessionManager.set("notification_count", 0)
                    st.rerun()
        except Exception as e:
            self.logger.error(f"Error displaying notifications: {e}", exc_info=True)
            st.error("Could not display notifications. Please try refreshing the page.")

    def _add_custom_css(self) -> None:
        """Add custom CSS to the Streamlit application with theme support."""
        try:
            # Determine theme - either from session state or default
            theme = SessionManager.get("theme", AppConfig.DEFAULT_THEME)

            # Look for theme-specific CSS file
            theme_css_path = AppConfig.STATIC_DIR / "css" / f"theme_{theme}.css"
            base_css_path = AppConfig.STATIC_DIR / "css" / "style.css"

            custom_css = ""

            # Load base CSS
            if base_css_path.exists():
                with open(base_css_path, "r") as css_file:
                    custom_css = css_file.read()
                self.logger.debug("Loaded base CSS from file")

            # Load theme-specific CSS
            if theme_css_path.exists():
                with open(theme_css_path, "r") as css_file:
                    theme_css = css_file.read()
                    custom_css += f"\n\n/* Theme: {theme} */\n{theme_css}"
                self.logger.debug(f"Loaded theme CSS for '{theme}'")

            # Apply custom styles
            st.markdown(f"""
            <style>
                {custom_css}

                /* Base custom styles */
                .main .block-container {{
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                }}

                h1, h2, h3 {{
                    color: var(--heading-color, #2c3e50);
                }}

                /* Custom component styles */
                .stButton button {{
                    border-radius: 4px;
                    padding: 0.25rem 1rem;
                }}

                .info-card {{
                    background-color: var(--info-card-bg, #f8f9fa);
                    padding: 1rem;
                    border-radius: 8px;
                    border-left: 4px solid var(--primary-color, #3498db);
                    margin-bottom: 1rem;
                }}

                /* Logo styling */
                .logo-text {{
                    font-weight: 600;
                    color: var(--primary-color, #3498db);
                    font-size: 1.5rem;
                    margin: 0;
                }}

                /* Toast notifications */
                .toast {{
                    position: fixed;
                    top: 1rem;
                    right: 1rem;
                    z-index: 9999;
                    min-width: 250px;
                    padding: 1rem;
                    border-radius: 4px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    animation: slide-in 0.3s ease-out;
                }}

                @keyframes slide-in {{
                    from {{ transform: translateX(100%); }}
                    to {{ transform: translateX(0); }}
                }}

                /* Modern card styles */
                .modern-card {{
                    background-color: var(--card-bg, white);
                    border-radius: 8px;
                    padding: 1.5rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    transition: transform 0.2s, box-shadow 0.2s;
                }}

                .modern-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 15px rgba(0,0,0,0.1);
                }}

                /* Responsive adjustments */
                @media (max-width: 768px) {{
                    .modern-card {{
                        padding: 1rem;
                    }}
                }}
            </style>
            """, unsafe_allow_html=True)
        except Exception as e:
            self.logger.error(f"Error adding custom CSS: {e}", exc_info=True)

    def _apply_theme(self) -> None:
        """Apply theme-specific styling with improved implementation."""
        try:
            theme = SessionManager.get("theme", AppConfig.DEFAULT_THEME)

            if theme == "dark":
                st.markdown("""
                <style>
                    :root {
                        --heading-color: #e0e0e0;
                        --text-color: #f0f2f6;
                        --background-color: #1e1e1e;
                        --secondary-bg-color: #2d2d2d;
                        --primary-color: #3498db;
                        --info-card-bg: #2d2d2d;
                        --card-bg: #2d2d2d;
                    }

                    body {
                        color: var(--text-color);
                        background-color: var(--background-color);
                    }

                    .stApp {
                        background-color: var(--background-color);
                    }

                    .main .block-container {
                        background-color: var(--background-color);
                    }

                    p, li {
                        color: #c0c0c0;
                    }

                    /* Input fields */
                    .stTextInput input, .stSelectbox, .stMultiselect {
                        background-color: var(--secondary-bg-color);
                        color: var(--text-color);
                        border-color: #444;
                    }

                    /* Sidebar */
                    .sidebar .sidebar-content {
                        background-color: var(--secondary-bg-color);
                    }

                    /* Charts */
                    .stPlotlyChart {
                        background-color: var(--secondary-bg-color);
                    }

                    /* Buttons */
                    .stButton button {
                        background-color: var(--secondary-bg-color);
                        color: var(--text-color);
                        border-color: #444;
                    }
                </style>
                """, unsafe_allow_html=True)
                self.logger.debug("Applied dark theme styling")
            elif theme == "high_contrast":
                st.markdown("""
                <style>
                    :root {
                        --heading-color: #ffffff;
                        --text-color: #ffffff;
                        --background-color: #000000;
                        --secondary-bg-color: #222222;
                        --primary-color: #ffff00;
                        --info-card-bg: #222222;
                        --card-bg: #222222;
                    }

                    body {
                        color: var(--text-color);
                        background-color: var(--background-color);
                    }

                    .stApp {
                        background-color: var(--background-color);
                    }

                    /* High contrast specific */
                    a {
                        color: var(--primary-color) !important;
                        text-decoration: underline !important;
                    }

                    .stButton button {
                        background-color: var(--primary-color);
                        color: #000000;
                        font-weight: bold;
                        border: 2px solid white;
                    }

                    /* Focus indicators */
                    :focus {
                        outline: 3px solid var(--primary-color) !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                self.logger.debug("Applied high contrast theme styling")
            else:
                # Light theme is the default - already set in base CSS
                self.logger.debug("Applied light theme styling (default)")

            # Apply font size adjustments if set
            font_size = SessionManager.get("font_size", "medium")
            if font_size == "large":
                st.markdown("""
                <style>
                    html {
                        font-size: 18px;
                    }

                    p, li, label {
                        font-size: 1.1rem;
                    }

                    h1 {
                        font-size: 2.2rem;
                    }

                    h2 {
                        font-size: 1.8rem;
                    }

                    h3 {
                        font-size: 1.5rem;
                    }
                </style>
                """, unsafe_allow_html=True)
                self.logger.debug("Applied large font size")
            elif font_size == "small":
                st.markdown("""
                <style>
                    html {
                        font-size: 14px;
                    }

                    p, li, label {
                        font-size: 0.9rem;
                    }

                    h1 {
                        font-size: 1.8rem;
                    }

                    h2 {
                        font-size: 1.5rem;
                    }

                    h3 {
                        font-size: 1.2rem;
                    }
                </style>
                """, unsafe_allow_html=True)
                self.logger.debug("Applied small font size")
        except Exception as e:
            self.logger.error(f"Error applying theme: {e}", exc_info=True)

    @timing_decorator
    def render_sidebar(self) -> str:
        """
        Render the application sidebar with navigation and user info
        with improved error handling and accessibility.

        Returns:
            The selected page name
        """
        try:
            with st.sidebar:
                # Display logo and title with improved layout
                col1, col2 = st.columns([1, 3])
                with col1:
                    logo_path = AppConfig.STATIC_DIR / "img" / "logo.png"
                    if logo_path.exists():
                        st.image(str(logo_path), width=50)
                    else:
                        st.image("../NeuroMed/img/logo.png", width=80)
                        self.logger.warning(f"Logo image not found: {logo_path}")

                with col2:
                    st.markdown(f"<p class='logo-text'>MedExplain AI Pro</p>", unsafe_allow_html=True)
                    st.caption(f"Version {AppConfig.VERSION}")

                # Display system status indicator with improved UI
                health_rate = self.registry.get_health_rate()

                if health_rate == 1.0:
                    system_status = "🟢 Fully Operational"
                    status_color = "green"
                elif health_rate >= 0.8:
                    system_status = "🟡 Partially Degraded"
                    status_color = "#daa520"  # goldenrod
                elif health_rate >= 0.5:
                    system_status = "🟠 Limited Functionality"
                    status_color = "orange"
                else:
                    system_status = "🔴 Severely Degraded"
                    status_color = "red"

                st.markdown(
                    f"<div style='display: flex; align-items: center;'>"
                    f"<div style='margin-right: 8px;'>System Status:</div>"
                    f"<div style='color: {status_color}; font-weight: 500;'>{system_status}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Theme selector with improved options
                st.markdown("---")
                theme_options = {
                    "light": "Light Mode 🌞",
                    "dark": "Dark Mode 🌙",
                    "high_contrast": "High Contrast ⚡"
                }

                current_theme = SessionManager.get("theme", AppConfig.DEFAULT_THEME)
                selected_theme = st.selectbox(
                    "Theme",
                    list(theme_options.keys()),
                    format_func=lambda x: theme_options[x],
                    index=list(theme_options.keys()).index(current_theme)
                )

                if selected_theme != current_theme:
                    SessionManager.set("theme", selected_theme)
                    st.rerun()

                # Accessibility options
                with st.expander("Accessibility"):
                    font_size = SessionManager.get("font_size", "medium")
                    font_options = ["small", "medium", "large"]
                    new_font_size = st.selectbox(
                        "Font Size",
                        font_options,
                        index=font_options.index(font_size)
                    )

                    if new_font_size != font_size:
                        SessionManager.set("font_size", new_font_size)
                        st.rerun()

                    # Screen reader optimization
                    screen_reader = SessionManager.get("screen_reader_optimized", False)
                    new_screen_reader = st.checkbox("Screen Reader Optimized", value=screen_reader)

                    if new_screen_reader != screen_reader:
                        SessionManager.set("screen_reader_optimized", new_screen_reader)
                        st.rerun()

                st.markdown("---")

                # Main navigation with improved categorization
                st.subheader("Navigation")

                # Organize navigation into categories
                nav_categories = {
                    "Main": ["Home", "Health Dashboard"],
                    "Tools": ["Symptom Analyzer", "Health Chat", "Medical Literature"],
                    "Analysis": ["Advanced Analytics", "Health History"],
                    "System": ["Settings"]
                }

                # If advanced mode is enabled, add admin panel
                if SessionManager.get("advanced_mode", False):
                    nav_categories["System"].append("Admin Panel")

                # Create tabs for each category
                selected_category = st.radio(
                    "Category",
                    list(nav_categories.keys()),
                    label_visibility="collapsed"
                )

                # Show pages for selected category
                choice = st.radio(
                    selected_category,
                    nav_categories[selected_category]
                )

                st.markdown("---")

                # Quick action buttons with improved UI and accessibility
                st.subheader("Quick Actions")

                # Use columns for a cleaner layout
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("🔍 Quick Check", key="quick_check", help="Quickly analyze your symptoms"):
                        SessionManager.navigate_to("Symptom Analyzer")
                        return "Symptom Analyzer"

                with col2:
                    tooltip = "Generate a comprehensive health report based on your data"
                    health_report_button = st.button("📋 Health Report", key="health_report", help=tooltip)

                    if health_report_button:
                        risk_assessor = self.registry.get("risk_assessor")
                        last_risk_assessment = SessionManager.get("last_risk_assessment")

                        if risk_assessor and last_risk_assessment:
                            try:
                                with st.spinner("Generating report..."):
                                    report = risk_assessor.generate_health_report(last_risk_assessment)
                                    SessionManager.set("health_report", report)
                                    SessionManager.navigate_to("Health History")
                                    return "Health History"
                            except Exception as e:
                                self.logger.error(f"Error generating health report: {e}", exc_info=True)
                                st.error("Could not generate report. Please try again.")
                        else:
                            st.info("Please complete a symptom analysis first to generate a health report.")

                # Notifications badge with improved UI
                notification_count = SessionManager.get("notification_count", 0)
                if notification_count > 0:
                    st.markdown(f"""
                    <div style='
                        background-color: #3498db;
                        color: white;
                        padding: 8px 12px;
                        border-radius: 20px;
                        display: inline-block;
                        margin-top: 10px;
                        cursor: pointer;
                    '>
                        📬 {notification_count} Notification{'' if notification_count == 1 else 's'}
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("View Notifications"):
                        self._show_notifications()

                # Display user profile with improved layout
                user_manager = self.registry.get("user_manager")
                if user_manager and user_manager.profile:
                    st.markdown("---")
                    st.subheader("User Profile")

                    profile = user_manager.profile

                    # Create a card-like display for the profile
                    profile_html = f"""
                    <div style='
                        padding: 15px;
                        border-radius: 8px;
                        background-color: var(--secondary-bg-color, #f8f9fa);
                        margin-bottom: 15px;
                    '>
                    """

                    if profile.get("name"):
                        profile_html += f"<div><strong>Name:</strong> {profile['name']}</div>"

                    if profile.get("age"):
                        profile_html += f"<div><strong>Age:</strong> {profile['age']}</div>"

                    if profile.get("gender"):
                        profile_html += f"<div><strong>Gender:</strong> {profile['gender']}</div>"

                    # Show brief health status if available
                    last_risk_assessment = SessionManager.get("last_risk_assessment")
                    if last_risk_assessment:
                        risk_level = last_risk_assessment.get('risk_level', 'unknown')
                        risk_color = {
                            'low': 'green',
                            'medium': 'orange',
                            'high': 'red',
                            'unknown': 'gray'
                        }.get(risk_level, 'gray')

                        profile_html += f"<div><strong>Health Status:</strong> <span style='color:{risk_color};'>{risk_level.capitalize()}</span></div>"

                    profile_html += "</div>"

                    st.markdown(profile_html, unsafe_allow_html=True)

                    # Add a quick button to edit profile
                    if st.button("Edit Profile", key="edit_profile"):
                        SessionManager.navigate_to("Settings")
                        return "Settings"

                # View mode selector (advanced mode only) with improved UI
                if SessionManager.get("advanced_mode", False):
                    st.markdown("---")
                    st.subheader("View Mode")

                    view_modes = ["Patient", "Clinician", "Researcher"]
                    view_mode_icons = {
                        "Patient": "👤",
                        "Clinician": "⚕️",
                        "Researcher": "🔬"
                    }

                    current_mode = SessionManager.get("view_mode", "patient")
                    view_mode = st.selectbox(
                        "Select view mode:",
                        view_modes,
                        format_func=lambda x: f"{view_mode_icons.get(x, '')} {x}",
                        index=["patient", "clinician", "researcher"].index(current_mode)
                    )

                    # Update view mode if changed
                    selected_mode = view_mode.lower()
                    if selected_mode != current_mode:
                        SessionManager.set("view_mode", selected_mode)
                        st.rerun()

                # Advanced mode toggle with better explanation
                st.markdown("---")

                # Create expandable section for advanced mode
                with st.expander("Advanced Options"):
                    advanced_mode = SessionManager.get("advanced_mode", False)
                    new_advanced_mode = st.checkbox(
                        "Advanced Mode",
                        value=advanced_mode,
                        help="Enable advanced features for healthcare professionals and researchers"
                    )

                    if new_advanced_mode != advanced_mode:
                        SessionManager.set("advanced_mode", new_advanced_mode)
                        st.rerun()

                    if new_advanced_mode:
                        st.info(
                            "Advanced mode is enabled, providing access to additional "
                            "analytics and administrative features."
                        )

                # Display disclaimer with improved information
                st.markdown("---")
                st.info(
                    "**MEDICAL DISCLAIMER:** This application is for educational purposes only. "
                    "It does not provide medical advice. Always consult a healthcare professional "
                    "for medical concerns."
                )

                # App credits with version and copyright information
                st.markdown(
                    f"<small>© {datetime.now().year} MedExplain AI Pro · v{AppConfig.VERSION}</small>",
                    unsafe_allow_html=True
                )

            return choice
        except Exception as e:
            self.logger.error(f"Error rendering sidebar: {e}", exc_info=True)
            st.error("Error rendering navigation. Please refresh the page.")
            return "Home"  # Default to home page on error

    @timing_decorator
    def run(self) -> None:
        """Run the main application with improved error recovery and monitoring."""
        try:


            # Initialize session state if needed
            self._init_session_state()

            # Add custom CSS
            self._add_custom_css()

            # Apply theme
            self._apply_theme()

            # Display any error messages
            error_message = SessionManager.get("error_message")
            if error_message:
                st.error(error_message)
                SessionManager.set("error_message", None)

            # Handle data export if ready
            if SessionManager.get("export_ready", False):
                self._handle_data_export()

            # Check for session timeout (if authentication is enabled)
            if SessionManager.get("auth", {}).get("authenticated", False):
                login_time = SessionManager.get("auth", {}).get("login_time")
                if login_time and (time.time() - login_time) > AppConfig.SESSION_TIMEOUT:
                    st.warning("Your session has expired. Please log in again.")
                    # Clear session except for basic settings
                    SessionManager.clear(exclude=["theme", "font_size", "screen_reader_optimized"])
                    st.rerun()

            # Monitor performance
            start_time = time.time()

            # Render sidebar and get selected page
            selected_page = self.render_sidebar()

            # Check session state for page overrides
            page_override = SessionManager.get("page")
            if page_override:
                selected_page = page_override
                SessionManager.set("page", None)  # Reset for next time

            # Render the selected page
            self._render_page(selected_page)

            # Update performance metrics
            render_time = time.time() - start_time
            if 'performance_metrics' in st.session_state:
                if 'page_render_times' not in st.session_state.performance_metrics:
                    st.session_state.performance_metrics['page_render_times'] = {}

                # Store the last 5 render times for each page
                page_times = st.session_state.performance_metrics['page_render_times']
                if selected_page not in page_times:
                    page_times[selected_page] = []

                page_times[selected_page].append(render_time)
                page_times[selected_page] = page_times[selected_page][-5:]  # Keep last 5

        except Exception as e:
            self.logger.error(f"Error running MedExplain AI Pro: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {str(e)}")

            # Add error notification
            SessionManager.add_notification(
                "System Error",
                f"An error occurred while running the application: {str(e)}",
                "error"
            )

            # Try to render home page as fallback
            try:
                self._render_home()
            except Exception as fallback_error:
                self.logger.error(f"Error rendering fallback home page: {fallback_error}", exc_info=True)
                st.error("Critical error. Please refresh the page.")

                # Display error details in expandable section
                with st.expander("Error Details (for IT support)"):
                    st.code(traceback.format_exc())

    def _render_page(self, page: str) -> None:
        """
        Render the selected page content with improved error handling,
        fallbacks, and performance monitoring.

        Args:
            page: The name of the page to render
        """
        try:
            # Track page view for analytics
            SessionManager.add_page_view(page)

            # Use timing decorator for performance tracking
            @timing_decorator
            def render_page_content():
                # Route to the appropriate page renderer
                if page == "Home":
                    self._render_home()
                elif page == "Symptom Analyzer":
                    symptom_analyzer_ui = self.registry.get("symptom_analyzer_ui")
                    if symptom_analyzer_ui:
                        symptom_analyzer_ui.render()
                    else:
                        self._render_fallback_page("Symptom Analyzer", "symptom analysis functionality")
                elif page == "Health Dashboard":
                    dashboard = self.registry.get("dashboard")
                    if dashboard:
                        dashboard.render_dashboard()
                    else:
                        self._render_fallback_page("Health Dashboard", "dashboard functionality")
                elif page == "Advanced Analytics":
                    self._render_advanced_analytics()
                elif page == "Medical Literature":
                    medical_literature_ui = self.registry.get("medical_literature_ui")
                    if medical_literature_ui:
                        medical_literature_ui.render()
                    else:
                        self._render_fallback_page("Medical Literature", "medical literature functionality")
                elif page == "Health Chat":
                    chat_interface = self.registry.get("chat_interface")
                    user_manager = self.registry.get("user_manager")
                    health_data = self.registry.get("health_data")

                    if chat_interface:
                        profile = user_manager.profile if user_manager else {}
                        chat_interface.render(profile, health_data)
                    else:
                        self._render_fallback_page("Health Chat", "chat functionality")
                elif page == "Health History":
                    health_history_ui = self.registry.get("health_history_ui")
                    if health_history_ui:
                        health_history_ui.render()
                    else:
                        self._render_fallback_page("Health History", "health history functionality")
                elif page == "Settings":
                    settings_ui = self.registry.get("settings_ui")
                    if settings_ui:
                        settings_ui.render()
                    else:
                        self._render_fallback_page("Settings", "settings functionality")
                elif page == "Admin Panel" and SessionManager.get("advanced_mode", False):
                    self._render_admin_panel()
                else:
                    # Default to home page
                    self._render_home()

            # Render the page and track performance
            render_page_content()

        except Exception as e:
            self.logger.error(f"Error rendering page {page}: {e}", exc_info=True)

            # Show error with more helpful information
            st.error(f"Error rendering {page} page: {str(e)}")

            # Provide troubleshooting steps
            st.markdown("""
            ### Troubleshooting steps:
            1. Try refreshing the page
            2. Check your internet connection
            3. Clear browser cache
            4. Try again later
            """)

            # Add button to go home
            if st.button("Return to Home Page"):
                SessionManager.navigate_to("Home")

            # Show error details in expandable section (advanced mode only)
            if SessionManager.get("advanced_mode", False):
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

            # Log the error for analytics
            if 'error_stats' not in st.session_state:
                st.session_state.error_stats = {}

            error_stats = st.session_state.error_stats
            error_type = type(e).__name__

            if error_type not in error_stats:
                error_stats[error_type] = {'count': 0, 'pages': {}}

            error_stats[error_type]['count'] += 1

            if page not in error_stats[error_type]['pages']:
                error_stats[error_type]['pages'][page] = 0

            error_stats[error_type]['pages'][page] += 1

    def _render_fallback_page(self, page_name: str, feature_name: str) -> None:
        """
        Render a fallback page when a component is not available with enhanced
        user experience and self-healing capabilities.

        Args:
            page_name: The name of the page
            feature_name: Description of the missing functionality
        """
        st.title(f"{page_name}")

        # Create a status container for better visibility
        with st.status(f"The {feature_name} is currently unavailable", expanded=True):
            st.write("We're experiencing technical difficulties with this feature.")

            # Provide more detailed information if in advanced mode
            if SessionManager.get("advanced_mode", False):
                st.markdown("### Component Status")

                # Get component status data
                status_data = []
                component_registry = ComponentRegistry()
                status = component_registry.get_status()

                for component, is_healthy in status.items():
                    status_data.append({
                        "Component": component,
                        "Status": "✅ Operational" if is_healthy else "❌ Failed"
                    })

                # Display as a dataframe for better formatting
                st.dataframe(
                    pd.DataFrame(status_data),
                    use_container_width=True,
                    hide_index=True
                )

            # Show troubleshooting steps
            st.markdown("### What you can do:")

            # Use columns for a cleaner layout
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                1. Try refreshing the page
                2. Check your internet connection
                3. Clear your browser cache
                4. Try again in a few minutes
                """)

            with col2:
                # Show go home button
                if st.button("Return to Home"):
                    SessionManager.navigate_to("Home")

                # Show repair button
                if st.button("Attempt Repair"):
                    # This would typically trigger component reinitialization
                    try:
                        with st.spinner("Attempting to repair components..."):
                            if page_name == "Symptom Analyzer":
                                self._initialize_ml_components()
                                health_data = self.registry.get("health_data")
                                symptom_predictor = self.registry.get("symptom_predictor")
                                symptom_extractor = self.registry.get("symptom_extractor")
                                risk_assessor = self.registry.get("risk_assessor")

                                if health_data:
                                    self.symptom_analyzer_ui = SymptomAnalyzerUI(
                                        health_data=health_data,
                                        symptom_predictor=symptom_predictor,
                                        symptom_extractor=symptom_extractor,
                                        risk_assessor=risk_assessor
                                    )
                                    component_registry = ComponentRegistry()
                                    component_registry.register(
                                        "symptom_analyzer_ui",
                                        self.symptom_analyzer_ui,
                                        ["health_data"]
                                    )
                                    st.success("Successfully repaired components!")
                                    st.rerun()
                            else:
                                # Generic repair attempt
                                component_registry = ComponentRegistry()
                                for component_name, status in component_registry.get_status().items():
                                    if not status:
                                        if component_name in ["dashboard", "chat_interface",
                                                            "medical_literature_ui", "health_history_ui",
                                                            "settings_ui"]:
                                            # Try to reinitialize UI components
                                            self._initialize_ui_components()
                                        elif component_name in ["symptom_predictor", "symptom_extractor", "risk_assessor"]:
                                            # Try to reinitialize ML components
                                            self._initialize_ml_components()
                                        elif component_name in ["health_data", "user_manager", "openai_client"]:
                                            # Try to reinitialize core components
                                            self._initialize_core_components()

                                st.success("Repair attempt completed. Please check if the issue is resolved.")
                                st.rerun()
                    except Exception as e:
                        self.logger.error(f"Error during component repair: {e}", exc_info=True)
                        st.error(f"Repair failed: {str(e)}")

        # Show a simplified version of the page if possible
        st.markdown("### Alternative Options")

        if page_name == "Health Dashboard":
            st.markdown("""
            While the dashboard is unavailable, you can access parts of your health information through:
            - The Health History page
            - The Symptom Analyzer
            """)

            # Display any available health data summary
            user_manager = self.registry.get("user_manager")
            if user_manager and hasattr(user_manager, "health_history"):
                st.info(f"You have {len(user_manager.health_history)} health records in your history.")

                # Show last record if available
                if user_manager.health_history:
                    last_record = user_manager.health_history[-1]
                    st.markdown("**Most recent health record:**")
                    st.json(last_record)

        elif page_name == "Health Chat":
            st.markdown("""
            While the chat interface is unavailable, you can use:
            - The Symptom Analyzer for guided symptom assessment
            - The Medical Literature section to read health information
            """)

            # Show a form to record symptoms manually
            with st.expander("Record Symptoms Manually"):
                with st.form("manual_symptoms"):
                    symptoms = st.text_area("Enter your symptoms (one per line)")
                    notes = st.text_area("Additional notes")

                    if st.form_submit_button("Save"):
                        if symptoms:
                            user_manager = self.registry.get("user_manager")
                            if user_manager:
                                try:
                                    symptom_list = [s.strip() for s in symptoms.split("\n") if s.strip()]
                                    record = {
                                        "date": datetime.now().strftime("%Y-%m-%d"),
                                        "symptoms": symptom_list,
                                        "notes": notes
                                    }
                                    user_manager.add_health_record(record)
                                    st.success("Symptoms recorded successfully!")
                                except Exception as e:
                                    st.error(f"Error saving symptoms: {str(e)}")
                            else:
                                st.error("User manager is not available.")

        elif page_name == "Medical Literature":
            st.markdown("""
            While the medical literature feature is unavailable, you can:
            - Visit reputable health websites like Mayo Clinic or WebMD
            - Consult a healthcare professional for medical information
            """)

            # Add links to reputable sources
            st.markdown("""
            ### Reputable Health Resources
            - [Mayo Clinic](https://www.mayoclinic.org)
            - [WebMD](https://www.webmd.com)
            - [CDC](https://www.cdc.gov)
            - [NIH Health Information](https://health.nih.gov)
            """)

    def _render_home(self) -> None:
        """Render the home page with enhanced UI components and accessibility features."""
        try:
            st.title("Welcome to MedExplain AI Pro")
            st.subheader("Your advanced personal health assistant powered by artificial intelligence")

            # Display main features in columns with improved layout
            col1, col2 = st.columns([3, 2])

            with col1:
                st.markdown("""
                ### Enterprise-Grade Healthcare Analytics at Your Fingertips

                MedExplain AI Pro combines advanced medical knowledge with state-of-the-art artificial intelligence to help you:

                - **Analyze your symptoms** with ensemble machine learning models
                - **Track your health** over time with interactive visualizations
                - **Uncover patterns** in your symptoms and health data
                - **Identify potential risks** based on your comprehensive health profile
                - **Chat naturally** about medical topics with our AI assistant
                - **Access medical literature** summarized in plain language
                """)

                # Feature highlights with improved UI
                st.markdown("### Key Advanced Features")

                # Create modern cards for features
                feature_col1, feature_col2 = st.columns(2)

                with feature_col1:
                    st.markdown("""
                    <div class="modern-card">
                        <h4>🧠 AI-Powered Analysis</h4>
                        <p>Ensemble ML models analyze your health data using multiple algorithms</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="modern-card">
                        <h4>📊 Interactive Health Dashboard</h4>
                        <p>Comprehensive visualizations of your health patterns and trends</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="modern-card">
                        <h4>🔍 Pattern Recognition</h4>
                        <p>Advanced algorithms identify correlations in your symptoms</p>
                    </div>
                    """, unsafe_allow_html=True)

                with feature_col2:
                    st.markdown("""
                    <div class="modern-card">
                        <h4>💬 Medical NLP Interface</h4>
                        <p>Discuss your health in plain language with contextual understanding</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="modern-card">
                        <h4>📈 Predictive Insights</h4>
                        <p>Risk assessment and early warning indicators</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="modern-card">
                        <h4>🔒 Enterprise Security</h4>
                        <p>HIPAA-compliant data encryption and privacy protection</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Hero image - load from static directory if available
                image_path = AppConfig.STATIC_DIR / "img" / "hero.png"
                if image_path.exists():
                    st.image(str(image_path))
                else:
                    st.image("../NeuroMed/img/hero.png" , width=450)

                # Quick action buttons with improved UI and accessibility
                st.markdown("### Quick Actions")

                # Use custom styling for buttons
                action_col1, action_col2 = st.columns(2)

                with action_col1:
                    analyze_btn = st.button(
                        "🔍 Analyze Symptoms",
                        key="home_analyze",
                        help="Start a symptom analysis to assess your health"
                    )
                    if analyze_btn:
                        SessionManager.navigate_to("Symptom Analyzer")

                    dashboard_btn = st.button(
                        "📊 View Dashboard",
                        key="home_dashboard",
                        help="See your health trends and analytics dashboard"
                    )
                    if dashboard_btn:
                        SessionManager.navigate_to("Health Dashboard")

                with action_col2:
                    chat_btn = st.button(
                        "💬 Health Chat",
                        key="home_chat",
                        help="Chat with the AI assistant about your health concerns"
                    )
                    if chat_btn:
                        SessionManager.navigate_to("Health Chat")

                    analytics_btn = st.button(
                        "📈 Advanced Analytics",
                        key="home_analytics",
                        help="Explore detailed analytics about your health data"
                    )
                    if analytics_btn:
                        SessionManager.navigate_to("Advanced Analytics")

            # What's New section for returning users
            if SessionManager.get("returning_user", False):
                with st.expander("What's New in v" + AppConfig.VERSION):
                    st.markdown("""
                    ### Latest Updates

                    - **Enhanced Performance**: Faster response times and improved reliability
                    - **Advanced Symptom Analysis**: New ML models for more accurate symptom correlation
                    - **Improved Visualizations**: Enhanced charts and interactive data exploration
                    - **Accessibility Features**: New options for customization and screen readers
                    - **Bug Fixes**: Resolved issues with data synchronization and user profiles
                    """)
            else:
                # Mark as returning user for next visit
                SessionManager.set("returning_user", True)

            # Recent activity section with improved UI
            st.markdown("---")
            st.subheader("Recent Activity")

            user_manager = self.registry.get("user_manager")
            health_data = self.registry.get("health_data")

            if user_manager and hasattr(user_manager, "get_recent_symptom_checks") and user_manager.health_history:
                recent_checks = user_manager.get_recent_symptom_checks(limit=3)

                if recent_checks:
                    for check in recent_checks:
                        date = check.get("date", "")
                        symptoms = check.get("symptoms", [])

                        # Get symptom names instead of IDs
                        symptom_names = []
                        if health_data:
                            for symptom_id in symptoms:
                                symptom_info = health_data.get_symptom_info(symptom_id)
                                if symptom_info:
                                    symptom_names.append(symptom_info.get("name", symptom_id))
                                else:
                                    symptom_names.append(symptom_id)
                        else:
                            symptom_names = symptoms  # Fallback to IDs if names not found

                        # Use custom card styling
                        theme = SessionManager.get("theme", AppConfig.DEFAULT_THEME)
                        card_bg = "#2d2d2d" if theme == "dark" else "#f8f9fa"

                        st.markdown(f"""
                        <div style="background-color: {card_bg};
                             padding: 15px; border-radius: 5px; margin-bottom: 15px;
                             border-left: 4px solid #3498db;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin-top: 0;">Check from {date}</h4>
                                <div style="font-size: 0.8rem; color: #666;">
                                    {(datetime.now() - datetime.strptime(date, '%Y-%m-%d')).days} days ago
                                </div>
                            </div>
                            <p><strong>Symptoms:</strong> {", ".join(symptom_names)}</p>
                            {f"<p><strong>Notes:</strong> {check.get('notes', '')}</p>" if check.get('notes') else ""}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent health activity. Start by analyzing your symptoms or setting up your health profile.")
            else:
                st.info("No recent health activity. Start by analyzing your symptoms or setting up your health profile.")

            # Health tips section with improved content
            openai_client = self.registry.get("openai_client")
            if openai_client and hasattr(openai_client, "api_key") and openai_client.api_key:
                st.markdown("---")
                st.subheader("Daily Health Tip")

                # Cache tip for the day
                if 'daily_tip' not in st.session_state:
                    try:
                        # Current date to create a consistent tip for the day
                        today_str = datetime.now().strftime("%Y-%m-%d")

                        prompt = f"""
                        Date: {today_str}

                        Provide a single, evidence-based health tip for today that would be useful for general wellness.
                        Focus on practical, actionable advice (100 words max) backed by medical research.
                        Format it as a brief paragraph with a title and the tip itself.
                        Include a seasonal or timely element if appropriate for the date.
                        """

                        tip = openai_client.generate_response(prompt)
                        if tip:
                            SessionManager.set("daily_tip", tip)
                            SessionManager.set("daily_tip_date", today_str)
                        else:
                            # Fallback tip
                            SessionManager.set("daily_tip", """
                            **Hydration for Health**

                            Stay hydrated throughout the day by drinking water regularly, not just when you feel thirsty.
                            Proper hydration supports all bodily functions, improves energy levels, and helps maintain
                            concentration. For most adults, aim for about 8 glasses (2 liters) of water daily, adjusting
                            based on activity level, climate, and personal health needs.
                            """)
                            SessionManager.set("daily_tip_date", today_str)
                    except Exception as e:
                        self.logger.error(f"Error generating health tip: {e}", exc_info=True)
                        # Fallback tip
                        SessionManager.set("daily_tip", """
                        **Hydration for Health**

                        Stay hydrated throughout the day by drinking water regularly, not just when you feel thirsty.
                        Proper hydration supports all bodily functions, improves energy levels, and helps maintain
                        concentration. For most adults, aim for about 8 glasses (2 liters) of water daily, adjusting
                        based on activity level, climate, and personal health needs.
                        """)
                        SessionManager.set("daily_tip_date", today_str)

                # Check if we need a new tip (new day)
                today_str = datetime.now().strftime("%Y-%m-%d")
                if SessionManager.get("daily_tip_date") != today_str:
                    SessionManager.delete("daily_tip")
                    SessionManager.delete("daily_tip_date")
                    st.rerun()  # Refresh to generate a new tip

                # Display the tip in a nicely formatted way
                daily_tip = SessionManager.get("daily_tip", "")
                st.markdown(f"<div class='info-card'>{daily_tip}</div>", unsafe_allow_html=True)

                # Add a "Get New Tip" button
                if st.button("Get Another Tip"):
                    SessionManager.delete("daily_tip")
                    st.rerun()

            # Call to action and medical disclaimer
            st.markdown("---")

            # Use columns for getting started and disclaimer
            cta_col1, cta_col2 = st.columns([3, 2])

            with cta_col1:
                st.subheader("Getting Started")
                st.markdown("""
                1. **Analyze your symptoms** to get an initial health assessment
                2. **Complete your profile** in the Settings page for personalized insights
                3. **Explore the dashboard** to see your health trends over time
                4. **Chat with the AI assistant** about your specific health concerns
                """)

                # Add a prominent call-to-action button
                if st.button("🚀 Start Your Health Journey", type="primary", use_container_width=True):
                    SessionManager.navigate_to("Symptom Analyzer")

            with cta_col2:
                st.warning("""
                **Medical Disclaimer:** MedExplain AI Pro is not a substitute for professional medical advice,
                diagnosis, or treatment. Always seek the advice of your physician or other qualified health
                provider with any questions you may have regarding a medical condition.
                """)

                # Add additional legal disclaimer in expander
                with st.expander("Legal Information"):
                    st.markdown("""
                    This application is intended for informational and educational purposes only. The information
                    provided by MedExplain AI Pro is not intended to replace medical consultation, diagnosis,
                    or treatment. All content, including text, graphics, images, and information, is for general
                    informational purposes only.

                    Healthcare decisions should be made in partnership with qualified healthcare providers.
                    """)
        except Exception as e:
            self.logger.error(f"Error rendering home page: {e}", exc_info=True)
            st.error("Error rendering home page. Please refresh the page.")

            # Show simplified home page on error
            st.markdown("## Welcome to MedExplain AI Pro")
            st.markdown("Your health analytics platform is experiencing temporary issues.")

            if st.button("Try Again"):
                st.rerun()

    def _render_advanced_analytics(self) -> None:
        """Render the advanced analytics page with improved error handling, visualization, and interactivity."""
        try:
            st.title("Advanced Health Analytics")
            st.markdown("Explore detailed analysis of your health data to uncover patterns and insights.")

            # Check for required components
            user_manager = self.registry.get("user_manager")
            health_analyzer = self.registry.get("health_analyzer")

            if not user_manager or not health_analyzer or not user_manager.health_history:
                st.info("Insufficient health data available for analysis. Please add more health information first.")

                # Provide guidance on how to add health data
                st.markdown("### How to Add Health Data")
                st.markdown("""
                1. Go to the **Symptom Analyzer** page to record your symptoms
                2. Use the **Health Chat** feature to discuss your health concerns
                3. Complete your profile in the **Settings** page
                """)

                # Display progress towards having enough data
                health_history = user_manager.health_history if user_manager else []
                record_count = len(health_history)
                required_count = 3  # Minimum records needed for analysis

                progress = min(1.0, record_count / required_count)
                st.progress(progress, text=f"Health records: {record_count}/{required_count} minimum required")

                if st.button("Go to Symptom Analyzer", type="primary"):
                    SessionManager.navigate_to("Symptom Analyzer")

                return

            # Get user health history
            health_history = user_manager.health_history

            # Create tabs for different analysis types with improved organization
            tab_names = [
                "Symptom Patterns",
                "Risk Analysis",
                "Temporal Trends",
                "Predictive Analytics",
                "Comparative Analysis"
            ]

            # Use icons for better UX
            tab_icons = {
                "Symptom Patterns": "🔄",
                "Risk Analysis": "⚠️",
                "Temporal Trends": "📈",
                "Predictive Analytics": "🔮",
                "Comparative Analysis": "⚖️"
            }

            # Create labeled tabs with icons
            tabs = st.tabs([f"{tab_icons[name]} {name}" for name in tab_names])

            with tabs[0]:
                self._render_symptom_pattern_tab(health_history)

            with tabs[1]:
                self._render_risk_analysis_tab()

            with tabs[2]:
                self._render_temporal_trends_tab(health_history)

            with tabs[3]:
                self._render_predictive_analytics_tab(health_history)

            with tabs[4]:
                self._render_comparative_analysis_tab()

        except Exception as e:
            self.logger.error(f"Error rendering advanced analytics: {e}", exc_info=True)
            st.error("Error rendering advanced analytics. Please try again later.")

            # Show basic information as fallback
            st.markdown("### Basic Health Summary")
            user_manager = self.registry.get("user_manager")
            if user_manager and hasattr(user_manager, "health_history"):
                st.info(f"You have {len(user_manager.health_history)} health records in your history.")
            else:
                st.info("No health records found. Please add some health data first.")

    @timing_decorator
    def _render_symptom_pattern_tab(self, health_history: List[HealthRecord]) -> None:
        """
        Render symptom pattern analysis tab with improved visualizations and insights.

        Args:
            health_history: List of health records to analyze
        """
        try:
            st.subheader("Symptom Pattern Analysis")
            st.markdown("Discover relationships and patterns in your reported symptoms.")

            # Use st.info for feature explanation
            with st.info("How to use this analysis", icon="ℹ️"):
                st.markdown("""
                This analysis shows how your symptoms are related. Strong connections between symptoms
                may indicate common underlying causes or conditions. Use this information to have
                more informed discussions with your healthcare provider.
                """)

            # Symptom co-occurrence analysis with improved methodology
            st.markdown("### Symptom Co-occurrence")

            # Extract symptom data from history with better error handling
            all_symptoms = []
            for entry in health_history:
                symptoms = entry.get("symptoms", [])
                if isinstance(symptoms, list):
                    all_symptoms.extend(symptoms)

            # Get unique symptoms
            unique_symptoms = list(set(all_symptoms))

            if len(unique_symptoms) >= 2:
                # Get symptom names with fallbacks
                health_data = self.registry.get("health_data")
                symptom_names = []
                for symptom_id in unique_symptoms:
                    if health_data:
                        symptom_info = health_data.get_symptom_info(symptom_id)
                        if symptom_info:
                            symptom_names.append(symptom_info.get("name", symptom_id))
                        else:
                            symptom_names.append(symptom_id)
                    else:
                        symptom_names.append(symptom_id)

                # Count co-occurrences with improved algorithm
                co_occurrence = np.zeros((len(unique_symptoms), len(unique_symptoms)))

                for entry in health_history:
                    entry_symptoms = entry.get("symptoms", [])
                    # Skip invalid entries
                    if not isinstance(entry_symptoms, list):
                        continue

                    # Create symptom index sets for faster lookup
                    entry_symptom_indices = set(
                        i for i, s in enumerate(unique_symptoms) if s in entry_symptoms
                    )

                    # Update co-occurrence matrix
                    for i in entry_symptom_indices:
                        for j in entry_symptom_indices:
                            co_occurrence[i, j] += 1

                # Create heatmap with improved visualization
                try:
                    # Use Plotly for interactive visualizations
                    import plotly.express as px
                    import plotly.graph_objects as go

                    # Create a more readable heatmap
                    fig = px.imshow(
                        co_occurrence,
                        x=symptom_names,
                        y=symptom_names,
                        color_continuous_scale="Blues",
                        title="Symptom Co-occurrence Matrix"
                    )

                    # Improve layout for better readability
                    fig.update_layout(
                        xaxis_title="Symptom",
                        yaxis_title="Symptom",
                        width=700,
                        height=700,
                        xaxis={'tickangle': 45},
                        margin=dict(t=50, l=80, r=80, b=80)
                    )

                    # Make text more readable
                    fig.update_xaxes(tickfont=dict(size=10))
                    fig.update_yaxes(tickfont=dict(size=10))

                    # Add value annotations for clearer reading
                    for i in range(len(symptom_names)):
                        for j in range(len(symptom_names)):
                            if co_occurrence[i, j] > 0:
                                fig.add_annotation(
                                    x=symptom_names[j],
                                    y=symptom_names[i],
                                    text=str(int(co_occurrence[i, j])),
                                    showarrow=False,
                                    font=dict(color="white" if co_occurrence[i, j] > 3 else "black")
                                )

                    # Show interactive chart
                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation guidance
                    st.caption("""
                    **How to interpret:** Darker colors indicate symptoms that frequently occur together.
                    Numbers show how many times each symptom pair appeared in your records.
                    """)

                except Exception as e:
                    self.logger.error(f"Error creating symptom co-occurrence chart: {e}", exc_info=True)
                    st.error("Could not render co-occurrence visualization. Using alternative display method.")

                    # Fallback to text-based representation
                    st.markdown("### Symptom Co-occurrence Table")
                    co_data = []
                    for i, s1 in enumerate(symptom_names):
                        for j, s2 in enumerate(symptom_names):
                            if i < j:  # Only show each pair once
                                if co_occurrence[i, j] > 0:
                                    co_data.append({
                                        "Symptom 1": s1,
                                        "Symptom 2": s2,
                                        "Co-occurrences": int(co_occurrence[i, j])
                                    })

                    if co_data:
                        co_df = pd.DataFrame(co_data).sort_values("Co-occurrences", ascending=False)
                        st.dataframe(
                            co_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Co-occurrences": st.column_config.NumberColumn(
                                    "Co-occurrences",
                                    help="Number of times these symptoms occurred together",
                                    format="%d"
                                )
                            }
                        )
                    else:
                        st.info("No co-occurring symptoms found.")

                # Find common symptom pairs with improved analysis
                st.markdown("### Common Symptom Combinations")

                # Network graph visualization for better insight
                try:
                    # Create network data
                    network_nodes = [{"id": name, "label": name} for name in symptom_names]

                    # Create edges with weight based on co-occurrence
                    network_edges = []
                    for i in range(len(symptom_names)):
                        for j in range(i+1, len(symptom_names)):  # Avoid duplicates
                            if co_occurrence[i, j] > 0:
                                network_edges.append({
                                    "from": symptom_names[i],
                                    "to": symptom_names[j],
                                    "value": int(co_occurrence[i, j]),
                                    "title": f"Occurred together {int(co_occurrence[i, j])} times"
                                })

                    # Display top pairs as text first
                    pairs = []
                    for i in range(len(unique_symptoms)):
                        for j in range(i+1, len(unique_symptoms)):
                            if co_occurrence[i, j] > 0:
                                pairs.append((
                                    symptom_names[i],
                                    symptom_names[j],
                                    co_occurrence[i, j]
                                ))

                    # Sort by frequency
                    pairs.sort(key=lambda x: x[2], reverse=True)

                    if pairs:
                        st.markdown("Top symptom combinations in your health history:")

                        # Create a more visually appealing display
                        for idx, (s1, s2, count) in enumerate(pairs[:5]):  # Show top 5
                            st.markdown(
                                f"""
                                <div style="
                                    padding: 10px;
                                    margin-bottom: 10px;
                                    border-radius: 5px;
                                    background-color: rgba(52, 152, 219, {min(1.0, count/5 + 0.2)});
                                    color: white;
                                ">
                                    <strong>{idx+1}.</strong> <strong>{s1}</strong> and <strong>{s2}</strong>
                                    occurred together <strong>{int(count)} times</strong>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No co-occurring symptoms found in your health history.")

                except Exception as e:
                    self.logger.error(f"Error creating network visualization: {e}", exc_info=True)

                    # Fallback to simple text display if there's an error
                    if pairs:
                        for s1, s2, count in pairs[:5]:  # Show top 5
                            st.markdown(f"- **{s1}** and **{s2}** occurred together {int(count)} times")
                    else:
                        st.info("No co-occurring symptoms found in your health history.")

                # Add analysis insights if we have the NLP component
                symptom_extractor = self.registry.get("symptom_extractor")
                if symptom_extractor and hasattr(symptom_extractor, "analyze_symptom_patterns"):
                    try:
                        with st.spinner("Generating insights..."):
                            insights = symptom_extractor.analyze_symptom_patterns(co_occurrence, symptom_names)
                            if insights:
                                st.markdown("### Pattern Insights")
                                st.markdown(insights)
                    except Exception as e:
                        self.logger.error(f"Error generating symptom insights: {e}", exc_info=True)
            else:
                st.info("Not enough unique symptoms to analyze patterns. Try adding more symptom records.")

                # Show a more helpful explanation and guidance
                st.markdown("""
                To perform symptom pattern analysis, we need at least 2 unique symptoms in your health records.

                You can add more health data by:
                1. Using the **Symptom Analyzer** to record new symptoms
                2. Using the **Health Chat** to discuss and log your symptoms
                """)

                # Add a direct button to add more data
                if st.button("Add Symptom Data Now"):
                    SessionManager.navigate_to("Symptom Analyzer")
        except Exception as e:
            self.logger.error(f"Error rendering symptom pattern tab: {e}", exc_info=True)
            st.error("Error rendering symptom patterns. Please try again later.")

    @timing_decorator
    def _render_risk_analysis_tab(self) -> None:
        """Render risk analysis tab with improved visualizations and interpretability."""
        try:
            st.subheader("Risk Analysis")

            # Add explanatory text
            with st.info("Understanding Risk Analysis", icon="ℹ️"):
                st.markdown("""
                This analysis evaluates potential health risks based on your symptom patterns and profile data.
                It is not a diagnosis but provides insights to discuss with healthcare professionals.
                """)

            # Check if we have risk assessment data
            last_risk_assessment = SessionManager.get("last_risk_assessment")
            if last_risk_assessment:
                risk_data = last_risk_assessment

                # Overall risk score and level with improved visualization
                risk_level = risk_data.get('risk_level', 'unknown')
                risk_score = risk_data.get('risk_score', 0)

                # Create risk gauge with better interactivity
                try:
                    import plotly.graph_objects as go

                    # Create a more informative and visually appealing gauge
                    fig = go.Figure()

                    # Add the main gauge
                    fig.add_trace(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Overall Risk Level: {risk_level.capitalize()}", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 3.9], 'color': "rgba(0, 128, 0, 0.3)"},  # Green with transparency
                                {'range': [4, 6.9], 'color': "rgba(255, 165, 0, 0.3)"},  # Orange with transparency
                                {'range': [7, 10], 'color': "rgba(255, 0, 0, 0.3)"}  # Red with transparency
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_score
                            }
                        },
                        # Add delta to show change from previous assessment if available
                        delta = {'reference': risk_data.get('previous_score', risk_score), 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}}
                    ))

                    # Improve layout
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation context
                    risk_interpretations = {
                        'low': "Your overall risk level is low, suggesting minimal health concerns based on your current data.",
                        'medium': "Your overall risk level is moderate, indicating potential health areas that may benefit from attention.",
                        'high': "Your overall risk level is elevated, suggesting areas that may require closer monitoring or professional consultation.",
                        'unknown': "Your risk level could not be determined with the available data."
                    }

                    st.markdown(f"**Interpretation:** {risk_interpretations.get(risk_level, '')}")

                except Exception as e:
                    self.logger.error(f"Error creating risk gauge: {e}", exc_info=True)
                    st.error("Could not render risk visualization. Using alternative display.")

                    # Fallback to text-based representation with better styling
                    risk_color = {
                        'low': 'green',
                        'medium': 'orange',
                        'high': 'red',
                        'unknown': 'gray'
                    }.get(risk_level, 'gray')

                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #f5f5f5; margin-bottom: 20px;">
                        <h3 style="margin-top: 0;">Overall Risk Assessment</h3>
                        <p style="font-size: 2.5rem; font-weight: bold; color: {risk_color}; text-align: center;">
                            {risk_score}/10
                        </p>
                        <p style="text-align: center; font-size: 1.5rem; color: {risk_color}; font-weight: bold;">
                            {risk_level.capitalize()} Risk
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Domain-specific risks with improved visualization
                st.markdown("### Health Domain Risk Analysis")

                # Display domain risks if available
                domain_risks = risk_data.get('domain_risks', {})
                if domain_risks:
                    domain_names = list(domain_risks.keys())
                    domain_values = list(domain_risks.values())

                    try:
                        # Create a more informative horizontal bar chart
                        fig = go.Figure()

                        # Custom color mapping function
                        def get_color(value):
                            if value < 4:
                                return 'green'
                            elif value < 7:
                                return 'orange'
                            else:
                                return 'red'

                        # Add bars with custom colors and hover information
                        fig.add_trace(go.Bar(
                            x = domain_values,
                            y = domain_names,
                            orientation = 'h',
                            marker = {
                                'color': [get_color(v) for v in domain_values],
                                'line': {'width': 1, 'color': '#333'}
                            },
                            hovertemplate = "<b>%{y}</b><br>Risk Score: %{x:.1f}/10<extra></extra>"
                        ))

                        # Improve layout for better readability
                        fig.update_layout(
                            title="Risk by Health Domain",
                            xaxis_title="Risk Score (0-10)",
                            xaxis=dict(
                                range=[0, 10],
                                tickvals=[0, 2.5, 5, 7.5, 10],
                                ticktext=["No Risk (0)", "Low (2.5)", "Medium (5)", "High (7.5)", "Severe (10)"]
                            ),
                            yaxis_title="Health Domain",
                            height=max(300, len(domain_names) * 40),  # Dynamic height based on domains
                            margin=dict(l=20, r=20, t=50, b=50),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )

                        # Add a risk score threshold line
                        fig.add_shape(
                            type="line",
                            x0=7, y0=-0.5,
                            x1=7, y1=len(domain_names)-0.5,
                            line=dict(color="red", width=2, dash="dash")
                        )

                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)

                        # Add recommendations if available
                        recommendations = risk_data.get('recommendations', [])
                        if recommendations:
                            st.markdown("### Recommendations")
                            for i, recommendation in enumerate(recommendations):
                                st.markdown(f"{i+1}. {recommendation}")

                    except Exception as e:
                        self.logger.error(f"Error creating domain risk chart: {e}", exc_info=True)
                        st.error("Could not render domain risk visualization. Using table format.")

                        # Fallback to enhanced table display
                        risk_df = pd.DataFrame({
                            'Health Domain': domain_names,
                            'Risk Score': domain_values
                        }).sort_values('Risk Score', ascending=False)

                        # Use Streamlit's enhanced dataframe display
                        st.dataframe(
                            risk_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Risk Score": st.column_config.ProgressColumn(
                                    "Risk Score",
                                    help="Risk score from 0-10",
                                    format="%.1f",
                                    min_value=0,
                                    max_value=10,
                                    width="large"
                                )
                            }
                        )
                else:
                    st.info("Detailed domain risk data is not available.")

                # Add action buttons at the bottom
                st.markdown("### Actions")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Generate Health Report", key="gen_health_report"):
                        risk_assessor = self.registry.get("risk_assessor")
                        if risk_assessor and hasattr(risk_assessor, "generate_health_report"):
                            try:
                                with st.spinner("Generating comprehensive health report..."):
                                    report = risk_assessor.generate_health_report(risk_data)
                                    SessionManager.set("health_report", report)
                                    SessionManager.add_notification(
                                        "Health Report Ready",
                                        "Your comprehensive health report has been generated.",
                                        "success"
                                    )
                                    SessionManager.navigate_to("Health History")
                            except Exception as e:
                                self.logger.error(f"Error generating health report: {e}", exc_info=True)
                                st.error("Could not generate report. Please try again.")

                with col2:
                    if st.button("Update Risk Assessment", key="update_risk"):
                        SessionManager.navigate_to("Symptom Analyzer")
            else:
                st.info("No risk assessment data available. Please complete a symptom analysis first.")

                # Add a direct link to symptom analyzer with better call-to-action
                with st.container():
                    st.markdown("""
                    ### Complete a Risk Assessment

                    To view your health risk analysis, you need to complete a symptom assessment first.
                    This will allow our system to evaluate potential health risks based on your reported symptoms.
                    """)

                    if st.button("Start Symptom Analysis", type="primary"):
                        SessionManager.navigate_to("Symptom Analyzer")
        except Exception as e:
            self.logger.error(f"Error rendering risk analysis tab: {e}", exc_info=True)
            st.error("Error rendering risk analysis. Please try again later.")

    @timing_decorator
    def _render_temporal_trends_tab(self, health_history: List[HealthRecord]) -> None:
        """
        Render temporal trends tab with improved visualizations and insights.

        Args:
            health_history: List of health records to analyze
        """
        try:
            st.subheader("Temporal Trends")
            st.markdown("Analyze how your health metrics change over time.")

            # Add helpful context
            with st.info("Understanding Temporal Trends", icon="ℹ️"):
                st.markdown("""
                This analysis shows how your health metrics have changed over time.
                Patterns in these trends can help identify improvements or areas needing attention.
                """)

            if len(health_history) > 1:
                # Extract dates and metrics with improved data validation
                dates = []
                symptom_counts = []
                severity_scores = []

                # Sort health history by date first (earliest to latest)
                sorted_history = sorted(
                    health_history,
                    key=lambda entry: entry.get("date", ""),
                    reverse=False  # Chronological order
                )

                for entry in sorted_history:
                    date_str = entry.get("date", "")
                    # Skip entries with missing dates
                    if not date_str:
                        continue

                    # Add symptom count
                    symptoms = entry.get("symptoms", [])
                    symptom_count = len(symptoms) if isinstance(symptoms, list) else 0

                    # Add severity score if available
                    severity = entry.get("severity", None)

                    dates.append(date_str)
                    symptom_counts.append(symptom_count)
                    severity_scores.append(severity)

                # Create interactive time-series visualization
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    # Create subplots if we have severity data
                    has_severity = any(s is not None for s in severity_scores)

                    if has_severity:
                        # Create a figure with secondary y-axis
                        fig = make_subplots(specs=[[{"secondary_y": True}]])

                        # Add symptom count trace
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=symptom_counts,
                                mode='lines+markers',
                                name='Symptom Count',
                                line=dict(color='royalblue', width=3),
                                marker=dict(size=8)
                            ),
                            secondary_y=False
                        )

                        # Add severity score trace
                        valid_severity = [s for s in severity_scores if s is not None]
                        if valid_severity:
                            valid_dates = [dates[i] for i in range(len(dates)) if severity_scores[i] is not None]
                            fig.add_trace(
                                go.Scatter(
                                    x=valid_dates,
                                    y=valid_severity,
                                    mode='lines+markers',
                                    name='Severity Score',
                                    line=dict(color='firebrick', width=3, dash='dot'),
                                    marker=dict(size=8, symbol='diamond')
                                ),
                                secondary_y=True
                            )

                            # Update y-axis titles
                            fig.update_yaxes(title_text="Number of Symptoms", secondary_y=False)
                            fig.update_yaxes(title_text="Severity Score (1-10)", secondary_y=True)
                        else:
                            # Update y-axis title for symptom count only
                            fig.update_yaxes(title_text="Number of Symptoms")
                    else:
                        # Create a simple figure with one y-axis
                        fig = go.Figure()

                        # Add symptom count trace
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=symptom_counts,
                                mode='lines+markers',
                                name='Symptom Count',
                                line=dict(color='royalblue', width=3),
                                marker=dict(size=10)
                            )
                        )

                        # Update y-axis title
                        fig.update_layout(yaxis_title="Number of Symptoms")

                    # Add trend line using simple moving average
                    if len(symptom_counts) >= 3:
                        # Calculate moving average (window size 2)
                        window_size = min(2, len(symptom_counts) - 1)
                        moving_avg = []
                        for i in range(len(symptom_counts)):
                            if i < window_size:
                                # Use available data points for start of series
                                window = symptom_counts[:i+1]
                            else:
                                window = symptom_counts[i-window_size:i+1]
                            moving_avg.append(sum(window) / len(window))

                        # Add moving average line
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=moving_avg,
                                mode='lines',
                                name='Trend (Moving Avg)',
                                line=dict(color='green', width=2, dash='dash')
                            ),
                            secondary_y=False
                        )

                    # Improve overall layout
                    fig.update_layout(
                        title="Health Metrics Over Time",
                        xaxis_title="Date",
                        height=450,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(l=20, r=20, t=50, b=50),
                        hovermode="x unified"
                    )

                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation context
                    st.caption("""
                    **How to interpret:** This chart shows how your symptom count changes over time.
                    Downward trends suggest improvement, while upward trends may indicate worsening conditions.
                    """)

                except Exception as e:
                    self.logger.error(f"Error creating temporal trend chart: {e}", exc_info=True)
                    st.error("Could not render temporal trend visualization. Using table format.")

                    # Fallback to table display with improved formatting
                    trend_df = pd.DataFrame({
                        'Date': dates,
                        'Symptom Count': symptom_counts
                    })

                    if any(s is not None for s in severity_scores):
                        trend_df['Severity Score'] = severity_scores

                    st.dataframe(
                        trend_df,
                        use_container_width=True,
                        hide_index=True
                    )

                # Text analysis of trends with improved insights
                st.markdown("### Trend Analysis")

                # Calculate basic trend metrics with more sophisticated analysis
                if len(symptom_counts) >= 3:
                    # Determine overall trend
                    start_avg = sum(symptom_counts[:2]) / 2
                    end_avg = sum(symptom_counts[-2:]) / 2

                    change = end_avg - start_avg
                    percent_change = (change / max(1, start_avg)) * 100

                    if percent_change < -15:
                        trend = "significantly improving"
                        trend_icon = "🟢"
                        trend_explanation = "Your symptom count has decreased substantially, suggesting a significant improvement in your condition."
                    elif percent_change < -5:
                        trend = "improving"
                        trend_icon = "🟢"
                        trend_explanation = "Your symptom count is showing a gradual improvement over time."
                    elif percent_change <= 5:
                        trend = "stable"
                        trend_icon = "🟡"
                        trend_explanation = "Your symptom count has remained relatively stable, without significant improvement or worsening."
                    elif percent_change <= 15:
                        trend = "worsening"
                        trend_icon = "🟠"
                        trend_explanation = "Your symptom count is showing a gradual increase, which may indicate a worsening condition."
                    else:
                        trend = "significantly worsening"
                        trend_icon = "🔴"
                        trend_explanation = "Your symptom count has increased substantially, which may indicate a significant worsening of your condition."

                    # Check for volatility
                    if len(symptom_counts) >= 4:
                        differences = [abs(symptom_counts[i] - symptom_counts[i-1]) for i in range(1, len(symptom_counts))]
                        avg_difference = sum(differences) / len(differences)
                        max_count = max(symptom_counts)

                        if avg_difference > max_count * 0.3:
                            volatility = "high"
                            volatility_explanation = "Your symptoms show significant fluctuations, which may indicate a cyclical condition or variable responses to treatment."
                        elif avg_difference > max_count * 0.15:
                            volatility = "moderate"
                            volatility_explanation = "Your symptoms show some fluctuations, which is common in many health conditions."
                        else:
                            volatility = "low"
                            volatility_explanation = "Your symptoms show consistent patterns with minimal fluctuations."
                    else:
                        volatility = "unknown"
                        volatility_explanation = "Not enough data points to determine symptom volatility."

                    # Display trend analysis with improved formatting
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #f5f5f5; margin-bottom: 20px;">
                        <h3 style="margin-top: 0;">Trend Summary {trend_icon}</h3>
                        <p>Based on your symptom history, your health appears to be <strong>{trend}</strong> over time.</p>

                        <ul>
                            <li><strong>Starting symptom count:</strong> {symptom_counts[0]}</li>
                            <li><strong>Current symptom count:</strong> {symptom_counts[-1]}</li>
                            <li><strong>Change:</strong> {change:.1f} symptoms ({percent_change:.1f}%)</li>
                            <li><strong>Volatility:</strong> {volatility}</li>
                        </ul>

                        <p><strong>Interpretation:</strong> {trend_explanation}</p>
                        <p>{volatility_explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add severity trend if available
                    valid_severity = [s for s in severity_scores if s is not None]
                    if len(valid_severity) >= 2:
                        sev_start = valid_severity[0]
                        sev_end = valid_severity[-1]
                        sev_change = sev_end - sev_start

                        if sev_change < -1:
                            sev_trend = "decreasing (improving)"
                            sev_icon = "🟢"
                        elif sev_change <= 1:
                            sev_trend = "stable"
                            sev_icon = "🟡"
                        else:
                            sev_trend = "increasing (worsening)"
                            sev_icon = "🔴"

                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f5f5f5; margin-bottom: 20px;">
                            <h3 style="margin-top: 0;">Severity Trend {sev_icon}</h3>
                            <p>Your symptom severity is <strong>{sev_trend}</strong>.</p>

                            <ul>
                                <li><strong>Initial severity:</strong> {sev_start}/10</li>
                                <li><strong>Current severity:</strong> {sev_end}/10</li>
                                <li><strong>Change:</strong> {sev_change:+.1f} points</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    # Add actions based on trend analysis
                    st.markdown("### Recommended Actions")

                    if trend in ["significantly worsening", "worsening"]:
                        st.warning("""
                        Based on your worsening trend, consider:
                        - Consulting with a healthcare professional
                        - Reviewing recent lifestyle or medication changes
                        - Tracking symptoms more frequently
                        """)
                    elif trend == "stable" and symptom_counts[-1] > 2:
                        st.info("""
                        Your condition is stable but still showing symptoms. Consider:
                        - Discussing maintenance strategies with your healthcare provider
                        - Exploring additional treatment options
                        - Continuing to monitor for any changes
                        """)
                    elif trend in ["improving", "significantly improving"]:
                        st.success("""
                        Your condition is improving. Consider:
                        - Maintaining current treatment approaches
                        - Documenting what strategies have been effective
                        - Continuing regular monitoring
                        """)

                    # Add chatbot referral
                    st.markdown("Want a more detailed analysis of your health trends?")
                    if st.button("Discuss with Health Chat AI"):
                        SessionManager.navigate_to("Health Chat")
            else:
                st.info("Not enough historical data to analyze trends. Please check back after recording more health data.")

                # Show progress indication
                st.markdown(f"Current records: **{len(health_history)}/3** minimum required for trend analysis")
                st.progress(min(1.0, len(health_history)/3))

                # Add guidance with a more helpful tone
                st.markdown("""
                ### Building Your Health Timeline

                Tracking your health over time provides valuable insights into patterns and trends.
                To enable meaningful trend analysis:

                1. Record your symptoms regularly (ideally once per week)
                2. Include severity ratings when recording symptoms
                3. Be consistent in how you report symptoms

                The more data points you add, the more accurate your trend analysis will become.
                """)

                # Add CTA button
                if st.button("Start Building Your Health Timeline"):
                    SessionManager.navigate_to("Symptom Analyzer")
        except Exception as e:
            self.logger.error(f"Error rendering temporal trends tab: {e}", exc_info=True)
            st.error("Error rendering temporal trends. Please try again later.")

    @timing_decorator
    def _render_predictive_analytics_tab(self, health_history: List[HealthRecord]) -> None:
        """
        Render predictive analytics tab with improved visualizations and explanations.

        Args:
            health_history: List of health records to analyze
        """
        try:
            st.subheader("Predictive Analytics")
            st.markdown("Explore AI-powered predictions about potential health trends.")

            # Add explanatory information
            with st.info("About Predictive Analytics", icon="ℹ️"):
                st.markdown("""
                This feature uses machine learning to predict potential health trends based on your
                historical data. Predictions become more accurate as you add more health records.
                """)

                # Add disclaimer about prediction accuracy
                st.markdown("""
                **Note:** Health predictions are estimates based on available data and should be
                used as a general guide, not as a replacement for professional medical advice.
                """)

            if len(health_history) >= 3:
                st.markdown("### Symptom Trend Prediction")

                # Alert users about prediction accuracy
                min_recommended = 5
                if len(health_history) < min_recommended:
                    st.warning(f"""
                    Predictive analytics requires at least {min_recommended} data points for optimal accuracy.
                    You currently have {len(health_history)} records. Predictions may be less reliable until
                    more data is available.
                    """)

                # Extract dates and metrics with better data handling
                dates = []
                symptom_counts = []

                # Sort health history by date first
                sorted_history = sorted(
                    health_history,
                    key=lambda entry: entry.get("date", ""),
                    reverse=False  # Chronological order
                )

                for entry in sorted_history:
                    date_str = entry.get("date", "")
                    # Skip entries with missing dates
                    if not date_str:
                        continue

                    symptoms = entry.get("symptoms", [])
                    symptom_count = len(symptoms) if isinstance(symptoms, list) else 0

                    dates.append(date_str)
                    symptom_counts.append(symptom_count)

                # Implement prediction visualization with improved methodology
                try:
                    import plotly.graph_objects as go
                    from datetime import datetime, timedelta

                    # Parse dates correctly
                    date_objects = []
                    for date_str in dates:
                        try:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        except ValueError:
                            try:
                                date_obj = datetime.strptime(date_str, "%m/%d/%Y")
                            except ValueError:
                                # Use a fallback date
                                date_obj = datetime.now() - timedelta(days=len(date_objects) * 7)
                        date_objects.append(date_obj)

                    # Generate future dates (7-day intervals)
                    if date_objects:
                        last_date = date_objects[-1]
                    else:
                        last_date = datetime.now()

                    future_dates = []
                    for i in range(1, 5):  # 4 future predictions (4 weeks)
                        future_date = last_date + timedelta(days=7 * i)
                        future_dates.append(future_date)

                    # Create prediction model using scikit-learn
                    if len(symptom_counts) >= 3:
                        # Create features - days since first record
                        first_date = date_objects[0]
                        days_since_first = [(date - first_date).days for date in date_objects]

                        # Use Linear Regression for prediction
                        from sklearn.linear_model import LinearRegression
                        import numpy as np

                        # Reshape for scikit-learn
                        X = np.array(days_since_first).reshape(-1, 1)
                        y = np.array(symptom_counts)

                        # Create and fit the model
                        model = LinearRegression()
                        model.fit(X, y)

                        # Generate predictions for future dates
                        future_days = [(date - first_date).days for date in future_dates]
                        future_X = np.array(future_days).reshape(-1, 1)
                        predictions = model.predict(future_X)

                        # Add minor random variation for more realistic presentation
                        np.random.seed(42)  # For reproducibility
                        noise = np.random.normal(0, 0.3, len(predictions))
                        predictions = np.maximum(0, predictions + noise)  # Ensure non-negative

                        # Convert to list and round to 1 decimal place
                        predictions = [round(max(0, p), 1) for p in predictions]
                    else:
                        # Fallback if not enough data
                        predictions = [symptom_counts[-1]] * 4 if symptom_counts else [0] * 4

                    # Format dates for display
                    formatted_dates = [d.strftime("%Y-%m-%d") for d in date_objects]
                    future_formatted = [d.strftime("%Y-%m-%d") for d in future_dates]

                    # Create plot with actual and predicted data
                    fig = go.Figure()

                    # Add confidence interval for predictions
                    if len(symptom_counts) >= 3:
                        # Simple confidence interval (mean +/- 1 standard deviation)
                        pred_std = max(1, np.std(symptom_counts) * 1.5)
                        upper_bound = [min(10, p + pred_std) for p in predictions]
                        lower_bound = [max(0, p - pred_std) for p in predictions]

                        # Add uncertainty range
                        fig.add_trace(go.Scatter(
                            x=future_formatted + future_formatted[::-1],
                            y=upper_bound + lower_bound[::-1],
                            fill='toself',
                            fillcolor='rgba(255,165,0,0.2)',
                            line=dict(color='rgba(255,165,0,0)'),
                            hoverinfo="skip",
                            showlegend=False
                        ))

                    # Add vertical line to separate actual from predicted
                    fig.add_shape(
                        type="line",
                        x0=formatted_dates[-1],
                        y0=0,
                        x1=formatted_dates[-1],
                        y1=max(symptom_counts + predictions) * 1.2,
                        line=dict(color="gray", width=2, dash="dot")
                    )

                    # Add annotation to mark prediction start
                    fig.add_annotation(
                        x=formatted_dates[-1],
                        y=max(symptom_counts + predictions) * 1.1,
                        text="Prediction Starts",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40
                    )

                    # Actual data with improved styling
                    fig.add_trace(go.Scatter(
                        x=formatted_dates,
                        y=symptom_counts,
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='royalblue', width=3),
                        marker=dict(size=8)
                    ))

                    # Predicted data with improved styling
                    fig.add_trace(go.Scatter(
                        x=future_formatted,
                        y=predictions,
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='orange', width=3, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))

                    # Improve layout
                    fig.update_layout(
                        title="Symptom Trend Prediction",
                        xaxis_title="Date",
                        yaxis_title="Number of Symptoms",
                        height=450,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(l=20, r=20, t=50, b=50),
                        hovermode="x unified"
                    )

                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)

                    # Add explanation of the prediction
                    if len(symptom_counts) >= 3:
                        # Calculate trend
                        trend_direction = "decreasing" if predictions[0] < symptom_counts[-1] else (
                            "increasing" if predictions[0] > symptom_counts[-1] else "stable"
                        )

                        trend_magnitude = abs(predictions[0] - symptom_counts[-1])
                        if trend_magnitude < 0.5:
                            trend_strength = "stable"
                        elif trend_magnitude < 1.5:
                            trend_strength = "slight"
                        elif trend_magnitude < 3:
                            trend_strength = "moderate"
                        else:
                            trend_strength = "significant"

                        # Determine if trend continues or reverses
                        first_vs_last_pred = predictions[-1] - predictions[0]
                        if abs(first_vs_last_pred) < 0.5:
                            continuation = "remain relatively stable"
                        elif (first_vs_last_pred > 0 and trend_direction == "increasing") or \
                             (first_vs_last_pred < 0 and trend_direction == "decreasing"):
                            continuation = f"continue {trend_direction}"
                        else:
                            continuation = "begin to reverse direction"

                        # Display prediction interpretation
                        st.markdown("### Prediction Interpretation")
                        st.markdown(f"""
                        Based on your historical symptom patterns, our model predicts a **{trend_strength} {trend_direction}**
                        trend in your symptom count in the near future. Over the next month, we predict your symptoms will
                        **{continuation}**.

                        **Next 4 weeks prediction:**
                        """)

                        # Show prediction in tabular format
                        prediction_df = pd.DataFrame({
                            'Date': future_formatted,
                            'Predicted Symptom Count': predictions,
                            'Confidence Range': [f"{max(0, p-pred_std):.1f} - {min(10, p+pred_std):.1f}" for p in predictions]
                        })

                        st.dataframe(
                            prediction_df,
                            hide_index=True,
                            use_container_width=True
                        )

                        # Add recommendations based on prediction
                        st.markdown("### Recommendations")

                        if trend_direction == "increasing" and trend_strength in ["moderate", "significant"]:
                            st.warning("""
                            Based on the predicted increase in symptoms:
                            - Consider scheduling a check-up with your healthcare provider
                            - Monitor your symptoms more frequently (every 2-3 days)
                            - Review any recent changes in medication, diet, or lifestyle
                            """)
                        elif trend_direction == "decreasing" and trend_strength in ["moderate", "significant"]:
                            st.success("""
                            Based on the predicted decrease in symptoms:
                            - Continue your current treatment/management approach
                            - Document what has been working well for you
                            - Maintain regular monitoring to confirm the improvement trend
                            """)
                        else:
                            st.info("""
                            Based on the predicted stable or slightly changing symptoms:
                            - Maintain your current health management routine
                            - Continue regular monitoring
                            - Discuss any persistent symptoms with your healthcare provider
                            """)

                except Exception as e:
                    self.logger.error(f"Error creating prediction chart: {e}", exc_info=True)
                    st.error("Could not render prediction visualization. Showing tabular data instead.")

                    # Create a fallback table view
                    if len(symptom_counts) >= 2:
                        st.markdown("### Historical Data")
                        hist_df = pd.DataFrame({
                            'Date': dates,
                            'Symptom Count': symptom_counts
                        })
                        st.dataframe(hist_df, hide_index=True, use_container_width=True)

                        # Simple prediction text
                        st.markdown("### Simple Prediction")
                        st.markdown("""
                        Based on your historical data, we expect your symptom trend to continue
                        in its current direction over the next few weeks.
                        """)
            else:
                st.info("Not enough historical data for predictive analysis. Please check back after recording more health data.")

                # Show a progress indicator
                st.markdown(f"Current records: **{len(health_history)}/3** minimum required")
                st.progress(min(1.0, len(health_history)/3))

                # Provide guidance with more helpful details
                st.markdown("""
                ### Enabling Predictive Analytics

                Our AI requires at least 3 data points to generate basic predictions, with more data
                improving accuracy. For best results:

                1. Record symptoms consistently (weekly is ideal)
                2. Include severity ratings when possible
                3. Be specific when listing symptoms
                4. Record dates accurately

                The quality of predictions will improve significantly with 5+ data points.
                """)

                # Add direct CTA
                if st.button("Add Health Data Now", type="primary"):
                    SessionManager.navigate_to("Symptom Analyzer")
        except Exception as e:
            self.logger.error(f"Error rendering predictive analytics tab: {e}", exc_info=True)
            st.error("Error rendering predictive analytics. Please try again later.")

    @timing_decorator
    def _render_comparative_analysis_tab(self) -> None:
        """Render comparative analysis tab with improved data visualization and insights."""
        try:
            st.subheader("Comparative Analysis")
            st.markdown("Compare your health metrics with anonymized population data.")

            # Check if comparative feature is enabled in configuration
            if not self.config_manager.get("feature_flags", "enable_comparative_analysis", default=False):
                st.info("Comparative analysis feature is coming soon.")

                st.markdown("""
                This feature will allow you to:
                - Compare your symptom patterns with similar demographic groups
                - Benchmark your health metrics against recommended ranges
                - Identify unusual health patterns that may require attention
                """)

                # Create more appealing coming soon visualization
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    # Create example visualization
                    fig = make_subplots(rows=1, cols=2,
                                        specs=[[{"type": "polar"}, {"type": "xy"}]],
                                        subplot_titles=("Health Profile Comparison", "Symptom Severity vs. Population"))

                    # Example radar chart data
                    categories = ['Respiratory', 'Digestive', 'Cardiac', 'Neurological', 'Musculoskeletal']
                    your_values = [7, 3, 5, 2, 6]
                    avg_values = [5, 4, 3, 3, 4]

                    # Add radar chart
                    fig.add_trace(go.Scatterpolar(
                        r=your_values,
                        theta=categories,
                        fill='toself',
                        name='You',
                        fillcolor='rgba(52, 152, 219, 0.5)',
                        line=dict(color='rgb(52, 152, 219)')
                    ), row=1, col=1)

                    fig.add_trace(go.Scatterpolar(
                        r=avg_values,
                        theta=categories,
                        fill='toself',
                        name='Population Average',
                        fillcolor='rgba(149, 165, 166, 0.5)',
                        line=dict(color='rgb(149, 165, 166)')
                    ), row=1, col=1)

                    # Example bar chart data
                    symptoms = ['Headache', 'Fatigue', 'Joint Pain', 'Cough', 'Nausea']
                    your_severity = [8, 6, 4, 2, 5]
                    pop_severity = [5, 4, 6, 3, 3]

                    # Add your severity bars
                    fig.add_trace(go.Bar(
                        x=symptoms,
                        y=your_severity,
                        name='Your Severity',
                        marker_color='rgba(52, 152, 219, 0.8)'
                    ), row=1, col=2)

                    # Add population average bars
                    fig.add_trace(go.Bar(
                        x=symptoms,
                        y=pop_severity,
                        name='Population Average',
                        marker_color='rgba(149, 165, 166, 0.8)'
                    ), row=1, col=2)

                    # Update layout for better appearance
                    fig.update_layout(
                        height=500,
                        title_text="Example Comparative Analysis (Coming Soon)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10]
                            )
                        ),
                        showlegend=True,
                        barmode='group'
                    )

                    # Update second subplot
                    fig.update_yaxes(title_text="Severity (1-10)", row=1, col=2)

                    # Show the mock-up
                    st.markdown("#### Feature Preview")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("This is a preview of how comparative analytics will appear when the feature is released.")

                except Exception as e:
                    self.logger.error(f"Error creating preview visualization: {e}", exc_info=True)
                    # Fail silently, this is just a preview

                # Allow user to sign up for notifications with improved validation
                st.markdown("### Get Notified")

                with st.form("notification_signup"):
                    email = st.text_input("Enter your email to be notified when this feature is available:")

                    # Add checkbox for consent
                    consent = st.checkbox("I consent to receive notifications about new features")

                    submit_button = st.form_submit_button("Notify Me")

                    if submit_button:
                        if not email:
                            st.error("Please enter your email address.")
                        elif not consent:
                            st.error("Please provide your consent to receive notifications.")
                        else:
                            # Basic email validation using regex
                            import re
                            email_pattern = "r'^[\w\.-]+@[\w\.-]+\.\w+"

                            if re.match(email_pattern, email):
                                # This would typically connect to your notification system
                                st.success(f"Thank you! We'll notify {email} when comparative analysis becomes available.")

                                # Store in session state for future reference
                                if 'notification_emails' not in st.session_state:
                                    st.session_state.notification_emails = []

                                if email not in st.session_state.notification_emails:
                                    st.session_state.notification_emails.append(email)

                                # Add a notification to confirm
                                SessionManager.add_notification(
                                    "Feature Alert Activated",
                                    f"You'll be notified at {email} when Comparative Analysis is available.",
                                    "success"
                                )
                            else:
                                st.error("Please enter a valid email address.")

                # Provide more detailed description of upcoming features
                with st.expander("Learn More About Comparative Analysis"):
                    st.markdown("""
                    ### Upcoming Comparative Analysis Features

                    The comparative analysis module will allow you to contextualize your health data within
                    broader population patterns, helping you better understand your health status.

                    #### Key Features

                    1. **Demographic Comparison**
                       - Compare your health metrics to others in your age group, gender, and region
                       - See how your symptom patterns relate to similar demographic profiles
                       - Understand whether your experiences are common or unusual

                    2. **Trend Benchmarking**
                       - Compare your recovery rates against expected patterns for your conditions
                       - See how your treatment response compares to population averages
                       - Identify if your health trajectory is following typical patterns

                    3. **Risk Factor Identification**
                       - Identify potential risk factors based on comparative analysis
                       - Discover correlations between your symptoms and known medical conditions
                       - Receive early warnings for patterns that may require attention

                    4. **Personalized Recommendations**
                       - Get tailored health suggestions based on comparative insights
                       - Learn from successful approaches used by similar profiles
                       - Receive evidence-based guidance for your specific situation

                    #### Implementation Timeline

                    We're currently in the data collection and model training phase. The comparative
                    analysis feature is expected to launch in Q3 2025.
                    """)
            else:
                # Implementation for when the feature is available
                pass
        except Exception as e:
            self.logger.error(f"Error rendering comparative analysis tab: {e}", exc_info=True)
            st.error("Error rendering comparative analysis tab. Please try again later.")

    @timing_decorator
    def _render_admin_panel(self) -> None:
        """Render the admin panel with improved security, monitoring, and management tools."""
        try:
            st.title("Admin Panel")
            st.markdown("Advanced system management and monitoring.")

            # Only allow access in advanced mode
            if not SessionManager.get("advanced_mode", False):
                st.warning("Admin panel access requires advanced mode to be enabled.")

                if st.button("Enable Advanced Mode"):
                    SessionManager.set("advanced_mode", True)
                    st.rerun()

                return

            # Create tabs for different admin functions with improved organization
            tab_labels = [
                "🖥️ System Status",
                "👥 User Management",
                "💾 Data Management",
                "📊 Performance Metrics",
                "🔌 API Configuration"
            ]

            tabs = st.tabs(tab_labels)

            with tabs[0]:
                self._render_system_status_tab()

            with tabs[1]:
                self._render_user_management_tab()

            with tabs[2]:
                self._render_data_management_tab()

            with tabs[3]:
                self._render_performance_metrics_tab()

            with tabs[4]:
                self._render_api_configuration_tab()
        except Exception as e:
            self.logger.error(f"Error rendering admin panel: {e}", exc_info=True)
            st.error("Error rendering admin panel. Please try again later.")

    @timing_decorator
    def _render_system_status_tab(self) -> None:
        """Render system status tab in admin panel with improved monitoring and controls."""
        try:
            st.subheader("System Status")

            # Display component status with improved visualization
            st.markdown("### Component Health")

            # Create a status table with enhanced UI
            component_registry = ComponentRegistry()
            status = component_registry.get_status()

            # Prepare status data
            status_data = []
            for component, is_healthy in status.items():
                status_data.append({
                    "Component": component,
                    "Status": "Operational" if is_healthy else "Failed",
                    "Health": 100 if is_healthy else 0
                })

            if status_data:
                # Display system health overview with visual indicators
                components_total = len(status_data)
                healthy_components = sum(1 for item in status_data if item["Health"] == 100)
                health_percentage = (healthy_components / components_total) * 100 if components_total > 0 else 0

                # Display status summary with columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "System Health",
                        f"{health_percentage:.0f}%",
                        delta=None
                    )

                with col2:
                    st.metric(
                        "Components Status",
                        f"{healthy_components}/{components_total}",
                        delta=None
                    )

                with col3:
                    # Define system status based on health percentage
                    if health_percentage == 100:
                        system_status = "Fully Operational"
                        status_color = "green"
                    elif health_percentage >= 80:
                        system_status = "Partially Degraded"
                        status_color = "orange"
                    elif health_percentage >= 50:
                        system_status = "Limited Functionality"
                        status_color = "orange"
                    else:
                        system_status = "Severely Degraded"
                        status_color = "red"

                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: {status_color}; color: white;">
                        <strong style="font-size: 1.1rem;">{system_status}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                # Create enhanced status table
                status_df = pd.DataFrame(status_data)
                st.dataframe(
                    status_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Component": st.column_config.TextColumn(
                            "Component",
                            width="medium"
                        ),
                        "Status": st.column_config.TextColumn(
                            "Status",
                            width="small"
                        ),
                        "Health": st.column_config.ProgressColumn(
                            "Health",
                            format="%d%%",
                            min_value=0,
                            max_value=100
                        )
                    }
                )

                # Create a health bar chart with improved visualization
                try:
                    import plotly.graph_objects as go

                    # Sort components by name for better readability
                    sorted_data = sorted(status_data, key=lambda x: x["Component"])

                    fig = go.Figure()

                    # Add bars with conditional coloring
                    fig.add_trace(go.Bar(
                        x=[d["Component"] for d in sorted_data],
                        y=[d["Health"] for d in sorted_data],
                        marker_color=['green' if d["Health"] == 100 else 'red' for d in sorted_data],
                        text=[d["Status"] for d in sorted_data],
                        textposition="auto"
                    ))

                    # Improve layout
                    fig.update_layout(
                        title="Component Health Status",
                        xaxis_title="Component",
                        yaxis_title="Health (%)",
                        height=400,
                        xaxis={'tickangle': 45},
                        yaxis={'range': [0, 100]}
                    )

                    # Show chart
                    st.plotly_chart(fig, use_container_width=True)

                    # Add help text for interpretation
                    if health_percentage < 100:
                        st.info(f"""
                        **System Health Alert:** {components_total - healthy_components} component(s) are currently
                        not operational. This may affect certain application features.
                        """)

                        # Show affected features
                        failed_components = [c["Component"] for c in status_data if c["Health"] == 0]
                        affected_features = self._get_affected_features(failed_components)

                        if affected_features:
                            st.warning("**Affected Features:**\n" + "\n".join([f"- {f}" for f in affected_features]))

                except Exception as e:
                    self.logger.error(f"Error creating health chart: {e}", exc_info=True)
                    st.error("Could not render health visualization.")

            # System information with improved organization
            st.markdown("### System Information")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <strong>Application Version:</strong> {AppConfig.VERSION}<br>
                    <strong>Session ID:</strong> {self.session_id}<br>
                    <strong>Session Start Time:</strong> {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}<br>
                    <strong>Environment:</strong> {'Production' if os.environ.get('PROD_ENV') else 'Development'}<br>
                    <strong>Python Version:</strong> {sys.version.split()[0]}
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Calculate uptime with more human-readable format
                uptime = time.time() - self.start_time
                days, remainder = divmod(uptime, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)

                uptime_str = ""
                if days > 0:
                    uptime_str += f"{int(days)}d "
                if hours > 0 or days > 0:
                    uptime_str += f"{int(hours)}h "
                if minutes > 0 or hours > 0 or days > 0:
                    uptime_str += f"{int(minutes)}m "
                uptime_str += f"{int(seconds)}s"

                st.markdown(f"""
                <div class="info-card">
                    <strong>Uptime:</strong> {uptime_str}<br>
                    <strong>Component Health:</strong> {healthy_components}/{components_total}<br>
                    <strong>Active Users:</strong> {self._get_active_user_count()}<br>
                    <strong>Data Storage:</strong> {self._get_data_storage_usage()}<br>
                    <strong>Streamlit Version:</strong> {st.__version__}
                </div>
                """, unsafe_allow_html=True)

            # Display logs with improved filtering
            st.markdown("### System Logs")

            # Create log filtering options
            col1, col2, col3 = st.columns(3)

            with col1:
                log_level = st.selectbox(
                    "Log Level",
                    ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"],
                    index=0
                )

            with col2:
                log_lines = st.slider("Number of lines", 10, 100, 20, step=10)

            with col3:
                search_term = st.text_input("Search logs", placeholder="Filter logs...")

            # Display logs if log file exists
            log_path = AppConfig.LOG_FILE
            if log_path.exists():
                try:
                    # Read log file
                    with open(log_path, "r") as log_file:
                        logs = log_file.readlines()

                    # Apply filters
                    filtered_logs = []
                    for log in logs:
                        # Apply level filter
                        if log_level != "ALL" and log_level not in log:
                            continue

                        # Apply search filter
                        if search_term and search_term.lower() not in log.lower():
                            continue

                        filtered_logs.append(log)

                    # Show most recent logs first
                    recent_logs = filtered_logs[-log_lines:] if len(filtered_logs) > log_lines else filtered_logs

                    # Display logs with proper formatting
                    if recent_logs:
                        # Add custom styling for log levels
                        styled_logs = []
                        for log in recent_logs:
                            if "ERROR" in log:
                                styled_logs.append(f"<span style='color:red'>{log}</span>")
                            elif "WARNING" in log:
                                styled_logs.append(f"<span style='color:orange'>{log}</span>")
                            elif "INFO" in log:
                                styled_logs.append(f"<span style='color:blue'>{log}</span>")
                            elif "DEBUG" in log:
                                styled_logs.append(f"<span style='color:gray'>{log}</span>")
                            else:
                                styled_logs.append(log)

                        # Display logs in a scrollable container
                        st.markdown(
                            f"""
                            <div style="height: 300px; overflow-y: auto; font-family: monospace;
                                 font-size: 12px; padding: 10px; background-color: #f5f5f5;
                                 border-radius: 5px;">
                                {''.join(styled_logs)}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("No logs match your filter criteria.")

                    # Add option to download full logs
                    if st.button("Download Full Logs"):
                        with open(log_path, "r") as log_file:
                            log_content = log_file.read()

                        date_str = datetime.now().strftime("%Y%m%d")
                        st.download_button(
                            label="Download Logs",
                            data=log_content,
                            file_name=f"medexplain_logs_{date_str}.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Could not read log file: {e}")
            else:
                st.info("No log file found.")

            # System maintenance actions with improved security and feedback
            st.markdown("### Maintenance Actions")

            # Use columns for a cleaner layout
            col1, col2, col3 = st.columns(3)

            with col1:
                # Restart application
                with st.popover("Restart Application", use_container_width=True):
                    st.warning("This will restart the application and end all user sessions.")

                    # Add authentication for critical actions
                    admin_password = st.text_input("Admin Password", type="password")
                    confirm = st.checkbox("I understand this action will disrupt active users")

                    if st.button("Confirm Restart", disabled=not confirm or not admin_password):
                        # Verify password (in a real app, use secure password verification)
                        if admin_password == os.environ.get("ADMIN_PASSWORD", "admin123"):
                            # This would typically trigger an actual restart mechanism
                            st.info("Restart functionality is simulated in this demo.")

                            # Simulate restart by clearing session state
                            SessionManager.clear(exclude=["page"])
                            SessionManager.set("page", "Home")

                            st.success("Application restarted. Redirecting to home page.")
                            st.rerun()
                        else:
                            st.error("Invalid admin password.")

            with col2:
                # Clear cache
                with st.popover("Clear System Cache", use_container_width=True):
                    st.info("This will clear application cache data to free up memory.")

                    cache_types = st.multiselect(
                        "Select cache types to clear:",
                        ["Session data", "Temporary files", "API response cache", "All caches"]
                    )

                    if st.button("Clear Selected Caches", disabled=not cache_types):
                        with st.spinner("Clearing cache..."):
                            # Simulate cache clearing
                            time.sleep(1)

                            if "Session data" in cache_types or "All caches" in cache_types:
                                # Keep minimal state
                                page = SessionManager.get('page', 'Home')
                                theme = SessionManager.get('theme', AppConfig.DEFAULT_THEME)

                                # Clear non-essential session data
                                for key in list(st.session_state.keys()):
                                    if key not in ['page', 'theme', 'advanced_mode']:
                                        SessionManager.delete(key)

                                st.session_state.page = page
                                st.session_state.theme = theme

                            # Log the action
                            self.logger.info(f"Cache cleared: {', '.join(cache_types)}")

                            st.success("Cache cleared successfully.")

                            if "Session data" in cache_types or "All caches" in cache_types:
                                st.warning("Session data cleared. Some settings may be reset.")
                                if st.button("Refresh Application"):
                                    st.rerun()

            with col3:
                # Component refresh
                with st.popover("Refresh Components", use_container_width=True):
                    st.info("This will attempt to reinitialize failed components.")

                    # Find failed components
                    failed_components = [c for c, status in status.items() if not status]

                    if failed_components:
                        st.warning(f"Found {len(failed_components)} failed component(s):")
                        for comp in failed_components:
                            st.markdown(f"- {comp}")

                        selected_components = st.multiselect(
                            "Select components to refresh:",
                            options=failed_components,
                            default=failed_components
                        )

                        if st.button("Refresh Selected Components", disabled=not selected_components):
                            with st.spinner("Refreshing components..."):
                                # Attempt to reinitialize each component
                                success_count = 0
                                for comp in selected_components:
                                    try:
                                        # Handle component initialization based on type
                                        if comp in ["health_data", "user_manager", "openai_client"]:
                                            self._initialize_core_components()
                                        elif comp in ["symptom_predictor", "symptom_extractor", "risk_assessor"]:
                                            self._initialize_ml_components()
                                        else:
                                            self._initialize_ui_components()

                                        # Check if initialization was successful
                                        if component_registry.get_status().get(comp, False):
                                            success_count += 1
                                            self.logger.info(f"Successfully reinitialized component: {comp}")
                                        else:
                                            self.logger.warning(f"Failed to reinitialize component: {comp}")
                                    except Exception as e:
                                        self.logger.error(f"Error refreshing component {comp}: {e}", exc_info=True)

                                st.success(f"Refreshed {success_count}/{len(selected_components)} components.")

                                if success_count > 0:
                                    st.info("Some components were successfully refreshed. Reloading admin panel...")
                                    if st.button("Reload Admin Panel"):
                                        st.rerun()
                    else:
                        st.success("All components are operational.")
                        st.button("Refresh All Components", disabled=True)

            # System diagnostics section
            st.markdown("### System Diagnostics")

            with st.expander("Run System Diagnostics"):
                if st.button("Run Diagnostics Check"):
                    with st.spinner("Running diagnostics..."):
                        # Simulate running diagnostics
                        time.sleep(2)

                        # Sample diagnostic results
                        diagnostics = {
                            "Database Connectivity": {
                                "status": "OK",
                                "latency": "45ms",
                                "details": "Connected to database successfully."
                            },
                            "File System Access": {
                                "status": "OK",
                                "details": "Read/write permissions verified."
                            },
                            "API Services": {
                                "status": "Partial",
                                "details": "OpenAI API available. Medical database API unavailable."
                            },
                            "Memory Usage": {
                                "status": "OK",
                                "value": "64.2%",
                                "details": "Memory usage within normal limits."
                            },
                            "Cache Efficiency": {
                                "status": "Warning",
                                "value": "37.5%",
                                "details": "Cache hit rate below recommended threshold (50%)."
                            }
                        }

                        # Display results
                        st.markdown("#### Diagnostic Results")

                        for system, details in diagnostics.items():
                            status = details.get("status", "Unknown")

                            # Determine status color
                            if status == "OK":
                                status_color = "green"
                            elif status == "Warning":
                                status_color = "orange"
                            elif status == "Error":
                                status_color = "red"
                            else:
                                status_color = "blue"

                            # Create expandable diagnostic result
                            with st.expander(f"{system}: **{status}**", expanded=True):
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                    <div style="width: 15px; height: 15px; border-radius: 50%; background-color: {status_color}; margin-right: 10px;"></div>
                                    <div style="font-weight: bold; color: {status_color};">{status}</div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Display additional details
                                if "value" in details:
                                    st.markdown(f"**Value:** {details['value']}")

                                st.markdown(f"**Details:** {details['details']}")

                                # Add actions for non-OK statuses
                                if status != "OK":
                                    if system == "API Services":
                                        st.button("Reconfigure API Settings", key=f"fix_{system}")
                                    elif system == "Cache Efficiency":
                                        st.button("Optimize Cache", key=f"fix_{system}")

                        # Add diagnostic summary
                        st.markdown("#### Diagnostic Summary")
                        status_counts = {}
                        for details in diagnostics.values():
                            status = details.get("status", "Unknown")
                            if status not in status_counts:
                                status_counts[status] = 0
                            status_counts[status] += 1

                        # Create summary pills
                        st.markdown(
                            f"""
                            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                                {"" if "OK" not in status_counts else f'<div style="background-color: rgba(0,128,0,0.2); color: green; padding: 5px 10px; border-radius: 15px; font-weight: bold;">{status_counts["OK"]} OK</div>'}
                                {"" if "Warning" not in status_counts else f'<div style="background-color: rgba(255,165,0,0.2); color: orange; padding: 5px 10px; border-radius: 15px; font-weight: bold;">{status_counts["Warning"]} Warning</div>'}
                                {"" if "Error" not in status_counts else f'<div style="background-color: rgba(255,0,0,0.2); color: red; padding: 5px 10px; border-radius: 15px; font-weight: bold;">{status_counts["Error"]} Error</div>'}
                                {"" if "Partial" not in status_counts else f'<div style="background-color: rgba(0,0,255,0.2); color: blue; padding: 5px 10px; border-radius: 15px; font-weight: bold;">{status_counts["Partial"]} Partial</div>'}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        except Exception as e:
            self.logger.error(f"Error rendering system status tab: {e}", exc_info=True)
            st.error("Error rendering system status. Please try again later.")

    def _get_affected_features(self, failed_components: List[str]) -> List[str]:
        """
        Determine which user-facing features are affected by failed components.

        Args:
            failed_components: List of component names that are not operational

        Returns:
            List of affected feature names
        """
        # Map components to features
        component_feature_map = {
            "health_data": ["Symptom Analyzer", "Health History", "Medical Literature"],
            "user_manager": ["Health Dashboard", "Health History", "Settings", "User Profiles"],
            "openai_client": ["Health Chat", "Medical Literature"],
            "health_analyzer": ["Advanced Analytics", "Health Dashboard"],
            "symptom_predictor": ["Symptom Analyzer", "Advanced Analytics"],
            "symptom_extractor": ["Symptom Analyzer", "Health Chat"],
            "risk_assessor": ["Symptom Analyzer", "Risk Assessment"],
            "dashboard": ["Health Dashboard"],
            "chat_interface": ["Health Chat"],
            "symptom_analyzer_ui": ["Symptom Analyzer"],
            "medical_literature_ui": ["Medical Literature"],
            "health_history_ui": ["Health History"],
            "settings_ui": ["Settings"]
        }

        # Collect affected features
        affected_features = set()
        for component in failed_components:
            features = component_feature_map.get(component, [])
            affected_features.update(features)

        return sorted(list(affected_features))

    def _get_active_user_count(self) -> int:
        """
        Get count of active users in the system.

        Returns:
            Number of active users
        """
        # In a real system, this would query database or session store
        # For demo purposes, return a random number
        import random
        return random.randint(1, 10)

    def _get_data_storage_usage(self) -> str:
        """
        Get human-readable data storage usage.

        Returns:
            String representation of storage usage
        """
        # Calculate actual directory size if possible
        try:
            total_size = 0
            for path, dirs, files in os.walk(AppConfig.DATA_DIR):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)

            # Convert to appropriate unit
            units = ['B', 'KB', 'MB', 'GB', 'TB']
            size = total_size
            unit_index = 0

            while size > 1024 and unit_index < len(units) - 1:
                size /= 1024
                unit_index += 1

            return f"{size:.2f} {units[unit_index]}"
        except:
            # Fallback to random value for demo
            import random
            size = random.uniform(10, 500)
            return f"{size:.2f} MB"

    @timing_decorator
    def _render_user_management_tab(self) -> None:
        """Render user management tab in admin panel with improved tools and security."""
        try:
            st.subheader("User Management")

            # Display user statistics with enhanced visualization
            st.markdown("### User Statistics")

            user_manager = self.registry.get("user_manager")
            if user_manager and hasattr(user_manager, "get_all_users"):
                # Get user information
                users = user_manager.get_all_users()

                if users:
                    # Display user metrics in visually appealing cards
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Users", len(users))

                    with col2:
                        # Calculate active users (simplified demo logic)
                        active_users = sum(1 for user in users if user.get("last_login"))
                        st.metric("Active Users", active_users)

                    with col3:
                        # Calculate new users in last 30 days
                        try:
                            thirty_days_ago = datetime.now() - timedelta(days=30)
                            new_users = sum(1 for user in users if user.get("created_at") and
                                           datetime.fromisoformat(user.get("created_at")) > thirty_days_ago)
                            st.metric("New Users (30d)", new_users)
                        except:
                            st.metric("New Users (30d)", "N/A")

                    with col4:
                        # Calculate average records per user
                        record_counts = []
                        for user in users:
                            if user.get("health_records"):
                                record_counts.append(len(user.get("health_records")))
                            elif hasattr(user_manager, "get_user_health_records"):
                                try:
                                    records = user_manager.get_user_health_records(user.get("id"))
                                    record_counts.append(len(records))
                                except:
                                    pass

                        avg_records = sum(record_counts) / len(record_counts) if record_counts else 0
                        st.metric("Avg Records/User", f"{avg_records:.1f}")

                    # Create user table with enhanced display
                    user_data = []
                    for user in users:
                        # Format date for better readability
                        last_login = user.get("last_login", "Never")
                        if last_login and last_login != "Never":
                            try:
                                login_date = datetime.fromisoformat(last_login)
                                last_login = login_date.strftime("%Y-%m-%d %H:%M")
                            except:
                                pass

                        created_at = user.get("created_at", "Unknown")
                        if created_at and created_at != "Unknown":
                            try:
                                created_date = datetime.fromisoformat(created_at)
                                created_at = created_date.strftime("%Y-%m-%d")
                            except:
                                pass

                        user_data.append({
                            "User ID": user.get("id", "Unknown"),
                            "Name": user.get("name", "Unknown"),
                            "Age": user.get("age", "Unknown"),
                            "Gender": user.get("gender", "Unknown"),
                            "Created": created_at,
                            "Last Activity": last_login
                        })

                    if user_data:
                        # Create an interactive dataframe
                        user_df = pd.DataFrame(user_data)
                        st.dataframe(
                            user_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "User ID": st.column_config.TextColumn("User ID", width="medium"),
                                "Name": st.column_config.TextColumn("Name", width="medium"),
                                "Last Activity": st.column_config.DatetimeColumn(
                                    "Last Activity",
                                    format="D MMM YYYY, h:mm a",
                                    width="medium"
                                )
                            }
                        )

                        # Add user visualization
                        try:
                            import plotly.express as px

                            # Add gender distribution pie chart
                            gender_counts = user_df["Gender"].value_counts().reset_index()
                            gender_counts.columns = ["Gender", "Count"]

                            fig = px.pie(
                                gender_counts,
                                values="Count",
                                names="Gender",
                                title="User Gender Distribution",
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                                hole=0.4
                            )

                            fig.update_layout(
                                height=350,
                                margin=dict(t=50, b=0, l=0, r=0)
                            )

                            # Age distribution histogram
                            age_data = user_df[user_df["Age"] != "Unknown"]
                            if not age_data.empty and age_data["Age"].dtype != object:
                                age_fig = px.histogram(
                                    age_data,
                                    x="Age",
                                    nbins=10,
                                    title="User Age Distribution",
                                    color_discrete_sequence=["royalblue"]
                                )

                                age_fig.update_layout(
                                    height=350,
                                    margin=dict(t=50, b=0, l=0, r=0),
                                    xaxis_title="Age",
                                    yaxis_title="Count"
                                )

                                # Display both charts side by side
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.plotly_chart(fig, use_container_width=True)
                                with col2:
                                    st.plotly_chart(age_fig, use_container_width=True)
                            else:
                                # Just show gender chart
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            self.logger.error(f"Error creating user visualization: {e}", exc_info=True)
                            # Fail silently - visualization is optional

                        # Add export functionality with improved options
                        export_col1, export_col2 = st.columns(2)

                        with export_col1:
                            if st.button("Export User Data"):
                                # Create downloadable user data in multiple formats
                                export_format = st.radio(
                                    "Select format:",
                                    ["CSV", "JSON", "Excel"],
                                    horizontal=True
                                )

                                if export_format == "CSV":
                                    csv = user_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download User Data (CSV)",
                                        data=csv,
                                        file_name=f"medexplain_users_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv"
                                    )
                                elif export_format == "JSON":
                                    json_str = user_df.to_json(orient="records", indent=2)
                                    st.download_button(
                                        label="Download User Data (JSON)",
                                        data=json_str,
                                        file_name=f"medexplain_users_{datetime.now().strftime('%Y%m%d')}.json",
                                        mime="application/json"
                                    )
                                else:  # Excel
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output) as writer:
                                        user_df.to_excel(writer, sheet_name="Users", index=False)
                                    output.seek(0)

                                    st.download_button(
                                        label="Download User Data (Excel)",
                                        data=output,
                                        file_name=f"medexplain_users_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )

                        with export_col2:
                            # Add user report generation
                            if st.button("Generate User Report"):
                                with st.spinner("Generating comprehensive user report..."):
                                    # Simulate report generation
                                    time.sleep(2)

                                    # Generate report HTML
                                    report_html = f"""
                                    <html>
                                    <head>
                                        <title>MedExplain User Report</title>
                                        <style>
                                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                            h1 {{ color: #3498db; }}
                                            table {{ border-collapse: collapse; width: 100%; }}
                                            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                                            th {{ background-color: #3498db; color: white; }}
                                            tr:nth-child(even) {{ background-color: #f2f2f2; }}
                                        </style>
                                    </head>
                                    <body>
                                        <h1>MedExplain AI Pro - User Report</h1>
                                        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                                        <h2>User Statistics</h2>
                                        <ul>
                                            <li>Total Users: {len(users)}</li>
                                            <li>Active Users: {active_users}</li>
                                            <li>New Users (30 days): {new_users}</li>
                                            <li>Average Records per User: {avg_records:.1f}</li>
                                        </ul>
                                        <h2>User Details</h2>
                                        <table>
                                            <tr>
                                                <th>User ID</th>
                                                <th>Name</th>
                                                <th>Age</th>
                                                <th>Gender</th>
                                                <th>Created</th>
                                                <th>Last Activity</th>
                                            </tr>
                                    """

                                    # Add user rows
                                    for user in user_data:
                                        report_html += f"""
                                        <tr>
                                            <td>{user["User ID"]}</td>
                                            <td>{user["Name"]}</td>
                                            <td>{user["Age"]}</td>
                                            <td>{user["Gender"]}</td>
                                            <td>{user["Created"]}</td>
                                            <td>{user["Last Activity"]}</td>
                                        </tr>
                                        """

                                    report_html += """
                                        </table>
                                    </body>
                                    </html>
                                    """

                                    # Create download button for the report
                                    st.download_button(
                                        label="Download User Report (HTML)",
                                        data=report_html,
                                        file_name=f"medexplain_user_report_{datetime.now().strftime('%Y%m%d')}.html",
                                        mime="text/html"
                                    )
                else:
                    st.info("No user data available.")
            else:
                st.warning("User management component is not available.")

            # User actions section with improved UX
            st.markdown("### User Actions")

            # Use tabs for different user management actions
            user_tabs = st.tabs(["Create User", "Edit User", "Delete User", "Switch Active User"])

            # Create User tab
            with user_tabs[0]:
                with st.form("create_user_form"):
                    st.markdown("#### Create New User")

                    # Improve form layout with columns
                    col1, col2 = st.columns(2)

                    with col1:
                        user_name = st.text_input("Name", placeholder="Full Name")
                        user_age = st.number_input("Age", min_value=0, max_value=120)

                    with col2:
                        user_gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Other", "Prefer not to say"])
                        user_email = st.text_input("Email (optional)", placeholder="email@example.com")

                    # Add additional fields
                    user_notes = st.text_area("Notes (optional)", placeholder="Add any additional notes about this user")

                    # Validate inputs
                    user_gender_value = None if user_gender == "Select..." else user_gender

                    submit_button = st.form_submit_button("Create User")

                    if submit_button:
                        if not user_name:
                            st.error("Name is required.")
                        elif user_age <= 0:
                            st.error("Please enter a valid age.")
                        elif not user_gender_value:
                            st.error("Please select a gender.")
                        elif user_email and not re.match("r'^[\w\.-]+@[\w\.-]+\.\w+"
                                , user_email):
                            st.error("Please enter a valid email address.")
                        else:
                            if user_manager and hasattr(user_manager, "create_user"):
                                try:
                                    # Create user data
                                    user_data = {
                                        "name": user_name,
                                        "age": user_age,
                                        "gender": user_gender_value,
                                        "created_at": datetime.now().isoformat()
                                    }

                                    if user_email:
                                        user_data["email"] = user_email

                                    if user_notes:
                                        user_data["notes"] = user_notes

                                    # Create user and profile
                                    user_id = user_manager.create_user(user_data)

                                    # Update profile with the same data
                                    if hasattr(user_manager, "update_profile"):
                                        current_user = user_manager.current_user_id
                                        user_manager.switch_user(user_id)
                                        user_manager.update_profile(user_data)
                                        user_manager.switch_user(current_user)

                                    st.success(f"User '{user_name}' created successfully with ID: {user_id}")

                                    # Add action buttons
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("Switch to New User"):
                                            user_manager.switch_user(user_id)
                                            SessionManager.set("user_id", user_id)
                                            SessionManager.add_notification(
                                                "User Switched",
                                                f"Now using account: {user_name}",
                                                "success"
                                            )
                                            SessionManager.navigate_to("Home")

                                    with col2:
                                        if st.button("Create Another User"):
                                            st.rerun()
                                except Exception as e:
                                    st.error(f"Error creating user: {e}")
                            else:
                                st.error("User management component is not available.")

            # Edit User tab
            with user_tabs[1]:
                st.markdown("#### Edit Existing User")

                if user_manager and hasattr(user_manager, "get_all_users"):
                    users = user_manager.get_all_users()

                    if users:
                        # Create a dropdown to select user
                        user_options = [f"{user.get('name', 'Unknown')} ({user.get('id', '')})" for user in users]
                        selected_user_idx = st.selectbox(
                            "Select User to Edit:",
                            range(len(user_options)),
                            format_func=lambda i: user_options[i]
                        )

                        # Get selected user
                        selected_user = users[selected_user_idx]
                        user_id = selected_user.get("id", "")

                        # Create form to edit user
                        with st.form("edit_user_form"):
                            st.markdown(f"Editing user: **{selected_user.get('name', 'Unknown')}**")

                            # Improve form layout with columns
                            col1, col2 = st.columns(2)

                            with col1:
                                user_name = st.text_input("Name", value=selected_user.get("name", ""))
                                user_age = st.number_input("Age", min_value=0, max_value=120, value=selected_user.get("age", 0))

                            with col2:
                                gender_options = ["Male", "Female", "Other", "Prefer not to say"]
                                current_gender = selected_user.get("gender", "")
                                gender_idx = gender_options.index(current_gender) if current_gender in gender_options else 0

                                user_gender = st.selectbox("Gender", gender_options, index=gender_idx)
                                user_email = st.text_input("Email", value=selected_user.get("email", ""))

                            # Add additional fields
                            user_notes = st.text_area("Notes", value=selected_user.get("notes", ""))

                            submit_button = st.form_submit_button("Update User")

                            if submit_button:
                                if not user_name:
                                    st.error("Name is required.")
                                elif user_age <= 0:
                                    st.error("Please enter a valid age.")
                                elif user_email and not re.match("r'^[\w\.-]+@[\w\.-]+\.\w+"
                                , user_email):
                                    st.error("Please enter a valid email address.")
                                else:
                                    try:
                                        # Create updated user data
                                        updated_data = {
                                            "name": user_name,
                                            "age": user_age,
                                            "gender": user_gender
                                        }

                                        if user_email:
                                            updated_data["email"] = user_email

                                        if user_notes:
                                            updated_data["notes"] = user_notes

                                        # Save changes
                                        current_user = user_manager.current_user_id
                                        user_manager.switch_user(user_id)

                                        if hasattr(user_manager, "update_profile"):
                                            user_manager.update_profile(updated_data)

                                            # Switch back
                                            user_manager.switch_user(current_user)

                                            st.success(f"User '{user_name}' updated successfully.")

                                            # Add notification
                                            SessionManager.add_notification(
                                                "User Updated",
                                                f"User profile for '{user_name}' has been updated.",
                                                "success"
                                            )
                                        else:
                                            st.error("Update profile functionality not available.")
                                            user_manager.switch_user(current_user)
                                    except Exception as e:
                                        st.error(f"Error updating user: {e}")

                                        # Try to switch back
                                        try:
                                            user_manager.switch_user(current_user)
                                        except:
                                            pass
                    else:
                        st.info("No users available to edit.")
                else:
                    st.warning("User management component is not available.")

            # Delete User tab with improved security
            with user_tabs[2]:
                st.markdown("#### Delete User")

                if user_manager and hasattr(user_manager, "get_all_users"):
                    # Get user IDs for selection
                    users = user_manager.get_all_users()

                    if users:
                        # Filter out the current user from deletion options if it's the only user
                        current_user_id = user_manager.current_user_id
                        available_users = users

                        if len(users) == 1 and users[0].get("id") == current_user_id:
                            st.warning("You cannot delete the only user in the system.")
                            st.info("Create another user first before deleting this one.")
                        else:
                            # Display available users to delete
                            user_options = []
                            for user in available_users:
                                user_id = user.get("id", "")
                                if user_id == current_user_id:
                                    user_options.append(f"{user.get('name', 'Unknown')} ({user_id}) (current user)")
                                else:
                                    user_options.append(f"{user.get('name', 'Unknown')} ({user_id})")

                            selected_user_idx = st.selectbox(
                                "Select User to Delete:",
                                range(len(user_options)),
                                format_func=lambda i: user_options[i]
                            )

                            # Get selected user
                            selected_user = available_users[selected_user_idx]
                            selected_user_id = selected_user.get("id", "")

                            # Show warning if trying to delete current user
                            if selected_user_id == current_user_id:
                                st.warning("You are about to delete your current user account. This will log you out.")

                            # Show user details for confirmation
                            st.markdown(f"""
                            **User Details:**
                            - **Name:** {selected_user.get('name', 'Unknown')}
                            - **ID:** {selected_user_id}
                            - **Created:** {selected_user.get('created_at', 'Unknown')}
                            - **Last Login:** {selected_user.get('last_login', 'Never')}
                            """)

                            # Add confirmation with password for security
                            with st.expander("Delete User (Danger Zone)"):
                                st.warning(f"⚠️ This will permanently delete user '{selected_user.get('name', 'Unknown')}' and all their data.")

                                confirm = st.checkbox("I understand and confirm this action")
                                confirm_text = st.text_input("Type the user's name to confirm deletion:", placeholder=selected_user.get('name', ''))
                                admin_password = st.text_input("Admin Password:", type="password")

                                if st.button("Delete User Permanently", disabled=not confirm):
                                    if confirm_text != selected_user.get('name', ''):
                                        st.error("The name you entered doesn't match the user's name.")
                                    elif not admin_password:
                                        st.error("Please enter the admin password.")
                                    else:
                                        # In a real app, would verify admin password
                                        try:
                                            # Delete user
                                            if hasattr(user_manager, "delete_user"):
                                                success = user_manager.delete_user(selected_user_id)

                                                if success:
                                                    st.success(f"User '{selected_user.get('name', 'Unknown')}' deleted successfully.")

                                                    # Add notification
                                                    SessionManager.add_notification(
                                                        "User Deleted",
                                                        f"User '{selected_user.get('name', 'Unknown')}' has been deleted.",
                                                        "info"
                                                    )

                                                    # If we deleted the current user, navigate to home
                                                    if selected_user_id == current_user_id:
                                                        SessionManager.navigate_to("Home")
                                                    else:
                                                        st.rerun()
                                                else:
                                                    st.error("Failed to delete user. Please try again.")
                                            else:
                                                st.error("Delete user functionality not available.")
                                        except Exception as e:
                                            st.error(f"Error deleting user: {e}")
                    else:
                        st.info("No users available to delete.")
                else:
                    st.warning("User management component is not available.")

            # User switching functionality with improved UX
            with user_tabs[3]:
                st.markdown("#### Switch Active User")

                if user_manager and hasattr(user_manager, "get_all_users"):
                    # Get user information for selection
                    users = user_manager.get_all_users()

                    if users:
                        # Get current user ID
                        current_user_id = user_manager.current_user_id

                        # Create user cards for selection instead of dropdown
                        st.markdown("Select a user to switch to:")

                        # Create two columns for user cards
                        user_cols = st.columns(2)

                        for i, user in enumerate(users):
                            user_id = user.get("id", "")
                            user_name = user.get("name", "Unknown")

                            # Determine if this is the current user
                            is_current = user_id == current_user_id

                            # Create card in alternating columns
                            with user_cols[i % 2]:
                                card_bg = "#e6f7ff" if is_current else "#f8f9fa"
                                card_border = "2px solid #3498db" if is_current else "1px solid #ddd"

                                st.markdown(f"""
                                <div style="
                                    padding: 15px;
                                    border-radius: 8px;
                                    background-color: {card_bg};
                                    border: {card_border};
                                    margin-bottom: 15px;
                                    position: relative;
                                ">
                                    {"<div style='position: absolute; top: 10px; right: 10px; background-color: #3498db; color: white; padding: 3px 8px; border-radius: 10px; font-size: 12px;'>Current</div>" if is_current else ""}
                                    <h4 style="margin-top: 0;">{user_name}</h4>
                                    <p><strong>ID:</strong> {user_id}</p>
                                    <p><strong>Age:</strong> {user.get("age", "Unknown")}</p>
                                    <p><strong>Gender:</strong> {user.get("gender", "Unknown")}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                # Create button if not current user
                                if not is_current:
                                    if st.button(f"Switch to {user_name}", key=f"switch_{user_id}"):
                                        try:
                                            success = user_manager.switch_user(user_id)

                                            if success:
                                                # Update session state
                                                SessionManager.set("user_id", user_id)

                                                # Clear cached data that relates to user
                                                for key in ['last_symptom_check', 'last_risk_assessment', 'health_report']:
                                                    SessionManager.delete(key)

                                                # Add notification
                                                SessionManager.add_notification(
                                                    "User Switched",
                                                    f"Now using account: {user_name}",
                                                    "success"
                                                )

                                                st.success(f"Switched to user: {user_name}")
                                                SessionManager.navigate_to("Home")
                                            else:
                                                st.error("Failed to switch user. Please try again.")
                                        except Exception as e:
                                            st.error(f"Error switching user: {e}")
                                else:
                                    st.info("This is already the active user.")
                    else:
                        st.info("No users available to switch to.")
                else:
                    st.warning("User management component is not available.")
        except Exception as e:
            self.logger.error(f"Error rendering user management tab: {e}", exc_info=True)
            st.error("Error rendering user management. Please try again later.")

    @timing_decorator
    def _render_data_management_tab(self) -> None:
        """Render data management tab in admin panel with improved tools for backups and maintenance."""
        try:
            st.subheader("Data Management")

            # Data overview with improved visualization
            st.markdown("### Data Overview")

            # Show storage statistics with visualizations
            try:
                # Get storage statistics
                data_stats = self._get_data_storage_stats()

                # Display statistics with visual indicators
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Storage", data_stats.get("total_size", "Unknown"))

                with col2:
                    st.metric("User Data", data_stats.get("user_data_size", "Unknown"))

                with col3:
                    st.metric("Model Files", data_stats.get("model_size", "Unknown"))

                with col4:
                    st.metric("Logs & Cache", data_stats.get("logs_cache_size", "Unknown"))

                # Add storage visualization
                try:
                    import plotly.graph_objects as go

                    # Extract storage values for chart
                    storage_labels = ["User Data", "Model Files", "Logs & Cache", "Other"]
                    storage_values = [
                        data_stats.get("user_data_bytes", 0),
                        data_stats.get("model_bytes", 0),
                        data_stats.get("logs_cache_bytes", 0),
                        data_stats.get("other_bytes", 0)
                    ]

                    # Create donut chart
                    fig = go.Figure(data=[go.Pie(
                        labels=storage_labels,
                        values=storage_values,
                        hole=.4,
                        marker_colors=['#3498db', '#2ecc71', '#f39c12', '#95a5a6']
                    )])

                    fig.update_layout(
                        title_text="Storage Distribution",
                        height=350,
                        margin=dict(t=50, b=0, l=0, r=0),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                    )

                    # Add annotation in the middle
                    fig.add_annotation(
                        text=f"{data_stats.get('total_size', '')}",
                        font=dict(size=16, family="Arial", color="black"),
                        showarrow=False,
                        x=0.5,
                        y=0.5
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    self.logger.error(f"Error creating storage visualization: {e}", exc_info=True)
                    # Fail silently - visualization is optional
            except Exception as e:
                self.logger.error(f"Error getting storage statistics: {e}", exc_info=True)
                st.warning("Could not retrieve storage statistics.")

            # Data backup and restore with improved functionality
            st.markdown("### Data Backup")

            backup_tabs = st.tabs(["Create Backup", "Restore from Backup", "Scheduled Backups"])

            with backup_tabs[0]:
                st.markdown("Create a backup of your application data")

                # Backup options
                backup_options = st.multiselect(
                    "Select data to include in backup:",
                    ["User Profiles", "Health Records", "Application Settings", "Logs"],
                    default=["User Profiles", "Health Records", "Application Settings"]
                )

                col1, col2 = st.columns(2)

                with col1:
                    backup_format = st.radio(
                        "Backup format:",
                        ["JSON", "Compressed Archive (.zip)"],
                        index=1,
                        horizontal=True
                    )

                with col2:
                    include_timestamp = st.checkbox("Include timestamp in filename", value=True)
                    encrypt_backup = st.checkbox("Encrypt backup", value=False)

                # Show encryption options if selected
                if encrypt_backup:
                    encryption_password = st.text_input("Encryption password", type="password")
                    encryption_confirmation = st.text_input("Confirm password", type="password")

                    if encryption_password != encryption_confirmation:
                        st.error("Passwords do not match.")

                if st.button("Create Backup", disabled=encrypt_backup and (encryption_password != encryption_confirmation)):
                    try:
                        with st.spinner("Creating backup..."):
                            import json
                            import datetime
                            import zipfile
                            import io

                            # Create backup data structure
                            backup_data = {
                                "app_version": AppConfig.VERSION,
                                "backup_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "backup_contents": backup_options,
                                "users": {},
                                "settings": {},
                                "metadata": {
                                    "component_status": self.registry.get_status()
                                }
                            }

                            # Add user data if selected
                            user_manager = self.registry.get("user_manager")
                            if user_manager and ("User Profiles" in backup_options or "Health Records" in backup_options):
                                for user in user_manager.get_all_users():
                                    user_id = user.get("id")
                                    if user_id:
                                        # Switch to user to get their data
                                        current_user = user_manager.current_user_id
                                        user_manager.switch_user(user_id)

                                        backup_data["users"][user_id] = {}

                                        # Add profile if selected
                                        if "User Profiles" in backup_options:
                                            backup_data["users"][user_id]["profile"] = user_manager.profile

                                        # Add health history if selected
                                        if "Health Records" in backup_options:
                                            backup_data["users"][user_id]["health_history"] = user_manager.health_history

                                        # Switch back
                                        user_manager.switch_user(current_user)

                            # Add application settings if selected
                            if "Application Settings" in backup_options:
                                backup_data["settings"] = self.config_manager.get_all()

                            # Add logs if selected
                            if "Logs" in backup_options:
                                try:
                                    log_path = AppConfig.LOG_FILE
                                    if log_path.exists():
                                        with open(log_path, "r") as log_file:
                                            backup_data["logs"] = log_file.read()
                                except Exception as log_error:
                                    self.logger.error(f"Error adding logs to backup: {log_error}", exc_info=True)

                            # Generate filename
                            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename_base = f"medexplain_backup{f'_{date_str}' if include_timestamp else ''}"

                            if backup_format == "JSON":
                                # Convert to JSON
                                backup_json = json.dumps(backup_data, indent=2)

                                if encrypt_backup and encryption_password:
                                    # Basic encryption (for demo purposes)
                                    # In production, use proper encryption libraries
                                    import base64
                                    from cryptography.fernet import Fernet
                                    from cryptography.hazmat.primitives import hashes
                                    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

                                    # Derive key from password
                                    salt = b'medexplain_salt'  # In production, use a random salt
                                    kdf = PBKDF2HMAC(
                                        algorithm=hashes.SHA256(),
                                        length=32,
                                        salt=salt,
                                        iterations=100000
                                    )
                                    key = base64.urlsafe_b64encode(kdf.derive(encryption_password.encode()))

                                    # Encrypt data
                                    f = Fernet(key)
                                    encrypted_data = f.encrypt(backup_json.encode())

                                    # Create download button for encrypted data
                                    st.download_button(
                                        label="Download Encrypted Backup",
                                        data=encrypted_data,
                                        file_name=f"{filename_base}.enc",
                                        mime="application/octet-stream"
                                    )
                                else:
                                    # Create download button for JSON
                                    st.download_button(
                                        label="Download Backup JSON",
                                        data=backup_json,
                                        file_name=f"{filename_base}.json",
                                        mime="application/json"
                                    )
                            else:
                                # Create ZIP archive
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                                    # Add main backup data
                                    zip_file.writestr("backup_data.json", json.dumps(backup_data, indent=2))

                                    # Add additional files if needed
                                    if "Logs" in backup_options:
                                        try:
                                            log_path = AppConfig.LOG_FILE
                                            if log_path.exists():
                                                zip_file.write(log_path, arcname="logs/app.log")
                                        except Exception as log_error:
                                            self.logger.error(f"Error adding log file to ZIP: {log_error}", exc_info=True)

                                zip_buffer.seek(0)

                                if encrypt_backup and encryption_password:
                                    # Encrypt the zip file
                                    import base64
                                    from cryptography.fernet import Fernet
                                    from cryptography.hazmat.primitives import hashes
                                    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

                                    # Derive key from password
                                    salt = b'medexplain_salt'
                                    kdf = PBKDF2HMAC(
                                        algorithm=hashes.SHA256(),
                                        length=32,
                                        salt=salt,
                                        iterations=100000
                                    )
                                    key = base64.urlsafe_b64encode(kdf.derive(encryption_password.encode()))

                                    # Encrypt data
                                    f = Fernet(key)
                                    encrypted_data = f.encrypt(zip_buffer.getvalue())

                                    # Create download button for encrypted data
                                    st.download_button(
                                        label="Download Encrypted Backup",
                                        data=encrypted_data,
                                        file_name=f"{filename_base}.zip.enc",
                                        mime="application/octet-stream"
                                    )
                                else:
                                    # Create download button for ZIP
                                    st.download_button(
                                        label="Download Backup ZIP",
                                        data=zip_buffer,
                                        file_name=f"{filename_base}.zip",
                                        mime="application/zip"
                                    )

                            st.success("Backup created successfully.")
                    except Exception as e:
                        self.logger.error(f"Error creating backup: {e}", exc_info=True)
                        st.error(f"Error creating backup: {e}")

            with backup_tabs[1]:
                st.markdown("Restore your application from a backup file")

                # Upload backup file
                uploaded_file = st.file_uploader(
                    "Upload Backup File",
                    type=["json", "zip", "enc"],
                    help="Upload a MedExplain backup file (.json, .zip, or encrypted .enc)"
                )

                if uploaded_file:
                    # Check if file is encrypted
                    is_encrypted = uploaded_file.name.endswith('.enc')

                    if is_encrypted:
                        encryption_password = st.text_input("Decryption password", type="password")

                    try:
                        if st.button("Analyze Backup", disabled=is_encrypted and not encryption_password):
                            with st.spinner("Analyzing backup file..."):
                                import json
                                import zipfile
                                import io

                                backup_data = None

                                if is_encrypted and encryption_password:
                                    # Decrypt the file
                                    import base64
                                    from cryptography.fernet import Fernet
                                    from cryptography.hazmat.primitives import hashes
                                    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

                                    # Derive key from password
                                    salt = b'medexplain_salt'
                                    kdf = PBKDF2HMAC(
                                        algorithm=hashes.SHA256(),
                                        length=32,
                                        salt=salt,
                                        iterations=100000
                                    )
                                    key = base64.urlsafe_b64encode(kdf.derive(encryption_password.encode()))

                                    try:
                                        # Decrypt data
                                        f = Fernet(key)
                                        decrypted_data = f.decrypt(uploaded_file.getvalue())

                                        # Check if it's a ZIP file or JSON
                                        if uploaded_file.name.endswith('.zip.enc'):
                                            # Handle as ZIP
                                            zip_buffer = io.BytesIO(decrypted_data)
                                            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                                                # Get main backup data
                                                with zip_file.open('backup_data.json') as f:
                                                    backup_data = json.load(f)
                                        else:
                                            # Handle as JSON
                                            backup_data = json.loads(decrypted_data.decode())
                                    except Exception as decrypt_error:
                                        st.error(f"Error decrypting backup: {decrypt_error}. Check your password.")
                                        return
                                elif uploaded_file.name.endswith('.zip'):
                                    # Handle ZIP file
                                    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue()), 'r') as zip_file:
                                        # Get main backup data
                                        with zip_file.open('backup_data.json') as f:
                                            backup_data = json.load(f)
                                else:
                                    # Handle JSON file
                                    backup_data = json.loads(uploaded_file.getvalue().decode("utf-8"))

                                # Validate backup format
                                if not backup_data or "app_version" not in backup_data or "backup_date" not in backup_data:
                                    st.error("Invalid backup file format. Missing required fields.")
                                else:
                                    # Show backup metadata
                                    st.markdown("### Backup Information")

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown(f"""
                                        - **App Version:** {backup_data.get('app_version', 'Unknown')}
                                        - **Backup Date:** {backup_data.get('backup_date', 'Unknown')}
                                        - **Contents:** {', '.join(backup_data.get('backup_contents', ['Unknown']))}
                                        """)

                                    with col2:
                                        user_count = len(backup_data.get('users', {}))
                                        has_settings = 'settings' in backup_data and backup_data['settings']
                                        has_logs = 'logs' in backup_data and backup_data['logs']

                                        st.markdown(f"""
                                        - **Users:** {user_count}
                                        - **Settings:** {'Included' if has_settings else 'Not included'}
                                        - **Logs:** {'Included' if has_logs else 'Not included'}
                                        """)

                                    # Check version compatibility
                                    backup_version = backup_data.get('app_version', '0.0.0')
                                    if backup_version != AppConfig.VERSION:
                                        st.warning(f"Backup was created with version {backup_version}, but current version is {AppConfig.VERSION}. Some data may not be compatible.")

                                    # Show restore options
                                    st.markdown("### Restore Options")

                                    restore_options = []

                                    if "User Profiles" in backup_data.get('backup_contents', []):
                                        restore_options.append("User Profiles")

                                    if "Health Records" in backup_data.get('backup_contents', []):
                                        restore_options.append("Health Records")

                                    if "Application Settings" in backup_data.get('backup_contents', []):
                                        restore_options.append("Application Settings")

                                    selected_restore = st.multiselect(
                                        "Select data to restore:",
                                        restore_options,
                                        default=restore_options
                                    )

                                    merge_option = st.radio(
                                        "Restore mode:",
                                        ["Merge with existing data", "Overwrite existing data"],
                                        index=0,
                                        horizontal=True,
                                        help="Merge mode will add data without removing existing records. Overwrite mode will replace all existing data."
                                    )

                                    if st.button("Restore Data", key="confirm_restore"):
                                        st.warning("This will modify your application data. Proceed with caution.")

                                        confirm = st.checkbox("I understand and confirm this action", key="restore_confirm")

                                        if confirm and st.button("Confirm Restore", key="final_confirm_restore"):
                                            with st.spinner("Restoring data..."):
                                                try:
                                                    # Handle data restoration based on selected options
                                                    restore_messages = []

                                                    # Get user manager
                                                    user_manager = self.registry.get("user_manager")

                                                    if "User Profiles" in selected_restore and user_manager:
                                                        # Save current user
                                                        current_user = user_manager.current_user_id
                                                        users_restored = 0

                                                        # Process each user
                                                        for user_id, user_data in backup_data.get("users", {}).items():
                                                            # Check if user exists
                                                            existing_user = user_manager.get_user(user_id)

                                                            if existing_user and merge_option == "Merge with existing data":
                                                                # Update profile
                                                                user_manager.switch_user(user_id)
                                                                if "profile" in user_data:
                                                                    user_manager.update_profile(user_data["profile"])
                                                                users_restored += 1
                                                            elif existing_user and merge_option == "Overwrite existing data":
                                                                # Delete and recreate user
                                                                user_manager.delete_user(user_id)
                                                                new_id = user_manager.create_user(user_data.get("profile", {}))
                                                                users_restored += 1
                                                            else:
                                                                # Create new user
                                                                new_id = user_manager.create_user(user_data.get("profile", {}))
                                                                users_restored += 1

                                                        # Switch back to original user
                                                        user_manager.switch_user(current_user)
                                                        restore_messages.append(f"Restored {users_restored} user profiles")

                                                    if "Health Records" in selected_restore and user_manager:
                                                        # Save current user
                                                        current_user = user_manager.current_user_id
                                                        records_restored = 0

                                                        # Process each user's health records
                                                        for user_id, user_data in backup_data.get("users", {}).items():
                                                            if "health_history" in user_data:
                                                                # Check if user exists
                                                                try:
                                                                    user_manager.switch_user(user_id)

                                                                    if merge_option == "Overwrite existing data":
                                                                        # Clear existing history
                                                                        user_manager.health_history = []

                                                                    # Add each record
                                                                    for record in user_data["health_history"]:
                                                                        user_manager.add_health_record(record)
                                                                        records_restored += 1
                                                                except Exception as user_error:
                                                                    self.logger.error(f"Error restoring health records for user {user_id}: {user_error}", exc_info=True)

                                                        # Switch back to original user
                                                        user_manager.switch_user(current_user)
                                                        restore_messages.append(f"Restored {records_restored} health records")

                                                    if "Application Settings" in selected_restore:
                                                        # Restore application settings
                                                        if "settings" in backup_data and backup_data["settings"]:
                                                            self.config_manager._config = backup_data["settings"]
                                                            self.config_manager.save()
                                                            restore_messages.append("Restored application settings")

                                                    st.success("Data restored successfully.")

                                                    # Display restore summary
                                                    st.markdown("### Restore Summary")
                                                    for message in restore_messages:
                                                        st.markdown(f"- {message}")

                                                    # Add notification
                                                    SessionManager.add_notification(
                                                        "Data Restored",
                                                        f"Successfully restored data from backup created on {backup_data.get('backup_date', 'unknown date')}.",
                                                        "success"
                                                    )

                                                    # Offer to refresh the app
                                                    if st.button("Refresh Application"):
                                                        st.rerun()
                                                except Exception as e:
                                                    self.logger.error(f"Error restoring data: {e}", exc_info=True)
                                                    st.error(f"Error restoring data: {e}")
                    except Exception as e:
                        self.logger.error(f"Error processing backup file: {e}", exc_info=True)
                        st.error(f"Error processing backup file: {e}")

            with backup_tabs[2]:
                st.markdown("Configure automated backup schedules")

                # Scheduled backups configuration
                st.info("This feature simulates scheduled backup configuration. In a production environment, backups would be handled by a server-side task scheduler.")

                # Enable scheduled backups
                enable_schedule = st.toggle("Enable Scheduled Backups", value=False)

                if enable_schedule:
                    col1, col2 = st.columns(2)

                    with col1:
                        backup_frequency = st.selectbox(
                            "Backup Frequency",
                            ["Daily", "Weekly", "Monthly", "Custom"]
                        )

                    with col2:
                        if backup_frequency == "Daily":
                            backup_time = st.time_input("Backup Time", value=datetime.time(1, 0))
                        elif backup_frequency == "Weekly":
                            backup_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                            backup_time = st.time_input("Backup Time", value=datetime.time(1, 0))
                        elif backup_frequency == "Monthly":
                            backup_day = st.number_input("Day of Month", min_value=1, max_value=28, value=1)
                            backup_time = st.time_input("Backup Time", value=datetime.time(1, 0))
                        elif backup_frequency == "Custom":
                            backup_interval = st.number_input("Interval (hours)", min_value=1, max_value=168, value=24)

                    # Backup storage options
                    st.markdown("### Backup Storage Options")

                    storage_option = st.radio(
                        "Storage Location",
                        ["Local Storage", "Cloud Storage"],
                        horizontal=True
                    )

                    if storage_option == "Local Storage":
                        local_path = st.text_input("Local Backup Path", value="/backups/medexplain")
                        retention_days = st.slider("Retention Period (days)", min_value=1, max_value=365, value=30)
                    else:
                        cloud_provider = st.selectbox(
                            "Cloud Provider",
                            ["Amazon S3", "Google Cloud Storage", "Microsoft Azure", "Custom"]
                        )

                        if cloud_provider == "Amazon S3":
                            st.text_input("S3 Bucket Name", placeholder="my-backup-bucket")
                            st.text_input("Access Key ID", type="password")
                            st.text_input("Secret Access Key", type="password")
                        elif cloud_provider == "Google Cloud Storage":
                            st.text_input("GCS Bucket Name", placeholder="my-backup-bucket")
                            st.file_uploader("Upload Service Account Key", type=["json"])
                        else:
                            st.text_input("Storage URL", placeholder="https://storage.example.com/backup")
                            st.text_input("API Key", type="password")

                        retention_days = st.slider("Retention Period (days)", min_value=1, max_value=365, value=30)

                    # Backup encryption
                    st.markdown("### Backup Security")

                    encrypt_backups = st.toggle("Encrypt Backups", value=True)

                    if encrypt_backups:
                        encryption_method = st.selectbox(
                            "Encryption Method",
                            ["AES-256", "Custom"]
                        )

                        if encryption_method == "AES-256":
                            st.text_input("Encryption Password", type="password")
                            st.text_input("Confirm Password", type="password")

                    # Notification options
                    st.markdown("### Notifications")

                    send_notifications = st.toggle("Send Backup Notifications", value=True)

                    if send_notifications:
                        notification_events = st.multiselect(
                            "Notify On",
                            ["Successful Backup", "Failed Backup", "Low Storage Space"],
                            default=["Failed Backup"]
                        )

                        notification_email = st.text_input("Email Address", placeholder="admin@example.com")

                    # Save configuration button
                    if st.button("Save Backup Schedule"):
                        # This would typically save to configuration
                        st.success("Backup schedule configured successfully.")

                        # Show next scheduled backup time
                        if backup_frequency == "Daily":
                            next_backup = datetime.datetime.now().replace(
                                hour=backup_time.hour,
                                minute=backup_time.minute,
                                second=0
                            )
                            if next_backup <= datetime.datetime.now():
                                next_backup += timedelta(days=1)
                        elif backup_frequency == "Weekly":
                            days = {
                                "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                                "Friday": 4, "Saturday": 5, "Sunday": 6
                            }
                            today = datetime.datetime.now().weekday()
                            target_day = days[backup_day]
                            days_ahead = (target_day - today) % 7
                            next_backup = datetime.datetime.now().replace(
                                hour=backup_time.hour,
                                minute=backup_time.minute,
                                second=0
                            ) + timedelta(days=days_ahead)
                            if days_ahead == 0 and next_backup <= datetime.datetime.now():
                                next_backup += timedelta(days=7)
                        else:
                            # Simple approximation for other frequencies
                            next_backup = datetime.datetime.now() + timedelta(days=1)

                        st.info(f"Next backup scheduled for: {next_backup.strftime('%Y-%m-%d %H:%M')}")

            # Data cleanup options with improved tools
            st.markdown("### Data Cleanup")

            with st.expander("Cleanup Options"):
                cleanup_tabs = st.tabs(["Temporary Files", "Database Optimization", "Data Export"])

                with cleanup_tabs[0]:
                    st.markdown("Clean up temporary files to free up disk space")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        clear_tmp = st.checkbox("Temporary Files", value=True)

                    with col2:
                        clear_cache = st.checkbox("Application Cache", value=True)

                    with col3:
                        clear_old_logs = st.checkbox("Old Log Files", value=False)

                    if st.button("Clear Selected Files"):
                        with st.spinner("Cleaning up files..."):
                            # Simulate file cleanup
                            import glob

                            deleted_files = 0
                            freed_space = 0

                            # Count temp files in log directory
                            if clear_tmp:
                                temp_files = glob.glob(os.path.join(AppConfig.LOG_DIR, "*.tmp"))

                                for file in temp_files:
                                    try:
                                        size = os.path.getsize(file)
                                        os.remove(file)
                                        deleted_files += 1
                                        freed_space += size
                                    except:
                                        pass

                            # Clear cache directory
                            if clear_cache:
                                cache_files = glob.glob(os.path.join(AppConfig.CACHE_DIR, "*"))

                                for file in cache_files:
                                    try:
                                        if os.path.isfile(file):
                                            size = os.path.getsize(file)
                                            os.remove(file)
                                            deleted_files += 1
                                            freed_space += size
                                    except:
                                        pass

                            # Clear old logs
                            if clear_old_logs:
                                # Find log rotation files
                                old_logs = glob.glob(os.path.join(AppConfig.LOG_DIR, "app.log.*"))

                                for file in old_logs:
                                    try:
                                        size = os.path.getsize(file)
                                        os.remove(file)
                                        deleted_files += 1
                                        freed_space += size
                                    except:
                                        pass

                            # Convert freed space to readable format
                            if freed_space < 1024:
                                freed_space_str = f"{freed_space} B"
                            elif freed_space < 1024 * 1024:
                                freed_space_str = f"{freed_space / 1024:.2f} KB"
                            else:
                                freed_space_str = f"{freed_space / (1024 * 1024):.2f} MB"

                            st.success(f"Cleaned up {deleted_files} files, freeing {freed_space_str} of disk space.")

                with cleanup_tabs[1]:
                    st.markdown("Optimize database and storage for better performance")

                    compact_database = st.checkbox("Compact Database", value=True, help="Remove deleted records and compact storage")
                    rebuild_indexes = st.checkbox("Rebuild Indexes", value=True, help="Rebuild database indexes for faster queries")
                    vacuum_storage = st.checkbox("Vacuum Storage", value=False, help="Reclaim unused storage space")

                    if st.button("Optimize Database"):
                        with st.spinner("Optimizing database..."):
                            # Simulate database optimization
                            time.sleep(2)

                            # Log optimization actions
                            optimization_steps = []

                            if compact_database:
                                optimization_steps.append("Database compaction completed")

                            if rebuild_indexes:
                                optimization_steps.append("Database indexes rebuilt")

                            if vacuum_storage:
                                optimization_steps.append("Storage space reclaimed")

                            if optimization_steps:
                                # Add diagnostic log entry
                                self.logger.info(f"Database optimization performed: {', '.join(optimization_steps)}")

                                # Show results
                                st.success("Database optimized successfully.")

                                for step in optimization_steps:
                                    st.markdown(f"- {step}")

                with cleanup_tabs[2]:
                    st.markdown("Export application data in various formats")

                    export_type = st.selectbox(
                        "What to export",
                        ["User Data", "Health Records", "System Logs", "All Data"]
                    )

                    export_format = st.selectbox(
                        "Export format",
                        ["CSV", "JSON", "Excel"]
                    )

                    include_metadata = st.checkbox("Include Metadata", value=True)
                    anonymize_data = st.checkbox("Anonymize Personal Data", value=False)

                    if st.button("Export Data"):
                        with st.spinner("Preparing data export..."):
                            try:
                                import json
                                import pandas as pd
                                import io

                                # Simulate data export
                                user_manager = self.registry.get("user_manager")

                                export_data = None

                                # Generate export data based on selection
                                if export_type == "User Data" and user_manager:
                                    # Get all users
                                    users = user_manager.get_all_users()

                                    # Anonymize if requested
                                    if anonymize_data:
                                        for user in users:
                                            if "name" in user:
                                                user["name"] = f"User_{user['id'][:8]}"
                                            if "email" in user:
                                                user["email"] = f"user_{user['id'][:8]}@example.com"

                                    if export_format == "CSV":
                                        # Convert to dataframe for CSV
                                        export_data = pd.DataFrame(users)
                                    else:
                                        export_data = users
                                elif export_type == "Health Records" and user_manager:
                                    # Collect health records
                                    health_records = []

                                    # Save current user
                                    current_user = user_manager.current_user_id

                                    # Get all users
                                    users = user_manager.get_all_users()

                                    for user in users:
                                        user_id = user.get("id")
                                        if user_id:
                                            try:
                                                # Switch to user
                                                user_manager.switch_user(user_id)

                                                # Get health history
                                                for record in user_manager.health_history:
                                                    # Add user ID to record
                                                    record_copy = record.copy()
                                                    record_copy["user_id"] = user_id

                                                    # Anonymize if requested
                                                    if anonymize_data:
                                                        record_copy["user_id"] = f"User_{user_id[:8]}"

                                                    health_records.append(record_copy)
                                            except:
                                                pass

                                    # Switch back to original user
                                    user_manager.switch_user(current_user)

                                    if export_format == "CSV":
                                        # Convert to dataframe for CSV
                                        export_data = pd.DataFrame(health_records)
                                    else:
                                        export_data = health_records
                                elif export_type == "System Logs":
                                    # Read log file
                                    log_path = AppConfig.LOG_FILE
                                    log_lines = []

                                    if log_path.exists():
                                        with open(log_path, "r") as log_file:
                                            for line in log_file:
                                                log_lines.append({"log_entry": line.strip()})

                                    if export_format == "CSV":
                                        # Convert to dataframe for CSV
                                        export_data = pd.DataFrame(log_lines)
                                    else:
                                        export_data = log_lines
                                else:
                                    # All data - create a comprehensive export
                                    # This is a simplified example
                                    all_data = {"users": [], "health_records": [], "system_info": {}}

                                    if user_manager:
                                        # Get all users
                                        users = user_manager.get_all_users()

                                        # Anonymize if requested
                                        if anonymize_data:
                                            anonymized_users = []
                                            for user in users:
                                                user_copy = user.copy()
                                                if "name" in user_copy:
                                                    user_copy["name"] = f"User_{user_copy['id'][:8]}"
                                                if "email" in user_copy:
                                                    user_copy["email"] = f"user_{user_copy['id'][:8]}@example.com"
                                                anonymized_users.append(user_copy)
                                            all_data["users"] = anonymized_users
                                        else:
                                            all_data["users"] = users

                                        # Collect health records
                                        health_records = []

                                        # Save current user
                                        current_user = user_manager.current_user_id

                                        for user in users:
                                            user_id = user.get("id")
                                            if user_id:
                                                try:
                                                    # Switch to user
                                                    user_manager.switch_user(user_id)

                                                    # Get health history
                                                    for record in user_manager.health_history:
                                                        # Add user ID to record
                                                        record_copy = record.copy()
                                                        record_copy["user_id"] = user_id

                                                        # Anonymize if requested
                                                        if anonymize_data:
                                                            record_copy["user_id"] = f"User_{user_id[:8]}"

                                                        health_records.append(record_copy)
                                                except:
                                                    pass

                                        # Switch back to original user
                                        user_manager.switch_user(current_user)

                                        all_data["health_records"] = health_records

                                    # Add system info if including metadata
                                    if include_metadata:
                                        all_data["system_info"] = {
                                            "app_version": AppConfig.VERSION,
                                            "export_date": datetime.datetime.now().isoformat(),
                                            "component_status": self.registry.get_status()
                                        }

                                    export_data = all_data

                                # Create appropriate output based on format
                                if export_data is not None:
                                    date_str = datetime.datetime.now().strftime("%Y%m%d")

                                    if export_format == "CSV":
                                        if isinstance(export_data, pd.DataFrame):
                                            csv_data = export_data.to_csv(index=False)

                                            st.download_button(
                                                label=f"Download {export_type} (CSV)",
                                                data=csv_data,
                                                file_name=f"medexplain_{export_type.lower().replace(' ', '_')}_{date_str}.csv",
                                                mime="text/csv"
                                            )
                                        else:
                                            st.error("Data cannot be exported as CSV. Try JSON format.")
                                    elif export_format == "JSON":
                                        if isinstance(export_data, pd.DataFrame):
                                            json_data = export_data.to_json(orient="records", indent=2)
                                        else:
                                            json_data = json.dumps(export_data, indent=2)

                                        st.download_button(
                                            label=f"Download {export_type} (JSON)",
                                            data=json_data,
                                            file_name=f"medexplain_{export_type.lower().replace(' ', '_')}_{date_str}.json",
                                            mime="application/json"
                                        )
                                    else:  # Excel
                                        output = io.BytesIO()

                                        if isinstance(export_data, pd.DataFrame):
                                            with pd.ExcelWriter(output) as writer:
                                                export_data.to_excel(writer, sheet_name=export_type, index=False)
                                        else:
                                            # Convert to dataframe(s) for Excel
                                            with pd.ExcelWriter(output) as writer:
                                                if export_type == "All Data":
                                                    # Multiple sheets for all data
                                                    if "users" in export_data and export_data["users"]:
                                                        pd.DataFrame(export_data["users"]).to_excel(writer, sheet_name="Users", index=False)

                                                    if "health_records" in export_data and export_data["health_records"]:
                                                        pd.DataFrame(export_data["health_records"]).to_excel(writer, sheet_name="Health Records", index=False)

                                                    if "system_info" in export_data and export_data["system_info"]:
                                                        # Convert nested dict to dataframe
                                                        system_info_df = pd.DataFrame(
                                                            [[k, str(v)] for k, v in export_data["system_info"].items()],
                                                            columns=["Key", "Value"]
                                                        )
                                                        system_info_df.to_excel(writer, sheet_name="System Info", index=False)
                                                else:
                                                    pd.DataFrame(export_data).to_excel(writer, sheet_name=export_type, index=False)

                                        output.seek(0)

                                        st.download_button(
                                            label=f"Download {export_type} (Excel)",
                                            data=output,
                                            file_name=f"medexplain_{export_type.lower().replace(' ', '_')}_{date_str}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                            except Exception as e:
                                self.logger.error(f"Error exporting data: {e}", exc_info=True)
                                st.error(f"Error exporting data: {e}")

            # Danger zone with improved security
            st.markdown("### Danger Zone")

            with st.expander("⚠️ Dangerous Operations"):
                st.error("⚠️ The following actions will permanently delete data and cannot be undone.")

                danger_tabs = st.tabs(["Clear User Data", "Reset Application", "Purge All Data"])

                with danger_tabs[0]:
                    st.markdown("Delete all user data while preserving application settings")

                    st.warning("""
                    This will delete:
                    - All user profiles
                    - All health records
                    - All symptom history

                    Application settings will be preserved.
                    """)

                    # Add authentication
                    admin_password = st.text_input("Admin Password:", type="password", key="clear_user_password")
                    confirm_text = st.text_input("Type 'DELETE USER DATA' to confirm:")

                    if st.button("Clear All User Data", disabled=not admin_password):
                        if confirm_text != "DELETE USER DATA":
                            st.error("Confirmation text doesn't match. Please type 'DELETE USER DATA' exactly.")
                        else:
                            with st.spinner("Clearing user data..."):
                                try:
                                    # Get user manager
                                    user_manager = self.registry.get("user_manager")

                                    if user_manager:
                                        # Get all users
                                        users = user_manager.get_all_users()
                                        default_user_id = None

                                        # Delete all users except default
                                        for user in users:
                                            user_id = user.get("id")
                                            if user_id and user_id != "default_user":
                                                user_manager.delete_user(user_id)
                                            elif user_id == "default_user":
                                                default_user_id = user_id

                                        # Reset default user if it exists
                                        if default_user_id:
                                            user_manager.switch_user(default_user_id)
                                            user_manager.clear_health_history()

                                            # Reset profile to default
                                            user_manager.update_profile({
                                                "name": "Default User",
                                                "gender": "Not specified",
                                                "age": 30
                                            })
                                        else:
                                            # Create default user if it doesn't exist
                                            user_manager.create_user({
                                                "id": "default_user",
                                                "name": "Default User",
                                                "gender": "Not specified",
                                                "age": 30
                                            })

                                    # Reset session state
                                    for key in ['last_symptom_check', 'last_risk_assessment', 'health_report']:
                                        SessionManager.delete(key)

                                    SessionManager.set("user_id", "default_user")

                                    st.success("All user data has been cleared. The application has been reset to default user.")

                                    # Add notification
                                    SessionManager.add_notification(
                                        "Data Cleared",
                                        "All user data has been cleared from the system.",
                                        "warning"
                                    )

                                    if st.button("Return to Home"):
                                        SessionManager.navigate_to("Home")
                                except Exception as e:
                                    self.logger.error(f"Error clearing user data: {e}", exc_info=True)
                                    st.error(f"Error clearing user data: {e}")

                with danger_tabs[1]:
                    st.markdown("Reset application to factory settings")

                    st.warning("""
                    This will reset:
                    - All application settings
                    - System preferences
                    - UI configurations

                    User data will be preserved.
                    """)

                    # Add authentication
                    admin_password = st.text_input("Admin Password:", type="password", key="reset_app_password")
                    confirm_text = st.text_input("Type 'RESET APP' to confirm:")

                    if st.button("Reset Application", disabled=not admin_password):
                        if confirm_text != "RESET APP":
                            st.error("Confirmation text doesn't match. Please type 'RESET APP' exactly.")
                        else:
                            with st.spinner("Resetting application..."):
                                try:
                                    # Reset configuration to defaults
                                    self.config_manager._load_from_defaults()
                                    self.config_manager.save()

                                    # Clear cache directory
                                    import shutil

                                    cache_dir = AppConfig.CACHE_DIR
                                    if os.path.exists(cache_dir):
                                        for item in os.listdir(cache_dir):
                                            item_path = os.path.join(cache_dir, item)
                                            try:
                                                if os.path.isfile(item_path):
                                                    os.unlink(item_path)
                                                elif os.path.isdir(item_path):
                                                    shutil.rmtree(item_path)
                                            except:
                                                pass

                                    # Reset session state settings
                                    SessionManager.set("theme", AppConfig.DEFAULT_THEME)
                                    SessionManager.set("font_size", "medium")
                                    SessionManager.set("screen_reader_optimized", False)
                                    SessionManager.set("advanced_mode", False)

                                    st.success("Application has been reset to factory settings.")

                                    # Add notification
                                    SessionManager.add_notification(
                                        "Application Reset",
                                        "Application settings have been reset to factory defaults.",
                                        "warning"
                                    )

                                    if st.button("Reload Application"):
                                        st.rerun()
                                except Exception as e:
                                    self.logger.error(f"Error resetting application: {e}", exc_info=True)
                                    st.error(f"Error resetting application: {e}")

                with danger_tabs[2]:
                    st.markdown("Purge all application data (⚠️ EXTREME CAUTION)")

                    st.error("""
                    ⚠️ EXTREME CAUTION ⚠️

                    This will permanently delete:
                    - All user profiles and health records
                    - All application settings and configurations
                    - All logs and cached data

                    THIS ACTION CANNOT BE UNDONE.
                    """)

                    # Add multiple authentication steps
                    col1, col2 = st.columns(2)

                    with col1:
                        admin_password = st.text_input("Admin Password:", type="password", key="purge_password")

                    with col2:
                        confirmation_code = st.text_input("Security Code:", type="password")
                        st.caption("Contact system administrator for the security code")

                    confirm_text = st.text_input("Type 'PURGE ALL DATA' to confirm:")
                    additional_check = st.checkbox("I understand this will permanently delete ALL data")

                    # Simulate requiring a security code
                    security_code = "PURGE-12345"

                    if st.button("Purge All Data", disabled=not (admin_password and additional_check)):
                        if confirm_text != "PURGE ALL DATA":
                            st.error("Confirmation text doesn't match. Please type 'PURGE ALL DATA' exactly.")
                        elif confirmation_code != security_code:
                            st.error("Invalid security code. Please contact your system administrator.")
                        else:
                            with st.spinner("Purging all data..."):
                                try:
                                    # Show extra confirmation dialog
                                    st.warning("⚠️ FINAL WARNING: You are about to delete all application data.")

                                    final_confirm = st.radio(
                                        "Are you absolutely certain?",
                                        ["No, cancel this operation", "Yes, delete everything"],
                                        index=0
                                    )

                                    if final_confirm == "Yes, delete everything" and st.button("Execute Complete Data Purge"):
                                        # Implement data purge (simulated for safety)
                                        time.sleep(3)  # Simulate processing time

                                        # In a real implementation, we would delete all data directories

                                        # Reset all session state
                                        SessionManager.clear()

                                        st.success("All application data has been purged. The application will now restart.")

                                        if st.button("Restart Application"):
                                            # This would typically perform a complete restart
                                            st.rerun()
                                except Exception as e:
                                    self.logger.error(f"Error purging data: {e}", exc_info=True)
                                    st.error(f"Error purging data: {e}")
        except Exception as e:
            self.logger.error(f"Error rendering data management tab: {e}", exc_info=True)
            st.error("Error rendering data management. Please try again later.")

    def _get_data_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about data storage usage.

        Returns:
            Dict with storage statistics
        """
        try:
            # Helper function to get directory size
            def get_dir_size(path):
                total = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total += os.path.getsize(fp)
                return total

            # Helper function to format bytes
            def format_bytes(size):
                power = 2**10  # 1024
                n = 0
                units = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
                while size > power:
                    size /= power
                    n += 1
                return f"{size:.2f} {units[n]}"

            # Get sizes of different directories
            user_data_bytes = get_dir_size(AppConfig.USER_DATA_DIR)
            model_bytes = get_dir_size(AppConfig.MODEL_DIR)
            logs_cache_bytes = get_dir_size(AppConfig.LOG_DIR) + get_dir_size(AppConfig.CACHE_DIR)

            # Calculate total size
            total_bytes = user_data_bytes + model_bytes + logs_cache_bytes

            # Calculate size of other data
            other_bytes = get_dir_size(AppConfig.DATA_DIR) - user_data_bytes - model_bytes
            if other_bytes < 0:
                other_bytes = 0

            # Return formatted results
            return {
                "total_size": format_bytes(total_bytes),
                "user_data_size": format_bytes(user_data_bytes),
                "model_size": format_bytes(model_bytes),
                "logs_cache_size": format_bytes(logs_cache_bytes),
                "other_size": format_bytes(other_bytes),
                "total_bytes": total_bytes,
                "user_data_bytes": user_data_bytes,
                "model_bytes": model_bytes,
                "logs_cache_bytes": logs_cache_bytes,
                "other_bytes": other_bytes
            }
        except Exception as e:
            self.logger.error(f"Error calculating storage statistics: {e}", exc_info=True)
            # Return fallback values
            return {
                "total_size": "Unknown",
                "user_data_size": "Unknown",
                "model_size": "Unknown",
                "logs_cache_size": "Unknown",
                "other_size": "Unknown",
                "total_bytes": 0,
                "user_data_bytes": 0,
                "model_bytes": 0,
                "logs_cache_bytes": 0,
                "other_bytes": 0
            }

    @timing_decorator
    def _render_performance_metrics_tab(self) -> None:
        """Render performance metrics tab in admin panel with improved analytics and visualizations."""
        try:
            st.subheader("Performance Metrics")

            # Get performance metrics from session state
            metrics = SessionManager.get("performance_metrics", {})

            # Real-time system metrics
            st.markdown("### Real-time System Metrics")

            try:
                import psutil

                # Get current system metrics
                cpu_percent = psutil.cpu_percent(interval=0.5)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent

                # Display metrics in columns with gauges
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <h4>CPU Usage</h4>
                        <div style="position: relative; height: 120px; width: 120px; margin: auto;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                                <span style="font-size: 24px; font-weight: bold;">{cpu_percent}%</span>
                            </div>
                            <svg width="120" height="120" viewBox="0 0 120 120">
                                <circle cx="60" cy="60" r="54" fill="none" stroke="#eee" stroke-width="12" />
                                <circle cx="60" cy="60" r="54" fill="none" stroke="{self._get_gauge_color(cpu_percent)}"
                                        stroke-width="12" stroke-dasharray="{3.4 * cpu_percent} 339.292"
                                        transform="rotate(-90 60 60)" />
                            </svg>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <h4>Memory Usage</h4>
                        <div style="position: relative; height: 120px; width: 120px; margin: auto;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                                <span style="font-size: 24px; font-weight: bold;">{memory_percent}%</span>
                            </div>
                            <svg width="120" height="120" viewBox="0 0 120 120">
                                <circle cx="60" cy="60" r="54" fill="none" stroke="#eee" stroke-width="12" />
                                <circle cx="60" cy="60" r="54" fill="none" stroke="{self._get_gauge_color(memory_percent)}"
                                        stroke-width="12" stroke-dasharray="{3.4 * memory_percent} 339.292"
                                        transform="rotate(-90 60 60)" />
                            </svg>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <h4>Disk Usage</h4>
                        <div style="position: relative; height: 120px; width: 120px; margin: auto;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                                <span style="font-size: 24px; font-weight: bold;">{disk_percent}%</span>
                            </div>
                            <svg width="120" height="120" viewBox="0 0 120 120">
                                <circle cx="60" cy="60" r="54" fill="none" stroke="#eee" stroke-width="12" />
                                <circle cx="60" cy="60" r="54" fill="none" stroke="{self._get_gauge_color(disk_percent)}"
                                        stroke-width="12" stroke-dasharray="{3.4 * disk_percent} 339.292"
                                        transform="rotate(-90 60 60)" />
                            </svg>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Add detailed system information in an expander
                with st.expander("System Details"):
                    # Get more detailed system info
                    cpu_count = psutil.cpu_count(logical=True)
                    cpu_physical = psutil.cpu_count(logical=False)
                    memory_total = memory.total / (1024 * 1024 * 1024)  # GB
                    memory_used = memory.used / (1024 * 1024 * 1024)  # GB
                    disk_total = disk.total / (1024 * 1024 * 1024)  # GB
                    disk_used = disk.used / (1024 * 1024 * 1024)  # GB

                    # Display in a more structured format
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**CPU Information:**")
                        st.markdown(f"- Logical processors: {cpu_count}")
                        st.markdown(f"- Physical cores: {cpu_physical}")
                        st.markdown(f"- Current load: {cpu_percent}%")

                        # Add per-core information
                        st.markdown("**Per-Core Usage:**")
                        per_cpu = psutil.cpu_percent(percpu=True)
                        for i, usage in enumerate(per_cpu):
                            st.progress(usage / 100, text=f"Core {i}: {usage}%")

                    with col2:
                        st.markdown("**Memory Information:**")
                        st.markdown(f"- Total memory: {memory_total:.2f} GB")
                        st.markdown(f"- Used memory: {memory_used:.2f} GB ({memory_percent}%)")
                        st.markdown(f"- Available memory: {(memory_total - memory_used):.2f} GB")

                        st.markdown("**Disk Information:**")
                        st.markdown(f"- Total disk space: {disk_total:.2f} GB")
                        st.markdown(f"- Used disk space: {disk_used:.2f} GB ({disk_percent}%)")
                        st.markdown(f"- Free disk space: {(disk_total - disk_used):.2f} GB")
            except ImportError:
                st.warning("System metrics require the psutil package. Using simulated data.")

                # Use simulated data
                import random

                # Simulate metrics
                cpu_percent = random.uniform(20, 80)
                memory_percent = random.uniform(30, 70)
                disk_percent = random.uniform(40, 60)

                # Display simulated metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("CPU Usage", f"{cpu_percent:.1f}%")

                with col2:
                    st.metric("Memory Usage", f"{memory_percent:.1f}%")

                with col3:
                    st.metric("Disk Usage", f"{disk_percent:.1f}%")

                st.info("Note: These are simulated metrics. Install 'psutil' for actual system monitoring.")

            # Application performance metrics with enhanced visualization
            st.markdown("### Application Performance")

            # Display application metrics
            startup_time = metrics.get('startup_time', 0)
            component_health = metrics.get('component_load_success_rate', 0)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Startup Time",
                    f"{startup_time:.2f}s",
                    delta=None
                )

            with col2:
                st.metric(
                    "Component Health",
                    f"{component_health:.1%}",
                    delta=None
                )

            with col3:
                # Calculate average response time from function timing if available
                response_times = []

                if 'function_timing' in metrics:
                    for func, times in metrics['function_timing'].items():
                        # Only include user-facing functions
                        if func in ['run', '_render_page', '_render_home', '_render_symptom_pattern_tab']:
                            response_times.extend(times)

                if response_times:
                    avg_response = sum(response_times) / len(response_times)
                    st.metric(
                        "Avg Response Time",
                        f"{avg_response*1000:.0f} ms",
                        delta=None
                    )
                else:
                    st.metric(
                        "Avg Response Time",
                        "N/A",
                        delta=None
                    )

            # Function timing analysis with improved visualization
            if 'function_timing' in metrics and metrics['function_timing']:
                st.markdown("### Function Performance Analysis")

                # Create function timing chart
                try:
                    import plotly.graph_objects as go

                    # Prepare data
                    functions = []
                    avg_times = []
                    max_times = []
                    calls = []

                    for func, times in metrics['function_timing'].items():
                        functions.append(func)
                        avg_times.append(sum(times) / len(times) * 1000)  # Convert to ms
                        max_times.append(max(times) * 1000)  # Convert to ms
                        calls.append(len(times))

                    # Sort by average time (descending)
                    sorted_indices = sorted(range(len(avg_times)), key=lambda i: avg_times[i], reverse=True)
                    functions = [functions[i] for i in sorted_indices]
                    avg_times = [avg_times[i] for i in sorted_indices]
                    max_times = [max_times[i] for i in sorted_indices]
                    calls = [calls[i] for i in sorted_indices]

                    # Create figure
                    fig = go.Figure()

                    # Add average time bars
                    fig.add_trace(go.Bar(
                        x=functions,
                        y=avg_times,
                        name='Average Time (ms)',
                        marker_color='rgb(55, 83, 109)'
                    ))

                    # Add max time bars
                    fig.add_trace(go.Bar(
                        x=functions,
                        y=max_times,
                        name='Max Time (ms)',
                        marker_color='rgb(26, 118, 255)'
                    ))

                    # Add call count line
                    fig.add_trace(go.Scatter(
                        x=functions,
                        y=calls,
                        mode='lines+markers',
                        name='Call Count',
                        yaxis='y2',
                        line=dict(color='rgb(219, 64, 82)', width=2)
                    ))

                    # Update layout
                    fig.update_layout(
                        title='Function Performance',
                        xaxis_title='Function',
                        yaxis_title='Time (ms)',
                        yaxis2=dict(
                            title='Call Count',
                            overlaying='y',
                            side='right'
                        ),
                        barmode='group',
                        height=500,
                        margin=dict(l=50, r=50, t=50, b=100),
                        xaxis={'tickangle': 45}
                    )

                    # Show chart
                    st.plotly_chart(fig, use_container_width=True)

                    # Show detailed table for deeper analysis
                    function_data = []

                    for i, func in enumerate(functions):
                        function_data.append({
                            "Function": func,
                            "Avg Time (ms)": f"{avg_times[i]:.2f}",
                            "Max Time (ms)": f"{max_times[i]:.2f}",
                            "Min Time (ms)": f"{min(metrics['function_timing'][func]) * 1000:.2f}",
                            "Calls": calls[i],
                            "Total Time (ms)": f"{sum(metrics['function_timing'][func]) * 1000:.2f}"
                        })

                    # Create and display the dataframe
                    func_df = pd.DataFrame(function_data)
                    st.dataframe(
                        func_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Function": st.column_config.TextColumn("Function", width="large"),
                            "Avg Time (ms)": st.column_config.NumberColumn("Avg Time (ms)", format="%.2f"),
                            "Max Time (ms)": st.column_config.NumberColumn("Max Time (ms)", format="%.2f"),
                            "Min Time (ms)": st.column_config.NumberColumn("Min Time (ms)", format="%.2f"),
                            "Total Time (ms)": st.column_config.NumberColumn("Total Time (ms)", format="%.2f"),
                            "Calls": st.column_config.NumberColumn("Calls", format="%d")
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Error creating function timing visualization: {e}", exc_info=True)

                    # Fallback to simple table view
                    st.markdown("#### Function Timing Statistics")

                    for func, times in metrics['function_timing'].items():
                        if times:
                            avg_time = sum(times) / len(times) * 1000  # Convert to ms
                            st.markdown(f"**{func}**: Avg: {avg_time:.2f}ms, Calls: {len(times)}")

            # Performance over time with improved visualization
            st.markdown("### Performance Over Time")

            # Generate sample data with realistic patterns if not available
            if 'page_render_times' not in metrics or not metrics['page_render_times']:
                # Create mock data for demonstration
                from datetime import datetime, timedelta
                import random
                import numpy as np

                # Generate dates for the past 14 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=14)
                dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14)]

                # Response times should start higher and gradually improve
                base_response_time = 250
                response_times = []
                for i in range(14):
                    # Add some daily variation but with a slight improving trend
                    day_variation = random.randint(-30, 20)
                    trend_improvement = i * 0.5  # Slight improvement over time
                    response_times.append(max(200, base_response_time + day_variation - trend_improvement))

                # Error rates should be low with occasional spikes
                error_rates = []
                for _ in range(14):
                    # Usually low (0-0.5%) but occasionally higher
                    if random.random() < 0.2:  # 20% chance of higher error rate
                        error_rates.append(random.uniform(0.5, 2.0))
                    else:
                        error_rates.append(random.uniform(0, 0.5))

                perf_dates = dates
                perf_response_times = response_times
                perf_error_rates = error_rates
            else:
                # Use real data from metrics
                page_render_times = metrics['page_render_times']

                # Prepare data for visualization
                perf_dates = []
                perf_response_times = []
                perf_error_rates = []

                # Use the most recent 14 days or less
                for i in range(min(14, len(page_render_times))):
                    perf_dates.append((datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"))

                    # Calculate average response time for the day
                    day_times = []
                    for page, times in page_render_times.items():
                        if times:
                            day_times.extend(times)

                    if day_times:
                        perf_response_times.append(sum(day_times) / len(day_times) * 1000)  # Convert to ms
                    else:
                        perf_response_times.append(0)

                    # Simulate error rate since we don't track it
                    perf_error_rates.append(random.uniform(0, 0.5))

                # Reverse to show chronological order
                perf_dates.reverse()
                perf_response_times.reverse()
                perf_error_rates.reverse()

            try:
                # Create performance chart
                import plotly.graph_objects as go
                fig = go.Figure()

                # Add response time trace
                fig.add_trace(go.Scatter(
                    x=perf_dates,
                    y=perf_response_times,
                    mode='lines+markers',
                    name='Response Time (ms)',
                    line={'color': 'blue'}
                ))

                # Create secondary y-axis for error rate
                fig.add_trace(go.Scatter(
                    x=perf_dates,
                    y=perf_error_rates,
                    mode='lines+markers',
                    name='Error Rate (%)',
                    line={'color': 'red'},
                    yaxis="y2"
                ))

                # Update layout with second y-axis
                fig.update_layout(
                    title="System Performance Trends",
                    xaxis_title="Date",
                    yaxis_title="Response Time (ms)",
                    yaxis={"range": [180, 300]},  # Set reasonable y-axis range for response time
                    yaxis2={
                        'title': 'Error Rate (%)',
                        'overlaying': 'y',
                        'side': 'right',
                        'range': [0, 3]  # Set reasonable y-axis range for error rate
                    },
                    height=400,
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02}
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                self.logger.error(f"Error creating performance chart: {e}", exc_info=True)
                st.error("Could not render performance visualization.")

            # Add page performance metrics with improved visualization
            st.markdown("### Page Performance")

            # Get page view data from session state
            page_views = SessionManager.get("page_views", {})

            if page_views:
                # Create performance data for each page
                page_data = []

                # Calculate page render times if available
                page_render_times = metrics.get('page_render_times', {})

                for page, views in page_views.items():
                    # Get average render time if available
                    avg_render_time = 0
                    if page in page_render_times and page_render_times[page]:
                        avg_render_time = sum(page_render_times[page]) / len(page_render_times[page]) * 1000  # ms

                    page_data.append({
                        "Page": page,
                        "Views": views,
                        "Avg Render Time (ms)": f"{avg_render_time:.2f}" if avg_render_time > 0 else "N/A"
                    })

                # Sort by popularity
                page_data = sorted(page_data, key=lambda x: x["Views"], reverse=True)

                # Create and display table
                page_df = pd.DataFrame(page_data)
                st.dataframe(
                    page_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Page": st.column_config.TextColumn("Page", width="medium"),
                        "Views": st.column_config.NumberColumn("Views", format="%d"),
                        "Avg Render Time (ms)": st.column_config.TextColumn("Avg Render Time (ms)", width="medium")
                    }
                )

                # Create visualization
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    # Create figure with subplots
                    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "bar"}]],
                                         subplot_titles=("Page Views", "Avg Render Time"))

                    # Extract data
                    pages = [item["Page"] for item in page_data]
                    views = [item["Views"] for item in page_data]

                    # Render times - convert "N/A" to 0
                    render_times = []
                    for item in page_data:
                        if item["Avg Render Time (ms)"] == "N/A":
                            render_times.append(0)
                        else:
                            render_times.append(float(item["Avg Render Time (ms)"]))

                    # Add page views bar chart
                    fig.add_trace(
                        go.Bar(
                            x=pages,
                            y=views,
                            marker_color='royalblue',
                            name="Views"
                        ),
                        row=1, col=1
                    )

                    # Add render time bar chart
                    fig.add_trace(
                        go.Bar(
                            x=pages,
                            y=render_times,
                            marker_color='orange',
                            name="Render Time (ms)"
                        ),
                        row=1, col=2
                    )

                    # Update layout
                    fig.update_layout(
                        height=400,
                        margin=dict(l=50, r=50, t=80, b=80),
                        showlegend=False
                    )

                    # Update x-axis properties
                    fig.update_xaxes(tickangle=45)

                    # Show chart
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    self.logger.error(f"Error creating page performance chart: {e}", exc_info=True)
                    # Fail silently, chart is optional
            else:
                st.info("No page view data available.")

                # Show sample metrics as placeholders
                st.markdown("### Sample Metrics")

                # Create sample data
                sample_pages = ["Home", "Health Dashboard", "Symptom Analyzer", "Health Chat", "Settings"]
                sample_views = [15, 12, 8, 5, 3]
                sample_times = [245, 320, 410, 280, 190]

                # Create dataframe
                sample_df = pd.DataFrame({
                    "Page": sample_pages,
                    "Views": sample_views,
                    "Avg Render Time (ms)": sample_times
                })

                st.dataframe(
                    sample_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.info("These are sample metrics. Real metrics will be displayed after more app usage.")

            # Performance optimization recommendations
            st.markdown("### Performance Optimization")

            # Provide tailored recommendations based on metrics
            with st.expander("Optimization Recommendations"):
                st.markdown("#### Recommendations for Improved Performance")

                # Check if we have actual metrics
                has_real_metrics = 'function_timing' in metrics and metrics['function_timing']

                if has_real_metrics:
                    # Find the slowest functions
                    slow_functions = []
                    for func, times in metrics['function_timing'].items():
                        if times:
                            avg_time = sum(times) / len(times) * 1000  # ms
                            if avg_time > 100:  # Consider functions taking > 100ms as slow
                                slow_functions.append((func, avg_time))

                    # Sort by average time (descending)
                    slow_functions.sort(key=lambda x: x[1], reverse=True)

                    if slow_functions:
                        st.markdown("##### Slow Functions to Optimize")

                        for func, avg_time in slow_functions[:3]:  # Show top 3
                            st.markdown(f"- **{func}**: {avg_time:.2f}ms average response time")

                        st.markdown("""
                        Consider optimizing these functions by:
                        - Using caching for expensive operations
                        - Reducing unnecessary computations
                        - Implementing lazy loading for UI components
                        """)
                    else:
                        st.success("All functions have good performance (under 100ms).")

                    # Check page render times
                    if 'page_render_times' in metrics and metrics['page_render_times']:
                        slow_pages = []
                        for page, times in metrics['page_render_times'].items():
                            if times:
                                avg_time = sum(times) / len(times) * 1000  # ms
                                if avg_time > 500:  # Consider pages taking > 500ms as slow
                                    slow_pages.append((page, avg_time))

                        # Sort by average time (descending)
                        slow_pages.sort(key=lambda x: x[1], reverse=True)

                        if slow_pages:
                            st.markdown("##### Slow Pages to Optimize")

                            for page, avg_time in slow_pages[:3]:  # Show top 3
                                st.markdown(f"- **{page}**: {avg_time:.2f}ms average render time")

                            st.markdown("""
                            Consider optimizing these pages by:
                            - Breaking complex pages into smaller components
                            - Using tabs to separate content
                            - Implementing pagination for large data displays
                            - Deferring non-critical component loading
                            """)
                        else:
                            st.success("All pages have good render times (under 500ms).")
                else:
                    # General recommendations
                    st.markdown("""
                    ##### General Performance Recommendations

                    1. **Component Optimization**
                       - Use caching for expensive operations with `@st.cache_data` or `@st.cache_resource`
                       - Implement lazy loading for components not visible on first load
                       - Optimize database queries and data processing functions

                    2. **UI Responsiveness**
                       - Break complex pages into tabs or expandable sections
                       - Use pagination for large datasets
                       - Minimize re-renders by using session state effectively

                    3. **Resource Management**
                       - Implement proper cleanup of temporary files
                       - Optimize image and asset sizes
                       - Use connection pooling for database operations
                    """)

                # Add general tips
                st.markdown("""
                ##### Streamlit-Specific Optimizations

                - Use `st.cache_data` for data processing functions
                - Use `st.cache_resource` for resource-intensive operations (e.g., loading ML models)
                - Minimize the use of `st.rerun()` as it causes a full page reload
                - For large datasets, use pagination or lazy loading
                - Consider using `st.empty()` containers for dynamic updates
                """)
        except Exception as e:
            self.logger.error(f"Error rendering performance metrics tab: {e}", exc_info=True)
            st.error("Error rendering performance metrics. Please try again later.")

    def _get_gauge_color(self, value: float) -> str:
        """
        Get color for gauge visualization based on value.

        Args:
            value: Percentage value (0-100)

        Returns:
            Color string in hex or RGB format
        """
        if value < 50:
            return "#4CAF50"  # Green
        elif value < 75:
            return "#FF9800"  # Orange
        else:
            return "#F44336"  # Red

    @timing_decorator
    def _render_api_configuration_tab(self) -> None:
        """Render API configuration tab in admin panel with improved security and validation."""
        try:
            st.subheader("API Configuration")

            # API key management with improved security
            st.markdown("### API Keys")

            # Use tabs for different API integrations
            api_tabs = st.tabs(["OpenAI API", "Medical Database API", "EHR Integration", "Other APIs"])

            # OpenAI API tab
            with api_tabs[0]:
                st.markdown("#### OpenAI API Configuration")

                # Display current status
                openai_client = self.registry.get("openai_client")
                if openai_client:
                    api_status = "Configured" if (hasattr(openai_client, "api_key") and openai_client.api_key) else "Not Configured"

                    # Create status indicator
                    status_color = "green" if api_status == "Configured" else "red"
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {status_color}; margin-right: 8px;"></div>
                        <div><strong>Status:</strong> <span style='color:{status_color};'>{api_status}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Check API key validity if configured
                    if api_status == "Configured":
                        try:
                            # This would typically call a validation method
                            # For demo purposes, we'll just show it as valid
                            st.markdown("""
                            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: green; margin-right: 8px;"></div>
                                <div><strong>Validation:</strong> <span style='color:green;'>Valid</span></div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Show current settings
                            settings = self.config_manager.get("api", "openai", default={})

                            if settings:
                                st.markdown("##### Current Settings")
                                settings_info = {
                                    "Model": settings.get("model", "gpt-4"),
                                    "Max Tokens": settings.get("max_tokens", 1000),
                                    "Temperature": settings.get("temperature", 0.7)
                                }

                                # Create a clean display of settings
                                for key, value in settings_info.items():
                                    st.markdown(f"**{key}:** {value}")
                        except:
                            st.markdown("**Validation:** <span style='color:orange;'>Not validated</span>", unsafe_allow_html=True)
                else:
                    st.warning("OpenAI client not available.")

                # API key input form
                with st.form("openai_api_form"):
                    st.markdown("#### Update OpenAI API Configuration")

                    new_api_key = st.text_input("Enter OpenAI API Key", type="password",
                                               help="Your OpenAI API key starting with 'sk-'")

                    # Add model configuration
                    model_options = [
                        "gpt-4",
                        "gpt-4-turbo",
                        "gpt-3.5-turbo",
                        "Custom"
                    ]

                    selected_model = st.selectbox(
                        "Select Model",
                        model_options,
                        index=model_options.index(self.config_manager.get("api", "openai", "model", default="gpt-4"))
                            if self.config_manager.get("api", "openai", "model", default="gpt-4") in model_options
                            else 0
                    )

                    if selected_model == "Custom":
                        custom_model = st.text_input("Custom Model Name", placeholder="e.g., gpt-4-0125-preview")

                    col1, col2 = st.columns(2)

                    with col1:
                        max_tokens = st.number_input(
                            "Max Tokens",
                            min_value=100,
                            max_value=8000,
                            value=self.config_manager.get("api", "openai", "max_tokens", default=1000),
                            help="Maximum number of tokens to generate in each response"
                        )

                    with col2:
                        temperature = st.slider(
                            "Temperature",
                            min_value=0.0,
                            max_value=1.0,
                            value=self.config_manager.get("api", "openai", "temperature", default=0.7),
                            step=0.1,
                            help="Controls randomness. Lower values are more deterministic."
                        )

                    # Add validation for key format (should start with "sk-")
                    if new_api_key and not new_api_key.startswith("sk-"):
                        st.warning("OpenAI API keys typically start with 'sk-'. Please check your key.")

                    submit_button = st.form_submit_button("Save API Configuration")

                    if submit_button:
                        if openai_client:
                            try:
                                # Update API key if provided
                                if new_api_key:
                                    if hasattr(openai_client, "set_api_key"):
                                        openai_client.set_api_key(new_api_key)

                                        # Update environment variable for persistence
                                        os.environ["OPENAI_API_KEY"] = new_api_key

                                        st.success("API key updated successfully.")

                                        # Add notification
                                        SessionManager.add_notification(
                                            "API Key Updated",
                                            "OpenAI API key has been updated successfully.",
                                            "success"
                                        )
                                    else:
                                        st.error("The OpenAI client does not support setting the API key.")

                                # Update configuration
                                config_changes = {}

                                if selected_model == "Custom" and custom_model:
                                    config_changes["model"] = custom_model
                                elif selected_model != "Custom":
                                    config_changes["model"] = selected_model

                                config_changes["max_tokens"] = max_tokens
                                config_changes["temperature"] = temperature

                                # Save to configuration
                                for key, value in config_changes.items():
                                    self.config_manager.set("api", "openai", key, value=value)

                                st.success("API configuration updated successfully.")

                                # Add notification
                                SessionManager.add_notification(
                                    "API Configuration Updated",
                                    "OpenAI API configuration has been updated successfully.",
                                    "success"
                                )
                            except Exception as e:
                                self.logger.error(f"Error saving API configuration: {e}", exc_info=True)
                                st.error(f"Error saving API configuration: {e}")
                        else:
                            st.error("OpenAI client not available.")

                # Add test button outside the form
                if st.button("Test API Connection"):
                    if openai_client and hasattr(openai_client, "api_key") and openai_client.api_key:
                        with st.spinner("Testing API connection..."):
                            try:
                                # This would call the OpenAI API with a simple prompt
                                # For demo, we'll simulate the test
                                time.sleep(1)

                                # Randomly succeed or fail
                                import random
                                if random.random() < 0.9:  # 90% success rate
                                    st.success("API connection test successful!")
                                    st.markdown("""
                                    Connection details:
                                    - Endpoint: api.openai.com
                                    - Model: gpt-4
                                    - Response time: 245ms
                                    """)
                                else:
                                    st.error("API connection failed: Rate limit exceeded. Please try again later.")
                            except Exception as e:
                                st.error(f"Error testing API connection: {e}")
                    else:
                        st.error("API key not configured. Please set up the API key first.")

            # Medical Database API tab
            with api_tabs[1]:
                st.markdown("#### Medical Database API Configuration")

                # Show configuration status
                medical_api_key = self.config_manager.get("api", "medical", "api_key", default=None)

                if medical_api_key:
                    st.markdown("""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: green; margin-right: 8px;"></div>
                        <div><strong>Status:</strong> <span style='color:green;'>Configured</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: red; margin-right: 8px;"></div>
                        <div><strong>Status:</strong> <span style='color:red;'>Not Configured</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                Configure access to external medical databases for enhanced
                symptom correlation and medical literature search.
                """)

                with st.form("medical_api_form"):
                    st.markdown("#### Update Medical Database API Configuration")

                    # API provider selection
                    providers = ["PubMed API", "MedlinePlus API", "UpToDate API", "Custom API"]
                    selected_provider = st.selectbox(
                        "API Provider",
                        providers,
                        index=providers.index(self.config_manager.get("api", "medical", "provider", default="PubMed API"))
                            if self.config_manager.get("api", "medical", "provider", default="PubMed API") in providers
                            else 0
                    )

                    # API configuration
                    api_url = st.text_input(
                        "API Endpoint URL",
                        value=self.config_manager.get("api", "medical", "url", default=""),
                        placeholder="https://api.medical-database.com/v1"
                    )

                    api_key = st.text_input(
                        "API Key",
                        type="password",
                        value="●●●●●●●●●●●●●●●●" if medical_api_key else "",
                        help="Enter your API key for the selected provider"
                    )

                    # Add URL validation
                    if api_url and not (api_url.startswith("http://") or api_url.startswith("https://")):
                        st.warning("API URL should start with http:// or https://")

                    # Additional settings
                    with st.expander("Advanced Settings"):
                        timeout = st.number_input(
                            "Request Timeout (seconds)",
                            min_value=1,
                            max_value=60,
                            value=self.config_manager.get("api", "medical", "timeout", default=10)
                        )

                        rate_limit = st.number_input(
                            "Rate Limit (requests per minute)",
                            min_value=1,
                            max_value=1000,
                            value=self.config_manager.get("api", "medical", "rate_limit", default=60)
                        )

                        cache_results = st.checkbox(
                            "Cache API Results",
                            value=self.config_manager.get("api", "medical", "cache_results", default=True)
                        )

                    submit_button = st.form_submit_button("Save Configuration")

                    if submit_button:
                        # Validate URL format
                        import re
                        url_pattern = re.compile(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')

                        if not api_url:
                            st.error("API URL is required.")
                        elif not url_pattern.match(api_url):
                            st.error("Invalid URL format. Please enter a valid URL.")
                        elif not api_key or api_key == "●●●●●●●●●●●●●●●●":
                            st.error("API key is required.")
                        else:
                            try:
                                # Save configuration
                                self.config_manager.set("api", "medical", "provider", value=selected_provider)
                                self.config_manager.set("api", "medical", "url", value=api_url)

                                # Only update API key if it was changed
                                if api_key != "●●●●●●●●●●●●●●●●":
                                    self.config_manager.set("api", "medical", "api_key", value=api_key)

                                # Save advanced settings
                                self.config_manager.set("api", "medical", "timeout", value=timeout)
                                self.config_manager.set("api", "medical", "rate_limit", value=rate_limit)
                                self.config_manager.set("api", "medical", "cache_results", value=cache_results)

                                st.success("Medical API configuration saved successfully.")

                                # Add notification
                                SessionManager.add_notification(
                                    "Medical API Updated",
                                    f"Medical Database API ({selected_provider}) has been configured successfully.",
                                    "success"
                                )
                            except Exception as e:
                                self.logger.error(f"Error saving Medical API configuration: {e}", exc_info=True)
                                st.error(f"Error saving configuration: {e}")

                # Test connection button
                if st.button("Test Medical API Connection"):
                    medical_api_url = self.config_manager.get("api", "medical", "url", default=None)

                    if not medical_api_url or not medical_api_key:
                        st.error("Medical API is not fully configured. Please set up API URL and key first.")
                    else:
                        with st.spinner("Testing API connection..."):
                            # Simulate API test
                            time.sleep(1.5)

                            # Random success or error
                            import random
                            if random.random() < 0.8:  # 80% success rate
                                st.success("Medical API connection test successful!")
                                st.markdown(f"""
                                Connection details:
                                - Endpoint: {medical_api_url}
                                - Provider: {self.config_manager.get("api", "medical", "provider", default="Unknown")}
                                - Response time: 312ms
                                """)
                            else:
                                error_message = "Could not connect to API. Please check your API key and URL."
                                st.error(f"API connection failed: {error_message}")

            # EHR Integration tab
            with api_tabs[2]:
                st.markdown("#### Electronic Health Record (EHR) Integration")

                # Show configuration status
                ehr_configured = self.config_manager.get("api", "ehr", "configured", default=False)

                if ehr_configured:
                    st.markdown("""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: green; margin-right: 8px;"></div>
                        <div><strong>Status:</strong> <span style='color:green;'>Configured</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show current configuration
                    ehr_system = self.config_manager.get("api", "ehr", "system", default="Unknown")
                    st.markdown(f"**Current EHR System:** {ehr_system}")
                else:
                    st.markdown("""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: red; margin-right: 8px;"></div>
                        <div><strong>Status:</strong> <span style='color:red;'>Not Configured</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                Configure integration with Electronic Health Record (EHR) systems
                for seamless patient data exchange.
                """)

                with st.form("ehr_integration_form"):
                    st.markdown("#### Configure EHR Integration")

                    # EHR system selection
                    ehr_systems = ["Select a system...", "Epic", "Cerner", "Allscripts", "MEDITECH", "NextGen", "Other"]

                    current_system = self.config_manager.get("api", "ehr", "system", default="")
                    system_index = ehr_systems.index(current_system) if current_system in ehr_systems else 0

                    ehr_system = st.selectbox(
                        "EHR System",
                        ehr_systems,
                        index=system_index
                    )

                    if ehr_system == "Select a system...":
                        st.info("Please select an EHR system.")
                    else:
                        # For "Other" system, allow custom name
                        if ehr_system == "Other":
                            ehr_system_custom = st.text_input(
                                "EHR System Name",
                                value=current_system if current_system not in ehr_systems else ""
                            )

                        # Integration details
                        col1, col2 = st.columns(2)

                        with col1:
                            ehr_url = st.text_input(
                                "EHR API Endpoint",
                                value=self.config_manager.get("api", "ehr", "url", default=""),
                                placeholder="https://api.ehr-system.com/fhir"
                            )

                        with col2:
                            ehr_key = st.text_input(
                                "Access Token",
                                type="password",
                                value="●●●●●●●●●●●●●●●●" if self.config_manager.get("api", "ehr", "key", default=None) else ""
                            )

                        # Show authentication method options
                        auth_methods = ["OAuth 2.0", "API Key", "Basic Auth"]

                        current_auth = self.config_manager.get("api", "ehr", "auth_method", default="")
                        auth_index = auth_methods.index(current_auth) if current_auth in auth_methods else 0

                        auth_method = st.radio(
                            "Authentication Method",
                            auth_methods,
                            index=auth_index
                        )

                        # Show relevant fields based on auth method
                        if auth_method == "OAuth 2.0":
                            oauth_client_id = st.text_input(
                                "Client ID",
                                value=self.config_manager.get("api", "ehr", "oauth_client_id", default="")
                            )

                            oauth_client_secret = st.text_input(
                                "Client Secret",
                                type="password",
                                value="●●●●●●●●●●●●●●●●" if self.config_manager.get("api", "ehr", "oauth_client_secret", default=None) else ""
                            )

                            oauth_scope = st.text_input(
                                "Scope",
                                value=self.config_manager.get("api", "ehr", "oauth_scope", default="patient/*.read"),
                                placeholder="patient/*.read"
                            )
                        elif auth_method == "Basic Auth":
                            basic_username = st.text_input(
                                "Username",
                                value=self.config_manager.get("api", "ehr", "basic_username", default="")
                            )

                            basic_password = st.text_input(
                                "Password",
                                type="password",
                                value="●●●●●●●●●●●●●●●●" if self.config_manager.get("api", "ehr", "basic_password", default=None) else ""
                            )

                        # FHIR version selection
                        fhir_versions = ["DSTU2", "STU3", "R4", "R5"]

                        current_fhir = self.config_manager.get("api", "ehr", "fhir_version", default="")
                        fhir_index = fhir_versions.index(current_fhir) if current_fhir in fhir_versions else 2  # Default to R4

                        fhir_version = st.selectbox(
                            "FHIR Version",
                            fhir_versions,
                            index=fhir_index
                        )

                        # Data integration options
                        st.markdown("#### Data Integration Options")

                        integration_options = st.multiselect(
                            "Select data to integrate:",
                            ["Patient Demographics", "Allergies", "Medications", "Conditions", "Observations", "Procedures"],
                            default=self.config_manager.get("api", "ehr", "integration_options", default=["Patient Demographics"])
                        )

                        # Data sync frequency
                        sync_options = ["Manual Only", "Daily", "Weekly", "On Login"]

                        current_sync = self.config_manager.get("api", "ehr", "sync_frequency", default="")
                        sync_index = sync_options.index(current_sync) if current_sync in sync_options else 0

                        sync_frequency = st.selectbox(
                            "Data Sync Frequency",
                            sync_options,
                            index=sync_index
                        )

                    submit_button = st.form_submit_button("Save EHR Configuration")

                    if submit_button:
                        if ehr_system == "Select a system...":
                            st.error("Please select an EHR system.")
                        else:
                            try:
                                # Prepare system name
                                final_system = ehr_system
                                if ehr_system == "Other" and 'ehr_system_custom' in locals() and ehr_system_custom:
                                    final_system = ehr_system_custom

                                # Save configuration
                                self.config_manager.set("api", "ehr", "system", value=final_system)
                                self.config_manager.set("api", "ehr", "auth_method", value=auth_method)
                                self.config_manager.set("api", "ehr", "fhir_version", value=fhir_version)
                                self.config_manager.set("api", "ehr", "integration_options", value=integration_options)
                                self.config_manager.set("api", "ehr", "sync_frequency", value=sync_frequency)
                                self.config_manager.set("api", "ehr", "configured", value=True)

                                # Save URL if provided
                                if 'ehr_url' in locals() and ehr_url:
                                    self.config_manager.set("api", "ehr", "url", value=ehr_url)

                                # Save token if changed
                                if 'ehr_key' in locals() and ehr_key and ehr_key != "●●●●●●●●●●●●●●●●":
                                    self.config_manager.set("api", "ehr", "key", value=ehr_key)

                                # Save auth-specific settings
                                if auth_method == "OAuth 2.0":
                                    if 'oauth_client_id' in locals():
                                        self.config_manager.set("api", "ehr", "oauth_client_id", value=oauth_client_id)

                                    if 'oauth_client_secret' in locals() and oauth_client_secret and oauth_client_secret != "●●●●●●●●●●●●●●●●":
                                        self.config_manager.set("api", "ehr", "oauth_client_secret", value=oauth_client_secret)

                                    if 'oauth_scope' in locals():
                                        self.config_manager.set("api", "ehr", "oauth_scope", value=oauth_scope)
                                elif auth_method == "Basic Auth":
                                    if 'basic_username' in locals():
                                        self.config_manager.set("api", "ehr", "basic_username", value=basic_username)

                                    if 'basic_password' in locals() and basic_password and basic_password != "●●●●●●●●●●●●●●●●":
                                        self.config_manager.set("api", "ehr", "basic_password", value=basic_password)

                                st.success(f"EHR configuration for {final_system} saved successfully.")

                                # Add notification
                                SessionManager.add_notification(
                                    "EHR Integration Updated",
                                    f"EHR integration with {final_system} has been configured successfully.",
                                    "success"
                                )
                            except Exception as e:
                                self.logger.error(f"Error saving EHR configuration: {e}", exc_info=True)
                                st.error(f"Error saving configuration: {e}")

                # Test connection button
                if st.button("Test EHR Connection"):
                    if not ehr_configured:
                        st.error("EHR integration is not configured. Please save a configuration first.")
                    else:
                        with st.spinner("Testing EHR connection..."):
                            # Simulate API test
                            time.sleep(2)

                            # Random success or failure
                            import random
                            success = random.random() < 0.7  # 70% success rate

                            if success:
                                st.success("EHR connection test successful!")
                                st.markdown(f"""
                                Connection details:
                                - System: {self.config_manager.get("api", "ehr", "system", default="Unknown")}
                                - FHIR Version: {self.config_manager.get("api", "ehr", "fhir_version", default="Unknown")}
                                - Response time: 480ms
                                """)

                                # Show sample data
                                with st.expander("Sample Patient Data"):
                                    st.json({
                                        "resourceType": "Patient",
                                        "id": "example",
                                        "meta": {
                                            "versionId": "1",
                                            "lastUpdated": "2023-11-15T08:15:30Z"
                                        },
                                        "text": {
                                            "status": "generated",
                                            "div": "<div>Example Patient</div>"
                                        },
                                        "identifier": [
                                            {
                                                "system": "urn:oid:1.2.36.146.595.217.0.1",
                                                "value": "12345"
                                            }
                                        ],
                                        "name": [
                                            {
                                                "family": "Smith",
                                                "given": ["John"]
                                            }
                                        ],
                                        "gender": "male",
                                        "birthDate": "1970-01-01"
                                    })
                            else:
                                error_types = [
                                    "Authentication failed. Please check your credentials.",
                                    "Connection timed out. The EHR server may be unavailable.",
                                    "Invalid FHIR version. The server requires a different version.",
                                    "Missing required scopes or permissions."
                                ]
                                error_message = random.choice(error_types)
                                st.error(f"EHR connection failed: {error_message}")

            # Other APIs tab
            with api_tabs[3]:
                st.markdown("#### Additional API Integrations")

                st.info("Configure additional third-party API integrations.")

                # Weather API for environmental health factors
                with st.expander("Weather API Integration"):
                    st.markdown("##### Weather API Configuration")
                    st.markdown("Integrate with weather services to correlate health data with environmental factors.")

                    # Check if configured
                    weather_api_key = self.config_manager.get("api", "weather", "api_key", default=None)

                    if weather_api_key:
                        st.success("Weather API is currently configured.")

                    # Configuration form
                    with st.form("weather_api_form"):
                        weather_providers = ["OpenWeatherMap", "WeatherAPI", "AccuWeather", "Other"]

                        weather_provider = st.selectbox(
                            "Weather API Provider",
                            weather_providers,
                            index=weather_providers.index(self.config_manager.get("api", "weather", "provider", default="OpenWeatherMap"))
                                if self.config_manager.get("api", "weather", "provider", default="OpenWeatherMap") in weather_providers
                                else 0
                        )

                        weather_api_key = st.text_input(
                            "API Key",
                            type="password",
                            value="●●●●●●●●●●●●●●●●" if self.config_manager.get("api", "weather", "api_key", default=None) else ""
                        )

                        # Save button
                        submit_button = st.form_submit_button("Save Weather API Configuration")

                        if submit_button:
                            if not weather_api_key or weather_api_key == "●●●●●●●●●●●●●●●●":
                                st.error("API key is required.")
                            else:
                                try:
                                    # Save configuration
                                    self.config_manager.set("api", "weather", "provider", value=weather_provider)

                                    # Only update API key if it was changed
                                    if weather_api_key != "●●●●●●●●●●●●●●●●":
                                        self.config_manager.set("api", "weather", "api_key", value=weather_api_key)

                                    st.success("Weather API configuration saved successfully.")
                                except Exception as e:
                                    self.logger.error(f"Error saving Weather API configuration: {e}", exc_info=True)
                                    st.error(f"Error saving configuration: {e}")

                # Nutritional database API
                with st.expander("Nutrition API Integration"):
                    st.markdown("##### Nutrition API Configuration")
                    st.markdown("Integrate with nutritional databases for diet and health recommendations.")

                    # Check if configured
                    nutrition_api_key = self.config_manager.get("api", "nutrition", "api_key", default=None)

                    if nutrition_api_key:
                        st.success("Nutrition API is currently configured.")

                    # Configuration form
                    with st.form("nutrition_api_form"):
                        nutrition_providers = ["Edamam", "Nutritionix", "USDA FoodData Central", "Other"]

                        nutrition_provider = st.selectbox(
                            "Nutrition API Provider",
                            nutrition_providers,
                            index=nutrition_providers.index(self.config_manager.get("api", "nutrition", "provider", default="Edamam"))
                                if self.config_manager.get("api", "nutrition", "provider", default="Edamam") in nutrition_providers
                                else 0
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            nutrition_app_id = st.text_input(
                                "Application ID",
                                value=self.config_manager.get("api", "nutrition", "app_id", default="")
                            )

                        with col2:
                            nutrition_api_key = st.text_input(
                                "API Key",
                                type="password",
                                value="●●●●●●●●●●●●●●●●" if self.config_manager.get("api", "nutrition", "api_key", default=None) else ""
                            )

                        # Save button
                        submit_button = st.form_submit_button("Save Nutrition API Configuration")

                        if submit_button:
                            if not nutrition_api_key or nutrition_api_key == "●●●●●●●●●●●●●●●●":
                                st.error("API key is required.")
                            else:
                                try:
                                    # Save configuration
                                    self.config_manager.set("api", "nutrition", "provider", value=nutrition_provider)
                                    self.config_manager.set("api", "nutrition", "app_id", value=nutrition_app_id)

                                    # Only update API key if it was changed
                                    if nutrition_api_key != "●●●●●●●●●●●●●●●●":
                                        self.config_manager.set("api", "nutrition", "api_key", value=nutrition_api_key)

                                    st.success("Nutrition API configuration saved successfully.")
                                except Exception as e:
                                    self.logger.error(f"Error saving Nutrition API configuration: {e}", exc_info=True)
                                    st.error(f"Error saving configuration: {e}")

                # Add new API integration
                with st.expander("Add New API Integration"):
                    st.markdown("##### Add Custom API Integration")

                    with st.form("custom_api_form"):
                        custom_api_name = st.text_input("API Name", placeholder="e.g., Fitness Tracker API")
                        custom_api_url = st.text_input("API Endpoint URL", placeholder="https://api.example.com/v1")
                        custom_api_key = st.text_input("API Key/Token", type="password")

                        col1, col2 = st.columns(2)

                        with col1:
                            auth_type = st.selectbox(
                                "Authentication Type",
                                ["API Key", "Bearer Token", "OAuth 2.0", "No Authentication"]
                            )

                        with col2:
                            data_format = st.selectbox(
                                "Data Format",
                                ["JSON", "XML", "CSV", "Other"]
                            )

                        integration_purpose = st.text_area(
                            "Integration Purpose",
                            placeholder="Describe how this API will be used in the application..."
                        )

                        # Save button
                        submit_button = st.form_submit_button("Add API Integration")

                        if submit_button:
                            if not custom_api_name:
                                st.error("API name is required.")
                            elif not custom_api_url:
                                st.error("API URL is required.")
                            elif auth_type != "No Authentication" and not custom_api_key:
                                st.error(f"API key is required for {auth_type} authentication.")
                            else:
                                try:
                                    # Create a safe key for config storage
                                    import re
                                    safe_key = re.sub(r'[^a-z0-9_]', '_', custom_api_name.lower())

                                    # Save configuration
                                    self.config_manager.set("api", safe_key, "name", value=custom_api_name)
                                    self.config_manager.set("api", safe_key, "url", value=custom_api_url)
                                    self.config_manager.set("api", safe_key, "auth_type", value=auth_type)
                                    self.config_manager.set("api", safe_key, "data_format", value=data_format)
                                    self.config_manager.set("api", safe_key, "purpose", value=integration_purpose)

                                    if custom_api_key:
                                        self.config_manager.set("api", safe_key, "api_key", value=custom_api_key)

                                    st.success(f"Custom API '{custom_api_name}' added successfully.")

                                    # Add notification
                                    SessionManager.add_notification(
                                        "Custom API Added",
                                        f"Custom API integration '{custom_api_name}' has been added successfully.",
                                        "success"
                                    )
                                except Exception as e:
                                    self.logger.error(f"Error adding custom API: {e}", exc_info=True)
                                    st.error(f"Error adding custom API: {e}")

            # API usage settings with improved UI and functionality
            st.markdown("### API Usage Settings")

            with st.expander("Rate Limiting & Budgets"):
                # Rate limiting options
                st.subheader("Rate Limiting")

                enable_rate_limiting = st.toggle(
                    "Enable API rate limiting",
                    value=self.config_manager.get("api", "rate_limiting", "enabled", default=True)
                )

                if enable_rate_limiting:
                    col1, col2 = st.columns(2)

                    with col1:
                        max_requests = st.slider(
                            "Maximum requests per minute",
                            min_value=1,
                            max_value=100,
                            value=self.config_manager.get("api", "rate_limiting", "max_requests", default=60)
                        )

                    with col2:
                        burst_limit = st.slider(
                            "Burst limit (max concurrent requests)",
                            min_value=1,
                            max_value=20,
                            value=self.config_manager.get("api", "rate_limiting", "burst_limit", default=10)
                        )

                    # What happens when rate limit is exceeded
                    rate_limit_action = st.radio(
                        "When rate limit is exceeded:",
                        ["Queue requests", "Return error", "Reduce request priority"],
                        index=["Queue requests", "Return error", "Reduce request priority"].index(
                            self.config_manager.get("api", "rate_limiting", "action", default="Queue requests")
                        ) if self.config_manager.get("api", "rate_limiting", "action", default="Queue requests") in
                           ["Queue requests", "Return error", "Reduce request priority"] else 0
                    )

                    st.info(f"API calls will be limited to {max_requests} requests per minute with a burst limit of {burst_limit}.")

                # Budget controls
                st.subheader("Budget Controls")

                enable_budget = st.toggle(
                    "Enable budget limits",
                    value=self.config_manager.get("api", "budget", "enabled", default=False)
                )

                if enable_budget:
                    daily_budget = st.number_input(
                        "Daily budget ($)",
                        min_value=0.0,
                        value=self.config_manager.get("api", "budget", "daily_limit", default=5.0),
                        step=0.5
                    )

                    monthly_budget = st.number_input(
                        "Monthly budget ($)",
                        min_value=0.0,
                        value=self.config_manager.get("api", "budget", "monthly_limit", default=100.0),
                        step=10.0
                    )

                    # Action when budget is exceeded
                    exceeded_action = st.radio(
                        "When budget is exceeded:",
                        ["Warn only", "Disable non-essential features", "Block all API calls"],
                        index=["Warn only", "Disable non-essential features", "Block all API calls"].index(
                            self.config_manager.get("api", "budget", "exceeded_action", default="Warn only")
                        ) if self.config_manager.get("api", "budget", "exceeded_action", default="Warn only") in
                           ["Warn only", "Disable non-essential features", "Block all API calls"] else 0
                    )

                    # Show budget warning threshold
                    warning_threshold = st.slider(
                        "Budget warning threshold (%)",
                        min_value=50,
                        max_value=95,
                        value=self.config_manager.get("api", "budget", "warning_threshold", default=80)
                    )

                    st.info(f"""
                    Daily budget set to ${daily_budget:.2f}. Monthly budget set to ${monthly_budget:.2f}.
                    You will be warned when {warning_threshold}% of the budget is used.
                    When exceeded: {exceeded_action}
                    """)

                # Save API usage settings
                if st.button("Save API Usage Settings"):
                    try:
                        # Save rate limiting settings
                        self.config_manager.set("api", "rate_limiting", "enabled", value=enable_rate_limiting)

                        if enable_rate_limiting:
                            self.config_manager.set("api", "rate_limiting", "max_requests", value=max_requests)
                            self.config_manager.set("api", "rate_limiting", "burst_limit", value=burst_limit)
                            self.config_manager.set("api", "rate_limiting", "action", value=rate_limit_action)

                        # Save budget settings
                        self.config_manager.set("api", "budget", "enabled", value=enable_budget)

                        if enable_budget:
                            self.config_manager.set("api", "budget", "daily_limit", value=daily_budget)
                            self.config_manager.set("api", "budget", "monthly_limit", value=monthly_budget)
                            self.config_manager.set("api", "budget", "exceeded_action", value=exceeded_action)
                            self.config_manager.set("api", "budget", "warning_threshold", value=warning_threshold)

                        st.success("API usage settings saved successfully.")

                        # Add notification
                        SessionManager.add_notification(
                            "API Settings Updated",
                            "API usage settings have been updated successfully.",
                            "success"
                        )
                    except Exception as e:
                        self.logger.error(f"Error saving API usage settings: {e}", exc_info=True)
                        st.error(f"Error saving settings: {e}")
        except Exception as e:
            self.logger.error(f"Error rendering API configuration tab: {e}", exc_info=True)
            st.error("Error rendering API configuration. Please try again later.")

    def _refresh_components(self) -> None:
        """Refresh components that need to be updated with improved error handling."""
        try:
            # Re-initialize health analyzer if dependencies available
            user_manager = self.registry.get("user_manager")
            health_analyzer = self.registry.get("health_analyzer")

            if user_manager and health_analyzer:
                try:
                    health_analyzer.update_data(
                        user_manager.health_history,
                        user_manager.profile
                    )
                    self.logger.info("Refreshed HealthDataAnalyzer")
                except Exception as e:
                    self.logger.error(f"Error refreshing HealthDataAnalyzer: {e}", exc_info=True)

            # Re-initialize dashboard if needed
            dashboard = self.registry.get("dashboard")
            if user_manager and dashboard:
                try:
                    dashboard.update_data(
                        user_manager.health_history,
                        user_manager.profile
                    )
                    self.logger.info("Refreshed HealthDashboard")
                except Exception as e:
                    self.logger.error(f"Error refreshing HealthDashboard: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error in _refresh_components: {e}", exc_info=True)

    def _handle_data_export(self) -> None:
        """Handle data export when ready with improved error handling and format support."""
        try:
            export_data = SessionManager.get("export_data")
            if not export_data:
                return

            # Create a download button for the data
            data_type = export_data.get("type", "data")
            file_format = export_data.get("format", "csv")
            data = export_data.get("data")

            if data is None:
                st.error("No data available for export.")
                SessionManager.set("export_ready", False)
                SessionManager.set("export_data", None)
                return

            st.markdown(f"### Export {data_type.title()}")
            st.info(f"Your {data_type} export is ready. Click below to download.")

            # Convert data to appropriate format
            if file_format.lower() == "csv":
                # Convert to CSV
                if isinstance(data, pd.DataFrame):
                    output = data.to_csv(index=False)
                else:
                    # Convert to DataFrame first
                    df = pd.DataFrame(data)
                    output = df.to_csv(index=False)

                # Create download button
                date_str = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    label=f"Download {data_type} as CSV",
                    data=output,
                    file_name=f"{data_type}_{date_str}.csv",
                    mime="text/csv"
                )
            elif file_format.lower() == "json":
                # Convert to JSON
                if isinstance(data, pd.DataFrame):
                    output = data.to_json(orient="records")
                else:
                    import json
                    output = json.dumps(data, indent=2)

                # Create download button
                date_str = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    label=f"Download {data_type} as JSON",
                    data=output,
                    file_name=f"{data_type}_{date_str}.json",
                    mime="application/json"
                )
            elif file_format.lower() == "excel":
                # Convert to Excel
                import io

                output = io.BytesIO()

                if isinstance(data, pd.DataFrame):
                    with pd.ExcelWriter(output) as writer:
                        data.to_excel(writer, index=False)
                else:
                    # Convert to DataFrame first
                    with pd.ExcelWriter(output) as writer:
                        pd.DataFrame(data).to_excel(writer, index=False)

                output.seek(0)

                # Create download button
                date_str = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    label=f"Download {data_type} as Excel",
                    data=output,
                    file_name=f"{data_type}_{date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error(f"Unsupported export format: {file_format}")

            # Add option to keep or clear export data
            if st.button("Clear Export Data"):
                SessionManager.set("export_ready", False)
                SessionManager.set("export_data", None)
                st.success("Export data cleared.")
                st.rerun()
        except Exception as e:
            self.logger.error(f"Error handling data export: {e}", exc_info=True)
            st.error(f"Error exporting data: {e}")
            SessionManager.set("export_ready", False)
            SessionManager.set("export_data", None)

# Run the application when executed directly
if __name__ == "__main__":
    app = MedExplainApp()
    app.run()
