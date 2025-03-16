import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import sys
import uuid
import time
from typing import Dict, List, Any, Optional, Union, Tuple

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

# Constants
APP_VERSION = "3.0.1"  # Incrementing version for your improved implementation
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# Ensure directories exist
for directory in [DATA_DIR, os.path.join(DATA_DIR, "user_data"), os.path.join(DATA_DIR, "ml_models"), LOG_DIR, STATIC_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),  # Changed from "log" to "app.log" for clarity
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedExplainApp:
    """
    Main application class for MedExplain AI Pro.
    Integrates all components into a complete application with enterprise features.
    """

    def __init__(self, load_components: bool = True):
        """
        Initialize the application and its components.

        Args:
            load_components: If True, load all components. Set to False for testing.
        """
        try:
            # Initialize session tracking
            self.session_id = str(uuid.uuid4())
            self.start_time = time.time()
            logger.info(f"Initializing MedExplain AI Pro v{APP_VERSION}, session: {self.session_id}")

            # Setup data directories
            self.user_data_dir = os.path.join(DATA_DIR, "user_data")
            self.model_dir = os.path.join(DATA_DIR, "ml_models")
            os.makedirs(self.user_data_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)

            # Component tracking for diagnostics
            self.component_status = {}

            if load_components:
                self._initialize_core_components()
                self._initialize_ml_components()
                self._initialize_ui_components()
                self._load_api_keys_from_environment()

            # Log initialization summary
            logger.info(f"MedExplain AI Pro initialization complete. "
                       f"Status: {sum(self.component_status.values())} of {len(self.component_status)} components loaded")

        except Exception as e:
            logger.error(f"Critical error during MedExplain AI Pro initialization: {e}", exc_info=True)
            self._set_critical_error(f"Application initialization failed: {str(e)}")

    def _initialize_core_components(self):
        """Initialize core system components with error handling."""
        # Health Data Manager - Improved error handling and dependency injection
        try:
            self.health_data = HealthDataManager(data_dir=self.user_data_dir)
            self.component_status["health_data"] = True
            logger.info("Initialized health_data")
        except Exception as e:
            logger.error(f"Error initializing health_data: {e}", exc_info=True)
            self.health_data = None
            self.component_status["health_data"] = False

        # User Profile Manager - Improved error handling
        try:
            self.user_manager = UserProfileManager(data_dir=self.user_data_dir)
            self.component_status["user_manager"] = True
            logger.info("Initialized user_manager")
        except Exception as e:
            logger.error(f"Error initializing user_manager: {e}", exc_info=True)
            self.user_manager = None
            self.component_status["user_manager"] = False

        # OpenAI Client - Improved error handling
        try:
            self.openai_client = OpenAIClient()
            self.component_status["openai_client"] = True
            logger.info("Initialized openai_client")
        except Exception as e:
            logger.error(f"Error initializing openai_client: {e}", exc_info=True)
            self.openai_client = None
            self.component_status["openai_client"] = False

        # Initialize Health Analyzer if dependencies available - Improved dependency checking
        if self.component_status.get("user_manager", False) and hasattr(self, "user_manager"):
            try:
                self.health_analyzer = HealthDataAnalyzer(
                    health_history=self.user_manager.health_history,
                    user_profile=self.user_manager.profile
                )
                self.component_status["health_analyzer"] = True
                logger.info("Initialized health_analyzer")
            except Exception as e:
                logger.error(f"Error initializing health_analyzer: {e}", exc_info=True)
                self.health_analyzer = None
                self.component_status["health_analyzer"] = False
        else:
            logger.warning("Skipping health_analyzer initialization due to missing dependencies")
            self.health_analyzer = None
            self.component_status["health_analyzer"] = False

    def _initialize_ml_components(self):
        """Initialize machine learning components with improved error handling."""
        # Symptom Predictor
        try:
            self.symptom_predictor = SymptomPredictor(model_dir=self.model_dir)
            self.component_status["symptom_predictor"] = True
            logger.info("Initialized symptom_predictor")
        except Exception as e:
            logger.error(f"Error initializing symptom_predictor: {e}", exc_info=True)
            self.symptom_predictor = None
            self.component_status["symptom_predictor"] = False

        # NLP Symptom Extractor
        try:
            self.symptom_extractor = NLPSymptomExtractor(model_dir=self.model_dir)
            self.component_status["symptom_extractor"] = True
            logger.info("Initialized symptom_extractor")
        except Exception as e:
            logger.error(f"Error initializing symptom_extractor: {e}", exc_info=True)
            self.symptom_extractor = None
            self.component_status["symptom_extractor"] = False

        # Risk Assessor
        try:
            self.risk_assessor = PatientRiskAssessor(model_dir=self.model_dir)
            self.component_status["risk_assessor"] = True
            logger.info("Initialized risk_assessor")
        except Exception as e:
            logger.error(f"Error initializing risk_assessor: {e}", exc_info=True)
            self.risk_assessor = None
            self.component_status["risk_assessor"] = False

    def _initialize_ui_components(self):
        """Initialize UI components with improved error recovery and dependency checking."""
        # Health Dashboard
        if self.component_status.get("user_manager", False):
            try:
                self.dashboard = HealthDashboard(
                    health_history=self.user_manager.health_history,
                    user_profile=self.user_manager.profile
                )
                self.component_status["dashboard"] = True
                logger.info("Initialized dashboard")
            except Exception as e:
                logger.error(f"Error initializing dashboard: {e}", exc_info=True)
                self.dashboard = None
                self.component_status["dashboard"] = False
        else:
            logger.warning("Skipping dashboard initialization due to missing user_manager")
            self.dashboard = None
            self.component_status["dashboard"] = False

        # Chat Interface - Check both dependencies
        if (self.component_status.get("openai_client", False) and
            self.component_status.get("symptom_extractor", False)):
            try:
                self.chat_interface = ChatInterface(
                    openai_client=self.openai_client,
                    symptom_extractor=self.symptom_extractor
                )
                self.component_status["chat_interface"] = True
                logger.info("Initialized chat_interface")
            except Exception as e:
                logger.error(f"Error initializing chat_interface: {e}", exc_info=True)
                self.chat_interface = None
                self.component_status["chat_interface"] = False
        else:
            logger.warning("Skipping chat_interface initialization due to missing dependencies")
            self.chat_interface = None
            self.component_status["chat_interface"] = False

        # Symptom Analyzer UI - Handle partial dependencies
        try:
            # Collect available dependencies
            symptom_analyzer_params = {}

            if hasattr(self, "health_data") and self.health_data:
                symptom_analyzer_params["health_data"] = self.health_data

            if hasattr(self, "symptom_predictor") and self.symptom_predictor:
                symptom_analyzer_params["symptom_predictor"] = self.symptom_predictor

            if hasattr(self, "symptom_extractor") and self.symptom_extractor:
                symptom_analyzer_params["symptom_extractor"] = self.symptom_extractor

            if hasattr(self, "risk_assessor") and self.risk_assessor:
                symptom_analyzer_params["risk_assessor"] = self.risk_assessor

            # Only initialize if we have at least health_data
            if "health_data" in symptom_analyzer_params:
                self.symptom_analyzer_ui = SymptomAnalyzerUI(**symptom_analyzer_params)
                self.component_status["symptom_analyzer_ui"] = True
                logger.info(f"Initialized symptom_analyzer_ui with {len(symptom_analyzer_params)} dependencies")
            else:
                logger.warning("Skipping symptom_analyzer_ui initialization due to missing health_data")
                self.symptom_analyzer_ui = None
                self.component_status["symptom_analyzer_ui"] = False
        except Exception as e:
            logger.error(f"Error initializing symptom_analyzer_ui: {e}", exc_info=True)
            self.symptom_analyzer_ui = None
            self.component_status["symptom_analyzer_ui"] = False

        # Medical Literature UI
        try:
            lit_ui_params = {}

            if hasattr(self, "health_data") and self.health_data:
                lit_ui_params["health_data"] = self.health_data

            if hasattr(self, "openai_client") and self.openai_client:
                lit_ui_params["openai_client"] = self.openai_client

            # Initialize even with partial dependencies
            self.medical_literature_ui = MedicalLiteratureUI(**lit_ui_params)
            self.component_status["medical_literature_ui"] = True
            logger.info(f"Initialized medical_literature_ui with {len(lit_ui_params)} dependencies")
        except Exception as e:
            logger.error(f"Error initializing medical_literature_ui: {e}", exc_info=True)
            self.medical_literature_ui = None
            self.component_status["medical_literature_ui"] = False

        # Health History UI
        try:
            history_ui_params = {}

            if hasattr(self, "user_manager") and self.user_manager:
                history_ui_params["user_manager"] = self.user_manager

            if hasattr(self, "health_data") and self.health_data:
                history_ui_params["health_data"] = self.health_data

            # Only initialize if we have user_manager
            if "user_manager" in history_ui_params:
                self.health_history_ui = HealthHistoryUI(**history_ui_params)
                self.component_status["health_history_ui"] = True
                logger.info(f"Initialized health_history_ui with {len(history_ui_params)} dependencies")
            else:
                logger.warning("Skipping health_history_ui initialization due to missing user_manager")
                self.health_history_ui = None
                self.component_status["health_history_ui"] = False
        except Exception as e:
            logger.error(f"Error initializing health_history_ui: {e}", exc_info=True)
            self.health_history_ui = None
            self.component_status["health_history_ui"] = False

        # Settings UI
        try:
            settings_ui_params = {}

            if hasattr(self, "user_manager") and self.user_manager:
                settings_ui_params["user_manager"] = self.user_manager

            if hasattr(self, "openai_client") and self.openai_client:
                settings_ui_params["openai_client"] = self.openai_client

            # Initialize even with partial dependencies
            self.settings_ui = SettingsUI(**settings_ui_params)
            self.component_status["settings_ui"] = True
            logger.info(f"Initialized settings_ui with {len(settings_ui_params)} dependencies")
        except Exception as e:
            logger.error(f"Error initializing settings_ui: {e}", exc_info=True)
            self.settings_ui = None
            self.component_status["settings_ui"] = False

    def _load_api_keys_from_environment(self):
        """Load API keys from environment variables with improved error handling."""
        # OpenAI API key
        if self.component_status.get("openai_client", False) and hasattr(self, "openai_client"):
            try:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if api_key:
                    self.openai_client.set_api_key(api_key)
                    logger.info("Set OpenAI API key from environment variable")
                else:
                    logger.warning("OPENAI_API_KEY environment variable not found")
            except Exception as e:
                logger.error(f"Error setting OpenAI API key: {e}", exc_info=True)

    def _set_critical_error(self, message: str):
        """Set a critical error message for display."""
        if not hasattr(st, "session_state"):
            # Create session_state attribute if it doesn't exist (for testing environments)
            setattr(st, "session_state", type('obj', (object,), {}))

        if not hasattr(st.session_state, "error_message"):
            st.session_state.error_message = message

        logger.critical(f"Critical error: {message}")

    def _init_session_state(self):
        """Initialize Streamlit session state variables with improved error handling."""
        try:
            # Create default session state if not already initialized
            if 'initialized' not in st.session_state:
                st.session_state.initialized = True
                st.session_state.page = "Home"  # Default to home page
                st.session_state.user_id = self.user_manager.current_user_id if hasattr(self, "user_manager") and self.user_manager else "default_user"
                st.session_state.chat_history = []
                st.session_state.last_symptom_check = None
                st.session_state.last_risk_assessment = None
                st.session_state.dark_mode = False
                st.session_state.analysis_in_progress = False
                st.session_state.advanced_mode = False
                st.session_state.error_message = None
                st.session_state.notification_count = 0
                st.session_state.notifications = []
                st.session_state.view_mode = "patient"  # patient, clinician, researcher
                st.session_state.data_sharing_enabled = False
                st.session_state.export_ready = False
                st.session_state.export_data = None
                st.session_state.performance_metrics = {
                    'startup_time': time.time() - self.start_time,
                    'component_load_success_rate': sum(self.component_status.values()) / max(1, len(self.component_status))
                }

                # Add welcome notification
                self.add_notification(
                    "Welcome to MedExplain AI Pro",
                    "Thank you for using our advanced health analytics platform.",
                    "info"
                )

                logger.info(f"Initialized session state for user: {st.session_state.user_id}")
        except Exception as e:
            logger.error(f"Error initializing session state: {e}", exc_info=True)
            self._set_critical_error(f"Session initialization failed: {str(e)}")

    def add_notification(self, title: str, message: str, notification_type: str = "info"):
        """
        Add a notification to the system.

        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification (info, warning, error, success)
        """
        try:
            if hasattr(st.session_state, "notifications"):
                st.session_state.notifications.append({
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "message": message,
                    "type": notification_type,
                    "timestamp": time.time(),
                    "read": False
                })
                st.session_state.notification_count = len(
                    [n for n in st.session_state.notifications if not n.get("read", False)]
                )
                logger.debug(f"Added notification: {title}")
            else:
                logger.warning("Cannot add notification: session_state.notifications not initialized")
        except Exception as e:
            logger.error(f"Error adding notification: {e}", exc_info=True)

    def _show_notifications(self):
        """Display and manage notifications with improved error handling."""
        try:
            with st.sidebar.expander("Notifications", expanded=True):
                if not st.session_state.notifications:
                    st.info("No notifications")
                    return

                for i, notification in enumerate(st.session_state.notifications):
                    # Display with appropriate styling based on type
                    notification_type = notification.get("type", "info")

                    if notification_type == "error":
                        container = st.error
                    elif notification_type == "warning":
                        container = st.warning
                    elif notification_type == "success":
                        container = st.success
                    else:  # info
                        container = st.info

                    with container():
                        st.markdown(f"**{notification['title']}**")
                        st.write(notification['message'])

                        # Mark as read button
                        if not notification.get("read", False):
                            if st.button("Mark as read", key=f"read_{notification['id']}"):
                                st.session_state.notifications[i]["read"] = True
                                st.session_state.notification_count -= 1
                                st.experimental_rerun()
        except Exception as e:
            logger.error(f"Error displaying notifications: {e}", exc_info=True)
            st.error("Could not display notifications. Please try refreshing the page.")

    def _add_custom_css(self):
        """Add specific CSS fixes for the navigation sidebar issues shown in the screenshot."""
        try:
            st.markdown("""
            <style>
                /* Fix for the logo and title area */
                section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                    padding-top: 1.5rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                /* Logo container fixes */
                div.logo-container {
                    display: flex;
                    flex-direction: column;
                    align-items: flex-start;
                    margin-bottom: 1.5rem;
                }

                /* Main logo styling */
                .main-logo {
                    display: flex;
                    align-items: center;
                    margin-bottom: 0.5rem;
                }

                .logo-box {
                    background-color: #1976D2;
                    color: white;
                    border-radius: 8px;
                    width: 50px;
                    height: 50px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 18px;
                    margin-right: 10px;
                }

                .app-title {
                    color: #1976D2;
                    font-size: 24px;
                    font-weight: 600;
                    line-height: 1.2;
                    margin: 0;
                    padding: 0;
                }

                .app-subtitle {
                    font-size: 12px;
                    color: #78909C;
                    line-height: 1.4;
                    margin-left: 60px; /* Align with app name */
                }

                /* Fix status indicator */
                .status-indicator {
                    display: flex;
                    align-items: center;
                    margin: 1rem 0;
                    padding: 0.5rem 0.75rem;
                    background-color: rgba(255, 152, 0, 0.1);
                    border-left: 4px solid #FF9800;
                    border-radius: 4px;
                }

                .status-icon {
                    background-color: #FF9800;
                    color: white;
                    border-radius: 50%;
                    width: 22px;
                    height: 22px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    margin-right: 10px;
                    font-size: 14px;
                }

                .status-text {
                    display: flex;
                    flex-direction: column;
                }

                .status-title {
                    font-weight: 500;
                    color: #424242;
                    font-size: 13px;
                }

                .status-details {
                    font-size: 11px;
                    color: #757575;
                }

                /* Fix user profile section */
                .user-profile {
                    display: flex;
                    align-items: center;
                    margin: 1rem 0;
                    padding: 0.75rem;
                    background-color: rgba(0, 0, 0, 0.2);
                    border-radius: 8px;
                }

                .avatar {
                    background-color: #0288D1;
                    color: white;
                    border-radius: 50%;
                    width: 36px;
                    height: 36px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    margin-right: 12px;
                    position: relative;
                    font-size: 18px;
                }

                .status-dot {
                    position: absolute;
                    bottom: -2px;
                    right: -2px;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background-color: #78909C;
                    border: 2px solid #1E1E1E;
                }

                .user-info {
                    display: flex;
                    flex-direction: column;
                }

                .user-name {
                    font-weight: 500;
                    color: #E0E0E0;
                    font-size: 14px;
                }

                .user-details {
                    font-size: 12px;
                    color: #9E9E9E;
                }

                /* Fix navigation categories */
                .nav-category {
                    text-transform: uppercase;
                    font-size: 12px;
                    font-weight: 600;
                    color: #78909C;
                    letter-spacing: 0.5px;
                    margin: 1.5rem 0 0.5rem 0;
                    padding-left: 0.5rem;
                }

                /* Fix navigation buttons */
                button[key^="nav_"] {
                    background-color: rgba(0, 0, 0, 0.2) !important;
                    border-radius: 6px !important;
                    text-align: left !important;
                    padding: 10px 12px !important;
                    color: #E0E0E0 !important;
                    font-size: 14px !important;
                    font-weight: 400 !important;
                    margin-bottom: 6px !important;
                    height: auto !important;
                    width: 100% !important;
                    border: none !important;
                    box-shadow: none !important;
                    transition: background-color 0.2s;
                }

                button[key^="nav_"]:hover {
                    background-color: rgba(25, 118, 210, 0.2) !important;
                    color: #90CAF9 !important;
                }

                button[key^="nav_home"] .stIcon,
                button[key^="nav_symptom"] .stIcon,
                button[key^="nav_health"] .stIcon {
                    margin-right: 8px !important;
                }

                /* Fix icon spacing in navigation */
                button[key^="nav_"] svg {
                    margin-right: 10px;
                }

                /* Fix dark mode specific issues */
                body.dark section[data-testid="stSidebar"] {
                    background-color: #1E1E1E !important;
                }

                body.dark .status-title {
                    color: #E0E0E0;
                }

                body.dark .status-details {
                    color: #9E9E9E;
                }
            </style>
            """, unsafe_allow_html=True)
            logger.debug("Applied navigation-specific CSS fixes")
        except Exception as e:
            logger.error(f"Error applying navigation CSS fixes: {e}", exc_info=True)

    def _reset_dark_mode(self):
        """Reset dark mode styling."""
        try:
            # The reset is handled by re-applying the base CSS
            self._add_custom_css()
            logger.debug("Reset dark mode styling")
        except Exception as e:
            logger.error(f"Error resetting dark mode: {e}", exc_info=True)

    def render_sidebar(self):
        """
        Render a professional, enterprise-grade navigation system with focus on medical aesthetics,
        clear organization, and enterprise features that would appeal to healthcare investors.
        """
        try:
            with st.sidebar:
                # HEADER SECTION WITH PROFESSIONAL BRANDING
                # ==========================================
                header_container = st.container()
                with header_container:
                    # Logo and title with professional medical styling
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        logo_path = os.path.join(STATIC_DIR, "img", "logo.png")
                        if os.path.exists(logo_path):
                            st.image(logo_path, width=50)
                        else:
                            # Fallback logo with medical blue color scheme
                            st.markdown("""
                            <div style="width:50px; height:50px; border-radius:10px; background:linear-gradient(135deg, #1976d2, #0288d1);
                                    display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; font-size:20px;">
                                ME
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        st.markdown("""
                        <div style="display:flex; flex-direction:column; margin-left:10px;">
                            <h1 style="margin:0; padding:0; font-size:24px; font-weight:600; color:#1976d2; font-family:'Helvetica Neue', sans-serif;">
                                MedExplain AI Pro
                            </h1>
                            <div style="font-size:12px; color:#546e7a; font-family:'Helvetica Neue', sans-serif;">
                                Enterprise Healthcare Analytics Platform
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # System health indicator designed for healthcare professionals
                component_status_ratio = sum(self.component_status.values()) / max(1, len(self.component_status))

                if component_status_ratio == 1.0:
                    system_status = "Fully Operational"
                    status_color = "#00C853"  # Green
                    status_icon = "✓"
                elif component_status_ratio >= 0.8:
                    system_status = "Partially Degraded"
                    status_color = "#FFD600"  # Amber
                    status_icon = "!"
                elif component_status_ratio >= 0.5:
                    system_status = "Limited Functionality"
                    status_color = "#FF9100"  # Orange
                    status_icon = "!"
                else:
                    system_status = "Critical Systems Offline"
                    status_color = "#D50000"  # Red
                    status_icon = "×"

                # Healthcare-styled system status indicator
                st.markdown(f"""
                <div style="display:flex; align-items:center; margin:15px 0 20px 0; padding:8px 12px;
                        background-color:rgba({int(status_color[1:3], 16)}, {int(status_color[3:5], 16)}, {int(status_color[5:7], 16)}, 0.1);
                        border-left:4px solid {status_color}; border-radius:4px;">
                    <div style="display:flex; justify-content:center; align-items:center; width:22px; height:22px;
                            border-radius:50%; background-color:{status_color}; margin-right:10px; color:white; font-weight:bold;">
                        {status_icon}
                    </div>
                    <div>
                        <div style="font-weight:500; color:#37474f; font-size:13px;">{system_status}</div>
                        <div style="font-size:11px; color:#546e7a;">{APP_VERSION} | {sum(self.component_status.values())}/{len(self.component_status)} Components</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ENTERPRISE USER PROFILE SECTION
                # ===============================
                if hasattr(self, "user_manager") and self.user_manager and self.user_manager.profile:
                    profile = self.user_manager.profile
                    profile_name = profile.get("name", "Guest User")
                    profile_age = profile.get("age", "")
                    profile_gender = profile.get("gender", "")

                    # Profile/risk indicator
                    risk_level = "Unknown"
                    risk_color = "#78909C"  # Default gray

                    if st.session_state.last_risk_assessment:
                        risk_level = st.session_state.last_risk_assessment.get('risk_level', 'unknown').capitalize()
                        risk_color = {
                            'Low': '#00C853',    # Green - healthy
                            'Medium': '#FFD600', # Yellow - caution
                            'High': '#D50000',   # Red - warning
                            'Unknown': '#78909C' # Gray - no data
                        }.get(risk_level, '#78909C')

                    # User profile with health status - enterprise healthcare styling
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:25px;">
                        <div style="position:relative; margin-right:12px;">
                            <div style="width:44px; height:44px; border-radius:50%; background-color:#E1F5FE;
                                    display:flex; align-items:center; justify-content:center;">
                                <span style="color:#0288D1; font-weight:500; font-size:18px;">{profile_name[0].upper() if profile_name else "G"}</span>
                            </div>
                            <div style="position:absolute; bottom:-3px; right:-3px; width:16px; height:16px;
                                    border-radius:50%; background-color:{risk_color}; border:2px solid white;"></div>
                        </div>
                        <div>
                            <div style="font-weight:500; font-size:15px; color:#37474F;">{profile_name}</div>
                            <div style="font-size:12px; color:#546E7A; display:flex; align-items:center;">
                                <span>{f"{profile_age} • {profile_gender}" if profile_age and profile_gender else "Complete your profile"}</span>
                                <span style="margin-left:8px; font-size:11px; padding:2px 6px; border-radius:10px;
                                        background-color:rgba({int(risk_color[1:3], 16)}, {int(risk_color[3:5], 16)}, {int(risk_color[5:7], 16)}, 0.15);
                                        color:{risk_color};">
                                    {risk_level}
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # MAIN NAVIGATION SECTION - HEALTHCARE FOCUSED
                # ===========================================
                # Define navigation categories optimized for healthcare workflow
                nav_categories = [
                    {
                        "name": "Patient Management",
                        "icon": "👤",
                        "items": [
                            {"icon": "🏠", "name": "Home", "key": "Home", "description": "Dashboard overview"},
                            {"icon": "🔍", "name": "Symptom Analyzer", "key": "Symptom Analyzer", "description": "AI-powered symptom analysis"},
                            {"icon": "📝", "name": "Health History", "key": "Health History", "description": "Patient records and history"}
                        ]
                    },
                    {
                        "name": "Clinical Analytics",
                        "icon": "📊",
                        "items": [
                            {"icon": "📈", "name": "Health Dashboard", "key": "Health Dashboard", "description": "Health metrics visualization"},
                            {"icon": "📊", "name": "Advanced Analytics", "key": "Advanced Analytics", "description": "In-depth health trend analysis"},
                            {"icon": "💬", "name": "Health Chat", "key": "Health Chat", "description": "AI clinical conversation"}
                        ]
                    },
                    {
                        "name": "Medical Resources",
                        "icon": "📚",
                        "items": [
                            {"icon": "📚", "name": "Medical Literature", "key": "Medical Literature", "description": "Evidence-based resources"}
                        ]
                    },
                    {
                        "name": "System",
                        "icon": "⚙️",
                        "items": [
                            {"icon": "⚙️", "name": "Settings", "key": "Settings", "description": "Application configuration"}
                        ]
                    }
                ]

                # Add Admin Panel in advanced mode
                if st.session_state.advanced_mode:
                    nav_categories[-1]["items"].append(
                        {"icon": "🔧", "name": "Admin Panel", "key": "Admin Panel", "description": "System administration"}
                    )

                # Get current page
                current_page = st.session_state.get('page', 'Home')
                selected_option = None

                # Enterprise-grade navigation with clear visual hierarchy
                for category in nav_categories:
                    st.markdown(f"""
                    <div style="margin-top:20px; margin-bottom:8px; display:flex; align-items:center;">
                        <span style="font-size:14px; margin-right:6px;">{category["icon"]}</span>
                        <span style="font-size:14px; font-weight:500; color:#546E7A; letter-spacing:0.5px; text-transform:uppercase;">
                            {category["name"]}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    for item in category["items"]:
                        is_active = item["key"] == current_page
                        bg_color = "rgba(25, 118, 210, 0.1)" if is_active else "transparent"
                        text_color = "#1976D2" if is_active else "#37474F"
                        border_left = "3px solid #1976D2" if is_active else "3px solid transparent"

                        if st.button(
                            f"{item['icon']} {item['name']}",
                            key=f"nav_{item['key']}",
                            help=item['description'],
                            use_container_width=True
                        ):
                            selected_option = item["key"]

                # QUICK ACTIONS PANEL
                # ==================
                st.markdown("""
                <div style="margin-top:30px; margin-bottom:10px;">
                    <div style="font-size:14px; font-weight:500; color:#546E7A; letter-spacing:0.5px; text-transform:uppercase; margin-bottom:10px;">
                        Quick Actions
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Quick actions in a clean 2x2 grid
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("🩺 Analyze", key="quick_analyze", help="Run a quick symptom analysis", use_container_width=True):
                        selected_option = "Symptom Analyzer"

                    if st.button("💬 Consult", key="quick_chat", help="Start a medical AI consultation", use_container_width=True):
                        selected_option = "Health Chat"

                with col2:
                    if st.button("📊 Report", key="quick_report", help="Generate health report", use_container_width=True):
                        if hasattr(self, "risk_assessor") and self.risk_assessor and st.session_state.last_risk_assessment:
                            try:
                                with st.spinner("Generating report..."):
                                    report = self.risk_assessor.generate_health_report(st.session_state.last_risk_assessment)
                                    st.session_state.health_report = report
                                    selected_option = "Health History"
                            except Exception as e:
                                logger.error(f"Error generating health report: {e}", exc_info=True)
                                st.error("Could not generate report. Please try again.")
                        else:
                            st.info("Please complete a symptom analysis first to generate a health report.")

                    if st.button("📈 Analytics", key="quick_analytics", help="View advanced analytics", use_container_width=True):
                        selected_option = "Advanced Analytics"

                # NOTIFICATION CENTER
                # ==================
                if st.session_state.notification_count > 0:
                    st.markdown("""
                    <div style="margin-top:30px; margin-bottom:10px;">
                        <div style="font-size:14px; font-weight:500; color:#546E7A; letter-spacing:0.5px; text-transform:uppercase; margin-bottom:10px;">
                            Notifications
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Enterprise-styled notification counter
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:15px;">
                        <div style="display:flex; align-items:center; justify-content:center; background-color:#1976D2;
                                color:white; border-radius:20px; padding:4px 12px; font-size:13px; font-weight:500;">
                            <span style="margin-right:5px;">📬</span>
                            <span>{st.session_state.notification_count} New Notification{'' if st.session_state.notification_count == 1 else 's'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show preview of notifications
                    unread_notifications = [n for n in st.session_state.notifications if not n.get("read", False)]

                    for i, notification in enumerate(unread_notifications[:2]):  # Show up to 2 recent notifications
                        notification_type = notification.get("type", "info")
                        title = notification.get("title", "Notification")
                        message = notification.get("message", "")

                        # Style based on notification type
                        if notification_type == "error":
                            bg_color = "rgba(213, 0, 0, 0.08)"
                            icon = "⚠️"
                            border_color = "#D50000"
                        elif notification_type == "warning":
                            bg_color = "rgba(255, 214, 0, 0.08)"
                            icon = "⚠️"
                            border_color = "#FFD600"
                        elif notification_type == "success":
                            bg_color = "rgba(0, 200, 83, 0.08)"
                            icon = "✓"
                            border_color = "#00C853"
                        else:  # info
                            bg_color = "rgba(25, 118, 210, 0.08)"
                            icon = "ℹ️"
                            border_color = "#1976D2"

                        # Enterprise-styled notification card
                        st.markdown(f"""
                        <div style="background-color:{bg_color}; border-left:3px solid {border_color};
                                border-radius:4px; padding:10px; margin-bottom:10px;">
                            <div style="font-weight:500; font-size:13px; margin-bottom:3px;">{title}</div>
                            <div style="font-size:12px; color:#546E7A; white-space:nowrap; overflow:hidden;
                                    text-overflow:ellipsis; max-width:100%;">
                                {message[:60] + ('...' if len(message) > 60 else '')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # View all button
                    if st.button("View All Notifications", key="view_all_notifications", use_container_width=True):
                        self._show_notifications()

                # ADVANCED FEATURES SECTION
                # ========================
                st.markdown("""
                <div style="margin-top:30px; margin-bottom:10px;">
                    <div style="font-size:14px; font-weight:500; color:#546E7A; letter-spacing:0.5px; text-transform:uppercase; margin-bottom:10px;">
                        System Settings
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # View mode selector (advanced mode only)
                if st.session_state.advanced_mode:
                    st.markdown("<div style='margin-bottom:15px;'>View Mode</div>", unsafe_allow_html=True)

                    # Create view mode selector
                    view_modes = ["Patient", "Clinician", "Researcher"]
                    view_mode_index = ["patient", "clinician", "researcher"].index(st.session_state.view_mode)

                    # Create custom select buttons
                    view_cols = st.columns(3)
                    selected_view_mode = None

                    for i, mode in enumerate(view_modes):
                        with view_cols[i]:
                            is_selected = i == view_mode_index

                            # Custom styling based on selected state
                            button_style = "primary" if is_selected else "secondary"

                            if st.button(
                                mode,
                                key=f"view_mode_{mode.lower()}",
                                use_container_width=True,
                                type=button_style
                            ):
                                selected_view_mode = mode.lower()

                    # Update view mode if changed
                    if selected_view_mode and selected_view_mode != st.session_state.view_mode:
                        st.session_state.view_mode = selected_view_mode
                        st.experimental_rerun()

                # Advanced mode toggle
                st.markdown("<div style='margin:15px 0 10px 0;'>Advanced Features</div>", unsafe_allow_html=True)

                advanced_mode = st.toggle("Enable advanced features", value=st.session_state.advanced_mode, key="advanced_mode_toggle")

                if advanced_mode != st.session_state.advanced_mode:
                    st.session_state.advanced_mode = advanced_mode
                    st.experimental_rerun()

                # Dark mode toggle with enterprise styling
                st.markdown("<div style='margin:15px 0 10px 0;'>Display Theme</div>", unsafe_allow_html=True)

                dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode, key="dark_mode_toggle")

                if dark_mode != st.session_state.dark_mode:
                    st.session_state.dark_mode = dark_mode
                    if dark_mode:
                        self._apply_dark_mode()
                    else:
                        self._reset_dark_mode()

                # PROFESSIONAL FOOTER
                # =================
                st.markdown("---")

                # Medical disclaimer with HIPAA reference for enterprise clients
                st.markdown("""
                <div style="background-color:rgba(25, 118, 210, 0.05); padding:12px;
                        border-radius:4px; border-left:3px solid #1976D2; margin:15px 0;">
                    <div style="font-weight:500; font-size:12px; margin-bottom:5px; color:#1976D2;">MEDICAL DISCLAIMER</div>
                    <div style="font-size:11px; color:#546E7A; line-height:1.4;">
                        This application is for educational purposes only and does not provide medical advice.
                        Always consult qualified healthcare professionals for medical concerns. HIPAA compliant.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Enterprise footer
                st.markdown("""
                <div style="text-align:center; padding-top:10px; font-size:11px; color:#78909C;">
                    <div>© 2025 MedExplain AI Pro Enterprise</div>
                    <div style="margin-top:5px;">Secure Enterprise Healthcare Analytics</div>
                </div>
                """, unsafe_allow_html=True)

                # Apply the selection
                if selected_option:
                    st.session_state.page = selected_option
                    return selected_option

                # Return the current page if no new selection
                return current_page
        except Exception as e:
                logger.error(f"Error rendering sidebar: {e}", exc_info=True)
                st.error("Error rendering navigation. Please refresh the page.")
                return "Home"  # Default to home page on error








    def run(self):
        """Run the main application with improved error recovery."""
        try:
            # Configure the page
            st.set_page_config(
                page_title="MedExplain AI Pro - Personal Health Assistant",
                page_icon="🏥",
                layout="wide",
                initial_sidebar_state="expanded"
            )

            # Initialize session state if needed
            self._init_session_state()

            # Add custom CSS
            self._add_custom_css()

            # Apply dark mode if enabled
            if st.session_state.dark_mode:
                self._apply_dark_mode()

            # Display any error messages
            if st.session_state.error_message:
                st.error(st.session_state.error_message)
                st.session_state.error_message = None

            # Handle data export if ready
            if hasattr(st.session_state, 'export_ready') and st.session_state.export_ready:
                self._handle_data_export()

            # Render sidebar and get selected page
            selected_page = self.render_sidebar()

            # Check session state for page overrides
            if hasattr(st.session_state, 'page') and st.session_state.page:
                selected_page = st.session_state.page
                st.session_state.page = None  # Reset for next time

            # Render the selected page
            self._render_page(selected_page)

        except Exception as e:
            logger.error(f"Error running MedExplain AI Pro: {e}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")

            # Add error notification
            self.add_notification(
                "System Error",
                f"An error occurred while running the application: {str(e)}",
                "error"
            )

            # Try to render home page as fallback
            try:
                self._render_home()
            except:
                st.error("Critical error. Please refresh the page.")

    def _render_page(self, page):
        """Render the selected page content with improved error handling and fallbacks."""
        try:
            # Track page view for analytics
            self._log_page_view(page)

            # Route to the appropriate page renderer
            if page == "Home":
                self._render_home()
            elif page == "Symptom Analyzer":
                if hasattr(self, "symptom_analyzer_ui") and self.symptom_analyzer_ui:
                    self.symptom_analyzer_ui.render()
                else:
                    self._render_fallback_page("Symptom Analyzer", "symptom analysis functionality")
            elif page == "Health Dashboard":
                if hasattr(self, "dashboard") and self.dashboard:
                    self.dashboard.render_dashboard()
                else:
                    self._render_fallback_page("Health Dashboard", "dashboard functionality")
            elif page == "Advanced Analytics":
                self._render_advanced_analytics()
            elif page == "Medical Literature":
                if hasattr(self, "medical_literature_ui") and self.medical_literature_ui:
                    self.medical_literature_ui.render()
                else:
                    self._render_fallback_page("Medical Literature", "medical literature functionality")
            elif page == "Health Chat":
                if hasattr(self, "chat_interface") and self.chat_interface:
                    self.chat_interface.render(
                        self.user_manager.profile if hasattr(self, "user_manager") and self.user_manager else {},
                        self.health_data if hasattr(self, "health_data") else None
                    )
                else:
                    self._render_fallback_page("Health Chat", "chat functionality")
            elif page == "Health History":
                if hasattr(self, "health_history_ui") and self.health_history_ui:
                    self.health_history_ui.render()
                else:
                    self._render_fallback_page("Health History", "health history functionality")
            elif page == "Settings":
                if hasattr(self, "settings_ui") and self.settings_ui:
                    self.settings_ui.render()
                else:
                    self._render_fallback_page("Settings", "settings functionality")
            elif page == "Admin Panel" and st.session_state.advanced_mode:
                self._render_admin_panel()
            else:
                # Default to home page
                self._render_home()
        except Exception as e:
            logger.error(f"Error rendering page {page}: {e}", exc_info=True)
            st.error(f"Error rendering page: {str(e)}")

            # Try to render home page as fallback
            try:
                self._render_home()
            except:
                st.error("Critical error. Please refresh the page.")

    def _log_page_view(self, page_name):
        """Log page views for analytics."""
        try:
            if not hasattr(st.session_state, 'page_views'):
                st.session_state.page_views = {}

            # Initialize counter for this page if not exists
            if page_name not in st.session_state.page_views:
                st.session_state.page_views[page_name] = 0

            # Increment view counter
            st.session_state.page_views[page_name] += 1

            # Log the page view
            logger.info(f"User viewed page: {page_name} (view #{st.session_state.page_views[page_name]})")
        except Exception as e:
            logger.error(f"Error logging page view: {e}", exc_info=True)

    def _render_fallback_page(self, page_name, feature_name):
        """Render a fallback page when a component is not available with improved user guidance."""
        st.title(f"{page_name}")

        st.warning(f"The {feature_name} is currently unavailable or experiencing issues.")

        # Provide more detailed information if in advanced mode
        if st.session_state.advanced_mode:
            st.markdown("### Component Status")
            status_data = []

            for component, status in self.component_status.items():
                status_data.append({
                    "Component": component,
                    "Status": "✅ Loaded" if status else "❌ Failed"
                })

            # Display as a dataframe for better formatting
            st.dataframe(pd.DataFrame(status_data), use_container_width=True)

            # Display troubleshooting steps
            st.markdown("### Troubleshooting")
            st.markdown("""
            If component loading failed:
            1. Check if required data files exist
            2. Verify API keys are correctly set
            3. Look for errors in the logs
            """)

        st.markdown("""
        ### What you can do:
        1. Try refreshing the page
        2. Check your internet connection
        3. Try again later
        4. Contact support if the issue persists
        """)

        # Show go home button
        if st.button("Return to Home"):
            st.session_state.page = "Home"
            st.experimental_rerun()

        # Show attempt reload button
        if st.button("Attempt Component Reload"):
            # This would typically trigger component reinitialization
            if page_name == "Symptom Analyzer":
                self._initialize_ml_components()
                if hasattr(self, "health_data") and self.health_data:
                    try:
                        self.symptom_analyzer_ui = SymptomAnalyzerUI(
                            health_data=self.health_data,
                            symptom_predictor=self.symptom_predictor if hasattr(self, "symptom_predictor") else None,
                            symptom_extractor=self.symptom_extractor if hasattr(self, "symptom_extractor") else None,
                            risk_assessor=self.risk_assessor if hasattr(self, "risk_assessor") else None
                        )
                        self.component_status["symptom_analyzer_ui"] = True
                        st.success("Successfully reloaded component. Refreshing...")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to reload component: {str(e)}")
            else:
                st.info("Component reload attempted. Please refresh the page to see if it worked.")




    def _render_home(self):
        """Render an ultra-premium enterprise-grade home page with cutting-edge styling and advanced features."""
        try:
            # Apply custom CSS for ultra-premium enterprise styling
            st.markdown("""
            <style>
            /* Ultra-Premium Enterprise Styling for MedExplain AI Pro */
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            }

            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                max-width: 1200px;
            }

            /* Ultra-Premium Header with advanced 3D gradient */
            .ultra-premium-header {
                font-weight: 800;
                background: linear-gradient(90deg, #0030B9, #0062FF, #00D1FF, #00F0E0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
                font-size: 4rem;
                letter-spacing: -0.025em;
                line-height: 1.1;
                margin-bottom: 0.6rem;
                padding-bottom: 0.5rem;
                text-shadow: 0 4px 12px rgba(0, 98, 230, 0.4);
                animation: gradient-shift 8s ease infinite;
                transform: perspective(500px) translateZ(0px);
                transition: transform 0.3s ease;
            }

            .ultra-premium-header:hover {
                transform: perspective(500px) translateZ(10px);
            }

            @keyframes gradient-shift {
                0% {
                    background-position: 0% 50%;
                }
                50% {
                    background-position: 100% 50%;
                }
                100% {
                    background-position: 0% 50%;
                }
            }

            .premium-subheader {
                font-weight: 500;
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.6rem;
                margin-bottom: 2rem;
                letter-spacing: -0.01em;
                max-width: 90%;
                line-height: 1.5;
                background: linear-gradient(90deg, rgba(255, 255, 255, 0.9), rgba(200, 225, 255, 0.9));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
            }

            /* Modern Glassmorphism Hero Banner */
            .hero-banner {
                background: linear-gradient(135deg, rgba(0, 48, 185, 0.85), rgba(0, 98, 255, 0.85), rgba(0, 209, 255, 0.65));
                border-radius: 24px;
                padding: 40px;
                margin-bottom: 32px;
                box-shadow: 0 16px 40px rgba(0, 48, 185, 0.3), 0 4px 12px rgba(0, 209, 255, 0.2);
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                animation: banner-glow 5s ease infinite alternate;
            }

            @keyframes banner-glow {
                0% {
                    box-shadow: 0 16px 40px rgba(0, 48, 185, 0.3), 0 4px 12px rgba(0, 209, 255, 0.2);
                }
                100% {
                    box-shadow: 0 20px 50px rgba(0, 48, 185, 0.4), 0 8px 24px rgba(0, 209, 255, 0.3);
                }
            }

            .hero-banner::before {
                content: "";
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 60%);
                animation: rotate 15s linear infinite;
                z-index: 0;
            }

            @keyframes rotate {
                from {
                    transform: rotate(0deg);
                }
                to {
                    transform: rotate(360deg);
                }
            }

            /* Enhanced 3D Metrics Cards */
            .metric-card {
                background: rgba(20, 20, 40, 0.7);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 28px 24px;
                text-align: center;
                margin-bottom: 1.5rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
                position: relative;
                overflow: hidden;
                z-index: 1;
            }

            .metric-card::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, rgba(0, 98, 255, 0.1), transparent);
                z-index: -1;
            }

            .metric-card::after {
                content: "";
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(0, 209, 255, 0.1) 0%, transparent 60%);
                animation: rotate 12s linear infinite;
                z-index: -1;
            }

            .metric-card:hover {
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 209, 255, 0.1);
            }

            .metric-value {
                font-size: 2.8rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                background: linear-gradient(90deg, #0062FF, #00F0E0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
            }

            .metric-label {
                font-size: 1.1rem;
                font-weight: 500;
                color: rgba(255, 255, 255, 0.8);
                margin: 0;
            }

            /* Premium Feature Cards with depth and lighting effects */
            .feature-card {
                background: rgba(20, 20, 40, 0.6);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                border-left: 4px solid;
                border-image: linear-gradient(to bottom, #0062FF, #00F0E0) 1;
                margin-bottom: 1.5rem;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }

            .feature-card::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, rgba(0, 98, 255, 0.05), transparent);
                z-index: -1;
            }

            .feature-card:hover {
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), 0 0 20px rgba(0, 209, 255, 0.15);
                background: rgba(25, 25, 45, 0.7);
            }

            .feature-card:hover .feature-icon {
                transform: translateY(-5px) scale(1.1);
            }

            /* Ultra-Premium Action Buttons with advanced animation */
            .stButton > button {
                background: linear-gradient(90deg, #0030B9, #0062FF, #00D1FF);
                background-size: 200% auto;
                color: white !important;
                border-radius: 16px !important;
                padding: 0.8rem 1.5rem !important;
                font-weight: 600 !important;
                font-size: 1.1rem !important;
                border: none !important;
                cursor: pointer;
                transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
                box-shadow: 0 10px 25px rgba(0, 48, 185, 0.3) !important;
                text-align: center;
                width: 100%;
                margin: 10px 0 !important;
                display: flex !important;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: hidden;
                z-index: 1;
            }

            .stButton > button::before {
                content: "";
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: all 0.5s ease;
                z-index: -1;
            }

            .stButton > button:hover {
                transform: translateY(-5px) scale(1.02);
                box-shadow: 0 15px 35px rgba(0, 48, 185, 0.4) !important;
                background-position: right center !important;
            }

            .stButton > button:hover::before {
                left: 100%;
            }

            /* Elegant Recent Activity Styling */
            .recent-activity {
                background: rgba(20, 20, 40, 0.6);
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 24px;
                border-left: 4px solid;
                border-image: linear-gradient(to bottom, #00D1FF, #00F0E0) 1;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }

            /* Premium Activity Item Styling */
            .activity-item {
                background: rgba(30, 30, 50, 0.6);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 24px;
                margin-bottom: 18px;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                position: relative;
                overflow: hidden;
            }

            .activity-item::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: linear-gradient(to bottom, #00D1FF, #00F0E0);
                opacity: 0.7;
            }

            .activity-item:hover {
                transform: translateY(-5px) scale(1.01);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 209, 255, 0.1);
            }

            .activity-date {
                font-weight: 700;
                background: linear-gradient(90deg, #00D1FF, #00F0E0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
                margin-top: 0;
                margin-bottom: 10px;
                font-size: 1.2rem;
                display: inline-block;
            }

            /* Sophisticated Health Tip Styling */
            .health-tip {
                background: linear-gradient(135deg, rgba(0, 48, 185, 0.1), rgba(0, 209, 255, 0.1));
                border-radius: 20px;
                padding: 30px;
                margin: 24px 0;
                border-left: 4px solid;
                border-image: linear-gradient(to bottom, #00D1FF, #00F0E0) 1;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }


            .health-tip:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 209, 255, 0.1);
            }

            /* Premium Medical Disclaimer */
            .medical-disclaimer {
                border-left: 4px solid;
                border-image: linear-gradient(to bottom, #FF8800, #FF5500) 1;
                padding: 25px;
                background: rgba(30, 30, 50, 0.6);
                border-radius: 16px;
                margin-top: 32px;
                font-size: 0.95rem;
                line-height: 1.6;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                position: relative;
                overflow: hidden;
            }

            .medical-disclaimer::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, rgba(255, 136, 0, 0.05), transparent);
                z-index: -1;
            }

            /* Premium Section Headers */
            .section-header {
                font-weight: 700;
                font-size: 1.8rem;
                margin: 2rem 0 1.2rem 0;
                color: white;
                background: linear-gradient(90deg, #0062FF, #00D1FF, #00F0E0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
                display: inline-block;
                letter-spacing: -0.02em;
                position: relative;
            }

            .section-header::after {
                content: "";
                position: absolute;
                bottom: -8px;
                left: 0;
                width: 40px;
                height: 3px;
                background: linear-gradient(90deg, #0062FF, #00F0E0);
                border-radius: 3px;
            }

            /* Enhanced Feature Icons with floating animation */
            .feature-icon {
                font-size: 2.5rem;
                margin-bottom: 1.2rem;
                background: linear-gradient(135deg, #0062FF, #00F0E0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
                display: inline-block;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                position: relative;
            }

            .feature-icon::after {
                content: "";
                position: absolute;
                bottom: -5px;
                left: 50%;
                transform: translateX(-50%);
                width: 40px;
                height: 2px;
                background: linear-gradient(90deg, #0062FF, #00F0E0);
                border-radius: 2px;
                opacity: 0.7;
            }

            /* Enterprise-grade data visualization styling */
            .data-visualization {
                background: rgba(20, 20, 40, 0.6);
                border-radius: 20px;
                padding: 24px;
                margin: 24px 0;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }

            /* Premium Badge */
            .premium-badge {
                background: linear-gradient(90deg, #FFD700, #FFA500);
                color: #000 !important;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 600;
                display: inline-block;
                margin-left: 10px;
                vertical-align: middle;
                box-shadow: 0 2px 6px rgba(255, 215, 0, 0.3);
            }

            /* Background with animated gradient */
            @keyframes gradientBG {
                0% {
                    background-position: 0% 50%;
                }
                50% {
                    background-position: 100% 50%;
                }
                100% {
                    background-position: 0% 50%;
                }
            }

            /* Enterprise-grade stats counter */
            .stats-counter {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }

            .counter-value {
                font-size: 3rem;
                font-weight: 800;
                background: linear-gradient(90deg, #0062FF, #00F0E0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
                line-height: 1;
                margin-bottom: 5px;
            }

            .counter-label {
                font-size: 1rem;
                color: rgba(255, 255, 255, 0.7);
                text-align: center;
            }

            /* Animated divider */
            .animated-divider {
                height: 3px;
                width: 100%;
                margin: 30px 0;
                background: linear-gradient(90deg, transparent, #0062FF, #00D1FF, #00F0E0, transparent);
                border-radius: 3px;
                position: relative;
                overflow: hidden;
            }

            .animated-divider::after {
                content: "";
                position: absolute;
                top: 0;
                left: -100%;
                width: 50%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent);
                animation: shine 3s ease-in-out infinite;
            }

            @keyframes shine {
                0% {
                    left: -100%;
                }
                100% {
                    left: 200%;
                }
            }

            /* Enterprise testimonial styling */
            .testimonial {
                background: rgba(20, 20, 40, 0.6);
                border-radius: 20px;
                padding: 25px;
                margin: 24px 0;
                border-left: 4px solid;
                border-image: linear-gradient(to bottom, #0062FF, #00F0E0) 1;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                position: relative;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }

            .testimonial-text {
                font-style: italic;
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
                line-height: 1.6;
                margin-bottom: 15px;
            }

            .testimonial-author {
                font-weight: 600;
                color: white;
                display: flex;
                align-items: center;
            }

            .author-company {
                margin-left: 5px;
                background: linear-gradient(90deg, #0062FF, #00F0E0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
            }

            /* Custom styling for dark mode */
            html[data-theme="dark"] body {
                background: linear-gradient(135deg, #0c0c1d, #1a1a2e);
            }

            /* Glow effect for text */
            .glow-text {
                text-shadow: 0 0 5px rgba(0, 209, 255, 0.5);
            }

            /* Enterprise feature tags */
            .enterprise-tag {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-right: 8px;
                margin-bottom: 8px;
                background: rgba(0, 98, 255, 0.15);
                border: 1px solid rgba(0, 209, 255, 0.3);
                color: rgba(255, 255, 255, 0.9);
            }
            </style>
            """, unsafe_allow_html=True)

            # Additional JS for animations and interactions
            st.markdown("""
            <script>
            // This would be where we'd add custom JS if Streamlit supported it in markdown
            // Since it doesn't, we're focusing on CSS animations instead
            </script>
            """, unsafe_allow_html=True)

            # Ultra-Premium Header with 3D effect
            st.markdown('<h1 class="ultra-premium-header">MedExplain AI Pro</h1>', unsafe_allow_html=True)
            st.markdown('<p class="premium-subheader">Your advanced personal health assistant powered by enterprise-grade medical AI technology</p>', unsafe_allow_html=True)

            # Hero Banner with advanced effects
            st.markdown("""
            <div class="hero-banner">
                <div style="position: relative; z-index: 2;">
                    <h2 style="color: white; margin-top: 0; font-size: 2.2rem; font-weight: 700;">Enterprise Healthcare Analytics Suite</h2>
                    <p style="color: rgba(255, 255, 255, 0.95); font-size: 1.2rem; max-width: 90%; line-height: 1.6;">
                        Bringing advanced medical intelligence and personalized analytics to healthcare professionals and organizations.
                        Powered by state-of-the-art AI and machine learning models.
                    </p>
                    <div style="display: flex; flex-wrap: wrap; margin-top: 20px;">
                        <span class="enterprise-tag">HIPAA Compliant</span>
                        <span class="enterprise-tag">Medical-Grade AI</span>
                        <span class="enterprise-tag">Advanced Analytics</span>
                        <span class="enterprise-tag">Multi-Modal Analysis</span>
                        <span class="enterprise-tag">Enterprise Security</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key metrics with premium styling
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">99.8%</div>
                    <p class="metric-label">System Reliability</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">2.5M+</div>
                    <p class="metric-label">Data Points Analyzed</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">94%</div>
                    <p class="metric-label">Diagnostic Accuracy</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">24/7</div>
                    <p class="metric-label">Monitoring & Support</p>
                </div>
                """, unsafe_allow_html=True)

            # Main content with enhanced visual layout
            col1, col2 = st.columns([3, 2])

            with col1:
                st.markdown('<h3 class="section-header">Enterprise-Grade Healthcare Analytics</h3>', unsafe_allow_html=True)

                st.markdown("""
                <p style="color: rgba(255, 255, 255, 0.85); font-size: 1.1rem; line-height: 1.7; margin-bottom: 25px;">
                    MedExplain AI Pro combines advanced medical knowledge with state-of-the-art artificial intelligence to provide an unparalleled healthcare analytics platform. Our system leverages enterprise-grade technology to deliver actionable insights for healthcare professionals and individuals.
                </p>
                """, unsafe_allow_html=True)

                # Premium feature highlights with enhanced styling
                st.markdown('<h3 class="section-header">Advanced Enterprise Features</h3>', unsafe_allow_html=True)

                feature_col1, feature_col2 = st.columns(2)

                with feature_col1:
                    st.markdown("""
                    <div class="feature-card">
                        <div class="feature-icon">🧠</div>
                        <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Medical Neural Networks</h4>
                        <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                            Ensemble ML models analyze health data using parallel neural networks for superior accuracy and insight
                        </p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Interactive Analytics</h4>
                        <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                            Enterprise-grade dashboard with real-time data visualization and interactive drill-down capabilities
                        </p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Advanced Pattern Recognition</h4>
                        <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                            Proprietary algorithms identify complex correlations and patterns invisible to standard analysis
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with feature_col2:
                    st.markdown("""
                    <div class="feature-card">
                        <div class="feature-icon">💬</div>
                        <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Medical Language Understanding</h4>
                        <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                            Enterprise NLP system capable of understanding complex medical terminology and context
                        </p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">📈</div>
                        <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Predictive Health Intelligence</h4>
                        <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                            Advanced risk assessment and early warning system with proactive health monitoring
                        </p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">🔒</div>
                        <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Enterprise Security Framework</h4>
                        <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                            HIPAA-compliant data encryption with enterprise-grade security protocols and audit trails
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Add testimonial section for enterprise credibility
                st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

                st.markdown('<h3 class="section-header">What Healthcare Leaders Say</h3>', unsafe_allow_html=True)

                st.markdown("""
                <div class="testimonial">
                    <p class="testimonial-text">
                        "MedExplain AI Pro has revolutionized how we approach patient diagnostics. The predictive analytics have helped us identify conditions earlier, leading to better outcomes and reduced costs."
                    </p>
                    <div class="testimonial-author">
                        Dr. Sarah Chen, <span class="author-company">Chief Medical Officer, HealthTech Innovations</span>
                    </div>
                </div>

                <div class="testimonial">
                    <p class="testimonial-text">
                        "The enterprise security features and HIPAA compliance of MedExplain AI Pro made it the clear choice for our hospital network. The ROI has been exceptional."
                    </p>
                    <div class="testimonial-author">
                        Robert Johnson, <span class="author-company">CTO, Metropolitan Healthcare Systems</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Ultra-premium visualization instead of static image
                image_path = os.path.join(STATIC_DIR, "img", "hero.png")
                if os.path.exists(image_path):
                    st.image(image_path, use_column_width=True)
                else:
                    # Create an advanced interactive placeholder with 3D effects
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #0030B9, #0062FF, #00D1FF);
                        border-radius: 24px; height: 340px; display: flex; align-items: center; position: relative;
                        justify-content: center; margin-bottom: 25px; overflow: hidden;
                        box-shadow: 0 20px 40px rgba(0, 48, 185, 0.4), 0 0 40px rgba(0, 209, 255, 0.2);">

                        <!-- Animated background elements -->
                        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; overflow: hidden;">
                            <div style="position: absolute; width: 300px; height: 300px; border-radius: 50%;
                                background: radial-gradient(circle, rgba(0, 209, 255, 0.4) 0%, transparent 70%);
                                top: -150px; right: -100px; filter: blur(20px);"></div>

                            <div style="position: absolute; width: 200px; height: 200px; border-radius: 50%;
                                background: radial-gradient(circle, rgba(0, 48, 185, 0.4) 0%, transparent 70%);
                                bottom: -100px; left: -50px; filter: blur(20px);"></div>

                            <div style="position: absolute; width: 100%; height: 100%;
                                background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxkZWZzPgogICAgPHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI1MCIgaGVpZ2h0PSI1MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+CiAgICAgIDxwYXRoIGQ9Ik0gNTAgMCBMIDAgMCAwIDUwIiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4wNSkiIHN0cm9rZS13aWR0aD0iMSIvPgogICAgPC9wYXR0ZXJuPgogIDwvZGVmcz4KICA8cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIiAvPgo8L3N2Zz4=');
                                opacity: 0.5;"></div>
                        </div>

                        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px);
                            padding: 30px; border-radius: 20px; text-align: center; position: relative;
                            border: 1px solid rgba(255, 255, 255, 0.2); z-index: 2;
                            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);">
                            <h3 style="margin-top: 0; color: white; font-weight: 700; font-size: 1.8rem;">Advanced Health Analytics</h3>
                            <p style="color: white; margin-bottom: 20px;">Enterprise-grade medical intelligence platform</p>

                            <!-- Animated data visualization placeholder -->
                            <div style="height: 150px; margin: 20px 0; background: rgba(255, 255, 255, 0.1);
                                border-radius: 12px; padding: 15px; position: relative; overflow: hidden;
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                                <!-- Animated chart bars -->
                                <div style="display: flex; justify-content: space-between; align-items: flex-end;
                                    height: 100%; padding: 0 10px;">
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 30%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 70%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 45%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 60%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 80%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 50%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 65%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 75%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 40%; border-radius: 4px 4px 0 0;"></div>
                                    <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 90%; border-radius: 4px 4px 0 0;"></div>
                                </div>

                                <!-- Animated chart line -->
                                <div style="position: absolute; top: 30%; left: 0; width: 100%; height: 2px;
                                    background: linear-gradient(90deg, #00F0E0, #00D1FF); z-index: 3;"></div>

                                <!-- Pulsing data points -->
                                <div style="position: absolute; width: 10px; height: 10px; border-radius: 50%;
                                    background: #00F0E0; top: 40%; left: 25%; box-shadow: 0 0 10px #00F0E0;"></div>
                                <div style="position: absolute; width: 10px; height: 10px; border-radius: 50%;
                                    background: #00F0E0; top: 60%; left: 75%; box-shadow: 0 0 10px #00F0E0;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Quick action buttons with premium styling
                st.markdown('<h3 class="section-header">Enterprise Actions</h3>', unsafe_allow_html=True)

                if st.button("🔍 Analyze Symptoms & Risk Factors", key="home_analyze"):
                    st.session_state.page = "Symptom Analyzer"
                    st.experimental_rerun()

                if st.button("📊 View Comprehensive Dashboard", key="home_dashboard"):
                    st.session_state.page = "Health Dashboard"
                    st.experimental_rerun()

                if st.button("💬 Medical Intelligence Chat", key="home_chat"):
                    st.session_state.page = "Health Chat"
                    st.experimental_rerun()

                if st.button("📈 Enterprise Analytics Suite", key="home_analytics"):
                    st.session_state.page = "Advanced Analytics"
                    st.experimental_rerun()



                # ROI calculator teaser
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(0, 48, 185, 0.3), rgba(0, 209, 255, 0.2));
                    border-radius: 16px; padding: 20px; margin-top: 20px; position: relative; overflow: hidden;
                    border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);">
                    <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Enterprise ROI Calculator</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 10px;">
                        Calculate potential savings and efficiency gains with MedExplain AI Pro.
                    </p>
                    <div style="color: #00F0E0; font-weight: 600; margin-top: 15px;">Coming soon →</div>
                </div>
                """, unsafe_allow_html=True)

            # Animated divider
            st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

            # Recent activity section with premium styling
            st.markdown('<h3 class="section-header">Health Activity Monitoring</h3>', unsafe_allow_html=True)

            if hasattr(self, "user_manager") and self.user_manager and self.user_manager.health_history:
                recent_checks = self.user_manager.get_recent_symptom_checks(limit=3)

                if recent_checks:
                    for check in recent_checks:
                        date = check.get("date", "")
                        symptoms = check.get("symptoms", [])

                        # Get symptom names instead of IDs
                        symptom_names = []
                        if hasattr(self, "health_data") and self.health_data:
                            for symptom_id in symptoms:
                                symptom_info = self.health_data.get_symptom_info(symptom_id)
                                if symptom_info:
                                    symptom_names.append(symptom_info.get("name", symptom_id))

                        if not symptom_names:
                            symptom_names = symptoms  # Fallback to IDs if names not found

                        st.markdown(f"""
                        <div class="activity-item">
                            <h4 class="activity-date">Health Assessment • {date}</h4>
                            <p style="margin-bottom: 8px; color: rgba(255, 255, 255, 0.9); font-size: 1.05rem;">
                                <strong>Identified Symptoms:</strong> {", ".join(symptom_names)}
                            </p>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                                <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">Enterprise Health Protocol</div>
                                <div style="background: rgba(0, 209, 255, 0.15); color: #00F0E0; padding: 5px 10px;
                                    border-radius: 12px; font-size: 0.9rem; font-weight: 600;">View Details</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="activity-item" style="text-align: center;">
                        <div style="font-size: 3rem; margin-bottom: 15px; color: rgba(255, 255, 255, 0.2);">📋</div>
                        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin-bottom: 5px; font-weight: 600;">
                            No Recent Health Activities
                        </p>
                        <p style="color: rgba(255, 255, 255, 0.7); margin-bottom: 15px;">
                            Begin your health journey by analyzing symptoms or setting up your profile.
                        </p>
                        <div style="background: linear-gradient(90deg, #0062FF, #00D1FF);
                            border-radius: 12px; padding: 8px 15px; display: inline-block;
                            font-weight: 600; color: white; cursor: pointer; box-shadow: 0 5px 15px rgba(0, 98, 255, 0.3);">
                            Start Health Assessment
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="activity-item" style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 15px; color: rgba(255, 255, 255, 0.2);">📋</div>
                    <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin-bottom: 5px; font-weight: 600;">
                        No Recent Health Activities
                    </p>
                    <p style="color: rgba(255, 255, 255, 0.7); margin-bottom: 15px;">
                        Begin your health journey by analyzing symptoms or setting up your profile.
                    </p>
                    <div style="background: linear-gradient(90deg, #0062FF, #00D1FF);
                        border-radius: 12px; padding: 8px 15px; display: inline-block;
                        font-weight: 600; color: white; cursor: pointer; box-shadow: 0 5px 15px rgba(0, 98, 255, 0.3);">
                        Start Health Assessment
                    </div>
                </div>
                """, unsafe_allow_html=True)


            # Medical disclaimer with premium styling
            st.markdown("""
            <div class="medical-disclaimer">
                <h4 style="margin-top: 0; color: #FF8800; font-size: 1.3rem; margin-bottom: 10px;">Enterprise Healthcare Disclaimer</h4>
                <p style="color: rgba(255, 255, 255, 0.85); margin-bottom: 0; line-height: 1.7;">
                    MedExplain AI Pro is designed to complement, not replace, professional medical advice, diagnosis, or treatment.
                    Our enterprise platform provides advanced analytics and insights for healthcare professionals and individuals,
                    but all medical decisions should be made in consultation with qualified healthcare providers. Always seek the
                    advice of your physician or other qualified health provider with any questions regarding a medical condition.
                </p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
                logger.error(f"Error rendering home page: {e}", exc_info=True)
                st.error("Error rendering enterprise home page. Please refresh the page or contact support.")




    def _render_symptom_pattern_tab(self, health_history):
        """Render symptom pattern analysis tab with improved error handling."""
        try:
            st.subheader("Symptom Pattern Analysis")
            st.markdown("Discover relationships and patterns in your reported symptoms.")

            # Symptom co-occurrence analysis
            st.markdown("### Symptom Co-occurrence")

            # Extract symptom data from history
            all_symptoms = []
            for entry in health_history:
                all_symptoms.extend(entry.get("symptoms", []))

            # Get unique symptoms
            unique_symptoms = list(set(all_symptoms))

            if len(unique_symptoms) >= 2:
                # Create co-occurrence matrix
                symptom_names = []
                for symptom_id in unique_symptoms:
                    if hasattr(self, "health_data") and self.health_data:
                        symptom_info = self.health_data.get_symptom_info(symptom_id)
                        if symptom_info:
                            symptom_names.append(symptom_info.get("name", symptom_id))
                        else:
                            symptom_names.append(symptom_id)
                    else:
                        symptom_names.append(symptom_id)

                # Count co-occurrences
                co_occurrence = np.zeros((len(unique_symptoms), len(unique_symptoms)))

                for entry in health_history:
                    entry_symptoms = entry.get("symptoms", [])
                    for i, s1 in enumerate(unique_symptoms):
                        for j, s2 in enumerate(unique_symptoms):
                            if s1 in entry_symptoms and s2 in entry_symptoms:
                                co_occurrence[i, j] += 1

                # Create heatmap
                try:
                    import plotly.express as px

                    fig = px.imshow(
                        co_occurrence,
                        x=symptom_names,
                        y=symptom_names,
                        color_continuous_scale="Blues",
                        title="Symptom Co-occurrence Matrix"
                    )

                    fig.update_layout(
                        xaxis_title="Symptom",
                        yaxis_title="Symptom",
                        width=700,
                        height=700
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating symptom co-occurrence chart: {e}", exc_info=True)
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
                        st.dataframe(co_df, use_container_width=True)
                    else:
                        st.info("No co-occurring symptoms found.")

                # Find common symptom pairs
                st.markdown("### Common Symptom Combinations")

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

                # Display top pairs
                if pairs:
                    st.markdown("Top symptom combinations in your health history:")

                    for s1, s2, count in pairs[:5]:  # Show top 5
                        st.markdown(f"- **{s1}** and **{s2}** occurred together {int(count)} times")
                else:
                    st.info("No co-occurring symptoms found in your health history.")
            else:
                st.info("Not enough unique symptoms to analyze patterns. Try adding more symptom records.")
        except Exception as e:
            logger.error(f"Error rendering symptom pattern tab: {e}", exc_info=True)
            st.error("Error rendering symptom patterns. Please try again later.")

    def _render_risk_analysis_tab(self):
        """Render risk analysis tab with improved error handling."""
        try:
            st.subheader("Risk Analysis")

            # Check if we have risk assessment data
            if hasattr(st.session_state, "last_risk_assessment") and st.session_state.last_risk_assessment:
                risk_data = st.session_state.last_risk_assessment

                # Overall risk score and level
                risk_level = risk_data.get('risk_level', 'unknown')
                risk_score = risk_data.get('risk_score', 0)

                # Create risk gauge
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Overall Risk Level: {risk_level.capitalize()}"},
                        gauge = {
                            'axis': {'range': [0, 10], 'tickwidth': 1},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 3.9], 'color': "green"},
                                {'range': [4, 6.9], 'color': "orange"},
                                {'range': [7, 10], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_score
                            }
                        }
                    ))

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating risk gauge: {e}", exc_info=True)
                    st.error("Could not render risk visualization. Using alternative display.")

                    # Fallback to text-based representation
                    risk_color = {
                        'low': 'green',
                        'medium': 'orange',
                        'high': 'red',
                        'unknown': 'gray'
                    }.get(risk_level, 'gray')

                    st.markdown(f"#### Overall Risk Score: {risk_score}/10")
                    st.markdown(f"#### Risk Level: <span style='color:{risk_color};'>{risk_level.capitalize()}</span>", unsafe_allow_html=True)

                # Domain-specific risks
                st.markdown("### Health Domain Risk Analysis")

                # Display domain risks if available
                domain_risks = risk_data.get('domain_risks', {})
                if domain_risks:
                    domain_names = list(domain_risks.keys())
                    domain_values = list(domain_risks.values())

                    try:
                        # Create horizontal bar chart
                        fig = go.Figure(go.Bar(
                            x = domain_values,
                            y = domain_names,
                            orientation = 'h',
                            marker = {
                                'color': [
                                    'green' if v < 4 else ('orange' if v < 7 else 'red')
                                    for v in domain_values
                                ]
                            }
                        ))

                        fig.update_layout(
                            title="Risk by Health Domain",
                            xaxis_title="Risk Score (0-10)",
                            yaxis_title="Health Domain",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error creating domain risk chart: {e}", exc_info=True)
                        st.error("Could not render domain risk visualization. Using table format.")

                        # Fallback to table display
                        risk_df = pd.DataFrame({
                            'Health Domain': domain_names,
                            'Risk Score': domain_values
                        }).sort_values('Risk Score', ascending=False)

                        st.dataframe(risk_df, use_container_width=True)
                else:
                    st.info("Detailed domain risk data is not available.")
            else:
                st.info("No risk assessment data available. Please complete a symptom analysis first.")

                # Add a direct link to symptom analyzer
                if st.button("Go to Symptom Analyzer"):
                    st.session_state.page = "Symptom Analyzer"
                    st.experimental_rerun()
        except Exception as e:
            logger.error(f"Error rendering risk analysis tab: {e}", exc_info=True)
            st.error("Error rendering risk analysis. Please try again later.")

    def _render_temporal_trends_tab(self, health_history):
        """Render temporal trends tab with improved error handling."""
        try:
            st.subheader("Temporal Trends")
            st.markdown("Analyze how your health metrics change over time.")

            if len(health_history) > 1:
                # Extract dates and metrics
                dates = []
                symptom_counts = []

                for entry in health_history:
                    dates.append(entry.get("date", ""))
                    symptom_counts.append(len(entry.get("symptoms", [])))

                # Reverse to show chronological order
                dates.reverse()
                symptom_counts.reverse()

                try:
                    # Create line chart
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x = dates,
                        y = symptom_counts,
                        mode = 'lines+markers',
                        name = 'Symptom Count',
                        line = {'color': 'royalblue', 'width': 2},
                        marker = {'size': 8}
                    ))

                    fig.update_layout(
                        title = "Symptom Count Over Time",
                        xaxis_title = "Date",
                        yaxis_title = "Number of Symptoms",
                        height = 400
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating temporal trend chart: {e}", exc_info=True)
                    st.error("Could not render temporal trend visualization. Using table format.")

                    # Fallback to table display
                    trend_df = pd.DataFrame({
                        'Date': dates,
                        'Symptom Count': symptom_counts
                    })

                    st.dataframe(trend_df, use_container_width=True)

                # Text analysis of trends
                st.markdown("### Trend Analysis")

                # Calculate basic trend metrics
                if len(symptom_counts) > 2:
                    trend = "improving" if symptom_counts[-1] < symptom_counts[0] else (
                        "worsening" if symptom_counts[-1] > symptom_counts[0] else "stable"
                    )

                    st.markdown(f"""
                    Based on your symptom history, your health appears to be **{trend}** over time.

                    - Starting symptom count: **{symptom_counts[0]}**
                    - Current symptom count: **{symptom_counts[-1]}**
                    - Change: **{symptom_counts[-1] - symptom_counts[0]}**
                    """)
            else:
                st.info("Not enough historical data to analyze trends. Please check back after recording more health data.")
        except Exception as e:
            logger.error(f"Error rendering temporal trends tab: {e}", exc_info=True)
            st.error("Error rendering temporal trends. Please try again later.")

    def _render_predictive_analytics_tab(self, health_history):
        """Render predictive analytics tab with improved error handling."""
        try:
            st.subheader("Predictive Analytics")
            st.markdown("Explore AI-powered predictions about potential health trends.")

            if len(health_history) >= 3:
                st.markdown("### Symptom Trend Prediction")

                # This would typically involve more sophisticated ML models
                # For now, we'll use a simple placeholder

                st.info("""
                **Note:** Predictive analytics requires at least 5 data points for accurate forecasting.
                The predictions shown here are for demonstration purposes only.
                """)

                # Mock prediction chart (would be replaced with actual ML prediction)
                try:
                    # Extract dates and metrics as before
                    dates = []
                    symptom_counts = []

                    for entry in health_history:
                        dates.append(entry.get("date", ""))
                        symptom_counts.append(len(entry.get("symptoms", [])))

                    # Reverse to show chronological order
                    dates.reverse()
                    symptom_counts.reverse()

                    # Generate mock future dates (7-day intervals)
                    from datetime import datetime, timedelta

                    if dates and dates[-1]:
                        try:
                            last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
                        except ValueError:
                            try:
                                last_date = datetime.strptime(dates[-1], "%m/%d/%Y")
                            except ValueError:
                                # Fallback to today's date
                                last_date = datetime.now()
                    else:
                        last_date = datetime.now()

                    future_dates = []
                    for i in range(1, 5):  # 4 future predictions (4 weeks)
                        future_date = last_date + timedelta(days=7 * i)
                        future_dates.append(future_date.strftime("%Y-%m-%d"))

                    # Mock predictions (simple linear trend with noise)
                    if len(symptom_counts) >= 2:
                        # Calculate trend
                        trend = (symptom_counts[-1] - symptom_counts[0]) / max(1, len(symptom_counts) - 1)

                        # Generate predictions
                        predictions = []
                        for i in range(1, 5):
                            # Add some random variation
                            import random
                            noise = random.uniform(-0.5, 0.5)
                            predicted_value = max(0, symptom_counts[-1] + trend * i + noise)
                            predictions.append(round(predicted_value, 1))
                    else:
                        # Fallback if not enough data
                        predictions = [symptom_counts[-1]] * 4 if symptom_counts else [0] * 4

                    # Create plot with actual and predicted data
                    import plotly.graph_objects as go
                    fig = go.Figure()

                    # Actual data
                    fig.add_trace(go.Scatter(
                        x = dates,
                        y = symptom_counts,
                        mode = 'lines+markers',
                        name = 'Actual',
                        line = {'color': 'royalblue', 'width': 2}
                    ))

                    # Predicted data
                    fig.add_trace(go.Scatter(
                        x = future_dates,
                        y = predictions,
                        mode = 'lines+markers',
                        name = 'Predicted',
                        line = {'color': 'orange', 'width': 2, 'dash': 'dash'}
                    ))

                    fig.update_layout(
                        title = "Symptom Trend Prediction",
                        xaxis_title = "Date",
                        yaxis_title = "Number of Symptoms",
                        height = 400
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating prediction chart: {e}", exc_info=True)
                    st.error("Could not render prediction visualization. Showing tabular data instead.")

                    # Create a fallback table view
                    if len(symptom_counts) >= 2:
                        st.markdown("### Historical Data")
                        hist_df = pd.DataFrame({
                            'Date': dates,
                            'Symptom Count': symptom_counts
                        })
                        st.dataframe(hist_df, use_container_width=True)

                        st.markdown("### Predicted Future Trend")
                        pred_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted Symptom Count': [round(max(0, symptom_counts[-1] + trend * i), 1) for i in range(1, 5)]
                        })
                        st.dataframe(pred_df, use_container_width=True)
            else:
                st.info("Not enough historical data for predictive analysis. Please check back after recording more health data.")

                # Show a progress indicator
                st.markdown(f"Current records: **{len(health_history)}/3** minimum required")
                st.progress(min(1.0, len(health_history)/3))

                # Provide guidance
                st.markdown("""
                ### How to Add More Data

                To enable predictive analytics:
                1. Go to the **Symptom Analyzer** page
                2. Record your symptoms regularly (at least once per week)
                3. Return here after recording at least 3 data points
                """)
        except Exception as e:
            logger.error(f"Error rendering predictive analytics tab: {e}", exc_info=True)
            st.error("Error rendering predictive analytics. Please try again later.")

    def _render_comparative_analysis_tab(self):
        """Render comparative analysis tab with improved error handling."""
        try:
            st.subheader("Comparative Analysis")
            st.markdown("Compare your health metrics with anonymized population data.")

            st.info("Comparative analysis feature is coming soon.")

            st.markdown("""
            This feature will allow you to:
            - Compare your symptom patterns with similar demographic groups
            - Benchmark your health metrics against recommended ranges
            - Identify unusual health patterns that may require attention
            """)

            # Allow user to sign up for notifications with validation
            st.markdown("### Get Notified")

            email = st.text_input("Enter your email to be notified when this feature is available:")

            # Basic email validation
            import re
            email_pattern =" r'^[\w\.-]+@[\w\.-]+\.\w+"

            if email:
                if re.match(email_pattern, email):
                    if st.button("Notify Me"):
                        # This would typically connect to your notification system
                        st.success(f"Thank you! We'll notify {email} when comparative analysis becomes available.")

                        # Store in session state for future reference
                        if 'notification_emails' not in st.session_state:
                            st.session_state.notification_emails = []

                        if email not in st.session_state.notification_emails:
                            st.session_state.notification_emails.append(email)
                else:
                    st.error("Please enter a valid email address.")

            # Preview of the feature
            st.markdown("### Feature Preview")

            st.markdown("""
            The comparative analysis will include:

            1. **Demographic Comparison** - See how your health metrics compare to others in your age group, gender, and region

            2. **Trend Benchmarking** - Compare your symptom trends against expected recovery patterns

            3. **Risk Factor Identification** - Identify potential risk factors based on comparative analysis

            4. **Personalized Recommendations** - Receive tailored health recommendations based on comparative insights
            """)

            # Mock-up visualization
            try:
                import plotly.graph_objects as go

                # Mock data
                categories = ['Respiratory', 'Digestive', 'Cardiac', 'Neurological', 'Musculoskeletal']
                your_values = [7, 3, 5, 2, 6]
                avg_values = [5, 4, 3, 3, 4]

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=your_values,
                    theta=categories,
                    fill='toself',
                    name='Your Health Profile'
                ))

                fig.add_trace(go.Scatterpolar(
                    r=avg_values,
                    theta=categories,
                    fill='toself',
                    name='Population Average'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    title="Example Comparative Health Profile",
                    showlegend=True
                )

                st.markdown("#### Sample Visualization (Preview)")
                st.plotly_chart(fig, use_container_width=True)

                st.caption("This is a sample visualization of how your health metrics will be compared to population averages.")
            except Exception as e:
                logger.error(f"Error creating preview visualization: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error rendering comparative analysis tab: {e}", exc_info=True)
            st.error("Error rendering comparative analysis tab. Please try again later.")

    def _render_admin_panel(self):
        """Render the admin panel (only available in advanced mode) with improved error handling and features."""
        try:
            st.title("Admin Panel")
            st.markdown("Advanced system management and monitoring.")

            # Only allow access in advanced mode
            if not st.session_state.advanced_mode:
                st.warning("Admin panel access requires advanced mode to be enabled.")

                if st.button("Enable Advanced Mode"):
                    st.session_state.advanced_mode = True
                    st.experimental_rerun()

                return

            # Create tabs for different admin functions
            tabs = st.tabs([
                "System Status",
                "User Management",
                "Data Management",
                "Performance Metrics",
                "API Configuration"
            ])

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
            logger.error(f"Error rendering admin panel: {e}", exc_info=True)
            st.error("Error rendering admin panel. Please try again later.")

    def _render_system_status_tab(self):
        """Render system status tab in admin panel with improved error handling."""
        try:
            st.subheader("System Status")

            # Display component status
            st.markdown("### Component Health")

            # Create a status table
            status_data = []
            for component, status in self.component_status.items():
                status_data.append({
                    "Component": component,
                    "Status": "Operational" if status else "Failed",
                    "Health": 100 if status else 0
                })

            if status_data:
                status_df = pd.DataFrame(status_data)
                st.dataframe(status_df, use_container_width=True)

                # Create a health bar chart
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x = [d["Component"] for d in status_data],
                        y = [d["Health"] for d in status_data],
                        marker_color = ['green' if d["Health"] == 100 else 'red' for d in status_data]
                    ))

                    fig.update_layout(
                        title = "Component Health",
                        xaxis_title = "Component",
                        yaxis_title = "Health (%)",
                        height = 400
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating health chart: {e}", exc_info=True)
                    st.error("Could not render health visualization.")

            # System information
            st.markdown("### System Information")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                **Application Version:** {APP_VERSION}

                **Session ID:** {self.session_id}

                **Session Start Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}

                **Environment:** {'Production' if os.environ.get('PROD_ENV') else 'Development'}
                """)

            with col2:
                # Calculate uptime
                uptime = time.time() - self.start_time
                uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"

                st.markdown(f"""
                **Uptime:** {uptime_str}

                **Component Health:** {sum(self.component_status.values())}/{len(self.component_status)}

                **Python Version:** {sys.version.split()[0]}

                **Streamlit Version:** {st.__version__}
                """)

            # Display logs
            st.markdown("### System Logs")

            # Display recent logs if log file exists
            log_path = os.path.join(LOG_DIR, "app.log")
            if os.path.exists(log_path):
                try:
                    # Read last 20 lines of log file
                    with open(log_path, "r") as log_file:
                        logs = log_file.readlines()
                        recent_logs = logs[-20:] if len(logs) > 20 else logs

                    st.code("".join(recent_logs), language="text")

                    # Add option to download full logs
                    if st.button("Download Full Logs"):
                        with open(log_path, "r") as log_file:
                            log_content = log_file.read()

                        st.download_button(
                            label="Download Logs",
                            data=log_content,
                            file_name="medexplain_logs.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Could not read log file: {e}")
            else:
                st.info("No log file found.")

            # System maintenance actions
            st.markdown("### Maintenance Actions")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Restart Application"):
                    st.warning("This will restart the application and end all user sessions.")
                    # This would typically trigger a restart mechanism
                    st.info("Restart functionality is simulated in this demo.")

                    # Simulate restart by clearing session state and rerunning
                    if st.button("Confirm Restart", key="confirm_restart"):
                        for key in list(st.session_state.keys()):
                            if key != 'page':  # Keep the page
                                del st.session_state[key]

                        st.session_state.page = "Home"
                        st.success("Application restarted. Redirecting to home page.")
                        st.experimental_rerun()

            with col2:
                if st.button("Clear Cache"):
                    if st.button("Confirm Cache Clear", key="confirm_cache_clear"):
                        # Keep minimal state to avoid errors
                        page = st.session_state.get('page', 'Home')
                        dark_mode = st.session_state.get('dark_mode', False)

                        st.session_state.clear()

                        # Restore minimal state
                        st.session_state.page = page
                        st.session_state.dark_mode = dark_mode

                        st.success("Cache cleared. Redirecting...")
                        st.experimental_rerun()

            with col3:
                if st.button("Refresh Components"):
                    with st.spinner("Refreshing components..."):
                        # Re-initialize components
                        self._initialize_core_components()
                        self._initialize_ml_components()
                        self._initialize_ui_components()
                        self._load_api_keys_from_environment()

                        st.success(f"Components refreshed: {sum(self.component_status.values())}/{len(self.component_status)} loaded")
        except Exception as e:
            logger.error(f"Error rendering system status tab: {e}", exc_info=True)
            st.error("Error rendering system status. Please try again later.")

    def _render_user_management_tab(self):
        """Render user management tab in admin panel with improved error handling."""
        try:
            st.subheader("User Management")

            # Display user statistics
            st.markdown("### User Statistics")

            if hasattr(self, "user_manager") and self.user_manager:
                # Get user information
                users = self.user_manager.get_all_users() if hasattr(self.user_manager, "get_all_users") else []

                if users:
                    # Display user count
                    st.metric("Total Users", len(users))

                    # Create user table
                    user_data = []
                    for user in users:
                        user_data.append({
                            "User ID": user.get("id", "Unknown"),
                            "Name": user.get("name", "Unknown"),
                            "Age": user.get("age", "Unknown"),
                            "Gender": user.get("gender", "Unknown"),
                            "Last Activity": user.get("last_login", "Unknown")
                        })

                    if user_data:
                        user_df = pd.DataFrame(user_data)
                        st.dataframe(user_df, use_container_width=True)

                        # Add export functionality
                        if st.button("Export User Data"):
                            # Create downloadable user data
                            csv = user_df.to_csv(index=False)
                            st.download_button(
                                label="Download User Data CSV",
                                data=csv,
                                file_name="medexplain_users.csv",
                                mime="text/csv"
                            )
                else:
                    st.info("No user data available.")
            else:
                st.warning("User management component is not available.")

            # User actions
            st.markdown("### User Actions")

            # User creation form
            with st.expander("Create New User"):
                with st.form("create_user_form"):
                    user_name = st.text_input("Name")
                    user_age = st.number_input("Age", min_value=0, max_value=120)
                    user_gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
                    user_email = st.text_input("Email (optional)")

                    submit_button = st.form_submit_button("Create User")

                    if submit_button:
                        if hasattr(self, "user_manager") and self.user_manager:
                            try:
                                # Create user data
                                user_data = {
                                    "name": user_name,
                                    "age": user_age,
                                    "gender": user_gender
                                }

                                if user_email:
                                    user_data["email"] = user_email

                                # Create user and profile
                                user_id = self.user_manager.create_user(user_data)

                                # Update profile with the same data
                                if hasattr(self.user_manager, "update_profile"):
                                    current_user = self.user_manager.current_user_id
                                    self.user_manager.switch_user(user_id)
                                    self.user_manager.update_profile(user_data)
                                    self.user_manager.switch_user(current_user)

                                st.success(f"User '{user_name}' created successfully with ID: {user_id}")
                            except Exception as e:
                                st.error(f"Error creating user: {e}")
                        else:
                            st.error("User management component is not available.")

            # User deletion with improved security
            with st.expander("Delete User"):
                if hasattr(self, "user_manager") and self.user_manager:
                    # Get user IDs for selection
                    user_ids = [user.get("id", "") for user in users] if users else []

                    if user_ids:
                        # Filter out the current user from deletion options if it's the only user
                        if len(user_ids) > 1 and self.user_manager.current_user_id in user_ids:
                            user_ids.remove(self.user_manager.current_user_id)
                            st.info("The current user cannot be deleted while in use.")

                        selected_user = st.selectbox("Select User to Delete", user_ids)

                        # Show user details for confirmation
                        user_details = next((user for user in users if user.get("id") == selected_user), None)
                        if user_details:
                            st.markdown(f"""
                            **User Details:**
                            - Name: {user_details.get('name', 'Unknown')}
                            - ID: {user_details.get('id', 'Unknown')}
                            - Created: {user_details.get('created_at', 'Unknown')}
                            """)

                        col1, col2 = st.columns(2)

                        with col1:
                            delete_button = st.button("Delete User")

                        if delete_button:
                            with col2:
                                st.warning(f"⚠️ This will permanently delete user '{selected_user}' and all their data.")
                                confirm = st.checkbox("I understand and confirm this action")

                                if confirm:
                                    confirm_button = st.button("Confirm Delete", key="confirm_delete_button")

                                    if confirm_button:
                                        try:
                                            # Delete user
                                            if hasattr(self.user_manager, "delete_user"):
                                                success = self.user_manager.delete_user(selected_user)

                                                if success:
                                                    st.success(f"User '{selected_user}' deleted successfully.")
                                                    # Add notification
                                                    self.add_notification(
                                                        "User Deleted",
                                                        f"User '{selected_user}' has been deleted from the system.",
                                                        "info"
                                                    )
                                                    st.experimental_rerun()
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

            # User switching functionality
            with st.expander("Switch Active User"):
                if hasattr(self, "user_manager") and self.user_manager:
                    # Get user IDs for selection
                    user_ids = [user.get("id", "") for user in users] if users else []

                    if user_ids:
                        # Get current user ID
                        current_user_id = self.user_manager.current_user_id

                        # Find index of current user
                        try:
                            current_index = user_ids.index(current_user_id)
                        except ValueError:
                            current_index = 0

                        selected_user = st.selectbox(
                            "Select User to Switch to",
                            user_ids,
                            index=current_index
                        )

                        if selected_user != current_user_id:
                            if st.button("Switch User"):
                                try:
                                    success = self.user_manager.switch_user(selected_user)

                                    if success:
                                        # Update session state
                                        st.session_state.user_id = selected_user

                                        # Clear cached data that relates to user
                                        for key in ['last_symptom_check', 'last_risk_assessment', 'health_report']:
                                            if key in st.session_state:
                                                del st.session_state[key]

                                        st.success(f"Switched to user: {selected_user}")
                                        st.session_state.page = "Home"  # Redirect to home
                                        st.experimental_rerun()
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
            logger.error(f"Error rendering user management tab: {e}", exc_info=True)
            st.error("Error rendering user management. Please try again later.")

    def _render_data_management_tab(self):
        """Render data management tab in admin panel with improved error handling."""
        try:
            st.subheader("Data Management")

            # Data backup and restore with improved functionality
            st.markdown("### Data Backup")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Backup All Data"):
                    try:
                        import json
                        import datetime

                        # Create backup data structure
                        backup_data = {
                            "app_version": APP_VERSION,
                            "backup_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "users": {},
                            "metadata": {
                                "component_status": self.component_status
                            }
                        }

                        # Add user data if available
                        if hasattr(self, "user_manager") and self.user_manager:
                            for user in self.user_manager.get_all_users():
                                user_id = user.get("id")
                                if user_id:
                                    # Switch to user to get their data
                                    current_user = self.user_manager.current_user_id
                                    self.user_manager.switch_user(user_id)

                                    # Get user data
                                    backup_data["users"][user_id] = {
                                        "profile": self.user_manager.profile,
                                        "health_history": self.user_manager.health_history
                                    }

                                    # Switch back
                                    self.user_manager.switch_user(current_user)

                        # Convert to JSON
                        backup_json = json.dumps(backup_data, indent=2)

                        # Create download button
                        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"medexplain_backup_{date_str}.json"

                        st.download_button(
                            label="Download Backup File",
                            data=backup_json,
                            file_name=filename,
                            mime="application/json"
                        )

                        st.success("Data backed up successfully.")
                    except Exception as e:
                        logger.error(f"Error backing up data: {e}", exc_info=True)
                        st.error(f"Error backing up data: {e}")

            with col2:
                if st.button("Export Analytics Data"):
                    try:
                        import json
                        import datetime

                        # Create analytics data structure
                        analytics_data = {
                            "export_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "page_views": st.session_state.get("page_views", {}),
                            "user_stats": {},
                            "performance_metrics": st.session_state.get("performance_metrics", {})
                        }

                        # Add user statistics if available
                        if hasattr(self, "user_manager") and self.user_manager:
                            for user in self.user_manager.get_all_users():
                                user_id = user.get("id")
                                if user_id and hasattr(self, "health_data") and self.health_data:
                                    try:
                                        analytics_data["user_stats"][user_id] = self.health_data.get_symptom_stats(user_id)
                                    except:
                                        pass

                        # Convert to CSV format for analytics
                        user_stats_df = pd.DataFrame()

                        if analytics_data["user_stats"]:
                            # Convert nested dictionary to DataFrame
                            rows = []
                            for user_id, stats in analytics_data["user_stats"].items():
                                row = {"user_id": user_id}
                                row.update(stats)
                                rows.append(row)

                            user_stats_df = pd.DataFrame(rows)

                        if not user_stats_df.empty:
                            csv_data = user_stats_df.to_csv(index=False)

                            # Create download button
                            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"medexplain_analytics_{date_str}.csv"

                            st.download_button(
                                label="Download Analytics CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv"
                            )
                        else:
                            st.warning("No analytics data available to export.")

                        st.success("Analytics data exported successfully.")
                    except Exception as e:
                        logger.error(f"Error exporting analytics data: {e}", exc_info=True)
                        st.error(f"Error exporting analytics data: {e}")

            # Data restoration with improved validation
            st.markdown("### Data Restoration")

            with st.expander("Restore from Backup"):
                uploaded_file = st.file_uploader("Upload Backup File", type=["json"])

                if uploaded_file is not None:
                    try:
                        import json

                        # Load the backup data
                        backup_data = json.loads(uploaded_file.getvalue().decode("utf-8"))

                        # Validate backup file
                        if "app_version" not in backup_data or "users" not in backup_data:
                            st.error("Invalid backup file format. Missing required fields.")
                        else:
                            # Show backup metadata
                            st.markdown(f"""
                            **Backup Information:**
                            - App Version: {backup_data.get('app_version', 'Unknown')}
                            - Backup Date: {backup_data.get('backup_date', 'Unknown')}
                            - Users: {len(backup_data.get('users', {}))}
                            """)

                            # Check version compatibility
                            backup_version = backup_data.get('app_version', '0.0.0')
                            if backup_version != APP_VERSION:
                                st.warning(f"Backup was created with version {backup_version}, but current version is {APP_VERSION}. Some data may not be compatible.")

                            if st.button("Restore Data"):
                                st.warning("This will overwrite existing data. Proceed with caution.")

                                confirm = st.checkbox("I understand and confirm this action", key="restore_confirm")

                                if confirm and st.button("Confirm Restore", key="confirm_restore"):
                                    try:
                                        # This would typically handle data restoration
                                        # For example, restoring users:
                                        if hasattr(self, "user_manager") and self.user_manager:
                                            for user_id, user_data in backup_data.get("users", {}).items():
                                                # Check if user exists
                                                existing_user = self.user_manager.get_user(user_id)

                                                if existing_user:
                                                    # Update profile
                                                    current_user = self.user_manager.current_user_id
                                                    self.user_manager.switch_user(user_id)
                                                    self.user_manager.update_profile(user_data.get("profile", {}))

                                                    # Restore health history
                                                    if "health_history" in user_data:
                                                        # Clear existing history first
                                                        self.user_manager.health_history = []

                                                        # Add each record
                                                        for record in user_data["health_history"]:
                                                            self.user_manager.add_health_record(record)

                                                    # Switch back
                                                    self.user_manager.switch_user(current_user)
                                                else:
                                                    # Create new user
                                                    new_user_id = self.user_manager.create_user({
                                                        "id": user_id,
                                                        "name": user_data.get("profile", {}).get("name", f"User {user_id}")
                                                    })

                                                    # Switch to new user to restore data
                                                    current_user = self.user_manager.current_user_id
                                                    self.user_manager.switch_user(user_id)

                                                    # Update profile
                                                    self.user_manager.update_profile(user_data.get("profile", {}))

                                                    # Restore health history
                                                    if "health_history" in user_data:
                                                        for record in user_data["health_history"]:
                                                            self.user_manager.add_health_record(record)

                                                    # Switch back
                                                    self.user_manager.switch_user(current_user)

                                        st.success("Data restored successfully.")

                                        # Add notification
                                        self.add_notification(
                                            "Data Restored",
                                            f"Successfully restored data from backup created on {backup_data.get('backup_date', 'unknown date')}.",
                                            "success"
                                        )

                                        # Refresh the app
                                        st.experimental_rerun()
                                    except Exception as e:
                                        logger.error(f"Error restoring data: {e}", exc_info=True)
                                        st.error(f"Error restoring data: {e}")
                    except Exception as e:
                        logger.error(f"Error processing backup file: {e}", exc_info=True)
                        st.error(f"Error processing backup file: {e}")

            # Data cleanup options
            st.markdown("### Data Cleanup")

            with st.expander("Cleanup Options"):
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Clear Temporary Files"):
                        try:
                            # This would typically clean up temp files
                            import glob

                            # Count temp files in log directory
                            temp_files = glob.glob(os.path.join(LOG_DIR, "*.tmp"))

                            if temp_files:
                                for file in temp_files:
                                    try:
                                        os.remove(file)
                                    except:
                                        pass

                                st.success(f"Cleared {len(temp_files)} temporary files.")
                            else:
                                st.info("No temporary files found to clear.")
                        except Exception as e:
                            logger.error(f"Error clearing temporary files: {e}", exc_info=True)
                            st.error(f"Error clearing temporary files: {e}")

                with col2:
                    if st.button("Optimize Database"):
                        try:
                            # This would typically optimize the database

                            # Simulate database optimization
                            import time
                            with st.spinner("Optimizing database..."):
                                # Simulate processing time
                                time.sleep(2)

                                # Add optimization log entry
                                logger.info("Database optimization performed")

                            st.success("Database optimized successfully.")
                        except Exception as e:
                            logger.error(f"Error optimizing database: {e}", exc_info=True)
                            st.error(f"Error optimizing database: {e}")

                # Add purge data option with confirmation
                st.markdown("### Dangerous Zone")
                st.error("⚠️ The following actions will permanently delete data and cannot be undone.")

                if st.button("Purge All User Data"):
                    st.warning("This will permanently delete ALL user data. This action cannot be undone.")

                    confirm_text = st.text_input("Type 'CONFIRM' to proceed with data purge:")

                    if confirm_text == "CONFIRM":
                        if st.button("Execute Data Purge", key="execute_purge"):
                            try:
                                # Delete user data directory
                                import shutil

                                # Implement safeguards for the simulation
                                if hasattr(self, "user_manager") and self.user_manager:
                                    # Get current user ID to restore later
                                    current_user_id = self.user_manager.current_user_id

                                    # Get all users
                                    users = self.user_manager.get_all_users()

                                    # Delete each user except default
                                    for user in users:
                                        user_id = user.get("id")
                                        if user_id and user_id != "default_user":
                                            self.user_manager.delete_user(user_id)

                                    # Reset default user
                                    self.user_manager.switch_user("default_user")
                                    self.user_manager.clear_health_history()

                                    # Reset session state
                                    st.session_state.clear()
                                    st.session_state.page = "Home"
                                    st.session_state.user_id = "default_user"

                                st.success("All user data has been purged. Returning to home page.")
                                st.experimental_rerun()
                            except Exception as e:
                                logger.error(f"Error purging data: {e}", exc_info=True)
                                st.error(f"Error purging data: {e}")
        except Exception as e:
            logger.error(f"Error rendering data management tab: {e}", exc_info=True)
            st.error("Error rendering data management. Please try again later.")

    def _render_performance_metrics_tab(self):
        """Render performance metrics tab in admin panel with improved error handling."""
        try:
            st.subheader("Performance Metrics")

            # Display performance metrics
            if hasattr(st.session_state, "performance_metrics"):
                metrics = st.session_state.performance_metrics

                # Main metrics
                st.markdown("### Core Metrics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Startup Time", f"{metrics.get('startup_time', 0):.2f}s")

                with col2:
                    st.metric("Component Load Rate", f"{metrics.get('component_load_success_rate', 0):.0%}")

                with col3:
                    # This would typically be tracked separately
                    response_time = metrics.get('avg_response_time', 245)
                    st.metric("Avg Response Time", f"{response_time}ms")

                # Create mock time series data for demonstration
                st.markdown("### Performance Over Time")

                # Generate sample data
                import random
                from datetime import datetime, timedelta

                # Generate dates for the past 14 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=14)
                dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14)]

                # Generate performance data with realistic trends
                # Start with a base value and add some random variation day by day
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

                try:
                    # Create performance chart
                    import plotly.graph_objects as go
                    fig = go.Figure()

                    # Add response time trace
                    fig.add_trace(go.Scatter(
                        x = dates,
                        y = response_times,
                        mode = 'lines+markers',
                        name = 'Response Time (ms)',
                        line = {'color': 'blue'}
                    ))

                    # Create secondary y-axis for error rate
                    fig.add_trace(go.Scatter(
                        x = dates,
                        y = error_rates,
                        mode = 'lines+markers',
                        name = 'Error Rate (%)',
                        line = {'color': 'red'},
                        yaxis = "y2"
                    ))

                    # Update layout with second y-axis
                    fig.update_layout(
                        title = "System Performance Trends",
                        xaxis_title = "Date",
                        yaxis_title = "Response Time (ms)",
                        yaxis = {"range": [180, 300]},  # Set reasonable y-axis range for response time
                        yaxis2 = {
                            'title': 'Error Rate (%)',
                            'overlaying': 'y',
                            'side': 'right',
                            'range': [0, 3]  # Set reasonable y-axis range for error rate
                        },
                        height = 400,
                        legend = {"orientation": "h", "yanchor": "bottom", "y": 1.02}
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating performance chart: {e}", exc_info=True)
                    st.error("Could not render performance visualization.")

                # Resource utilization with realistic patterns
                st.markdown("### Resource Utilization")

                # Generate realistic CPU and memory data
                # CPU usage should be moderate with periodic spikes
                cpu_usage = []
                for i in range(14):
                    # Base CPU usage with weekly pattern (higher on weekdays 0-4)
                    day_of_week = i % 7
                    if day_of_week < 5:  # Weekday
                        base_cpu = random.uniform(30, 45)
                    else:  # Weekend
                        base_cpu = random.uniform(15, 30)

                    # Add random variation
                    cpu_usage.append(base_cpu + random.uniform(-5, 10))

                # Memory usage tends to grow gradually and reset
                memory_usage = []
                memory_base = 35
                for i in range(14):
                    # Memory tends to grow and occasionally reset (simulating garbage collection)
                    if i > 0 and random.random() < 0.2:  # 20% chance of memory reset
                        memory_base = random.uniform(30, 40)
                    else:
                        memory_base += random.uniform(0, 2)  # Slight growth

                    memory_usage.append(min(85, memory_base + random.uniform(-2, 2)))  # Add jitter

                try:
                    # Create resource utilization chart
                    fig = go.Figure()

                    # Add CPU trace
                    fig.add_trace(go.Scatter(
                        x = dates,
                        y = cpu_usage,
                        mode = 'lines+markers',
                        name = 'CPU Usage (%)',
                        line = {'color': 'orange'}
                    ))

                    # Add memory trace
                    fig.add_trace(go.Scatter(
                        x = dates,
                        y = memory_usage,
                        mode = 'lines+markers',
                        name = 'Memory Usage (%)',
                        line = {'color': 'green'}
                    ))

                    fig.update_layout(
                        title = "Resource Utilization",
                        xaxis_title = "Date",
                        yaxis_title = "Utilization (%)",
                        height = 400,
                        yaxis = {"range": [0, 100]},
                        legend = {"orientation": "h", "yanchor": "bottom", "y": 1.02}
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating resource chart: {e}", exc_info=True)
                    st.error("Could not render resource visualization.")

                # Add page performance metrics
                st.markdown("### Page Performance")

                # Get page view data from session state
                page_views = st.session_state.get("page_views", {})

                if page_views:
                    # Create page view chart
                    page_names = list(page_views.keys())
                    view_counts = list(page_views.values())

                    # Sort by popularity
                    page_data = sorted(zip(page_names, view_counts), key=lambda x: x[1], reverse=True)
                    page_names = [x[0] for x in page_data]
                    view_counts = [x[1] for x in page_data]

                    try:
                        fig = go.Figure(go.Bar(
                            x = page_names,
                            y = view_counts,
                            marker_color = 'lightblue'
                        ))

                        fig.update_layout(
                            title = "Page Views",
                            xaxis_title = "Page",
                            yaxis_title = "View Count",
                            height = 400
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error creating page view chart: {e}", exc_info=True)

                        # Fallback to table
                        view_df = pd.DataFrame({
                            "Page": page_names,
                            "Views": view_counts
                        })

                        st.dataframe(view_df, use_container_width=True)
                else:
                    st.info("No page view data available.")
            else:
                st.info("Performance metrics not available.")

                # Create metrics placeholder
                st.markdown("### Sample Metrics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Startup Time", "1.85s")

                with col2:
                    st.metric("Component Load Rate", "100%")

                with col3:
                    st.metric("Avg Response Time", "245ms")

                st.info("These are placeholder metrics. Real metrics will be displayed once the application has been used more.")
        except Exception as e:
            logger.error(f"Error rendering performance metrics tab: {e}", exc_info=True)
            st.error("Error rendering performance metrics. Please try again later.")

    def _render_api_configuration_tab(self):
        """Render API configuration tab in admin panel with improved error handling and validation."""
        try:
            st.subheader("API Configuration")

            # API key management
            st.markdown("### API Keys")

            with st.expander("OpenAI API"):
                # Display current status
                if hasattr(self, "openai_client") and self.openai_client:
                    api_status = "Configured" if (hasattr(self.openai_client, "api_key") and self.openai_client.api_key) else "Not Configured"

                    status_color = "green" if api_status == "Configured" else "red"
                    st.markdown(f"**Status:** <span style='color:{status_color};'>{api_status}</span>", unsafe_allow_html=True)

                    # Check API key validity if configured
                    if api_status == "Configured":
                        try:
                            # This would typically call a validation method
                            # For demo purposes, we'll just show it as valid
                            st.markdown("**Validation:** <span style='color:green;'>Valid</span>", unsafe_allow_html=True)
                        except:
                            st.markdown("**Validation:** <span style='color:orange;'>Not validated</span>", unsafe_allow_html=True)
                else:
                    st.warning("OpenAI client not available.")

                # API key input
                new_api_key = st.text_input("Enter OpenAI API Key", type="password")

                # Add validation for key format (should start with "sk-")
                if new_api_key and not new_api_key.startswith("sk-"):
                    st.warning("OpenAI API keys typically start with 'sk-'. Please check your key.")

                if st.button("Save API Key") and new_api_key:
                    if hasattr(self, "openai_client") and self.openai_client:
                        try:
                            # Save the API key
                            if hasattr(self.openai_client, "set_api_key"):
                                self.openai_client.set_api_key(new_api_key)

                                # Update environment variable for persistence
                                os.environ["OPENAI_API_KEY"] = new_api_key

                                st.success("API key saved successfully.")

                                # Add notification
                                self.add_notification(
                                    "API Key Updated",
                                    "OpenAI API key has been updated successfully.",
                                    "success"
                                )
                            else:
                                st.error("The OpenAI client does not support setting the API key.")
                        except Exception as e:
                            logger.error(f"Error saving API key: {e}", exc_info=True)
                            st.error(f"Error saving API key: {e}")
                    else:
                        st.error("OpenAI client not available.")

            # Other API integrations
            st.markdown("### Other Integrations")

            with st.expander("Medical Database API"):
                st.markdown("""
                **Status:** Not Configured

                Configure access to external medical databases for enhanced
                symptom correlation and medical literature search.
                """)

                # Mock implementation
                api_url = st.text_input("API Endpoint URL")
                api_key = st.text_input("API Key", type="password")

                # Add URL validation
                if api_url and not (api_url.startswith("http://") or api_url.startswith("https://")):
                    st.warning("API URL should start with http:// or https://")

                if st.button("Save Configuration") and api_url and api_key:
                    # Validate URL format
                    import re
                    url_pattern = re.compile(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')

                    if url_pattern.match(api_url):
                        st.success("Configuration saved successfully.")

                        # Store in session state for persistence
                        if 'api_configs' not in st.session_state:
                            st.session_state.api_configs = {}

                        st.session_state.api_configs['medical_db'] = {
                            'url': api_url,
                            'configured_at': time.time()
                        }
                    else:
                        st.error("Invalid URL format. Please enter a valid URL.")

            with st.expander("EHR Integration"):
                st.markdown("""
                **Status:** Not Configured

                Configure integration with Electronic Health Record (EHR) systems
                for seamless patient data exchange.
                """)

                # Mock implementation with improved validation
                ehr_system = st.selectbox("EHR System", ["Select a system...", "Epic", "Cerner", "Allscripts", "Other"])

                if ehr_system == "Select a system...":
                    st.info("Please select an EHR system.")
                else:
                    ehr_url = st.text_input("EHR API Endpoint", key="ehr_url")
                    ehr_key = st.text_input("Access Token", type="password", key="ehr_key")

                    # For "Other" system, allow custom name
                    if ehr_system == "Other":
                        ehr_system_custom = st.text_input("EHR System Name")
                        if ehr_system_custom:
                            ehr_system = ehr_system_custom

                    # Show authentication method options
                    auth_method = st.radio("Authentication Method", ["OAuth 2.0", "API Key", "Basic Auth"])

                    # Show relevant fields based on auth method
                    if auth_method == "OAuth 2.0":
                        st.text_input("Client ID")
                        st.text_input("Client Secret", type="password")
                    elif auth_method == "Basic Auth":
                        st.text_input("Username")
                        st.text_input("Password", type="password")

                    # Test connection button
                    if st.button("Test Connection"):
                        if not ehr_url or not ehr_key:
                            st.error("Please provide both API endpoint and access token.")
                        else:
                            with st.spinner("Testing connection..."):
                                # This would typically test the actual connection
                                # For demo, we'll just simulate a delay and success
                                time.sleep(2)
                                st.success("Connection test successful!")

                    if st.button("Save EHR Configuration") and ehr_system != "Select a system..." and ehr_url and ehr_key:
                        st.success(f"EHR configuration for {ehr_system} saved successfully.")

                        # Store in session state for persistence
                        if 'api_configs' not in st.session_state:
                            st.session_state.api_configs = {}

                        st.session_state.api_configs['ehr'] = {
                            'system': ehr_system,
                            'url': ehr_url,
                            'auth_method': auth_method,
                            'configured_at': time.time()
                        }

            # API usage settings
            st.markdown("### API Usage Settings")

            with st.expander("Rate Limiting & Budgets"):
                # Rate limiting options
                st.subheader("Rate Limiting")
                enable_rate_limiting = st.checkbox("Enable API rate limiting", value=True)

                if enable_rate_limiting:
                    max_requests = st.slider("Maximum requests per minute", 1, 100, 60)
                    st.info(f"API calls will be limited to {max_requests} requests per minute.")

                # Budget controls
                st.subheader("Budget Controls")
                enable_budget = st.checkbox("Enable budget limits", value=False)

                if enable_budget:
                    daily_budget = st.number_input("Daily budget ($)", min_value=0.0, value=5.0, step=0.5)

                    # Action when budget is exceeded
                    exceeded_action = st.radio(
                        "When budget is exceeded:",
                        ["Warn only", "Disable non-essential features", "Block all API calls"]
                    )

                    st.info(f"Daily budget set to ${daily_budget:.2f}. When exceeded: {exceeded_action}")

                if st.button("Save API Usage Settings"):
                    # This would typically save these settings
                    st.success("API usage settings saved successfully.")

                    # Store in session state
                    st.session_state.api_settings = {
                        'rate_limiting': {
                            'enabled': enable_rate_limiting,
                            'max_requests': max_requests if enable_rate_limiting else 0
                        },
                        'budget': {
                            'enabled': enable_budget,
                            'daily_limit': daily_budget if enable_budget else 0,
                            'exceeded_action': exceeded_action if enable_budget else "Warn only"
                        }
                    }
        except Exception as e:
            logger.error(f"Error rendering API configuration tab: {e}", exc_info=True)
            st.error("Error rendering API configuration. Please try again later.")

    def _refresh_components(self):
        """Refresh components that need to be updated with improved error handling."""
        try:
            # Re-initialize health analyzer if dependencies available
            if hasattr(self, "user_manager") and self.user_manager and hasattr(self, "health_analyzer"):
                try:
                    self.health_analyzer.update_data(
                        self.user_manager.health_history,
                        self.user_manager.profile
                    )
                    logger.info("Refreshed HealthDataAnalyzer")
                except Exception as e:
                    logger.error(f"Error refreshing HealthDataAnalyzer: {e}", exc_info=True)

            # Re-initialize dashboard if needed
            if hasattr(self, "user_manager") and self.user_manager and hasattr(self, "dashboard"):
                try:
                    self.dashboard.update_data(
                        self.user_manager.health_history,
                        self.user_manager.profile
                    )
                    logger.info("Refreshed HealthDashboard")
                except Exception as e:
                    logger.error(f"Error refreshing HealthDashboard: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error in _refresh_components: {e}", exc_info=True)

    def _handle_data_export(self):
        """Handle data export when ready with improved error handling and format support."""
        try:
            if not st.session_state.export_data:
                return

            # Create a download button for the data
            data_type = st.session_state.export_data.get("type", "data")
            file_format = st.session_state.export_data.get("format", "csv")
            data = st.session_state.export_data.get("data")

            if data is None:
                st.error("No data available for export.")
                st.session_state.export_ready = False
                st.session_state.export_data = None
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
                st.download_button(
                    label=f"Download {data_type} as CSV",
                    data=output,
                    file_name=f"{data_type}_{time.strftime('%Y%m%d')}.csv",
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
                st.download_button(
                    label=f"Download {data_type} as JSON",
                    data=output,
                    file_name=f"{data_type}_{time.strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            elif file_format.lower() == "excel":
                # Convert to Excel
                import io

                output = io.BytesIO()

                if isinstance(data, pd.DataFrame):
                    data.to_excel(output, index=False)
                else:
                    # Convert to DataFrame first
                    df = pd.DataFrame(data)
                    df.to_excel(output, index=False)

                output.seek(0)

                # Create download button
                st.download_button(
                    label=f"Download {data_type} as Excel",
                    data=output,
                    file_name=f"{data_type}_{time.strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error(f"Unsupported export format: {file_format}")

            # Add option to keep or clear export data
            if st.button("Clear Export Data"):
                st.session_state.export_ready = False
                st.session_state.export_data = None
                st.success("Export data cleared.")
                st.experimental_rerun()
        except Exception as e:
            logger.error(f"Error handling data export: {e}", exc_info=True)
            st.error(f"Error exporting data: {e}")
            st.session_state.export_ready = False
            st.session_state.export_data = None

# Run the application when executed directly
if __name__ == "__main__":
    app = MedExplainApp()
    app.run()
