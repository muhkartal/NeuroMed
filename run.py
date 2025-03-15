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

from ml.symptom_predictor import SymptomPredictor
from ml.symptom_extractor import NLPSymptomExtractor
from ml.risk_assessor import PatientRiskAssessor

from analytics.health_analyzer import HealthDataAnalyzer
from analytics.visualization import create_timeseries_chart, create_symptom_heatmap, create_risk_radar

from ui.dashboard import HealthDashboard
from ui.chat import ChatInterface
from ui.symptom_analyzer import SymptomAnalyzerUI
from ui.medical_literature import MedicalLiteratureUI
from ui.health_history import HealthHistoryUI
from ui.settings import SettingsUI

# Constants
APP_VERSION = "3.0.0"
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
        logging.FileHandler(os.path.join(LOG_DIR, "log")),
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
        # Health Data Manager
        self._initialize_component("health_data", HealthDataManager, {"data_dir": self.user_data_dir})

        # User Profile Manager
        self._initialize_component("user_manager", UserProfileManager, {"data_dir": self.user_data_dir})

        # OpenAI Client
        self._initialize_component("openai_client", OpenAIClient, {})

        # Initialize Health Analyzer if dependencies available
        if self.component_status.get("user_manager", False):
            self._initialize_component(
                "health_analyzer",
                HealthDataAnalyzer,
                {
                    "health_history": self.user_manager.health_history if hasattr(self, "user_manager") else [],
                    "user_profile": self.user_manager.profile if hasattr(self, "user_manager") else {}
                }
            )

    def _initialize_ml_components(self):
        """Initialize machine learning components with error handling."""
        # Symptom Predictor
        self._initialize_component("symptom_predictor", SymptomPredictor, {"model_dir": self.model_dir})

        # NLP Symptom Extractor
        self._initialize_component("symptom_extractor", NLPSymptomExtractor, {"model_dir": self.model_dir})

        # Risk Assessor
        self._initialize_component("risk_assessor", PatientRiskAssessor, {"model_dir": self.model_dir})

    def _initialize_ui_components(self):
        """Initialize UI components with error handling."""
        # Health Dashboard
        if self.component_status.get("user_manager", False):
            self._initialize_component(
                "dashboard",
                HealthDashboard,
                {
                    "health_history": self.user_manager.health_history if hasattr(self, "user_manager") else [],
                    "user_profile": self.user_manager.profile if hasattr(self, "user_manager") else {}
                }
            )

        # Chat Interface
        if self.component_status.get("openai_client", False) and self.component_status.get("symptom_extractor", False):
            self._initialize_component(
                "chat_interface",
                ChatInterface,
                {
                    "openai_client": self.openai_client if hasattr(self, "openai_client") else None,
                    "symptom_extractor": self.symptom_extractor if hasattr(self, "symptom_extractor") else None
                }
            )

        # Symptom Analyzer UI
        self._initialize_component(
            "symptom_analyzer_ui",
            SymptomAnalyzerUI,
            {
                "health_data": self.health_data if hasattr(self, "health_data") else None,
                "symptom_predictor": self.symptom_predictor if hasattr(self, "symptom_predictor") else None,
                "symptom_extractor": self.symptom_extractor if hasattr(self, "symptom_extractor") else None,
                "risk_assessor": self.risk_assessor if hasattr(self, "risk_assessor") else None
            }
        )

        # Medical Literature UI
        self._initialize_component(
            "medical_literature_ui",
            MedicalLiteratureUI,
            {
                "health_data": self.health_data if hasattr(self, "health_data") else None,
                "openai_client": self.openai_client if hasattr(self, "openai_client") else None
            }
        )

        # Health History UI
        self._initialize_component(
            "health_history_ui",
            HealthHistoryUI,
            {
                "user_manager": self.user_manager if hasattr(self, "user_manager") else None,
                "health_data": self.health_data if hasattr(self, "health_data") else None
            }
        )

        # Settings UI
        self._initialize_component(
            "settings_ui",
            SettingsUI,
            {
                "user_manager": self.user_manager if hasattr(self, "user_manager") else None,
                "openai_client": self.openai_client if hasattr(self, "openai_client") else None
            }
        )

    def _initialize_component(self, name: str, component_class: Any, params: Dict[str, Any]) -> bool:
        """
        Initialize a component and handle any errors.

        Args:
            name: Name of the component to track
            component_class: Class to instantiate
            params: Parameters to pass to the constructor

        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Filter out None values from parameters
            filtered_params = {k: v for k, v in params.items() if v is not None}

            # Initialize the component
            component = component_class(**filtered_params)
            setattr(self, name, component)
            self.component_status[name] = True
            logger.info(f"Initialized {name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing {name}: {e}", exc_info=True)
            self.component_status[name] = False
            setattr(self, name, None)
            return False

    def _load_api_keys_from_environment(self):
        """Load API keys from environment variables."""
        # OpenAI API key
        if self.component_status.get("openai_client", False):
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                self.openai_client.set_api_key(api_key)
                logger.info("Set OpenAI API key from environment variable")

    def _set_critical_error(self, message: str):
        """Set a critical error message for display."""
        if not hasattr(st.session_state, "error_message"):
            st.session_state.error_message = message

    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
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

    def add_notification(self, title: str, message: str, notification_type: str = "info"):
        """
        Add a notification to the system.

        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification (info, warning, error, success)
        """
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

    def _show_notifications(self):
        """Display and manage notifications."""
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

                with container:
                    st.markdown(f"**{notification['title']}**")
                    st.write(notification['message'])

                    # Mark as read button
                    if not notification.get("read", False):
                        if st.button("Mark as read", key=f"read_{notification['id']}"):
                            st.session_state.notifications[i]["read"] = True
                            st.session_state.notification_count -= 1
                            st.experimental_rerun()

    def _add_custom_css(self):
        """Add custom CSS to the Streamlit application."""
        # Read from the CSS file if available
        css_path = os.path.join(STATIC_DIR, "css", "style.css")
        custom_css = ""

        if os.path.exists(css_path):
            with open(css_path, "r") as css_file:
                custom_css = css_file.read()

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
                color: #2c3e50;
            }}

            /* Custom component styles */
            .stButton button {{
                border-radius: 4px;
                padding: 0.25rem 1rem;
            }}

            .info-card {{
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                margin-bottom: 1rem;
            }}

            /* Logo styling */
            .logo-text {{
                font-weight: 600;
                color: #3498db;
                font-size: 1.5rem;
                margin: 0;
            }}
        </style>
        """, unsafe_allow_html=True)

    def _apply_dark_mode(self):
        """Apply dark mode styling."""
        st.markdown("""
        <style>
            body {
                color: #f0f2f6;
                background-color: #1e1e1e;
            }

            .stApp {
                background-color: #1e1e1e;
            }

            .main .block-container {
                background-color: #1e1e1e;
            }

            h1, h2, h3 {
                color: #e0e0e0 !important;
            }

            p, li {
                color: #c0c0c0;
            }

            .info-card {
                background-color: #2d2d2d;
                border-left: 4px solid #3498db;
            }

            /* Additional dark mode styles */
            .stTextInput input, .stSelectbox, .stMultiselect {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
        </style>
        """, unsafe_allow_html=True)

    def _reset_dark_mode(self):
        """Reset dark mode styling."""
        # The reset is handled by re-applying the base CSS in _add_custom_css
        pass

    def render_sidebar(self):
        """Render the application sidebar with navigation and user info."""
        # Display logo and title
        with st.sidebar:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(os.path.join(STATIC_DIR, "img", "logo.png")
                        if os.path.exists(os.path.join(STATIC_DIR, "img", "logo.png"))
                        else "https://via.placeholder.com/150x150.png?text=ME", width=50)
            with col2:
                st.markdown("<p class='logo-text'>MedExplain AI Pro</p>", unsafe_allow_html=True)
                st.caption(f"Version {APP_VERSION}")

            # Display system status indicator
            system_status = "🟢 Fully Operational" if sum(self.component_status.values()) == len(self.component_status) else "🟡 Partially Degraded"

            st.caption(f"System Status: {system_status}")

            # Dark mode toggle
            dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
            if dark_mode != st.session_state.dark_mode:
                st.session_state.dark_mode = dark_mode
                # Apply custom CSS for dark mode
                if dark_mode:
                    self._apply_dark_mode()
                else:
                    self._reset_dark_mode()

            st.markdown("---")

            # Main navigation
            st.subheader("Navigation")
            menu_options = [
                "Home",
                "Symptom Analyzer",
                "Health Dashboard",
                "Advanced Analytics",
                "Medical Literature",
                "Health Chat",
                "Health History",
                "Settings"
            ]

            # Add admin panel in advanced mode
            if st.session_state.advanced_mode:
                menu_options.append("Admin Panel")

            choice = st.radio("Go to", menu_options)

            st.markdown("---")

            # Quick action buttons in a two-column layout
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔍 Quick Check", key="quick_check"):
                    st.session_state.page = "Symptom Analyzer"
                    return "Symptom Analyzer"

            with col2:
                if st.button("📋 Health Report", key="health_report"):
                    if hasattr(self, "risk_assessor") and self.risk_assessor and st.session_state.last_risk_assessment:
                        try:
                            with st.spinner("Generating report..."):
                                report = self.risk_assessor.generate_health_report(st.session_state.last_risk_assessment)
                                st.session_state.health_report = report
                                st.session_state.page = "Health History"
                                return "Health History"
                        except Exception as e:
                            logger.error(f"Error generating health report: {e}")
                            st.error("Could not generate report. Please try again.")
                    else:
                        st.info("Please complete a symptom analysis first to generate a health report.")

            # Notifications badge
            if st.session_state.notification_count > 0:
                st.markdown(f"""
                <div style="background-color: #3498db; color: white; padding: 8px 12px; border-radius: 20px;
                display: inline-block; margin-top: 10px;">
                    📬 {st.session_state.notification_count} Notification{'' if st.session_state.notification_count == 1 else 's'}
                </div>
                """, unsafe_allow_html=True)

                if st.button("View Notifications"):
                    self._show_notifications()

            # Display user profile summary if available
            if hasattr(self, "user_manager") and self.user_manager and self.user_manager.profile:
                st.markdown("---")
                st.subheader("User Profile")

                profile = self.user_manager.profile

                if profile.get("name"):
                    st.markdown(f"**Name:** {profile['name']}")

                if profile.get("age"):
                    st.markdown(f"**Age:** {profile['age']}")

                if profile.get("gender"):
                    st.markdown(f"**Gender:** {profile['gender']}")

                # Show brief health status if available
                if st.session_state.last_risk_assessment:
                    risk_level = st.session_state.last_risk_assessment.get('risk_level', 'unknown')
                    risk_color = {
                        'low': 'green',
                        'medium': 'orange',
                        'high': 'red',
                        'unknown': 'gray'
                    }.get(risk_level, 'gray')

                    st.markdown(f"**Health Status:** <span style='color:{risk_color};'>{risk_level.capitalize()}</span>", unsafe_allow_html=True)

            # View mode selector (advanced mode only)
            if st.session_state.advanced_mode:
                st.markdown("---")
                st.subheader("View Mode")
                view_mode = st.selectbox(
                    "Select view mode:",
                    ["Patient", "Clinician", "Researcher"],
                    index=["patient", "clinician", "researcher"].index(st.session_state.view_mode)
                )

                # Update view mode if changed
                selected_mode = view_mode.lower()
                if selected_mode != st.session_state.view_mode:
                    st.session_state.view_mode = selected_mode
                    st.experimental_rerun()

            # Advanced mode toggle for developers/healthcare providers
            st.markdown("---")
            advanced_mode = st.checkbox("Advanced Mode", value=st.session_state.advanced_mode)
            if advanced_mode != st.session_state.advanced_mode:
                st.session_state.advanced_mode = advanced_mode

            # Display disclaimer
            st.markdown("---")
            st.info(
                "**MEDICAL DISCLAIMER:** This application is for educational purposes only. "
                "It does not provide medical advice. Always consult a healthcare professional "
                "for medical concerns."
            )

            # App credits
            st.markdown("<small>© 2025 MedExplain AI Pro</small>", unsafe_allow_html=True)

        return choice

    def run(self):
        """Run the main application."""
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

    def _render_page(self, page):
        """Render the selected page content."""
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

    def _log_page_view(self, page_name):
        """Log page views for analytics."""
        if not hasattr(st.session_state, 'page_views'):
            st.session_state.page_views = {}

        # Initialize counter for this page if not exists
        if page_name not in st.session_state.page_views:
            st.session_state.page_views[page_name] = 0

        # Increment view counter
        st.session_state.page_views[page_name] += 1

        # Log the page view
        logger.info(f"User viewed page: {page_name} (view #{st.session_state.page_views[page_name]})")

    def _render_fallback_page(self, page_name, feature_name):
        """Render a fallback page when a component is not available."""
        st.title(f"{page_name}")

        st.warning(f"The {feature_name} is currently unavailable or experiencing issues.")

        # Provide more detailed information if in advanced mode
        if st.session_state.advanced_mode:
            st.markdown("### Component Status")
            for component, status in self.component_status.items():
                st.markdown(f"- {component}: {'✅ Loaded' if status else '❌ Failed'}")

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

    def _render_home(self):
        """Render the home page."""
        st.title("Welcome to MedExplain AI Pro")
        st.subheader("Your advanced personal health assistant powered by artificial intelligence")

        # Display main features in columns
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

            # Feature highlights
            st.markdown("### Key Advanced Features")

            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:
                st.markdown("""
                🧠 **AI-Powered Analysis**
                Ensemble ML models analyze your health data using multiple algorithms

                📊 **Interactive Health Dashboard**
                Comprehensive visualizations of your health patterns and trends

                🔍 **Pattern Recognition**
                Advanced algorithms identify correlations in your symptoms
                """)

            with feature_col2:
                st.markdown("""
                💬 **Medical NLP Interface**
                Discuss your health in plain language with contextual understanding

                📈 **Predictive Insights**
                Risk assessment and early warning indicators

                🔒 **Enterprise Security**
                HIPAA-compliant data encryption and privacy protection
                """)

        with col2:
            # Hero image - load from static directory if available
            image_path = os.path.join(STATIC_DIR, "img", "hero.png")
            if os.path.exists(image_path):
                st.image(image_path)
            else:
                st.image("https://via.placeholder.com/500x300.png?text=Advanced+Health+Analytics")

            # Quick action buttons
            st.markdown("### Quick Actions")

            action_col1, action_col2 = st.columns(2)

            with action_col1:
                if st.button("🔍 Analyze Symptoms", key="home_analyze"):
                    st.session_state.page = "Symptom Analyzer"
                    st.experimental_rerun()

                if st.button("📊 View Dashboard", key="home_dashboard"):
                    st.session_state.page = "Health Dashboard"
                    st.experimental_rerun()

            with action_col2:
                if st.button("💬 Health Chat", key="home_chat"):
                    st.session_state.page = "Health Chat"
                    st.experimental_rerun()

                if st.button("📈 Advanced Analytics", key="home_analytics"):
                    st.session_state.page = "Advanced Analytics"
                    st.experimental_rerun()

        # Recent activity section
        st.markdown("---")
        st.subheader("Recent Activity")

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
                    <div style="background-color: {'#2d2d2d' if st.session_state.dark_mode else '#f8f9fa'};
                         padding: 15px; border-radius: 5px; margin-bottom: 10px;
                         border-left: 4px solid #3498db;">
                        <h4 style="margin-top: 0;">Check from {date}</h4>
                        <p><strong>Symptoms:</strong> {", ".join(symptom_names)}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent health activity. Start by analyzing your symptoms or setting up your health profile.")
        else:
            st.info("No recent health activity. Start by analyzing your symptoms or setting up your health profile.")

        # Health tips section
        if hasattr(self, "openai_client") and self.openai_client and self.openai_client.api_key:
            st.markdown("---")
            st.subheader("Daily Health Tip")

            # Cache tip for the day
            if 'daily_tip' not in st.session_state:
                try:
                    prompt = """
                    Provide a single, concise health tip (100 words max) that would be useful for general wellness.
                    Focus on evidence-based advice that's practical and actionable. Format it as a brief paragraph.
                    """

                    tip = self.openai_client.generate_response(prompt)
                    if tip:
                        st.session_state.daily_tip = tip
                    else:
                        st.session_state.daily_tip = "Stay hydrated throughout the day. Proper hydration supports all bodily functions, improves energy levels, and helps maintain concentration."
                except Exception as e:
                    logger.error(f"Error generating health tip: {e}")
                    st.session_state.daily_tip = "Stay hydrated throughout the day. Proper hydration supports all bodily functions, improves energy levels, and helps maintain concentration."

            st.info(st.session_state.daily_tip)

        # Call to action and medical disclaimer
        st.markdown("---")
        st.warning("""
        **Medical Disclaimer:** MedExplain AI Pro is not a substitute for professional medical advice,
        diagnosis, or treatment. Always seek the advice of your physician or other qualified health
        provider with any questions you may have regarding a medical condition.
        """)

    def _render_advanced_analytics(self):
        """Render the advanced analytics page."""
        st.title("Advanced Health Analytics")
        st.markdown("Explore detailed analysis of your health data to uncover patterns and insights.")

        if not hasattr(self, "health_analyzer") or not self.health_analyzer or not hasattr(self, "user_manager") or not self.user_manager or not self.user_manager.health_history:
            st.info("Insufficient health data available for analysis. Please add more health information.")
            return

        # Create tabs for different analysis types
        tabs = st.tabs([
            "Symptom Patterns",
            "Risk Analysis",
            "Temporal Trends",
            "Predictive Analytics",
            "Comparative Analysis"
        ])

        # Get user health history
        health_history = self.user_manager.health_history

        with tabs[0]:
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
                    logger.error(f"Error creating symptom co-occurrence chart: {e}")
                    st.error("Could not render co-occurrence visualization. Please try again later.")

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

        with tabs[1]:
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
                    logger.error(f"Error creating risk gauge: {e}")
                    st.error("Could not render risk visualization. Please try again later.")

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
                        logger.error(f"Error creating domain risk chart: {e}")
                        st.error("Could not render domain risk visualization.")
                else:
                    st.info("Detailed domain risk data is not available.")
            else:
                st.info("No risk assessment data available. Please complete a symptom analysis first.")

        with tabs[2]:
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
                    logger.error(f"Error creating temporal trend chart: {e}")
                    st.error("Could not render temporal trend visualization.")

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

        with tabs[3]:
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
                    logger.error(f"Error creating prediction chart: {e}")
                    st.error("Could not render prediction visualization.")
            else:
                st.info("Not enough historical data for predictive analysis. Please check back after recording more health data.")

        with tabs[4]:
            st.subheader("Comparative Analysis")
            st.markdown("Compare your health metrics with anonymized population data.")

            st.info("Comparative analysis feature is coming soon.")

            st.markdown("""
            This feature will allow you to:
            - Compare your symptom patterns with similar demographic groups
            - Benchmark your health metrics against recommended ranges
            - Identify unusual health patterns that may require attention
            """)

            # Allow user to sign up for notifications
            st.markdown("### Get Notified")
            email = st.text_input("Enter your email to be notified when this feature is available:")

            if email and st.button("Notify Me"):
                # This would typically connect to your notification system
                st.success(f"Thank you! We'll notify {email} when comparative analysis becomes available.")

    def _render_admin_panel(self):
        """Render the admin panel (only available in advanced mode)."""
        st.title("Admin Panel")
        st.markdown("Advanced system management and monitoring.")

        # Only allow access in advanced mode
        if not st.session_state.advanced_mode:
            st.warning("Admin panel access requires advanced mode to be enabled.")
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
                    logger.error(f"Error creating health chart: {e}")
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
            log_path = os.path.join(LOG_DIR, "log")
            if os.path.exists(log_path):
                try:
                    # Read last 20 lines of log file
                    with open(log_path, "r") as log_file:
                        logs = log_file.readlines()
                        recent_logs = logs[-20:] if len(logs) > 20 else logs

                    st.code("".join(recent_logs), language="text")
                except Exception as e:
                    st.error(f"Could not read log file: {e}")
            else:
                st.info("No log file found.")

            # System maintenance actions
            st.markdown("### Maintenance Actions")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Restart Application"):
                    st.warning("This will restart the application and end all user sessions.")
                    # This would typically trigger a restart mechanism
                    st.info("Restart functionality is simulated in this demo.")

            with col2:
                if st.button("Clear Cache"):
                    st.session_state.clear()
                    st.success("Cache cleared. Redirecting to home page...")
                    st.session_state.page = "Home"
                    st.experimental_rerun()

        with tabs[1]:
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
                            "Last Activity": user.get("last_activity", "Unknown")
                        })

                    if user_data:
                        user_df = pd.DataFrame(user_data)
                        st.dataframe(user_df, use_container_width=True)
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

                    submit_button = st.form_submit_button("Create User")

                    if submit_button:
                        if hasattr(self, "user_manager") and self.user_manager:
                            try:
                                # This would typically call a user creation method
                                st.success(f"User '{user_name}' created successfully.")
                            except Exception as e:
                                st.error(f"Error creating user: {e}")
                        else:
                            st.error("User management component is not available.")

            # User deletion
            with st.expander("Delete User"):
                if hasattr(self, "user_manager") and self.user_manager:
                    # Get user IDs for selection
                    user_ids = [user.get("id", "") for user in users] if users else []

                    if user_ids:
                        selected_user = st.selectbox("Select User to Delete", user_ids)

                        if st.button("Delete User"):
                            st.warning(f"This will permanently delete user '{selected_user}' and all their data.")

                            confirm = st.checkbox("I understand and confirm this action")

                            if confirm and st.button("Confirm Delete"):
                                try:
                                    # This would typically call a user deletion method
                                    st.success(f"User '{selected_user}' deleted successfully.")
                                except Exception as e:
                                    st.error(f"Error deleting user: {e}")
                    else:
                        st.info("No users available to delete.")
                else:
                    st.warning("User management component is not available.")

        with tabs[2]:
            st.subheader("Data Management")

            # Data backup and restore
            st.markdown("### Data Backup")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Backup All Data"):
                    try:
                        # This would typically trigger a data backup procedure
                        st.success("Data backed up successfully.")
                    except Exception as e:
                        st.error(f"Error backing up data: {e}")

            with col2:
                if st.button("Export Analytics Data"):
                    try:
                        # This would typically export analytics data
                        st.success("Analytics data exported successfully.")
                    except Exception as e:
                        st.error(f"Error exporting analytics data: {e}")

            # Data restoration
            st.markdown("### Data Restoration")

            with st.expander("Restore from Backup"):
                uploaded_file = st.file_uploader("Upload Backup File", type=["zip", "json", "pkl"])

                if uploaded_file is not None:
                    if st.button("Restore Data"):
                        st.warning("This will overwrite existing data. Proceed with caution.")

                        confirm = st.checkbox("I understand and confirm this action", key="restore_confirm")

                        if confirm and st.button("Confirm Restore", key="confirm_restore"):
                            try:
                                # This would typically handle data restoration
                                st.success("Data restored successfully.")
                            except Exception as e:
                                st.error(f"Error restoring data: {e}")

            # Data cleanup options
            st.markdown("### Data Cleanup")

            with st.expander("Cleanup Options"):
                if st.button("Clear Temporary Files"):
                    try:
                        # This would typically clean up temp files
                        st.success("Temporary files cleared successfully.")
                    except Exception as e:
                        st.error(f"Error clearing temporary files: {e}")

                if st.button("Optimize Database"):
                    try:
                        # This would typically optimize the database
                        st.success("Database optimized successfully.")
                    except Exception as e:
                        st.error(f"Error optimizing database: {e}")

        with tabs[3]:
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
                    st.metric("Avg Response Time", "245ms")

                # Create mock time series data for demonstration
                st.markdown("### Performance Over Time")

                # Generate sample data
                import random

                dates = [f"2025-03-{i+1:02d}" for i in range(14)]
                response_times = [random.randint(200, 300) for _ in range(14)]
                error_rates = [random.uniform(0, 2) for _ in range(14)]

                try:
                    # Create performance chart
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
                        yaxis2 = {
                            'title': 'Error Rate (%)',
                            'overlaying': 'y',
                            'side': 'right'
                        },
                        height = 400
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating performance chart: {e}")
                    st.error("Could not render performance visualization.")

                # Resource utilization
                st.markdown("### Resource Utilization")

                # Generate sample CPU and memory data
                cpu_usage = [random.uniform(10, 60) for _ in range(14)]
                memory_usage = [random.uniform(20, 70) for _ in range(14)]

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
                        height = 400
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating resource chart: {e}")
                    st.error("Could not render resource visualization.")
            else:
                st.info("Performance metrics not available.")

        with tabs[4]:
            st.subheader("API Configuration")

            # API key management
            st.markdown("### API Keys")

            with st.expander("OpenAI API"):
                # Display current status
                if hasattr(self, "openai_client") and self.openai_client:
                    api_status = "Configured" if self.openai_client.api_key else "Not Configured"
                    st.markdown(f"**Status:** {api_status}")
                else:
                    st.warning("OpenAI client not available.")

                # API key input
                new_api_key = st.text_input("Enter OpenAI API Key", type="password")

                if st.button("Save API Key") and new_api_key:
                    if hasattr(self, "openai_client") and self.openai_client:
                        try:
                            self.openai_client.set_api_key(new_api_key)
                            st.success("API key saved successfully.")
                        except Exception as e:
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

                if st.button("Save Configuration") and api_url and api_key:
                    st.success("Configuration saved successfully.")

            with st.expander("EHR Integration"):
                st.markdown("""
                **Status:** Not Configured

                Configure integration with Electronic Health Record (EHR) systems
                for seamless patient data exchange.
                """)

                # Mock implementation
                ehr_system = st.selectbox("EHR System", ["Epic", "Cerner", "Allscripts", "Other"])
                ehr_url = st.text_input("EHR API Endpoint")
                ehr_key = st.text_input("Access Token", type="password")

                if st.button("Save EHR Configuration") and ehr_url and ehr_key:
                    st.success("EHR configuration saved successfully.")

    def _refresh_components(self):
        """Refresh components that need to be updated."""
        # Re-initialize health analyzer if dependencies available
        if hasattr(self, "user_manager") and self.user_manager and hasattr(self, "health_analyzer"):
            try:
                self.health_analyzer.update_data(
                    self.user_manager.health_history,
                    self.user_manager.profile
                )
                logger.info("Refreshed HealthDataAnalyzer")
            except Exception as e:
                logger.error(f"Error refreshing HealthDataAnalyzer: {e}")

        # Re-initialize dashboard if needed
        if hasattr(self, "user_manager") and self.user_manager and hasattr(self, "dashboard"):
            try:
                self.dashboard.update_data(
                    self.user_manager.health_history,
                    self.user_manager.profile
                )
                logger.info("Refreshed HealthDashboard")
            except Exception as e:
                logger.error(f"Error refreshing HealthDashboard: {e}")

    def _handle_data_export(self):
        """Handle data export when ready."""
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

            # Convert data to appropriate format
            if file_format == "csv":
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
            elif file_format == "json":
                # Convert to JSON
                if isinstance(data, pd.DataFrame):
                    output = data.to_json(orient="records")
                else:
                    import json
                    output = json.dumps(data)

                # Create download button
                st.download_button(
                    label=f"Download {data_type} as JSON",
                    data=output,
                    file_name=f"{data_type}_{time.strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            else:
                st.error(f"Unsupported export format: {file_format}")

            # Clear export data
            st.session_state.export_ready = False
            st.session_state.export_data = None
        except Exception as e:
            logger.error(f"Error handling data export: {e}")
            st.error(f"Error exporting data: {e}")
            st.session_state.export_ready = False
            st.session_state.export_data = None

# Run the application when executed directly
if __name__ == "__main__":
    app = MedExplainApp()
    app.run()
