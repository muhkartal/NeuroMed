"""
Settings UI for MedExplain AI Pro.

This module provides the user interface for configuring application
settings and managing user profiles.
"""

import streamlit as st
import json
import logging
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

class SettingsUI:
    """
    Class for the Settings UI, providing an interface for configuring
    application settings and managing user profiles.
    """

    def __init__(self, user_manager=None, openai_client=None):
        """
        Initialize the SettingsUI.

        Args:
            user_manager: The user profile manager instance
            openai_client: The OpenAI client instance
        """
        self.user_manager = user_manager
        self.openai_client = openai_client

    def render(self) -> None:
        """Render the settings interface."""
        st.title("Settings")
        st.markdown("Configure your personal information and application settings.")

        # Use session state components if not explicitly provided
        user_manager = self.user_manager or st.session_state.get("user_manager")
        openai_client = self.openai_client or st.session_state.get("openai_client")

        # Check if user manager is available
        if not user_manager:
            st.error("Application not properly initialized. Please refresh the page.")
            return

        # Create tabs for different settings sections
        tabs = st.tabs(["Profile", "API Configuration", "Preferences", "Data Management"])

        with tabs[0]:
            self._render_profile_settings(user_manager)

        with tabs[1]:
            self._render_api_settings(openai_client)

        with tabs[2]:
            self._render_preferences()

        with tabs[3]:
            self._render_data_management(user_manager)

    def _render_profile_settings(self, user_manager) -> None:
        """
        Render the profile settings tab.

        Args:
            user_manager: The user profile manager instance
        """
        st.subheader("Personal Profile")
        st.markdown("This information helps provide more personalized health information.")

        try:
            profile = user_manager.profile

            col1, col2 = st.columns(2)

            with col1:
                profile["name"] = st.text_input("Name", value=profile.get("name", ""))
                profile["age"] = st.number_input("Age", min_value=0, max_value=120, value=int(profile.get("age", 0) or 0))
                profile["gender"] = st.selectbox(
                    "Gender",
                    ["", "Male", "Female", "Non-binary", "Prefer not to say"],
                    index=0 if not profile.get("gender") else ["", "Male", "Female", "Non-binary", "Prefer not to say"].index(profile.get("gender"))
                )

            with col2:
                profile["height"] = st.text_input("Height", value=profile.get("height", ""), placeholder="e.g., 5'10\" or 178 cm")
                profile["weight"] = st.text_input("Weight", value=profile.get("weight", ""), placeholder="e.g., 165 lbs or 75 kg")
                profile["blood_type"] = st.selectbox(
                    "Blood Type",
                    ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                    index=0 if not profile.get("blood_type") else ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].index(profile.get("blood_type", ""))
                )

            st.subheader("Additional Health Information")

            # Add fields for additional information
            profile["exercise_frequency"] = st.selectbox(
                "Exercise Frequency",
                ["", "Sedentary", "Light (1-2 days/week)", "Moderate (3-4 days/week)", "Active (5+ days/week)"],
                index=0 if not profile.get("exercise_frequency") else ["", "Sedentary", "Light (1-2 days/week)", "Moderate (3-4 days/week)", "Active (5+ days/week)"].index(profile.get("exercise_frequency", ""))
            )

            profile["smoking_status"] = st.selectbox(
                "Smoking Status",
                ["", "Never smoked", "Former smoker", "Current smoker"],
                index=0 if not profile.get("smoking_status") else ["", "Never smoked", "Former smoker", "Current smoker"].index(profile.get("smoking_status", ""))
            )

            # Allergies
            st.subheader("Allergies")
            self._render_profile_list_editor(user_manager, "allergies", "Allergy")

            # Chronic conditions
            st.subheader("Chronic Conditions")
            self._render_profile_list_editor(user_manager, "chronic_conditions", "Chronic Condition")

            # Medications
            st.subheader("Medications")
            self._render_profile_list_editor(user_manager, "medications", "Medication")

            if st.button("Save Profile", key="save_profile"):
                user_manager.save_profile()
                st.success("Profile saved successfully!")
        except Exception as e:
            logger.error("Error rendering profile settings: %s", str(e))
            st.error(f"Error saving profile settings: {str(e)}")

    def _render_profile_list_editor(self, user_manager, list_name, item_label) -> None:
        """
        Render an editor for a list in the user profile.

        Args:
            user_manager: The user profile manager instance
            list_name: Name of the list to edit
            item_label: Label for the item type
        """
        profile = user_manager.profile

        # Initialize list if it doesn't exist
        if list_name not in profile:
            profile[list_name] = []

        # Add new item
        new_item = st.text_input(f"Add {item_label}", key=f"add_{list_name}")
        if st.button("Add", key=f"add_{list_name}_btn") and new_item:
            if new_item not in profile[list_name]:
                profile[list_name].append(new_item)

        # Display current items
        if profile[list_name]:
            st.markdown(f"**Current {item_label}s:**")
            for i, item in enumerate(profile[list_name]):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"- {item}")
                with col2:
                    if st.button("Remove", key=f"remove_{list_name}_{i}"):
                        profile[list_name].pop(i)
                        st.experimental_rerun()

    def _render_api_settings(self, openai_client) -> None:
        """
        Render the API configuration tab.

        Args:
            openai_client: The OpenAI client instance
        """
        st.subheader("API Configuration")
        st.markdown("Configure API keys for enhanced functionality.")

        # OpenAI API key
        current_key = openai_client.api_key if openai_client else ""
        openai_key = st.text_input(
            "OpenAI API Key",
            value=current_key,
            type="password",
            help="Enter your OpenAI API key to enable AI-powered symptom analysis."
        )

        if st.button("Save API Key", key="save_api_key"):
            try:
                if openai_key:
                    if openai_client and openai_client.set_api_key(openai_key):
                        st.success("API key saved successfully!")
                    else:
                        st.error("Failed to save API key. Please try again.")
                else:
                    st.warning("Please enter an API key.")
            except Exception as e:
                logger.error("Error saving API key: %s", str(e))
                st.error(f"Error saving API key: {str(e)}")

        st.markdown("""
        **Note:** The OpenAI API key is used for advanced symptom analysis and explanation.
        If you don't have a key, the application will still function, but with limited capabilities.
        Your API key is stored locally and is never sent to our servers.
        """)

        # Add model selection if OpenAI client is available
        if openai_client and openai_client.is_available():
            st.subheader("AI Model Settings")

            current_model = openai_client.model
            model_options = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]

            selected_model = st.selectbox(
                "AI Model",
                options=model_options,
                index=model_options.index(current_model) if current_model in model_options else 0,
                help="Select the AI model to use for generating explanations."
            )

            if selected_model != current_model and st.button("Update Model"):
                try:
                    openai_client.set_model(selected_model)
                    st.success(f"Model updated to {selected_model}")
                except Exception as e:
                    logger.error("Error updating model: %s", str(e))
                    st.error(f"Error updating model: {str(e)}")

    def _render_preferences(self) -> None:
        """Render the application preferences tab."""
        st.subheader("Application Preferences")

        # UI Theme
        theme = st.selectbox(
            "UI Theme",
            ["Light", "Dark", "System Default"],
            index=0
        )

        # Language
        language = st.selectbox(
            "Language",
            ["English", "Spanish", "French", "German", "Turkish"],
            index=0
        )

        # Dashboard settings
        st.subheader("Dashboard Settings")
        dashboard_theme = st.selectbox(
            "Dashboard Theme",
            ["Default", "Plotly", "Streamlit", "Minimal"],
            index=0
        )

        # Enable features
        st.subheader("Feature Settings")
        enable_chat = st.checkbox("Enable Health Chat", value=True)
        enable_voice = st.checkbox("Enable Voice Input (Beta)", value=False)
        enable_advanced_analytics = st.checkbox("Enable Advanced Analytics", value=True)

        # Data storage
        st.subheader("Data Storage")
        store_locally = st.checkbox("Store health data locally", value=True, help="Your health data is stored only on your device.")
        enable_stats = st.checkbox("Enable anonymous usage statistics", value=False, help="Help us improve by sending anonymous usage data.")

        if st.button("Save Preferences", key="save_preferences"):
            # In a production app, this would save to a config file or database
            st.success("Preferences saved successfully!")

            # Update session state with new preferences
            if "preferences" not in st.session_state:
                st.session_state.preferences = {}

            st.session_state.preferences.update({
                "theme": theme,
                "language": language,
                "dashboard_theme": dashboard_theme,
                "enable_chat": enable_chat,
                "enable_voice": enable_voice,
                "enable_advanced_analytics": enable_advanced_analytics,
                "store_locally": store_locally,
                "enable_stats": enable_stats
            })

            st.info("Note: Some preferences may require restarting the application to take effect.")

    def _render_data_management(self, user_manager) -> None:
        """
        Render the data management tab.

        Args:
            user_manager: The user profile manager instance
        """
        st.subheader("Data Management")
        st.markdown("Manage your data stored in the application.")

        # Data backup
        st.markdown("### Backup Your Data")
        st.markdown("Create a backup of all your health data and settings.")

        if st.button("Create Backup", key="create_backup"):
            try:
                # Prepare data for backup
                backup_data = user_manager.export_user_data()

                # Add version and timestamp
                backup_data["app_version"] = st.session_state.get("app_version", "1.0.0")
                backup_data["backup_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Convert to JSON
                backup_json = json.dumps(backup_data, indent=4)

                # Create download button
                st.download_button(
                    label="Download Backup",
                    data=backup_json,
                    file_name=f"medexplain_backup_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

                st.success("Backup created successfully!")
            except Exception as e:
                logger.error("Error creating backup: %s", str(e))
                st.error(f"Error creating backup: {str(e)}")

        # Data restore
        st.markdown("### Restore From Backup")
        st.markdown("Restore your data from a previous backup.")

        backup_file = st.file_uploader("Upload Backup File", type=["json"])

        if backup_file and st.button("Restore Data", key="restore_data"):
            try:
                # Read the backup file
                backup_data = json.load(backup_file)

                # Validate backup data
                required_keys = ["profile", "health_history"]
                if not all(key in backup_data for key in required_keys):
                    st.error("Invalid backup file format. Could not restore data.")
                    return

                # Import user data
                success = user_manager.import_user_data(backup_data)

                if success:
                    st.success("Data restored successfully!")
                    st.info("Please refresh the application to see the restored data.")
                else:
                    st.error("Failed to restore data. Please try again.")
            except Exception as e:
                logger.error("Error restoring data: %s", str(e))
                st.error(f"Error restoring data: {str(e)}")

        # Data deletion
        st.markdown("### Delete All Data")
        st.markdown("Delete all your data from the application. This action cannot be undone.")

        with st.expander("Delete All Data"):
            st.warning("This will permanently delete all your data, including your profile and health history. This action cannot be undone.")

            confirmation = st.text_input("Type 'DELETE' to confirm deletion:", key="delete_confirmation")

            if st.button("Delete All Data", key="delete_all_data") and confirmation == "DELETE":
                try:
                    # Clear profile and health history
                    user_manager.profile = user_manager._create_default_profile()
                    user_manager.health_history = []

                    # Save empty data
                    user_manager.save_profile()
                    user_manager.save_health_history()

                    st.success("All data has been deleted successfully.")
                    st.info("Please refresh the application to complete the process.")
                except Exception as e:
                    logger.error("Error deleting data: %s", str(e))
                    st.error(f"Error deleting data: {str(e)}")


# For backward compatibility with the functional approach
def render_settings() -> None:
    """Render the settings interface using the class-based implementation."""
    settings_ui = SettingsUI()
    settings_ui.render()
