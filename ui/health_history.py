"""
Health History UI for MedExplain AI Pro.

This module provides the user interface for viewing and managing
the user's health history records.
"""

import streamlit as st
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import visualization functionality
from medexplain.analytics.visualization import (
    create_symptom_frequency_chart,
    create_symptom_timeline_chart
)

# Configure logger
logger = logging.getLogger(__name__)

def render_health_history() -> None:
    """Render the health history interface."""
    st.title("Health History")
    st.markdown("View and track your symptom history over time.")

    # Check if user manager is in session state
    if "user_manager" not in st.session_state:
        st.error("Application not properly initialized. Please refresh the page.")
        return

    user_manager = st.session_state.user_manager
    health_data_manager = st.session_state.get("health_data_manager")

    # Get health history
    history = user_manager.health_history

    if not history:
        st.info("No health history available yet. Use the Symptom Analyzer to start tracking your symptoms.")
        return

    # Create tabs for different views
    tabs = st.tabs(["Timeline", "Symptom Trends", "Detailed History", "Export Data"])

    with tabs[0]:
        _render_timeline_view(history, health_data_manager)

    with tabs[1]:
        _render_trends_view(history, health_data_manager)

    with tabs[2]:
        _render_detailed_history(history, health_data_manager)

    with tabs[3]:
        _render_export_options(history, health_data_manager)

def _render_timeline_view(history, health_data_manager) -> None:
    """
    Render the timeline view of health history.

    Args:
        history: List of health history entries
        health_data_manager: Optional health data manager for symptom names
    """
    st.subheader("Timeline View")

    # Get symptom names if health data manager is available
    symptom_names = {}
    if health_data_manager:
        # Collect all unique symptoms
        all_symptoms = set()
        for entry in history:
            all_symptoms.update(entry.get("symptoms", []))

        # Get names for all symptoms
        for symptom_id in all_symptoms:
            symptom_info = health_data_manager.get_symptom_info(symptom_id)
            if symptom_info and "name" in symptom_info:
                symptom_names[symptom_id] = symptom_info["name"]

    # Display recent entries in reverse chronological order
    for entry in reversed(history):
        date = entry.get("date", "")
        symptoms = entry.get("symptoms", [])
        results = entry.get("analysis_results", {})

        # Get symptom names
        symptom_display_names = []
        for symptom_id in symptoms:
            name = symptom_names.get(symptom_id, symptom_id)
            symptom_display_names.append(name)

        st.markdown(f"""
        <div class="symptom-card">
            <h4>Check from {date}</h4>
            <p><strong>Symptoms:</strong> {", ".join(symptom_display_names)}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show the top condition that was matched, if any
        if results:
            top_condition_id = next(iter(results))
            top_condition = results[top_condition_id]

            st.markdown(f"""
            <div style="margin-left: 20px; margin-bottom: 20px;">
                <p>Top potential condition: <strong>{top_condition['name']}</strong> (Confidence: {top_condition['confidence']}%)</p>
            </div>
            """, unsafe_allow_html=True)

def _render_trends_view(history, health_data_manager) -> None:
    """
    Render the trends view of health history.

    Args:
        history: List of health history entries
        health_data_manager: Optional health data manager for symptom names
    """
    st.subheader("Symptom Trends")

    # Get symptom names if health data manager is available
    symptom_names = {}
    if health_data_manager:
        # Collect all unique symptoms
        all_symptoms = set()
        for entry in history:
            all_symptoms.update(entry.get("symptoms", []))

        # Get names for all symptoms
        for symptom_id in all_symptoms:
            symptom_info = health_data_manager.get_symptom_info(symptom_id)
            if symptom_info and "name" in symptom_info:
                symptom_names[symptom_id] = symptom_info["name"]

    # Prepare data for trends chart
    symptom_counts = {}
    for entry in history:
        for symptom_id in entry.get("symptoms", []):
            if symptom_id not in symptom_counts:
                symptom_counts[symptom_id] = 0
            symptom_counts[symptom_id] += 1

    # Convert symptom IDs to names if available
    named_counts = {}
    for symptom_id, count in symptom_counts.items():
        name = symptom_names.get(symptom_id, symptom_id)
        named_counts[name] = count

    # Create frequency chart
    try:
        if named_counts:
            st.markdown("### Symptom Frequency")
            fig = create_symptom_frequency_chart(named_counts)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No symptom data available for frequency analysis.")
    except Exception as e:
        logger.error("Error creating frequency chart: %s", str(e))
        st.error(f"Error creating frequency chart: {str(e)}")

    # Create timeline chart
    try:
        st.markdown("### Symptoms Over Time")
        fig = create_symptom_timeline_chart(history, symptom_names)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error("Error creating timeline chart: %s", str(e))
        st.error(f"Error creating timeline chart: {str(e)}")

    # Add more sophisticated analysis link
    st.markdown("---")
    st.markdown("For a more detailed breakdown of your symptom patterns over time, visit the [Health Dashboard](#) page.")

def _render_detailed_history(history, health_data_manager) -> None:
    """
    Render the detailed history view.

    Args:
        history: List of health history entries
        health_data_manager: Optional health data manager for symptom names
    """
    st.subheader("Detailed History")

    # Get symptom names if health data manager is available
    symptom_names = {}
    if health_data_manager:
        # Collect all unique symptoms
        all_symptoms = set()
        for entry in history:
            all_symptoms.update(entry.get("symptoms", []))

        # Get names for all symptoms
        for symptom_id in all_symptoms:
            symptom_info = health_data_manager.get_symptom_info(symptom_id)
            if symptom_info and "name" in symptom_info:
                symptom_names[symptom_id] = symptom_info["name"]

    # Create expandable sections for each entry
    for i, entry in enumerate(reversed(history)):
        date = entry.get("date", "")
        symptoms = entry.get("symptoms", [])
        results = entry.get("analysis_results", {})

        # Get symptom names
        symptom_display_names = []
        for symptom_id in symptoms:
            name = symptom_names.get(symptom_id, symptom_id)
            symptom_display_names.append(name)

        with st.expander(f"Check from {date}"):
            st.markdown(f"**Symptoms:** {', '.join(symptom_display_names)}")

            # Display additional information if available
            if "symptom_duration" in entry:
                st.markdown(f"**Duration:** {entry['symptom_duration']}")

            if "symptom_severity" in entry:
                st.markdown(f"**Severity:** {entry['symptom_severity']}")

            if "had_fever" in entry and entry["had_fever"]:
                st.markdown(f"**Fever:** Yes, {entry.get('fever_temp', 'temperature not recorded')}°F")

            # Display risk assessment if available
            if "risk_assessment" in entry:
                risk = entry["risk_assessment"]
                st.markdown(f"**Risk Level:** {risk.get('risk_level', 'unknown').capitalize()}")

                if "risk_factors" in risk and risk["risk_factors"]:
                    st.markdown("**Risk Factors:**")
                    for factor in risk["risk_factors"]:
                        st.markdown(f"- {factor}")

            # Display potential conditions
            if results:
                st.markdown("**Potential Conditions:**")

                for condition_id, condition_data in results.items():
                    st.markdown(f"""
                    - **{condition_data['name']}** (Confidence: {condition_data['confidence']}%)
                      - Severity: {condition_data['severity'].capitalize()}
                      - Description: {condition_data['description']}
                    """)
            else:
                st.info("No specific conditions were matched for these symptoms.")

def _render_export_options(history, health_data_manager) -> None:
    """
    Render data export options.

    Args:
        history: List of health history entries
        health_data_manager: Optional health data manager for symptom names
    """
    st.subheader("Export Health Data")

    st.markdown("""
    You can export your health history data in various formats for your records or to share with healthcare providers.
    """)

    # Get symptom names if health data manager is available
    symptom_names = {}
    if health_data_manager:
        # Collect all unique symptoms
        all_symptoms = set()
        for entry in history:
            all_symptoms.update(entry.get("symptoms", []))

        # Get names for all symptoms
        for symptom_id in all_symptoms:
            symptom_info = health_data_manager.get_symptom_info(symptom_id)
            if symptom_info and "name" in symptom_info:
                symptom_names[symptom_id] = symptom_info["name"]

    export_format = st.selectbox(
        "Export Format",
        ["JSON", "CSV", "PDF Report"]
    )

    if st.button("Export Data"):
        _process_data_export(history, export_format, symptom_names)

    # Option to delete history
    st.markdown("---")
    with st.expander("Delete History Data"):
        st.warning("This action will permanently delete your health history. This cannot be undone.")
        if st.button("Delete All Health History", key="delete_history"):
            try:
                # Clear history
                user_manager = st.session_state.user_manager
                user_manager.health_history = []
                user_manager.save_health_history()
                st.success("Health history has been deleted.")
                st.experimental_rerun()
            except Exception as e:
                logger.error("Error deleting health history: %s", str(e))
                st.error(f"Error deleting health history: {str(e)}")

def _process_data_export(history, export_format, symptom_names) -> None:
    """
    Process the export of health history data.

    Args:
        history: List of health history entries
        export_format: Format to export in (JSON, CSV, PDF)
        symptom_names: Dictionary mapping symptom IDs to display names
    """
    try:
        if export_format == "JSON":
            # Convert history to JSON
            json_str = json.dumps(history, indent=4)

            # Create download button
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="health_history.json",
                mime="application/json"
            )

            st.success("JSON export ready for download.")

        elif export_format == "CSV":
            # Create a simplified DataFrame for CSV export
            csv_data = []

            for entry in history:
                date = entry.get("date", "")
                symptoms = entry.get("symptoms", [])

                # Get symptom names
                symptom_display_names = []
                for symptom_id in symptoms:
                    name = symptom_names.get(symptom_id, symptom_id)
                    symptom_display_names.append(name)

                # Get top condition if available
                top_condition = "None"
                top_confidence = 0

                if "analysis_results" in entry and entry["analysis_results"]:
                    results = entry["analysis_results"]
                    top_condition_id = next(iter(results))
                    top_condition = results[top_condition_id]["name"]
                    top_confidence = results[top_condition_id]["confidence"]

                # Get risk level if available
                risk_level = "Unknown"
                if "risk_assessment" in entry and entry["risk_assessment"]:
                    risk_level = entry["risk_assessment"].get("risk_level", "Unknown").capitalize()

                csv_data.append({
                    "Date": date,
                    "Symptoms": ", ".join(symptom_display_names),
                    "Duration": entry.get("symptom_duration", ""),
                    "Severity": entry.get("symptom_severity", ""),
                    "Top Condition": top_condition,
                    "Confidence": top_confidence,
                    "Risk Level": risk_level
                })

            # Convert to DataFrame
            csv_df = pd.DataFrame(csv_data)

            # Convert to CSV
            csv = csv_df.to_csv(index=False)

            # Create download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="health_history.csv",
                mime="text/csv"
            )

            st.success("CSV export ready for download.")

        elif export_format == "PDF Report":
            st.info("In a production version, this would generate a comprehensive PDF report of your health history that you could share with healthcare providers.")

            # Mock PDF generation feedback
            st.markdown("""
            📋 **PDF Report Would Include:**
            - Complete symptom history with dates
            - Trend analysis of your most common symptoms
            - Summary of identified conditions
            - Risk assessments over time
            - Visualizations of your health patterns
            - Recommendations based on your health history
            """)
    except Exception as e:
        logger.error("Error exporting data: %s", str(e))
        st.error(f"Error exporting data: {str(e)}")
