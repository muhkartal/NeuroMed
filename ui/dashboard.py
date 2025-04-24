"""
Dashboard UI for MedExplain AI Pro.

This module provides the user interface for the health dashboard,
allowing users to visualize and interact with their health data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Import analytics functionality
from analytics.visualization import (
    create_symptom_frequency_chart,
    create_symptom_timeline_chart,
    create_correlation_heatmap,
    create_risk_assessment_chart,
    create_monthly_trend_chart,
    create_dashboard
)

# Configure logger
logger = logging.getLogger(__name__)

class HealthDashboard:
    """
    Interactive health dashboard for visualizing health data and trends.

    This class is responsible for:
    - Rendering the main dashboard interface
    - Providing different views of health data
    - Creating interactive visualizations
    """

    def __init__(self, health_history: Optional[List[Dict[str, Any]]] = None,
                user_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the health dashboard.

        Args:
            health_history: List of health history entries
            user_profile: User profile data
        """
        self.health_history = health_history or []
        self.user_profile = user_profile or {}
        self.theme = "default"

        logger.info("HealthDashboard initialized")

    def set_theme(self, theme: str) -> None:
        """
        Set the dashboard visualization theme.

        Args:
            theme: Theme name (e.g., 'default', 'plotly', 'streamlit', 'minimal')
        """
        self.theme = theme
        logger.debug("Dashboard theme set to %s", theme)

    def render_dashboard(self) -> None:
        """Render the main dashboard with key health metrics."""
        st.title("Health Analytics Dashboard")

        if not self.health_history:
            st.info("No health data available yet. Use the Symptom Analyzer to start tracking your health.")
            return

        # Create tabs for different dashboard views
        tabs = st.tabs(["Overview", "Symptom Trends", "Condition Analysis", "Timeline"])

        with tabs[0]:
            self._render_overview_tab()

        with tabs[1]:
            self._render_symptom_trends_tab()

        with tabs[2]:
            self._render_condition_analysis_tab()

        with tabs[3]:
            self._render_timeline_tab()

    def _render_overview_tab(self) -> None:
        """Render the overview dashboard tab with key metrics."""
        st.subheader("Health Overview")

        # Calculate key metrics
        total_checks = len(self.health_history)
        unique_symptoms = set()
        for entry in self.health_history:
            unique_symptoms.update(entry.get("symptoms", []))

        unique_symptom_count = len(unique_symptoms)

        # Get most recent check
        most_recent = self.health_history[-1] if self.health_history else None
        most_recent_date = most_recent.get("date", "N/A") if most_recent else "N/A"

        # Create summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Total Health Checks",
                value=total_checks
            )

        with col2:
            st.metric(
                label="Unique Symptoms Tracked",
                value=unique_symptom_count
            )

        with col3:
            st.metric(
                label="Last Check",
                value=most_recent_date
            )

        # Create comprehensive dashboard
        try:
            # Get risk assessment if available
            risk_assessment = {}
            for entry in reversed(self.health_history):
                if "risk_assessment" in entry:
                    risk_assessment = entry["risk_assessment"]
                    break

            # If no risk assessment found, create a default one
            if not risk_assessment:
                risk_assessment = {
                    "overall_risk": "unknown",
                    "risk_factors": [],
                    "protective_factors": []
                }

            # Create symptom names mapping if available
            symptom_names = {}
            health_data_manager = st.session_state.get("health_data_manager")
            if health_data_manager:
                for symptom_id in unique_symptoms:
                    symptom_info = health_data_manager.get_symptom_info(symptom_id)
                    if symptom_info and "name" in symptom_info:
                        symptom_names[symptom_id] = symptom_info["name"]

            # Create dashboard figure
            fig = create_dashboard(self.health_history, risk_assessment, symptom_names)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error("Error creating dashboard: %s", str(e))
            st.error(f"Error creating dashboard visualization: {str(e)}")

            # Fallback to individual charts
            self._render_fallback_overview()

    def _render_fallback_overview(self) -> None:
        """Render individual charts as fallback if dashboard creation fails."""
        # Create symptom summary
        st.subheader("Symptom Summary")

        symptom_counts = {}
        for entry in self.health_history:
            for symptom in entry.get("symptoms", []):
                if symptom not in symptom_counts:
                    symptom_counts[symptom] = 0
                symptom_counts[symptom] += 1

        if symptom_counts:
            # Sort by count (descending)
            symptom_counts = dict(sorted(
                symptom_counts.items(),
                key=lambda item: item[1],
                reverse=True
            ))

            # Create chart
            fig = create_symptom_frequency_chart(symptom_counts)
            st.plotly_chart(fig, use_container_width=True)

    def _render_symptom_trends_tab(self) -> None:
        """Render the symptom trends tab."""
        st.subheader("Symptom Trends Over Time")

        if not self.health_history:
            st.info("No symptom history available yet.")
            return

        # Create symptom names mapping if available
        symptom_names = {}
        health_data_manager = st.session_state.get("health_data_manager")
        if health_data_manager:
            all_symptoms = set()
            for entry in self.health_history:
                all_symptoms.update(entry.get("symptoms", []))

            for symptom_id in all_symptoms:
                symptom_info = health_data_manager.get_symptom_info(symptom_id)
                if symptom_info and "name" in symptom_info:
                    symptom_names[symptom_id] = symptom_info["name"]

        # Create timeline chart
        try:
            st.markdown("### Symptoms Over Time")
            timeline_fig = create_symptom_timeline_chart(self.health_history, symptom_names)
            st.plotly_chart(timeline_fig, use_container_width=True)
        except Exception as e:
            logger.error("Error creating timeline chart: %s", str(e))
            st.error(f"Error creating timeline visualization: {str(e)}")

        # Create monthly trend chart
        try:
            st.markdown("### Monthly Symptom Patterns")
            monthly_fig = create_monthly_trend_chart(self.health_history, symptom_names)
            st.plotly_chart(monthly_fig, use_container_width=True)
        except Exception as e:
            logger.error("Error creating monthly trend chart: %s", str(e))
            st.error(f"Error creating monthly trend visualization: {str(e)}")

    def _render_condition_analysis_tab(self) -> None:
        """Render the condition analysis tab."""
        st.subheader("Condition Analysis")

        # Extract conditions from analysis results
        conditions = []
        confidence_values = []
        dates = []

        for entry in self.health_history:
            entry_date = entry.get("date", "")
            results = entry.get("analysis_results", {})

            for condition_id, condition_data in results.items():
                conditions.append(condition_data.get("name", condition_id))
                confidence_values.append(condition_data.get("confidence", 0))
                dates.append(entry_date)

        if not conditions:
            st.info("No condition analysis data available yet.")
            return

        # Create DataFrame
        df = pd.DataFrame({
            "condition": conditions,
            "confidence": confidence_values,
            "date": dates
        })

        # Group by condition and calculate average confidence
        condition_summary = df.groupby("condition")["confidence"].mean().reset_index()
        condition_summary = condition_summary.sort_values("confidence", ascending=False)

        # Create bar chart of average confidence by condition
        st.markdown("### Average Confidence by Condition")

        try:
            fig = px.bar(
                condition_summary,
                x="condition",
                y="confidence",
                color="confidence",
                color_continuous_scale="Viridis",
                title="Average Confidence by Condition"
            )

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error("Error creating confidence bar chart: %s", str(e))

            # Fallback to a simple table
            st.dataframe(condition_summary)

        # Create condition confidence over time chart
        st.markdown("### Condition Confidence Over Time")

        try:
            confidence_fig = create_condition_confidence_chart(self.health_history)
            st.plotly_chart(confidence_fig, use_container_width=True)
        except Exception as e:
            logger.error("Error creating confidence chart: %s", str(e))
            st.error(f"Error creating confidence visualization: {str(e)}")

    def _render_timeline_tab(self) -> None:
        """Render the timeline tab with chronological health events."""
        st.subheader("Health Timeline")

        if not self.health_history:
            st.info("No health timeline data available yet.")
            return

        # Get symptom names mapping if available
        symptom_names = {}
        health_data_manager = st.session_state.get("health_data_manager")
        if health_data_manager:
            all_symptoms = set()
            for entry in self.health_history:
                all_symptoms.update(entry.get("symptoms", []))

            for symptom_id in all_symptoms:
                symptom_info = health_data_manager.get_symptom_info(symptom_id)
                if symptom_info and "name" in symptom_info:
                    symptom_names[symptom_id] = symptom_info["name"]

        # Sort entries by date
        sorted_entries = sorted(
            self.health_history,
            key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d %H:%M:%S") if isinstance(x["date"], str) else datetime.now(),
            reverse=True
        )

        # Create a visual timeline
        for i, entry in enumerate(sorted_entries):
            # Get date and symptoms
            date = entry.get("date", "")
            symptoms = entry.get("symptoms", [])
            results = entry.get("analysis_results", {})

            # Get symptom names
            symptom_display_names = []
            for symptom_id in symptoms:
                name = symptom_names.get(symptom_id, symptom_id)
                symptom_display_names.append(name)

            # Create expandable section for each entry
            with st.expander(f"Health Check: {date}", expanded=(i == 0)):
                # Display symptoms
                if symptoms:
                    st.markdown("**Symptoms Reported:**")
                    st.write(", ".join(symptom_display_names))

                # Display analysis results
                if results:
                    st.markdown("**Potential Conditions:**")

                    for condition_id, condition_data in results.items():
                        condition_name = condition_data.get("name", condition_id)
                        confidence = condition_data.get("confidence", 0)
                        description = condition_data.get("description", "")

                        st.markdown(f"- **{condition_name}** (Confidence: {confidence}%)")
                        if description:
                            st.markdown(f"  *{description}*")

                # Display recommendations (if any)
                if "recommendations" in entry:
                    st.markdown("**Recommendations:**")
                    for rec in entry["recommendations"]:
                        st.markdown(f"- {rec}")

                # Display risk assessment (if any)
                if "risk_assessment" in entry:
                    risk = entry["risk_assessment"]
                    st.markdown("**Risk Assessment:**")
                    st.markdown(f"- Risk Level: **{risk.get('overall_risk', 'unknown').capitalize()}**")

                    if "risk_factors" in risk and risk["risk_factors"]:
                        st.markdown("- Risk Factors:")
                        for factor in risk["risk_factors"]:
                            st.markdown(f"  - {factor}")


def render_dashboard(health_history: Optional[List[Dict[str, Any]]] = None,
                   user_profile: Optional[Dict[str, Any]] = None) -> None:
    """
    Render the health dashboard interface.

    Args:
        health_history: List of health history entries
        user_profile: User profile data
    """
    # Use data from session state if not provided
    if health_history is None and "user_manager" in st.session_state:
        health_history = st.session_state.user_manager.health_history

    if user_profile is None and "user_manager" in st.session_state:
        user_profile = st.session_state.user_manager.profile

    # Create and render dashboard
    dashboard = HealthDashboard(health_history, user_profile)
    dashboard.render_dashboard()
