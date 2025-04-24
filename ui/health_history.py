import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple, TypedDict
import logging

class HealthHistoryUI:
    """
    UI component for displaying and managing health history records.
    Provides functionalities for viewing, filtering, and analyzing health records.
    """

    def __init__(self, user_manager=None, health_data=None):
        """
        Initialize the Health History UI component.

        Args:
            user_manager: UserProfileManager instance to access user health records
            health_data: HealthDataManager instance to access symptom information
        """
        self.user_manager = user_manager
        self.health_data = health_data
        self.logger = logging.getLogger("medexplain.health_history")

    def _format_date(self, date_str: str) -> str:
        """Format date string into a more readable format."""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%B %d, %Y")
        except:
            return date_str

    def _get_symptom_name(self, symptom_id: str) -> str:
        """Get the readable name of a symptom from its ID."""
        if not self.health_data:
            return symptom_id

        symptom_info = self.health_data.get_symptom_info(symptom_id)
        if symptom_info:
            return symptom_info.get("name", symptom_id)
        return symptom_id

    def _create_health_record_display(self, record: Dict[str, Any]) -> None:
        """Create a visual display for a single health record."""
        record_date = record.get("date", "Unknown date")
        symptoms = record.get("symptoms", [])
        formatted_date = self._format_date(record_date)

        # Format symptoms
        symptom_names = [self._get_symptom_name(s) for s in symptoms]

        # Get severity if available
        severity = record.get("severity")
        severity_html = ""
        if severity is not None:
            # Determine severity color
            if severity <= 3:
                severity_color = "#2ecc71"  # Green
            elif severity <= 6:
                severity_color = "#f39c12"  # Orange
            else:
                severity_color = "#e74c3c"  # Red

            severity_html = f"""
            <div style="display: flex; align-items: center; margin-top: 10px;">
                <div style="font-size: 0.9rem; color: #7f8c8d; margin-right: 10px;">Severity:</div>
                <div style="background-color: {severity_color};
                            width: 35px;
                            height: 35px;
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: white;
                            font-weight: bold;">
                    {severity}
                </div>
                <div style="width: 100px;
                            height: 8px;
                            background-color: #f1f1f1;
                            border-radius: 4px;
                            margin-left: 10px;
                            overflow: hidden;">
                    <div style="width: {severity*10}%;
                                height: 100%;
                                background-color: {severity_color};">
                    </div>
                </div>
            </div>
            """

        # Get notes if available
        notes = record.get("notes", "")
        notes_html = ""
        if notes:
            notes_html = f"""
            <div style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 5px;">Notes:</div>
                <div style="font-size: 0.95rem; color: #2c3e50;">{notes}</div>
            </div>
            """

        # Create the record card
        st.markdown(f"""
        <div style="background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                    border: 1px solid #eaecef;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #2c3e50; font-size: 1.2rem;">{formatted_date}</h3>
                <div style="background-color: rgba(52, 152, 219, 0.1);
                            border-radius: 15px;
                            padding: 5px 10px;
                            font-size: 0.8rem;
                            color: #2980b9;">
                    {len(symptoms)} symptom{'' if len(symptoms) == 1 else 's'}
                </div>
            </div>

            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px;">
                {' '.join([f'<div style="background-color: rgba(52, 152, 219, 0.1); border-radius: 15px; padding: 5px 10px; font-size: 0.9rem; color: #2c3e50;">{name}</div>' for name in symptom_names])}
            </div>

            {severity_html}
            {notes_html}
        </div>
        """, unsafe_allow_html=True)

    def _create_health_report_section(self) -> None:
        """Create a section for comprehensive health reports if available."""
        health_report = st.session_state.get("health_report")
        if not health_report:
            return

        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 30px; border: 1px solid #e0e0e0;">
            <h3 style="margin-top: 0; color: #2c3e50; font-size: 1.3rem; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-bottom: 15px;">
                Comprehensive Health Report
            </h3>
        """, unsafe_allow_html=True)

        # Parse the health report and display it with improved formatting
        if isinstance(health_report, str):
            # Split the report into sections
            sections = re.split(r'#+\s+', health_report)

            # Process each section
            for section in sections[1:]:  # Skip the first empty section
                # Split into title and content
                parts = section.split('\n', 1)
                if len(parts) < 2:
                    continue

                title = parts[0].strip()
                content = parts[1].strip()

                 # Use a raw string (r-string) to handle backslashes
                markdown_html = f"""
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #2c3e50; font-size: 1.1rem; margin-bottom: 10px;">{title}</h4>
                    <div style="color: #34495e; font-size: 0.95rem; line-height: 1.5;">
                        {content.replace('n', '<br>')}
                    </div>
                </div>
                """
                st.markdown(markdown_html, unsafe_allow_html=True)
        else:
            # If the report is a dictionary or other format, display it as is
            st.write(health_report)

        st.markdown("</div>", unsafe_allow_html=True)

        # Add button to clear the report
        if st.button("Clear Health Report", key="clear_health_report"):
            st.session_state.pop("health_report", None)
            st.rerun()

    def _create_timeline_visualization(self, health_history: List[Dict[str, Any]]) -> None:
        """Create a visual timeline of health records."""
        if not health_history:
            return

        # Sort records by date
        sorted_history = sorted(
            health_history,
            key=lambda entry: entry.get("date", ""),
            reverse=False  # Oldest to newest
        )

        # Extract dates and symptom counts
        dates = []
        symptom_counts = []
        severity_scores = []

        for record in sorted_history:
            date = record.get("date", "")
            if not date:
                continue

            symptoms = record.get("symptoms", [])
            symptom_count = len(symptoms) if isinstance(symptoms, list) else 0

            severity = record.get("severity")

            dates.append(date)
            symptom_counts.append(symptom_count)
            severity_scores.append(severity)

        if not dates:
            return

        # Create interactive time-series visualization
        try:
            # Create a figure with secondary y-axis if severity data exists
            fig = go.Figure()

            # Add symptom count trace
            fig.add_trace(go.Scatter(
                x=dates,
                y=symptom_counts,
                mode='lines+markers',
                name='Symptom Count',
                line=dict(color='royalblue', width=3),
                marker=dict(size=10)
            ))

            # Add severity score trace if available
            valid_severity_data = [s for s in severity_scores if s is not None]
            if valid_severity_data:
                # Filter dates to match valid severity data
                severity_dates = []
                valid_severity_values = []

                for i, severity in enumerate(severity_scores):
                    if severity is not None:
                        severity_dates.append(dates[i])
                        valid_severity_values.append(severity)

                fig.add_trace(go.Scatter(
                    x=severity_dates,
                    y=valid_severity_values,
                    mode='lines+markers',
                    name='Severity',
                    line=dict(color='firebrick', width=3, dash='dot'),
                    marker=dict(size=10, symbol='diamond')
                ))

            # Improve layout
            fig.update_layout(
                title="Health Metrics Over Time",
                xaxis_title="Date",
                yaxis_title="Count / Severity",
                height=450,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            self.logger.error(f"Error creating timeline visualization: {e}")
            st.error("Could not generate the health timeline visualization. Please try again later.")

    def _create_symptom_frequency_chart(self, health_history: List[Dict[str, Any]]) -> None:
        """Create a chart showing frequency of different symptoms."""
        if not health_history:
            return

        # Collect all symptoms
        symptom_counts = {}

        for record in health_history:
            symptoms = record.get("symptoms", [])
            if not isinstance(symptoms, list):
                continue

            for symptom_id in symptoms:
                symptom_name = self._get_symptom_name(symptom_id)
                if symptom_name in symptom_counts:
                    symptom_counts[symptom_name] += 1
                else:
                    symptom_counts[symptom_name] = 1

        if not symptom_counts:
            return

        # Sort symptoms by frequency
        sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
        top_symptoms = sorted_symptoms[:10]  # Get top 10 symptoms

        symptom_names = [item[0] for item in top_symptoms]
        frequencies = [item[1] for item in top_symptoms]

        try:
            # Create horizontal bar chart
            fig = go.Figure()

            # Add bars
            fig.add_trace(go.Bar(
                y=symptom_names,
                x=frequencies,
                orientation='h',
                marker_color='rgba(52, 152, 219, 0.8)',
                text=frequencies,
                textposition='auto'
            ))

            # Improve layout
            fig.update_layout(
                title="Most Frequent Symptoms",
                xaxis_title="Frequency",
                yaxis=dict(
                    title="Symptom",
                    categoryorder='total ascending'  # Sort by frequency
                ),
                height=400,
                margin=dict(l=10, r=10, t=50, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            self.logger.error(f"Error creating symptom frequency chart: {e}")
            st.error("Could not generate the symptom frequency chart. Please try again later.")

    def _create_health_summary(self, health_history: List[Dict[str, Any]]) -> None:
        """Create a summary of health statistics."""
        if not health_history:
            return

        # Calculate basic statistics
        total_records = len(health_history)

        # Get date range
        dates = [record.get("date") for record in health_history if record.get("date")]
        try:
            date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in dates if date]
            if date_objects:
                first_date = min(date_objects)
                last_date = max(date_objects)
                days_tracked = (last_date - first_date).days + 1
            else:
                days_tracked = 0
        except:
            days_tracked = 0

        # Count total symptoms
        total_symptoms = 0
        unique_symptoms = set()
        for record in health_history:
            symptoms = record.get("symptoms", [])
            if isinstance(symptoms, list):
                total_symptoms += len(symptoms)
                unique_symptoms.update(symptoms)

        # Average symptoms per record
        avg_symptoms = total_symptoms / total_records if total_records > 0 else 0

        # Get average severity if available
        severity_values = [record.get("severity") for record in health_history if record.get("severity") is not None]
        avg_severity = sum(severity_values) / len(severity_values) if severity_values else None

        # Display statistics in a modern grid layout
        st.markdown("""
        <h3 style="color: #2c3e50; font-size: 1.3rem; margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px;">
            Health History Summary
        </h3>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style="background-color: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                        border: 1px solid #eaecef;
                        text-align: center;">
                <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 5px;">Total Records</div>
                <div style="font-size: 2rem; font-weight: bold; color: #3498db;">{total_records}</div>
                <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 5px;">health entries</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background-color: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                        border: 1px solid #eaecef;
                        text-align: center;">
                <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 5px;">Tracking Period</div>
                <div style="font-size: 2rem; font-weight: bold; color: #3498db;">{days_tracked}</div>
                <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 5px;">days monitored</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background-color: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                        border: 1px solid #eaecef;
                        text-align: center;">
                <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 5px;">Unique Symptoms</div>
                <div style="font-size: 2rem; font-weight: bold; color: #3498db;">{len(unique_symptoms)}</div>
                <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 5px;">different symptoms</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div style="background-color: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                        border: 1px solid #eaecef;
                        text-align: center;">
                <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 5px;">Avg Symptoms Per Record</div>
                <div style="font-size: 2rem; font-weight: bold; color: #3498db;">{avg_symptoms:.1f}</div>
                <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 5px;">symptoms per entry</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if avg_severity is not None:
                # Determine severity color
                if avg_severity <= 3:
                    severity_color = "#2ecc71"  # Green
                elif avg_severity <= 6:
                    severity_color = "#f39c12"  # Orange
                else:
                    severity_color = "#e74c3c"  # Red

                st.markdown(f"""
                <div style="background-color: white;
                            border-radius: 10px;
                            padding: 20px;
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                            border: 1px solid #eaecef;
                            text-align: center;">
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 5px;">Average Severity</div>
                    <div style="font-size: 2rem; font-weight: bold; color: {severity_color};">{avg_severity:.1f}</div>
                    <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 5px;">out of 10</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: white;
                            border-radius: 10px;
                            padding: 20px;
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                            border: 1px solid #eaecef;
                            text-align: center;">
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 5px;">Average Severity</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #95a5a6;">Not Available</div>
                    <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 5px;">no severity data</div>
                </div>
                """, unsafe_allow_html=True)

    def _export_health_data(self, health_history: List[Dict[str, Any]]) -> None:
        """Create export functionality for health data."""
        if not health_history:
            return

        st.markdown("""
        <h3 style="color: #2c3e50; font-size: 1.3rem; margin: 30px 0 20px 0; border-bottom: 1px solid #eee; padding-bottom: 10px;">
            Export Health Data
        </h3>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "Excel"],
                key="export_format"
            )

        with col2:
            include_details = st.checkbox("Include detailed symptom info", value=True, key="include_details")

        with col3:
            anonymize = st.checkbox("Anonymize data", value=False, key="anonymize_export")

        if st.button("Export Health History", key="export_button"):
            try:
                # Prepare data for export
                export_data = []

                for record in health_history:
                    export_record = record.copy()

                    # Process symptoms to include names if requested
                    if include_details and self.health_data:
                        symptoms = export_record.get("symptoms", [])
                        if isinstance(symptoms, list):
                            symptom_details = []
                            for symptom_id in symptoms:
                                symptom_info = self.health_data.get_symptom_info(symptom_id)
                                if symptom_info:
                                    symptom_details.append({
                                        "id": symptom_id,
                                        "name": symptom_info.get("name", symptom_id),
                                        "category": symptom_info.get("category", "Unknown")
                                    })
                                else:
                                    symptom_details.append({"id": symptom_id, "name": symptom_id})

                            export_record["symptom_details"] = symptom_details

                    # Anonymize if requested
                    if anonymize:
                        if "user_id" in export_record:
                            export_record["user_id"] = f"user_{hash(export_record['user_id']) % 10000}"

                        if "name" in export_record:
                            export_record["name"] = "Anonymous"

                    export_data.append(export_record)

                # Prepare export based on format
                if export_format == "CSV":
                    # For CSV, need to flatten the data
                    flat_data = []
                    for record in export_data:
                        flat_record = record.copy()

                        # Flatten symptom list to comma-separated string
                        if "symptoms" in flat_record:
                            flat_record["symptoms"] = ",".join(flat_record["symptoms"])

                        # Remove nested symptom details
                        if "symptom_details" in flat_record:
                            del flat_record["symptom_details"]

                        flat_data.append(flat_record)

                    # Convert to DataFrame
                    df = pd.DataFrame(flat_data)

                    # Set export data in session state
                    st.session_state.export_data = {
                        "type": "health_history",
                        "format": "csv",
                        "data": df
                    }
                elif export_format == "JSON":
                    # Set export data in session state
                    st.session_state.export_data = {
                        "type": "health_history",
                        "format": "json",
                        "data": export_data
                    }
                else:  # Excel
                    # Convert to DataFrame with some processing
                    flat_data = []
                    for record in export_data:
                        flat_record = record.copy()

                        # Flatten symptom list to comma-separated string
                        if "symptoms" in flat_record:
                            flat_record["symptoms"] = ",".join(flat_record["symptoms"])

                        # Remove nested symptom details for main sheet
                        # (they'll go in a separate sheet)
                        if "symptom_details" in flat_record:
                            del flat_record["symptom_details"]

                        flat_data.append(flat_record)

                    # Create main DataFrame
                    df = pd.DataFrame(flat_data)

                    # Set export data in session state
                    st.session_state.export_data = {
                        "type": "health_history",
                        "format": "excel",
                        "data": df
                    }

                # Mark export as ready
                st.session_state.export_ready = True
                st.success("Export ready! Click the download button below.")
                st.rerun()
            except Exception as e:
                self.logger.error(f"Error exporting health data: {e}")
                st.error("There was an error preparing your export. Please try again.")

    def _filter_health_history(self, health_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply filters to health history."""
        if not health_history:
            return []

        # Add filtering options
        st.markdown("""
        <h3 style="color: #2c3e50; font-size: 1.3rem; margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px;">
            Filter Health Records
        </h3>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        # Extract date range
        all_dates = [datetime.strptime(record["date"], "%Y-%m-%d") for record in health_history if "date" in record]
        if all_dates:
            min_date = min(all_dates).date()
            max_date = max(all_dates).date()
        else:
            min_date = datetime.now().date() - timedelta(days=365)
            max_date = datetime.now().date()

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="filter_start_date"
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="filter_end_date"
            )

        with col3:
            # Collect all symptoms
            all_symptoms = set()
            for record in health_history:
                symptoms = record.get("symptoms", [])
                if isinstance(symptoms, list):
                    all_symptoms.update(symptoms)

            # Create map of symptom IDs to names
            symptom_id_to_name = {}
            for symptom_id in all_symptoms:
                name = self._get_symptom_name(symptom_id)
                symptom_id_to_name[symptom_id] = name

            # Sort symptom names for better UX
            sorted_symptom_ids = sorted(symptom_id_to_name.keys(), key=lambda x: symptom_id_to_name[x])

            # Convert to options for multiselect
            symptom_options = [(symptom_id, symptom_id_to_name[symptom_id]) for symptom_id in sorted_symptom_ids]

            selected_symptom_ids = st.multiselect(
                "Filter by Symptoms",
                options=[s[0] for s in symptom_options],
                format_func=lambda x: next((s[1] for s in symptom_options if s[0] == x), x),
                key="filter_symptoms"
            )

        # Add severity filter
        severity_filter = st.slider(
            "Minimum Severity",
            min_value=0,
            max_value=10,
            value=0,
            key="filter_severity"
        )

        # Apply filters
        filtered_history = []
        for record in health_history:
            # Check date filter
            record_date_str = record.get("date")
            if not record_date_str:
                continue

            try:
                record_date = datetime.strptime(record_date_str, "%Y-%m-%d").date()
                if record_date < start_date or record_date > end_date:
                    continue
            except:
                continue

            # Check symptom filter
            if selected_symptom_ids:
                record_symptoms = record.get("symptoms", [])
                if not any(s in record_symptoms for s in selected_symptom_ids):
                    continue

            # Check severity filter
            if severity_filter > 0:
                record_severity = record.get("severity")
                if record_severity is None or record_severity < severity_filter:
                    continue

            # Record passed all filters
            filtered_history.append(record)

        # Display filter stats
        if len(filtered_history) != len(health_history):
            st.info(f"Showing {len(filtered_history)} of {len(health_history)} health records based on filters.")

        return filtered_history

    def render(self) -> None:
        """Render the health history UI."""
        try:
            st.title("Health History")

            # Check if we have the necessary components
            if not self.user_manager:
                st.error("User management component is not available. Cannot display health history.")
                return

            # Get health history from user manager
            health_history = self.user_manager.health_history if hasattr(self.user_manager, "health_history") else []

            if not health_history:
                st.info("No health records found. Start by adding health data through the Symptom Analyzer.")

                # Show a button to navigate to symptom analyzer
                if st.button("Go to Symptom Analyzer", key="health_history_to_analyzer"):
                    st.session_state.page = "Symptom Analyzer"
                    st.rerun()
                return

            # Show health report if available
            self._create_health_report_section()

            # Create tab-based interface for different views
            main_tabs = st.tabs(["Records", "Analytics", "Export"])

            with main_tabs[0]:  # Records tab
                # Add health summary statistics
                self._create_health_summary(health_history)

                # Add filtering options
                filtered_history = self._filter_health_history(health_history)

                if not filtered_history:
                    st.warning("No records match your filter criteria.")
                    return

                # Sort records by date (newest first)
                sorted_records = sorted(
                    filtered_history,
                    key=lambda record: record.get("date", ""),
                    reverse=True
                )

                # Display the records
                for record in sorted_records:
                    self._create_health_record_display(record)

            with main_tabs[1]:  # Analytics tab
                if len(health_history) < 2:
                    st.info("You need at least 2 health records to see analytics. Add more health data for insights.")
                    return

                # Create visualizations
                st.markdown("""
                <h3 style="color: #2c3e50; font-size: 1.3rem; margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px;">
                    Health Trends Analysis
                </h3>
                """, unsafe_allow_html=True)

                # Create timeline visualization
                self._create_timeline_visualization(health_history)

                # Create symptom frequency chart
                self._create_symptom_frequency_chart(health_history)

                # Add insights if we have the analyzer component
                if "health_analyzer" in st.session_state:
                    analyzer = st.session_state.health_analyzer
                    try:
                        with st.spinner("Generating health insights..."):
                            insights = analyzer.generate_insights(health_history)

                            st.markdown("""
                            <h3 style="color: #2c3e50; font-size: 1.3rem; margin: 30px 0 20px 0; border-bottom: 1px solid #eee; padding-bottom: 10px;">
                                Health Insights
                            </h3>
                            """, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div style="background-color: white;
                                        border-radius: 10px;
                                        padding: 20px;
                                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                                        border: 1px solid #eaecef;">
                                {insights}
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        self.logger.error(f"Error generating health insights: {e}")

                # Add option to generate a comprehensive health report
                risk_assessor = None
                if "risk_assessor" in st.session_state:
                    risk_assessor = st.session_state.risk_assessor

                st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

                if st.button("Generate Comprehensive Health Report", key="gen_health_report"):
                    last_risk_assessment = st.session_state.get("last_risk_assessment")

                    if risk_assessor and last_risk_assessment:
                        try:
                            with st.spinner("Generating comprehensive health report..."):
                                report = risk_assessor.generate_health_report(last_risk_assessment)
                                st.session_state.health_report = report
                                st.success("Health report generated successfully. View it in the Records tab.")
                                st.rerun()
                        except Exception as e:
                            self.logger.error(f"Error generating health report: {e}")
                            st.error("Could not generate report. Please try again.")
                    else:
                        st.warning("Risk assessment data is not available. Please complete a symptom analysis first.")

            with main_tabs[2]:  # Export tab
                # Create export functionality
                self._export_health_data(health_history)

        except Exception as e:
            self.logger.error(f"Error rendering health history: {e}")
            st.error("Error displaying health history. Please try again later.")

            # Show error details in debug mode
            if st.session_state.get("advanced_mode", False):
                st.exception(e)
