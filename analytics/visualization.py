"""
Visualization utilities for MedExplain AI Pro.

This module provides functions for creating interactive visualizations
of health data, including charts, graphs, and dashboards.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logger
logger = logging.getLogger(__name__)

def create_symptom_frequency_chart(symptom_counts: Dict[str, int],
                                 title: str = "Symptom Frequency",
                                 color_scale: str = "Blues") -> go.Figure:
    """
    Create a bar chart of symptom frequency.

    Args:
        symptom_counts: Dictionary mapping symptom names to counts
        title: Chart title
        color_scale: Color scale for the chart

    Returns:
        Plotly figure object
    """
    try:
        # Create dataframe from symptom counts
        df = pd.DataFrame({
            "Symptom": list(symptom_counts.keys()),
            "Count": list(symptom_counts.values())
        })

        # Sort by count descending
        df = df.sort_values("Count", ascending=False)

        # Create bar chart
        fig = px.bar(
            df,
            x="Symptom",
            y="Count",
            title=title,
            color="Count",
            color_continuous_scale=color_scale
        )

        # Update layout
        fig.update_layout(
            xaxis_title="Symptom",
            yaxis_title="Number of Occurrences",
            height=500
        )

        logger.debug("Created symptom frequency chart with %d symptoms", len(symptom_counts))
        return fig
    except Exception as e:
        logger.error("Error creating symptom frequency chart: %s", str(e))
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_symptom_timeline_chart(health_history: List[Dict[str, Any]],
                                symptom_names: Dict[str, str] = None) -> go.Figure:
    """
    Create a timeline chart of symptoms over time.

    Args:
        health_history: List of health history entries
        symptom_names: Optional dictionary mapping symptom IDs to display names

    Returns:
        Plotly figure object
    """
    try:
        # Extract dates and symptoms
        timeline_data = []

        for entry in health_history:
            date_str = entry.get("date", "")
            if not date_str:
                continue

            try:
                # Parse date
                if isinstance(date_str, str):
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:
                    date = date_str

                # Add each symptom as a row
                for symptom_id in entry.get("symptoms", []):
                    # Get display name if available
                    symptom_name = symptom_names.get(symptom_id, symptom_id) if symptom_names else symptom_id

                    timeline_data.append({
                        "date": date,
                        "symptom": symptom_name,
                        "present": 1
                    })
            except Exception as e:
                logger.error("Error processing entry date: %s", str(e))
                continue

        # Create dataframe
        if not timeline_data:
            raise ValueError("No timeline data available")

        df = pd.DataFrame(timeline_data)

        # Create scatter plot
        fig = px.scatter(
            df,
            x="date",
            y="symptom",
            size="present",
            size_max=10,
            color="symptom",
            title="Symptoms Over Time"
        )

        # Add lines connecting the same symptoms
        for symptom in df["symptom"].unique():
            symptom_df = df[df["symptom"] == symptom].sort_values("date")
            fig.add_trace(
                go.Scatter(
                    x=symptom_df["date"],
                    y=[symptom] * len(symptom_df),
                    mode="lines",
                    line=dict(width=1),
                    showlegend=False,
                    opacity=0.5
                )
            )

        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Symptom",
            height=600,
            showlegend=True
        )

        logger.debug("Created symptom timeline chart with %d data points", len(timeline_data))
        return fig
    except Exception as e:
        logger.error("Error creating symptom timeline chart: %s", str(e))
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_correlation_heatmap(correlation_data: Dict[str, float],
                             symptom_names: Dict[str, str] = None) -> go.Figure:
    """
    Create a heatmap of symptom correlations.

    Args:
        correlation_data: Dictionary mapping paired symptom IDs to correlation values
        symptom_names: Optional dictionary mapping symptom IDs to display names

    Returns:
        Plotly figure object
    """
    try:
        # Extract unique symptoms
        unique_symptoms = set()
        for pair in correlation_data.keys():
            s1, s2 = pair.split("_")
            unique_symptoms.add(s1)
            unique_symptoms.add(s2)

        # Convert to list and sort
        symptoms = sorted(list(unique_symptoms))

        # Create correlation matrix
        n = len(symptoms)
        corr_matrix = np.zeros((n, n))

        # Map symptoms to indices
        symptom_to_idx = {symptom: i for i, symptom in enumerate(symptoms)}

        # Fill correlation matrix
        for pair, corr in correlation_data.items():
            s1, s2 = pair.split("_")
            i, j = symptom_to_idx[s1], symptom_to_idx[s2]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # Mirror

        # Set diagonal to 1
        np.fill_diagonal(corr_matrix, 1.0)

        # Get display names if available
        labels = [symptom_names.get(s, s) if symptom_names else s for s in symptoms]

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            x=labels,
            y=labels,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Symptom Correlation Heatmap"
        )

        # Update layout
        fig.update_layout(
            height=600,
            width=700
        )

        logger.debug("Created correlation heatmap with %d symptoms", n)
        return fig
    except Exception as e:
        logger.error("Error creating correlation heatmap: %s", str(e))
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_risk_assessment_chart(risk_assessment: Dict[str, Any]) -> go.Figure:
    """
    Create a visualization of risk assessment data.

    Args:
        risk_assessment: Risk assessment data

    Returns:
        Plotly figure object
    """
    try:
        # Extract risk factors and protective factors
        risk_factors = risk_assessment.get("risk_factors", [])
        protective_factors = risk_assessment.get("protective_factors", [])
        overall_risk = risk_assessment.get("overall_risk", "unknown")
        risk_score = risk_assessment.get("risk_score", 0)

        # Create gauge chart for overall risk
        risk_level_map = {
            "low": {"color": "green", "value": 25},
            "moderate": {"color": "orange", "value": 50},
            "high": {"color": "red", "value": 75},
            "unknown": {"color": "gray", "value": 50}
        }

        # Get value and color for current risk level
        risk_info = risk_level_map.get(overall_risk, risk_level_map["unknown"])

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "xy"}],
                   [{"type": "table", "colspan": 2}, None]],
            row_heights=[0.6, 0.4],
            column_widths=[0.4, 0.6],
            subplot_titles=["Overall Risk Level", "Risk Factors vs. Protective Factors",
                           "Risk Assessment Details"]
        )

        # Add gauge chart for overall risk
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_info["value"],
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_info["color"]},
                    "steps": [
                        {"range": [0, 33], "color": "lightgreen"},
                        {"range": [33, 66], "color": "lightyellow"},
                        {"range": [66, 100], "color": "lightcoral"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": risk_info["value"]
                    }
                },
                title={"text": f"Risk Level: {overall_risk.capitalize()}"}
            ),
            row=1, col=1
        )

        # Add bar chart for risk vs. protective factors
        factors_data = [
            {"category": "Risk Factors", "count": len(risk_factors)},
            {"category": "Protective Factors", "count": len(protective_factors)}
        ]

        fig.add_trace(
            go.Bar(
                x=[d["category"] for d in factors_data],
                y=[d["count"] for d in factors_data],
                marker_color=["red", "green"]
            ),
            row=1, col=2
        )

        # Create table for risk factors and protective factors
        table_data = []

        # Add risk factors
        for factor in risk_factors:
            table_data.append(["Risk Factor", factor])

        # Add protective factors
        for factor in protective_factors:
            table_data.append(["Protective Factor", factor])

        # Add table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Type", "Description"],
                    fill_color="lightgrey",
                    align="left"
                ),
                cells=dict(
                    values=list(zip(*table_data)) if table_data else [[], []],
                    fill_color=[["lightcoral" if row[0] == "Risk Factor" else "lightgreen"
                               for row in table_data]] if table_data else None,
                    align="left"
                )
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=900,
            title_text="Health Risk Assessment",
            showlegend=False
        )

        logger.debug("Created risk assessment chart")
        return fig
    except Exception as e:
        logger.error("Error creating risk assessment chart: %s", str(e))
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_monthly_trend_chart(health_history: List[Dict[str, Any]],
                              symptom_names: Dict[str, str] = None) -> go.Figure:
    """
    Create a chart showing monthly trends in symptoms.

    Args:
        health_history: List of health history entries
        symptom_names: Optional dictionary mapping symptom IDs to display names

    Returns:
        Plotly figure object
    """
    try:
        # Extract dates and symptoms
        monthly_data = []

        for entry in health_history:
            date_str = entry.get("date", "")
            if not date_str:
                continue

            try:
                # Parse date
                if isinstance(date_str, str):
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:
                    date = date_str

                # Extract year and month
                year_month = date.strftime("%Y-%m")

                # Add each symptom
                for symptom_id in entry.get("symptoms", []):
                    # Get display name if available
                    symptom_name = symptom_names.get(symptom_id, symptom_id) if symptom_names else symptom_id

                    monthly_data.append({
                        "year_month": year_month,
                        "month": date.strftime("%b %Y"),  # Abbreviated month name and year
                        "symptom": symptom_name,
                        "count": 1
                    })
            except Exception as e:
                logger.error("Error processing entry date: %s", str(e))
                continue

        # Create dataframe
        if not monthly_data:
            raise ValueError("No monthly data available")

        df = pd.DataFrame(monthly_data)

        # Group by month and symptom
        monthly_counts = df.groupby(["month", "symptom"])["count"].sum().reset_index()

        # Pivot data for heatmap
        pivot_df = monthly_counts.pivot(index="symptom", columns="month", values="count")

        # Fill NaN with 0
        pivot_df = pivot_df.fillna(0)

        # Create heatmap
        fig = px.imshow(
            pivot_df,
            title="Monthly Symptom Patterns",
            color_continuous_scale="Blues",
            labels=dict(x="Month", y="Symptom", color="Count")
        )

        # Update layout
        fig.update_layout(
            height=600,
            width=800,
            xaxis=dict(
                tickangle=45,
                title_standoff=25
            )
        )

        logger.debug("Created monthly trend chart")
        return fig
    except Exception as e:
        logger.error("Error creating monthly trend chart: %s", str(e))
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_condition_confidence_chart(health_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a chart showing confidence in identified conditions over time.

    Args:
        health_history: List of health history entries

    Returns:
        Plotly figure object
    """
    try:
        # Extract dates, conditions, and confidence values
        condition_data = []

        for entry in health_history:
            date_str = entry.get("date", "")
            if not date_str or "analysis_results" not in entry:
                continue

            try:
                # Parse date
                if isinstance(date_str, str):
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:
                    date = date_str

                # Add each condition
                for condition_id, condition_info in entry["analysis_results"].items():
                    condition_name = condition_info.get("name", condition_id)
                    confidence = condition_info.get("confidence", 0)

                    condition_data.append({
                        "date": date,
                        "condition": condition_name,
                        "confidence": confidence
                    })
            except Exception as e:
                logger.error("Error processing entry: %s", str(e))
                continue

        # Create dataframe
        if not condition_data:
            raise ValueError("No condition data available")

        df = pd.DataFrame(condition_data)

        # Create line chart
        fig = px.line(
            df,
            x="date",
            y="confidence",
            color="condition",
            markers=True,
            title="Condition Confidence Over Time",
            labels={"confidence": "Confidence (%)", "date": "Date", "condition": "Condition"}
        )

        # Update layout
        fig.update_layout(
            height=600,
            width=800,
            yaxis=dict(
                range=[0, 100]
            )
        )

        logger.debug("Created condition confidence chart with %d data points", len(condition_data))
        return fig
    except Exception as e:
        logger.error("Error creating condition confidence chart: %s", str(e))
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_symptom_network_graph(symptom_correlations: Dict[str, float],
                               symptom_names: Dict[str, str] = None,
                               min_correlation: float = 0.3) -> go.Figure:
    """
    Create a network graph of symptom relationships.

    Args:
        symptom_correlations: Dictionary mapping paired symptom IDs to correlation values
        symptom_names: Optional dictionary mapping symptom IDs to display names
        min_correlation: Minimum correlation value to include in the graph

    Returns:
        Plotly figure object
    """
    try:
        # Extract nodes (symptoms) and edges (correlations)
        nodes = set()
        edges = []

        for pair, corr in symptom_correlations.items():
            # Skip weak correlations
            if abs(corr) < min_correlation:
                continue

            s1, s2 = pair.split("_")
            nodes.add(s1)
            nodes.add(s2)

            # Add edge with correlation value
            edges.append((s1, s2, abs(corr)))

        # Convert nodes to list and sort
        node_list = sorted(list(nodes))

        # Create node positions using a simple circular layout
        n = len(node_list)
        node_positions = {}

        for i, node in enumerate(node_list):
            angle = 2 * np.pi * i / n
            node_positions[node] = (np.cos(angle), np.sin(angle))

        # Create figure
        fig = go.Figure()

        # Add edges (lines)
        for s1, s2, corr in edges:
            x0, y0 = node_positions[s1]
            x1, y1 = node_positions[s2]

            # Line width based on correlation strength
            width = corr * 5

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=width, color="rgba(150, 150, 150, 0.6)"),
                    hoverinfo="none",
                    showlegend=False
                )
            )

        # Add nodes (points)
        node_x = []
        node_y = []
        node_text = []

        for node in node_list:
            x, y = node_positions[node]
            node_x.append(x)
            node_y.append(y)

            # Get display name if available
            name = symptom_names.get(node, node) if symptom_names else node
            node_text.append(name)

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=15, color="skyblue", line=dict(width=2, color="darkblue")),
                text=node_text,
                textposition="top center",
                hoverinfo="text"
            )
        )

        # Update layout
        fig.update_layout(
            title="Symptom Relationship Network",
            showlegend=False,
            height=700,
            width=700,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        logger.debug("Created symptom network graph with %d nodes and %d edges", len(node_list), len(edges))
        return fig
    except Exception as e:
        logger.error("Error creating symptom network graph: %s", str(e))
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating graph: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
def create_timeseries_chart(data: List[Dict[str, Any]],
                            x_field: str,
                            y_field: str,
                            title: str = "Time Series Analysis",
                            color_field: str = None,
                            line_dash_field: str = None,
                            date_format: str = "%Y-%m-%d %H:%M:%S") -> go.Figure:
    """
    Create a time series chart for analyzing trends over time.

    Args:
        data: List of data points containing time and value fields
        x_field: Field name for time values
        y_field: Field name for metric values
        title: Chart title
        color_field: Optional field name for color grouping
        line_dash_field: Optional field name for line style grouping
        date_format: Format string for parsing date strings

    Returns:
        Plotly figure object
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Convert string dates to datetime if necessary
        if isinstance(df[x_field].iloc[0], str):
            df[x_field] = pd.to_datetime(df[x_field], format=date_format)

        # Create line chart
        if color_field and line_dash_field:
            fig = px.line(
                df,
                x=x_field,
                y=y_field,
                color=color_field,
                line_dash=line_dash_field,
                title=title,
                markers=True
            )
        elif color_field:
            fig = px.line(
                df,
                x=x_field,
                y=y_field,
                color=color_field,
                title=title,
                markers=True
            )
        else:
            fig = px.line(
                df,
                x=x_field,
                y=y_field,
                title=title,
                markers=True
            )

        # Update layout
        fig.update_layout(
            xaxis_title=x_field.capitalize(),
            yaxis_title=y_field.capitalize(),
            height=500,
            width=800,
            hovermode="closest"
        )

        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        logger.debug(f"Created time series chart for {len(data)} data points")
        return fig
    except Exception as e:
        logger.error(f"Error creating time series chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_symptom_heatmap(symptom_data: List[Dict[str, Any]],
                           time_dimension: str = "day",
                           symptom_names: Dict[str, str] = None,
                           title: str = "Symptom Intensity Heatmap") -> go.Figure:
    """
    Create a heatmap visualization showing symptom intensity over time.

    Args:
        symptom_data: List of symptom records with date, symptom ID, and intensity values
        time_dimension: Time grouping dimension ("day", "week", or "month")
        symptom_names: Optional dictionary mapping symptom IDs to display names
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        # Create dataframe from symptom data
        df = pd.DataFrame(symptom_data)

        # Ensure required columns exist
        required_cols = ["date", "symptom_id", "intensity"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Convert string dates to datetime if necessary
        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])

        # Apply symptom name mapping if provided
        if symptom_names:
            df['symptom'] = df['symptom_id'].map(lambda x: symptom_names.get(x, x))
        else:
            df['symptom'] = df['symptom_id']

        # Create time period column based on specified dimension
        if time_dimension == "day":
            df['time_period'] = df['date'].dt.strftime('%Y-%m-%d')
        elif time_dimension == "week":
            df['time_period'] = df['date'].dt.strftime('%Y-W%U')
        elif time_dimension == "month":
            df['time_period'] = df['date'].dt.strftime('%Y-%m')
        else:
            raise ValueError(f"Invalid time dimension: {time_dimension}. Use 'day', 'week', or 'month'.")

        # Group by time period and symptom, taking the average intensity
        grouped_df = df.groupby(['time_period', 'symptom'])['intensity'].mean().reset_index()

        # Pivot the data for the heatmap
        pivot_df = grouped_df.pivot(index='symptom', columns='time_period', values='intensity')

        # Fill NaN with 0
        pivot_df = pivot_df.fillna(0)

        # Sort columns chronologically
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

        # Create heatmap
        fig = px.imshow(
            pivot_df,
            title=title,
            labels=dict(x="Time Period", y="Symptom", color="Intensity"),
            color_continuous_scale="YlOrRd",
            aspect="auto"
        )

        # Customize layout
        fig.update_layout(
            height=600,
            width=900,
            xaxis=dict(
                tickangle=45,
                title_standoff=25
            )
        )

        logger.debug(f"Created symptom heatmap with {len(pivot_df.index)} symptoms across {len(pivot_df.columns)} time periods")
        return fig

    except Exception as e:
        logger.error(f"Error creating symptom heatmap: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
def create_risk_radar():
    return

def create_dashboard(health_history: List[Dict[str, Any]],
                    risk_assessment: Dict[str, Any],
                    symptom_names: Dict[str, str] = None) -> go.Figure:
    """
    Create a comprehensive health dashboard with multiple visualizations.

    Args:
        health_history: List of health history entries
        risk_assessment: Risk assessment data
        symptom_names: Optional dictionary mapping symptom IDs to display names

    Returns:
        Plotly figure object
    """
    try:
        # Check if we have enough data
        if not health_history:
            raise ValueError("No health history data available")

        # Extract symptom frequency data
        symptom_counts = {}
        for entry in health_history:
            for symptom_id in entry.get("symptoms", []):
                if symptom_id not in symptom_counts:
                    symptom_counts[symptom_id] = 0
                symptom_counts[symptom_id] += 1

        # Convert symptom IDs to names if available
        if symptom_names:
            named_counts = {}
            for symptom_id, count in symptom_counts.items():
                name = symptom_names.get(symptom_id, symptom_id)
                named_counts[name] = count
            symptom_counts = named_counts

        # Create dashboard with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=[
                "Overall Health Risk",
                "Top Symptoms",
                "Recent Symptom Timeline",
                "Monthly Symptom Patterns"
            ]
        )

        # 1. Add risk indicator (top left)
        # Extract risk level
        overall_risk = risk_assessment.get("overall_risk", "unknown")
        risk_level_map = {
            "low": {"color": "green", "value": 25},
            "moderate": {"color": "orange", "value": 50},
            "high": {"color": "red", "value": 75},
            "unknown": {"color": "gray", "value": 50}
        }
        risk_info = risk_level_map.get(overall_risk, risk_level_map["unknown"])

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_info["value"],
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_info["color"]},
                    "steps": [
                        {"range": [0, 33], "color": "lightgreen"},
                        {"range": [33, 66], "color": "lightyellow"},
                        {"range": [66, 100], "color": "lightcoral"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": risk_info["value"]
                    }
                },
                title={"text": f"Risk Level: {overall_risk.capitalize()}"}
            ),
            row=1, col=1
        )

        # 2. Add top symptoms bar chart (top right)
        # Sort symptoms by frequency
        sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
        # Take top 5
        top_symptoms = sorted_symptoms[:5] if len(sorted_symptoms) > 5 else sorted_symptoms

        fig.add_trace(
            go.Bar(
                x=[item[0] for item in top_symptoms],
                y=[item[1] for item in top_symptoms],
                marker_color="lightblue"
            ),
            row=1, col=2
        )

        # 3. Add recent timeline (bottom left)
        # Get most recent entries
        recent_entries = health_history[-10:] if len(health_history) > 10 else health_history

        # Prepare timeline data
        timeline_data = []
        for entry in recent_entries:
            date_str = entry.get("date", "")
            if not date_str:
                continue

            try:
                # Parse date
                if isinstance(date_str, str):
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:
                    date = date_str

                # Add symptoms
                for symptom_id in entry.get("symptoms", []):
                    name = symptom_names.get(symptom_id, symptom_id) if symptom_names else symptom_id
                    timeline_data.append({
                        "date": date,
                        "symptom": name
                    })
            except Exception as e:
                logger.error("Error processing entry date: %s", str(e))
                continue

        if timeline_data:
            df = pd.DataFrame(timeline_data)

            # Get unique symptoms
            unique_symptoms = df["symptom"].unique()

            # Create scatter plot for each symptom
            for i, symptom in enumerate(unique_symptoms):
                symptom_df = df[df["symptom"] == symptom]

                fig.add_trace(
                    go.Scatter(
                        x=symptom_df["date"],
                        y=[symptom] * len(symptom_df),
                        mode="markers",
                        name=symptom,
                        marker=dict(size=10)
                    ),
                    row=2, col=1
                )

        # 4. Add monthly heat map (bottom right)
        # Create monthly data
        monthly_data = []
        for entry in health_history:
            date_str = entry.get("date", "")
            if not date_str:
                continue

            try:
                # Parse date
                if isinstance(date_str, str):
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:
                    date = date_str

                # Extract month
                month = date.strftime("%b")

                # Add symptoms
                for symptom_id in entry.get("symptoms", []):
                    name = symptom_names.get(symptom_id, symptom_id) if symptom_names else symptom_id
                    monthly_data.append({
                        "month": month,
                        "symptom": name,
                        "count": 1
                    })
            except Exception as e:
                continue

        if monthly_data:
            df = pd.DataFrame(monthly_data)

            # Group by month and symptom
            monthly_counts = df.groupby(["month", "symptom"])["count"].sum().reset_index()

            # Get top symptoms for heatmap (to avoid overcrowding)
            top_symptoms_list = [item[0] for item in top_symptoms]
            filtered_counts = monthly_counts[monthly_counts["symptom"].isin(top_symptoms_list)]

            # Pivot data for heatmap
            pivot_df = filtered_counts.pivot(index="symptom", columns="month", values="count")

            # Fill NaN with 0
            pivot_df = pivot_df.fillna(0)

            # Add heatmap
            months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            available_months = [m for m in months_order if m in pivot_df.columns]

            if available_months and not pivot_df.empty:
                # Sort columns by month
                pivot_df = pivot_df[available_months]

                # Convert to matrix
                z_data = pivot_df.values

                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=z_data,
                        x=available_months,
                        y=pivot_df.index.tolist(),
                        colorscale="Blues"
                    ),
                    row=2, col=2
                )

        # Update layout for the entire dashboard
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Health Dashboard",
            showlegend=False
        )

        # Update axes
        fig.update_xaxes(title_text="Symptom", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Symptom", row=2, col=1)
        fig.update_yaxes(title_text="Symptom", row=2, col=2)

        logger.debug("Created comprehensive health dashboard")
        return fig
    except Exception as e:
        logger.error("Error creating dashboard: %s", str(e))
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating dashboard: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
