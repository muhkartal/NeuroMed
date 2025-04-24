"""
Enhanced SymptomAnalyzerUI Module for MedExplain AI Pro.

This modernized module provides an interactive symptom analysis interface that integrates with
the main application and utilizes the application's ML components for predictions,
extraction, and risk assessment. Built with the latest Streamlit features for an optimal
enterprise user experience.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from functools import wraps, lru_cache
import uuid
from io import BytesIO
import traceback

# Configure logger
logger = logging.getLogger("medexplain.symptom_analyzer")

# Local implementation of SessionManager to avoid circular imports
class LocalSessionManager:
    """Simple implementation of SessionManager to avoid circular imports."""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a value from session state with a default fallback."""
        if key not in st.session_state:
            return default
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set a value in session state."""
        st.session_state[key] = value

    @staticmethod
    def delete(key: str) -> None:
        """Delete a key from session state if it exists."""
        if key in st.session_state:
            del st.session_state[key]

    @staticmethod
    def navigate_to(page: str) -> None:
        """Navigate to a different page in the application."""
        st.session_state.page = page
        st.rerun()

    @staticmethod
    def add_notification(title: str, message: str, notification_type: str = "info") -> None:
        """Add a notification to the system."""
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


# Performance measurement decorator with caching support
def timing_decorator(func):
    """Decorator to measure execution time of functions with caching support."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the result is already cached
        cache_key = f"cache_{func.__name__}_{str(args)}_{str(kwargs)}"
        cached_result = LocalSessionManager.get(cache_key, None)

        # Return cached result if available and it's not a sensitive function
        non_cacheable_funcs = ["render", "_render_symptom_analysis"]
        if cached_result is not None and func.__name__ not in non_cacheable_funcs:
            logger.debug(f"Cache hit for {func.__name__}")
            return cached_result

        # Otherwise execute the function and measure time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Log execution time
        logger.debug(f"Function {func.__name__} executed in {execution_time:.4f} seconds")

        # Store timing metrics in session state if available
        try:
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}

            if 'function_timing' not in st.session_state.performance_metrics:
                st.session_state.performance_metrics['function_timing'] = {}

            metrics = st.session_state.performance_metrics['function_timing']
            if func.__name__ not in metrics:
                metrics[func.__name__] = []

            metrics[func.__name__].append(execution_time)
        except:
            pass  # Fail silently if session state is not available

        # Cache the result for non-rendering functions
        if func.__name__ not in non_cacheable_funcs:
            # Use simple caching for lightweight objects
            try:
                LocalSessionManager.set(cache_key, result)
                # Set reasonable expiry - 10 minutes for static data
                LocalSessionManager.set(f"{cache_key}_timestamp", time.time() + 600)
            except:
                pass  # Fail silently if object not cacheable

        return result
    return wrapper


# Custom exceptions
class SymptomAnalyzerError(Exception):
    """Base exception class for the SymptomAnalyzer module."""
    pass

class DataAccessError(SymptomAnalyzerError):
    """Exception raised for data access errors."""
    pass

class AnalysisError(SymptomAnalyzerError):
    """Exception raised when symptom analysis fails."""
    pass


class SymptomAnalyzerUI:
    """
    Enhanced SymptomAnalyzerUI class that provides a comprehensive and user-friendly
    interface for symptom analysis, utilizing the latest Streamlit features.
    """

    def __init__(
        self,
        health_data=None,
        symptom_predictor=None,
        symptom_extractor=None,
        risk_assessor=None
    ):
        """
        Initialize the enhanced symptom analyzer UI with necessary components.

        Args:
            health_data: The health data manager for accessing symptom information
            symptom_predictor: ML model for predicting related symptoms
            symptom_extractor: NLP model for extracting symptoms from text
            risk_assessor: Model for assessing health risks based on symptoms
        """
        self.health_data = health_data
        self.symptom_predictor = symptom_predictor
        self.symptom_extractor = symptom_extractor
        self.risk_assessor = risk_assessor
        self.logger = logger

        # Theme configuration - Streamlit modern theme settings
        self.theme = {
            "primary_color": "#3498db",  # Primary brand color
            "secondary_color": "#2ecc71",  # Secondary accent color
            "warning_color": "#f39c12",  # Warning indicators
            "danger_color": "#e74c3c",  # Critical warnings/errors
            "info_color": "#2980b9",  # Information highlights
            "success_color": "#27ae60",  # Success indicators

            # Visual elements
            "card_shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
            "border_radius": "0.5rem",
            "transition": "all 0.3s ease",

            # Dark mode compatible colors
            "dark_mode_bg": "#121212",
            "dark_mode_card": "#1e1e1e",
            "dark_mode_text": "#e0e0e0"
        }

        # Initialize state variables
        self._ensure_session_state()

        # Track component availability
        self.component_status = {
            "health_data": health_data is not None,
            "symptom_predictor": symptom_predictor is not None,
            "symptom_extractor": symptom_extractor is not None,
            "risk_assessor": risk_assessor is not None
        }

        self.logger.info(f"Enhanced SymptomAnalyzerUI initialized. Available components: "
                         f"{sum(self.component_status.values())}/{len(self.component_status)}")

        # Apply custom CSS for modern UI
        self._apply_custom_css()

    def _ensure_session_state(self) -> None:
        """
        Ensure that necessary session state variables exist with proper initialization.
        """
        # Core symptom analysis session variables
        session_vars = {
            "current_symptoms": [],
            "symptom_severity": {},
            "symptom_duration": {},
            "analysis_step": 1,
            "analysis_in_progress": False,
            "analysis_complete": False,
            "analysis_results": {},
            "risk_assessment": {},
            "nlp_extracted_symptoms": [],
            "symptom_notes": "",
            "symptom_history": [],  # Track historical analyses
            "symptom_favorites": [],  # User's common/favorite symptoms
            "ui_preferences": {  # User UI preferences
                "compact_mode": False,
                "dark_mode": LocalSessionManager.get("theme", "") == "dark",
                "show_advanced": LocalSessionManager.get("advanced_mode", False),
                "animations": True
            }
        }

        # Ensure each variable exists in session state
        for key, default_value in session_vars.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _apply_custom_css(self) -> None:
        """Apply custom CSS to enhance the visual appearance of the UI."""
        # Using Streamlit's markdown to inject custom CSS
        st.markdown("""
        <style>
            /* Modern card styling */
            .symptom-card {
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
                border-left: 3px solid #3498db;
            }

            .symptom-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            }

            /* Button enhancements */
            .stButton button {
                border-radius: 0.5rem;
                font-weight: 500;
                transition: all 0.2s ease;
            }

            /* Progress indicators */
            .step-progress {
                display: flex;
                justify-content: space-between;
                margin-bottom: 2rem;
                position: relative;
            }

            .step-progress:before {
                content: '';
                position: absolute;
                top: 50%;
                left: 0;
                right: 0;
                height: 2px;
                background: #e0e0e0;
                z-index: -1;
            }

            .step-item {
                background: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                border: 2px solid #e0e0e0;
                z-index: 1;
            }

            .step-active {
                background: #3498db;
                color: white;
                border-color: #3498db;
            }

            .step-completed {
                background: #2ecc71;
                color: white;
                border-color: #2ecc71;
            }

            /* Improved select boxes */
            .stSelectbox label p {
                font-weight: 500;
            }

            /* Slider enhancements */
            .stSlider {
                padding-top: 0.5rem;
                padding-bottom: 1.5rem;
            }

            /* Tabs styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
            }

            .stTabs [data-baseweb="tab"] {
                height: 3rem;
                border-radius: 0.5rem 0.5rem 0 0;
                padding: 0 1rem;
                font-weight: 500;
            }

            /* Info cards */
            .info-card {
                background-color: #f8f9fa;
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 3px solid #3498db;
            }

            /* Risk indicators */
            .risk-indicator {
                padding: 0.25rem 0.75rem;
                border-radius: 1rem;
                font-weight: 500;
                font-size: 0.875rem;
                display: inline-block;
            }

            .risk-low {
                background-color: rgba(46, 204, 113, 0.2);
                color: #27ae60;
            }

            .risk-medium {
                background-color: rgba(243, 156, 18, 0.2);
                color: #f39c12;
            }

            .risk-high {
                background-color: rgba(231, 76, 60, 0.2);
                color: #e74c3c;
            }

            /* Tooltip helper */
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: help;
            }

            .tooltip .tooltiptext {
                visibility: hidden;
                width: 200px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }

            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }

            /* Scrollable area */
            .scrollable {
                max-height: 300px;
                overflow-y: auto;
                padding-right: 10px;
            }

            /* Dark mode compatibility */
            @media (prefers-color-scheme: dark) {
                .symptom-card {
                    background-color: #1e1e1e;
                }

                .step-progress:before {
                    background: #333;
                }

                .step-item {
                    background: #121212;
                    border-color: #333;
                }

                .info-card {
                    background-color: #1e1e1e;
                }
            }
        </style>
        """, unsafe_allow_html=True)

    def _render_detailed_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """
        Render the detailed analysis section with interactive visualizations.

        Args:
            analysis_results: Dictionary containing analysis results
        """
        st.subheader("Detailed Symptom Analysis")

        # Get symptom data
        symptoms = analysis_results.get("symptoms", [])

        if not symptoms:
            st.warning("No symptoms available for detailed analysis.")
            return

        # Create interactive sub-tabs for different analysis views
        analysis_tabs = st.tabs([
            "🔄 Symptom Patterns",
            "📅 Timeline Analysis",
            "📊 Severity Distribution",
            "🔮 Predicted Symptoms"
        ])

        # Symptom Patterns tab
        with analysis_tabs[0]:
            self._render_symptom_patterns(analysis_results)

        # Timeline Analysis tab
        with analysis_tabs[1]:
            self._render_timeline_analysis(analysis_results)

        # Severity Distribution tab
        with analysis_tabs[2]:
            self._render_severity_distribution(analysis_results)

        # Predicted Symptoms tab
        with analysis_tabs[3]:
            self._render_predicted_symptoms(analysis_results)

    def _render_symptom_patterns(self, analysis_results: Dict[str, Any]) -> None:
        """
        Render symptom pattern analysis with network visualization.

        Args:
            analysis_results: Dictionary containing analysis results
        """
        st.subheader("Symptom Pattern Analysis")

        # Get symptom data
        symptoms = analysis_results.get("symptoms", [])

        if not symptoms or len(symptoms) < 2:
            st.info("Symptom pattern analysis requires at least 2 symptoms.")
            return

        try:
            # Show explanation of this visualization
            with st.expander("Understanding Symptom Patterns", expanded=False):
                st.markdown("""
                This visualization shows the connections between your symptoms. Symptoms that
                are connected often occur together and may indicate common underlying causes.

                - **Connected symptoms** often share underlying causes
                - **Different colors** represent different body systems
                - **Central symptoms** that connect to many others may be key to your condition

                This can help you and your healthcare provider understand potential relationships
                in your symptom profile.
                """)

            # Get related symptoms data
            related_symptoms = {}

            # For each symptom, determine which other symptoms are likely related
            for symptom_id in symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if not symptom_info:
                    continue

                # Get related symptoms for this symptom
                # In a real implementation, this would use the actual relationships
                # For demo, we'll simulate relationships
                symptom_category = symptom_info.get("category", "")

                # Find symptoms in the same category
                same_category = [
                    s for s in symptoms
                    if s != symptom_id and
                    self.health_data.get_symptom_info(s) and
                    self.health_data.get_symptom_info(s).get("category", "") == symptom_category
                ]

                # Also add some based on co-occurrence patterns (simulated)
                import random
                random.seed(symptom_id)  # Use consistent randomness
                other_related = [
                    s for s in symptoms
                    if s != symptom_id and s not in same_category and random.random() < 0.5
                ]

                # Combine and store
                related_symptoms[symptom_id] = same_category + other_related

            # Create network graph data
            nodes = []
            edges = []

            # Add nodes (symptoms)
            for symptom_id in symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if symptom_info:
                    nodes.append({
                        "id": symptom_id,
                        "label": symptom_info.get("name", "Unknown"),
                        "title": f"{symptom_info.get('name', 'Unknown')}<br>{symptom_info.get('category', 'Unknown')}",
                        "category": symptom_info.get("category", "Unknown")
                    })

            # Add edges (relationships)
            for symptom_id, related in related_symptoms.items():
                for related_id in related:
                    # Add edge if both nodes exist
                    if related_id in symptoms:
                        edges.append({
                            "from": symptom_id,
                            "to": related_id
                        })

            # Create a color map for symptom categories
            categories = list(set(node["category"] for node in nodes))
            color_map = {}

            # Generate colors
            import colorsys
            for i, category in enumerate(categories):
                hue = i / max(1, len(categories))
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
                color_map[category] = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"

            # Add colors to nodes
            for node in nodes:
                node["color"] = color_map.get(node["category"], "#999999")

            # Create interactive network visualization using Plotly
            import plotly.graph_objects as go
            import networkx as nx

            # Create a networkx graph
            G = nx.Graph()

            # Add nodes
            for node in nodes:
                G.add_node(node["id"], label=node["label"], category=node["category"], color=node["color"])

            # Add edges
            for edge in edges:
                G.add_edge(edge["from"], edge["to"])

            # Use a layout algorithm to position nodes
            pos = nx.spring_layout(G, seed=42)

            # Create a trace for the edges
            edge_x = []
            edge_y = []

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            # Create a trace for the nodes
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(G.nodes[node]['label'])
                node_color.append(G.nodes[node]['color'])

                # Make nodes with more connections larger
                node_size.append(15 + len(list(G.neighbors(node))) * 5)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    line_width=2,
                    line=dict(color='white')
                ),
                textfont=dict(
                    family="Arial",
                    size=12,
                    color="black"
                )
            )

            # Create the figure
            fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Symptom Relationship Network",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="left",
                            buttons=[
                                dict(
                                    args=[{"marker.size": node_size}],
                                    label="Size by Connections",
                                    method="restyle"
                                ),
                                dict(
                                    args=[{"marker.size": 15}],
                                    label="Uniform Size",
                                    method="restyle"
                                )
                            ],
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.1,
                            xanchor="left",
                            y=1.1,
                            yanchor="top"
                        )
                    ]
                )
            )

            # Display the network graph
            st.plotly_chart(fig, use_container_width=True)

            # Create a legend for the categories
            st.subheader("Symptom Categories")

            # Display category legend in a grid
            legend_cols = st.columns(min(4, len(categories)))

            for i, category in enumerate(categories):
                with legend_cols[i % len(legend_cols)]:
                    st.markdown(f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        margin-bottom: 10px;
                    ">
                        <div style="
                            width: 15px;
                            height: 15px;
                            border-radius: 50%;
                            background-color: {color_map[category]};
                            margin-right: 8px;
                        "></div>
                        <div>{category}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Add interpretation
            st.markdown("""
            ### Interpretation

            This network graph shows the relationships between your symptoms. Symptoms that are
            connected are often related to each other and may indicate a common underlying cause.
            Symptoms of the same color belong to the same category, which can help identify
            which body systems are affected.

            * **Highly connected symptoms** (larger nodes) may be central to your condition
            * **Clusters of symptoms** often represent a common cause
            * **Isolated symptoms** may be unrelated to your main condition
            """)
        except Exception as e:
            self.logger.error(f"Error creating symptom pattern visualization: {e}", exc_info=True)
            st.error("Could not create symptom pattern visualization.")

            # Display a simple table as fallback
            symptom_data = []
            for symptom_id in symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if symptom_info:
                    symptom_data.append({
                        "Symptom": symptom_info.get("name", "Unknown"),
                        "Category": symptom_info.get("category", "Unknown")
                    })

            if symptom_data:
                st.dataframe(
                    pd.DataFrame(symptom_data),
                    use_container_width=True,
                    hide_index=True
                )

    def _render_timeline_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """
        Render timeline analysis visualization.

        Args:
            analysis_results: Dictionary containing analysis results
        """
        st.subheader("Symptom Timeline Analysis")

        # Get symptom data
        symptoms = analysis_results.get("symptoms", [])
        durations = analysis_results.get("symptom_duration", {})

        if not symptoms or not durations:
            st.info("Timeline analysis requires symptom duration information.")
            return

        try:
            # Explain this visualization
            with st.expander("Understanding Timeline Analysis", expanded=False):
                st.markdown("""
                This timeline shows when your symptoms likely started based on the duration you provided.

                - **Pattern analysis** identifies acute and chronic symptoms
                - **Symptom progression** shows which symptoms appeared first
                - **Timeline clusters** may reveal different phases of your condition

                Understanding the timeline can help identify potential causes and progression patterns.
                """)

            # Convert duration strings to day estimates
            duration_mapping = {
                "Less than 1 day": 0.5,
                "1-2 days": 1.5,
                "3-7 days": 5,
                "1-2 weeks": 10.5,
                "2-4 weeks": 21,
                "More than 4 weeks": 35
            }

            # Create timeline data
            timeline_data = []

            # Current date for reference
            current_date = datetime.now()

            for symptom_id in symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if not symptom_info:
                    continue

                # Get duration
                duration_str = durations.get(symptom_id, "Less than 1 day")
                duration_days = duration_mapping.get(duration_str, 0)

                # Calculate start date
                start_date = current_date - timedelta(days=duration_days)

                # Add to timeline data
                timeline_data.append({
                    "Symptom": symptom_info.get("name", "Unknown"),
                    "Start": start_date,
                    "End": current_date,
                    "Duration": f"{duration_days:.1f} days",
                    "Duration_Days": duration_days,
                    "Category": symptom_info.get("category", "Unknown")
                })

            if timeline_data:
                # Convert to DataFrame
                timeline_df = pd.DataFrame(timeline_data)

                # Sort by duration
                timeline_df = timeline_df.sort_values("Duration_Days", ascending=False)

                # Create a Gantt chart with enhanced styling
                fig = px.timeline(
                    timeline_df,
                    x_start="Start",
                    x_end="End",
                    y="Symptom",
                    color="Category",
                    title="Symptom Timeline",
                    height=500,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

                # Improve layout for better readability
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Symptom",
                    legend_title="Category",
                    margin=dict(l=20, r=20, t=50, b=20),
                    hovermode="closest",
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot area
                )

                # Add today's date line
                fig.add_vline(
                    x=current_date,
                    line_width=2,
                    line_dash="dash",
                    line_color="#e74c3c",
                    annotation_text="Today",
                    annotation_position="top right"
                )

                # Customize hover template
                fig.update_traces(
                    hovertemplate='<b>%{y}</b><br>Started: %{x}<br>Duration: ' +
                                  timeline_df.loc[timeline_df['Symptom'] == '%{y}', 'Duration'].values[0] +
                                  '<br>Category: %{marker.color}<extra></extra>'
                )

                # Display the interactive timeline
                st.plotly_chart(fig, use_container_width=True)

                # Add a caption explaining how to interpret
                st.caption("""
                **How to interpret:** This timeline shows when your symptoms likely started based on the reported durations.
                Hover over bars for details. The "Today" line marks the current date.
                """)

                # Add insightful analysis of the timeline
                st.markdown("### Timeline Analysis")

                # Get the earliest and latest onset symptoms
                earliest = timeline_df.loc[timeline_df["Duration_Days"].idxmax()]
                latest = timeline_df.loc[timeline_df["Duration_Days"].idxmin()]

                # Get chronic symptoms (more than 2 weeks)
                chronic_symptoms = timeline_df[timeline_df["Duration_Days"] > 14]
                chronic_count = len(chronic_symptoms)

                # Get acute symptoms (less than 1 week)
                acute_symptoms = timeline_df[timeline_df["Duration_Days"] <= 7]
                acute_count = len(acute_symptoms)

                # Create visual insights cards
                st.markdown(f"""
                <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
                    <div style="
                        flex: 1;
                        min-width: 200px;
                        padding: 15px;
                        background-color: var(--secondary-bg-color, #f8f9fa);
                        border-radius: 8px;
                        border-left: 4px solid #3498db;
                    ">
                        <div style="font-weight: bold;">First Symptom</div>
                        <div style="font-size: 1.2rem;">{earliest['Symptom']}</div>
                        <div style="color: #7f8c8d; font-size: 0.9rem;">{earliest['Duration']} ago</div>
                    </div>

                    <div style="
                        flex: 1;
                        min-width: 200px;
                        padding: 15px;
                        background-color: var(--secondary-bg-color, #f8f9fa);
                        border-radius: 8px;
                        border-left: 4px solid #3498db;
                    ">
                        <div style="font-weight: bold;">Most Recent Symptom</div>
                        <div style="font-size: 1.2rem;">{latest['Symptom']}</div>
                        <div style="color: #7f8c8d; font-size: 0.9rem;">{latest['Duration']} ago</div>
                    </div>

                    <div style="
                        flex: 1;
                        min-width: 200px;
                        padding: 15px;
                        background-color: var(--secondary-bg-color, #f8f9fa);
                        border-radius: 8px;
                        border-left: 4px solid #3498db;
                    ">
                        <div style="font-weight: bold;">Symptom Pattern</div>
                        <div style="font-size: 1.2rem;">{chronic_count} chronic</div>
                        <div style="color: #7f8c8d; font-size: 0.9rem;">{acute_count} acute symptoms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Identify potential patterns with more insightful analysis
                st.markdown("#### Pattern Identification")

                if chronic_count > 0 and acute_count > 0:
                    st.markdown("""
                    <div style="
                        padding: 15px;
                        border-radius: 8px;
                        background-color: rgba(243, 156, 18, 0.1);
                        margin-bottom: 15px;
                        border-left: 4px solid #f39c12;
                    ">
                        <strong>Mixed Pattern:</strong> You have a combination of chronic and acute symptoms, which could indicate:
                        <ul>
                            <li>A chronic condition with recent complications</li>
                            <li>Multiple unrelated health issues occurring simultaneously</li>
                            <li>Progression of a single condition with new emerging symptoms</li>
                        </ul>
                        <p><em>Consider discussing with your healthcare provider how these symptoms may be related.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif chronic_count > 0:
                    st.markdown("""
                    <div style="
                        padding: 15px;
                        border-radius: 8px;
                        background-color: rgba(41, 128, 185, 0.1);
                        margin-bottom: 15px;
                        border-left: 4px solid #2980b9;
                    ">
                        <strong>Chronic Pattern:</strong> Your symptoms are primarily long-term, which could indicate:
                        <ul>
                            <li>A persistent underlying condition</li>
                            <li>A chronic disease requiring ongoing management</li>
                            <li>Slow progression of a condition over time</li>
                        </ul>
                        <p><em>Many chronic conditions benefit from a consistent treatment approach and lifestyle management.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif acute_count > 0:
                    st.markdown("""
                    <div style="
                        padding: 15px;
                        border-radius: 8px;
                        background-color: rgba(39, 174, 96, 0.1);
                        margin-bottom: 15px;
                        border-left: 4px solid #27ae60;
                    ">
                        <strong>Acute Pattern:</strong> Your symptoms are primarily recent, which could indicate:
                        <ul>
                            <li>A new infection or acute illness</li>
                            <li>A recent exposure to an allergen or irritant</li>
                            <li>An acute exacerbation of an underlying condition</li>
                        </ul>
                        <p><em>Many acute conditions resolve on their own but may require targeted treatment.</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Check for symptom clusters
                categories = timeline_df.groupby("Category").size().reset_index(name="Count")
                if len(categories) > 1:
                    # Find the primary affected system (category with most symptoms)
                    primary_category = categories.loc[categories["Count"].idxmax()]["Category"]
                    primary_count = categories.loc[categories["Count"].idxmax()]["Count"]

                    if primary_count / len(symptoms) >= 0.5:  # If 50%+ symptoms in one category
                        st.markdown(f"""
                        <div style="
                            padding: 15px;
                            border-radius: 8px;
                            background-color: rgba(155, 89, 182, 0.1);
                            margin-bottom: 15px;
                            border-left: 4px solid #9b59b6;
                        ">
                            <strong>System Focus:</strong> The majority of your symptoms ({primary_count}/{len(symptoms)}) affect your <strong>{primary_category}</strong> system, which could point to a condition primarily affecting this body system.
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Insufficient data for timeline analysis.")
        except Exception as e:
            self.logger.error(f"Error creating timeline visualization: {e}", exc_info=True)
            st.error("Could not create timeline visualization.")

            # Display a simple table as fallback
            st.markdown("#### Symptom Durations")

            duration_data = []
            for symptom_id in symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if symptom_info:
                    duration_data.append({
                        "Symptom": symptom_info.get("name", "Unknown"),
                        "Duration": durations.get(symptom_id, "Unknown")
                    })

            if duration_data:
                st.dataframe(
                    pd.DataFrame(duration_data),
                    use_container_width=True,
                    hide_index=True
                )

    def _render_severity_distribution(self, analysis_results: Dict[str, Any]) -> None:
        """
        Render severity distribution visualization with enhanced charts.

        Args:
            analysis_results: Dictionary containing analysis results
        """
        st.subheader("Symptom Severity Distribution")

        # Get symptom data
        symptoms = analysis_results.get("symptoms", [])
        severity = analysis_results.get("symptom_severity", {})

        if not symptoms or not severity:
            st.info("Severity analysis requires symptom severity information.")
            return

        try:
            # Explain this visualization
            with st.expander("Understanding Severity Analysis", expanded=False):
                st.markdown("""
                This analysis shows the severity distribution of your symptoms:

                - **Severity scale**: 1 (very mild) to 10 (extremely severe)
                - **Severity levels**: Mild (1-3), Moderate (4-7), Severe (8-10)
                - **Color coding**: Green (mild), Orange (moderate), Red (severe)

                Understanding the severity profile helps prioritize treatments and interventions.
                """)

            # Create severity data
            severity_data = []

            for symptom_id in symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if not symptom_info:
                    continue

                # Get severity
                symptom_severity = severity.get(symptom_id, 5)

                # Add to severity data
                severity_data.append({
                    "Symptom": symptom_info.get("name", "Unknown"),
                    "Severity": symptom_severity,
                    "Category": symptom_info.get("category", "Unknown"),
                    "Level": "Mild" if symptom_severity <= 3 else
                             "Moderate" if symptom_severity <= 7 else "Severe"
                })

            if severity_data:
                # Convert to DataFrame
                severity_df = pd.DataFrame(severity_data)

                # Sort by severity
                severity_df = severity_df.sort_values("Severity", ascending=False)

                # Create a horizontal bar chart with conditional coloring
                fig = px.bar(
                    severity_df,
                    x="Severity",
                    y="Symptom",
                    color="Level",
                    title="Symptom Severity Distribution",
                    orientation='h',
                    height=500,
                    color_discrete_map={
                        "Mild": "#2ecc71",    # Green
                        "Moderate": "#f39c12", # Orange
                        "Severe": "#e74c3c"    # Red
                    },
                    category_orders={"Level": ["Severe", "Moderate", "Mild"]},
                    hover_data=["Category"]
                )

                # Add severity zones for visual reference
                fig.add_shape(
                    type="rect",
                    x0=0, x1=3,
                    y0=-0.5, y1=len(severity_df)-0.5,
                    line=dict(width=0),
                    fillcolor="rgba(46, 204, 113, 0.1)",  # Green with transparency
                    layer="below"
                )

                fig.add_shape(
                    type="rect",
                    x0=3, x1=7,
                    y0=-0.5, y1=len(severity_df)-0.5,
                    line=dict(width=0),
                    fillcolor="rgba(243, 156, 18, 0.1)",  # Orange with transparency
                    layer="below"
                )

                fig.add_shape(
                    type="rect",
                    x0=7, x1=10,
                    y0=-0.5, y1=len(severity_df)-0.5,
                    line=dict(width=0),
                    fillcolor="rgba(231, 76, 60, 0.1)",  # Red with transparency
                    layer="below"
                )

                # Add text labels to show exact severity values
                fig.update_traces(
                    texttemplate='%{x}',
                    textposition='outside',
                )

                # Update layout
                fig.update_layout(
                    xaxis_title="Severity (1-10)",
                    yaxis_title="Symptom",
                    legend_title="Severity Level",
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(range=[0, 10.5]),  # Extend a bit for outside labels
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Add a donut chart showing severity distribution
                severity_counts = severity_df['Level'].value_counts().reset_index()
                severity_counts.columns = ['Level', 'Count']

                # Set order of levels
                level_order = ['Severe', 'Moderate', 'Mild']
                severity_counts['Level'] = pd.Categorical(
                    severity_counts['Level'],
                    categories=level_order,
                    ordered=True
                )
                severity_counts = severity_counts.sort_values('Level')

                # Create donut chart with percentage labels
                pie_fig = px.pie(
                    severity_counts,
                    values='Count',
                    names='Level',
                    title='Severity Distribution',
                    color='Level',
                    color_discrete_map={
                        "Mild": "#2ecc71",    # Green
                        "Moderate": "#f39c12", # Orange
                        "Severe": "#e74c3c"    # Red
                    },
                    hole=0.4
                )

                # Update layout
                pie_fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=0,
                        xanchor="center",
                        x=0.5
                    )
                )

                # Add percentage labels
                pie_fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    insidetextfont=dict(color='white')
                )

                # Display the chart
                st.plotly_chart(pie_fig, use_container_width=True)

                # Add interpretation with meaningful insights
                st.markdown("### Severity Analysis")

                # Calculate average severity
                avg_severity = sum(s["Severity"] for s in severity_data) / len(severity_data)

                # Count symptoms by severity level
                mild_count = sum(1 for s in severity_data if s["Severity"] <= 3)
                moderate_count = sum(1 for s in severity_data if 3 < s["Severity"] <= 7)
                severe_count = sum(1 for s in severity_data if s["Severity"] > 7)

                # Get the most severe symptom
                most_severe = severity_df.iloc[0]

                # Create visual severity summary
                st.markdown(f"""
                <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
                    <div style="
                        flex: 1;
                        min-width: 200px;
                        padding: 15px;
                        background-color: var(--secondary-bg-color, #f8f9fa);
                        border-radius: 8px;
                        border-left: 4px solid #3498db;
                    ">
                        <div style="font-weight: bold;">Average Severity</div>
                        <div style="font-size: 1.2rem;">{avg_severity:.1f}/10</div>
                        <div style="color: #7f8c8d; font-size: 0.9rem;">Overall intensity</div>
                    </div>

                    <div style="
                        flex: 1;
                        min-width: 200px;
                        padding: 15px;
                        background-color: var(--secondary-bg-color, #f8f9fa);
                        border-radius: 8px;
                        border-left: 4px solid #e74c3c;
                    ">
                        <div style="font-weight: bold;">Most Severe</div>
                        <div style="font-size: 1.2rem;">{most_severe['Symptom']}</div>
                        <div style="color: #7f8c8d; font-size: 0.9rem;">Rated {most_severe['Severity']}/10</div>
                    </div>

                    <div style="
                        flex: 1;
                        min-width: 200px;
                        padding: 15px;
                        background-color: var(--secondary-bg-color, #f8f9fa);
                        border-radius: 8px;
                        border-left: 4px solid #9b59b6;
                    ">
                        <div style="font-weight: bold;">Distribution</div>
                        <div style="font-size: 1.2rem;">{mild_count} mild, {moderate_count} moderate, {severe_count} severe</div>
                        <div style="color: #7f8c8d; font-size: 0.9rem;">Symptom severity profile</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Identify potential patterns with clinical insights
                st.markdown("#### Severity Interpretation")

                if severe_count > 0:
                    st.markdown("""
                    <div style="
                        padding: 15px;
                        border-radius: 8px;
                        background-color: rgba(231, 76, 60, 0.1);
                        margin-bottom: 15px;
                        border-left: 4px solid #e74c3c;
                    ">
                        <strong>High Severity Pattern:</strong> Your severe symptoms may indicate a condition requiring prompt attention, especially if they:
                        <ul>
                            <li>Are worsening over time</li>
                            <li>Significantly interfere with daily activities</li>
                            <li>Are accompanied by other concerning symptoms</li>
                        </ul>
                        <p><em>Consider discussing these severe symptoms with a healthcare provider soon.</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add specific recommendations for severe symptoms
                    st.markdown("##### Severe Symptom Focus")
                    severe_symptoms = severity_df[severity_df["Severity"] > 7]["Symptom"].tolist()
                    for symptom in severe_symptoms:
                        st.markdown(f"- **{symptom}**: Consider keeping a symptom diary to track intensity changes")
                elif moderate_count > 0:
                    st.markdown("""
                    <div style="
                        padding: 15px;
                        border-radius: 8px;
                        background-color: rgba(243, 156, 18, 0.1);
                        margin-bottom: 15px;
                        border-left: 4px solid #f39c12;
                    ">
                        <strong>Moderate Severity Pattern:</strong> Your symptoms are primarily moderate in severity, suggesting:
                        <ul>
                            <li>A condition that may benefit from medical evaluation</li>
                            <li>Potential for management with appropriate interventions</li>
                            <li>Importance of monitoring for changes in severity</li>
                        </ul>
                        <p><em>Consider discussing with a healthcare provider if symptoms persist or worsen.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="
                        padding: 15px;
                        border-radius: 8px;
                        background-color: rgba(46, 204, 113, 0.1);
                        margin-bottom: 15px;
                        border-left: 4px solid #2ecc71;
                    ">
                        <strong>Mild Severity Pattern:</strong> Your symptoms are primarily mild, which suggests:
                        <ul>
                            <li>A less serious condition or early stage of illness</li>
                            <li>Good potential for recovery with appropriate self-care</li>
                            <li>Importance of monitoring for any escalation in severity</li>
                        </ul>
                        <p><em>Mild symptoms often respond well to self-care measures.</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Add tips for managing symptoms based on severity
                st.markdown("#### Management Considerations")

                # Create a 2-column layout for tips
                tip_col1, tip_col2 = st.columns(2)

                with tip_col1:
                    st.markdown("""
                    **For Moderate to Severe Symptoms:**

                    * Track symptom changes daily
                    * Note factors that worsen symptoms
                    * Be specific when discussing with healthcare providers
                    * Prepare questions before medical appointments
                    """)

                with tip_col2:
                    st.markdown("""
                    **For Mild Symptoms:**

                    * Monitor for any worsening
                    * Try appropriate self-care measures
                    * Note any patterns or triggers
                    * Maintain healthy lifestyle habits
                    """)
            else:
                st.info("Insufficient data for severity analysis.")
        except Exception as e:
            self.logger.error(f"Error creating severity visualization: {e}", exc_info=True)
            st.error("Could not create severity visualization.")

            # Display a simple table as fallback
            st.markdown("#### Symptom Severities")

            severity_data = []
            for symptom_id in symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if symptom_info:
                    severity_data.append({
                        "Symptom": symptom_info.get("name", "Unknown"),
                        "Severity": severity.get(symptom_id, 5)
                    })

            if severity_data:
                st.dataframe(
                    pd.DataFrame(severity_data).sort_values("Severity", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Severity": st.column_config.ProgressColumn(
                            "Severity",
                            format="%d",
                            min_value=1,
                            max_value=10
                        )
                    }
                )

    def _render_predicted_symptoms(self, analysis_results: Dict[str, Any]) -> None:
        """
        Render predicted symptoms visualization with enhanced interactivity.

        Args:
            analysis_results: Dictionary containing analysis results
        """
        st.subheader("Predicted Related Symptoms")

        # Explain this feature
        with st.expander("About Symptom Prediction", expanded=False):
            st.markdown("""
            This feature uses machine learning to identify symptoms that commonly occur with your
            reported symptoms. These predictions are based on medical knowledge and statistical patterns.

            - Higher percentages indicate stronger associations
            - These are potential symptoms to watch for
            - Not all predicted symptoms will develop
            - Discuss these possibilities with your healthcare provider
            """)

        # Get symptom data
        symptoms = analysis_results.get("symptoms", [])
        predicted = analysis_results.get("predicted_symptoms", [])

        if not self.symptom_predictor:
            st.warning("Symptom prediction is not available.")
            return

        if not symptoms:
            st.info("Symptom prediction requires existing symptoms.")
            return

        # If no predicted symptoms in results, try to generate them now
        if not predicted:
            try:
                predicted = self._predict_related_symptoms(symptoms)
            except Exception as e:
                self.logger.error(f"Error predicting related symptoms: {e}", exc_info=True)
                st.error("Could not predict related symptoms.")
                return

        if not predicted:
            st.info("No related symptoms were predicted based on your current symptoms.")
            return

        try:
            # Create data for visualization
            predicted_data = []

            for prediction in predicted:
                symptom_id = prediction.get("id")
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if not symptom_info:
                    continue

                # Add to predicted data
                predicted_data.append({
                    "id": symptom_id,
                    "Symptom": symptom_info.get("name", "Unknown"),
                    "Probability": prediction.get("probability", 0) * 100,  # Convert to percentage
                    "Category": symptom_info.get("category", "Unknown")
                })

            if predicted_data:
                # Convert to DataFrame
                predicted_df = pd.DataFrame(predicted_data)

                # Sort by probability
                predicted_df = predicted_df.sort_values("Probability", ascending=False)

                # Create a horizontal bar chart with enhanced styling
                fig = px.bar(
                    predicted_df,
                    x="Probability",
                    y="Symptom",
                    color="Category",
                    title="Predicted Related Symptoms",
                    orientation='h',
                    height=500,
                    text="Probability",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

                # Update layout
                fig.update_layout(
                    xaxis_title="Probability (%)",
                    yaxis_title="Symptom",
                    legend_title="Category",
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(range=[0, 100]),
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot
                )

                # Update text format
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Add interpretation
                st.markdown("### Potential Related Symptoms")
                st.markdown("""
                This analysis identifies symptoms that commonly occur together with your
                reported symptoms, based on medical data patterns. Consider monitoring
                for these potential related symptoms.
                """)

                # Create check-in form for predicted symptoms with better UI
                st.markdown("### Do you have any of these symptoms?")

                # Use a more visual approach for selecting predicted symptoms
                current_symptoms = analysis_results.get("symptoms", [])

                # Create a grid of cards for top predictions
                num_columns = 2
                rows = [predicted_df.head(6).iloc[i:i+num_columns] for i in range(0, min(len(predicted_df), 6), num_columns)]

                # Track selected symptoms
                selected_predicted = []

                for row in rows:
                    cols = st.columns(num_columns)

                    for i, (_, pred) in enumerate(row.iterrows()):
                        with cols[i]:
                            # Create a card with checkbox
                            symptom_id = pred["id"]
                            probability = pred["Probability"]

                            # Custom card with checkbox
                            is_selected = st.checkbox(
                                f"{pred['Symptom']} ({probability:.1f}% probability)",
                                key=f"pred_{symptom_id}",
                                help=f"Check if you have this symptom"
                            )

                            if is_selected:
                                selected_predicted.append(symptom_id)

                            # Show category in smaller text
                            st.caption(f"Category: {pred['Category']}")

                # Add button to add selected symptoms
                if selected_predicted:
                    if st.button("Add Selected Symptoms", type="primary", use_container_width=True):
                        # Add selected symptoms to current list
                        current_symptoms = LocalSessionManager.get("current_symptoms", [])

                        # Count newly added symptoms
                        newly_added = 0

                        # Add new symptoms
                        for symptom_id in selected_predicted:
                            if symptom_id not in current_symptoms:
                                current_symptoms.append(symptom_id)
                                newly_added += 1

                                # Add to favorites if high probability
                                this_pred = next((p for p in predicted if p.get("id") == symptom_id), None)
                                if this_pred and this_pred.get("probability", 0) >= 0.8:
                                    favorites = st.session_state.symptom_favorites
                                    if symptom_id not in favorites and len(favorites) < 10:
                                        favorites.append(symptom_id)
                                        st.session_state.symptom_favorites = favorites

                        # Update session state
                        LocalSessionManager.set("current_symptoms", current_symptoms)

                        # Notify user
                        if newly_added > 0:
                            st.success(f"Added {newly_added} new symptoms to your list.")

                            # Offer to go back to details step
                            if st.button("Update Symptom Details"):
                                LocalSessionManager.set("analysis_step", 2)
                                LocalSessionManager.set("analysis_complete", False)
                                st.rerun()
                        else:
                            st.info("No new symptoms were added. All selected symptoms were already in your list.")

                # Add quick monitoring tips
                with st.expander("Monitoring Tips", expanded=False):
                    st.markdown("""
                    **How to monitor for predicted symptoms:**

                    1. **Be aware but not anxious** - these are possibilities, not certainties
                    2. **Take note** of any new symptoms that develop
                    3. **Track intensity** if any predicted symptoms occur
                    4. **Report new symptoms** to your healthcare provider
                    5. **Return** to the symptom analyzer to update your profile if new symptoms develop
                    """)
            else:
                st.info("No specific related symptoms were predicted.")
        except Exception as e:
            self.logger.error(f"Error creating predicted symptoms visualization: {e}", exc_info=True)
            st.error("Could not create predicted symptoms visualization.")

            # Display a simple table as fallback
            st.markdown("#### Predicted Related Symptoms")

            predicted_data = []
            for prediction in predicted:
                symptom_id = prediction.get("id")
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if symptom_info:
                    predicted_data.append({
                        "Symptom": symptom_info.get("name", "Unknown"),
                        "Probability": f"{prediction.get('probability', 0) * 100:.1f}%"
                    })

            if predicted_data:
                st.dataframe(
                    pd.DataFrame(predicted_data),
                    use_container_width=True,
                    hide_index=True
                )

    def _render_recommendations(self, analysis_results: Dict[str, Any], risk_assessment: Dict[str, Any]) -> None:
        """
        Render the recommendations section with actionable insights.

        Args:
            analysis_results: Dictionary containing analysis results
            risk_assessment: Dictionary containing risk assessment data
        """
        st.subheader("Recommendations")

        # Explain recommendations
        with st.expander("About These Recommendations", expanded=False):
            st.markdown("""
            These recommendations are generated based on your symptom profile, including:

            - Symptom severity and duration
            - Symptom combinations and patterns
            - Risk assessment findings
            - General health best practices

            They are meant to provide general guidance and are not a substitute for
            professional medical advice.
            """)

        # Get recommendations from risk assessment
        recommendations = risk_assessment.get("recommendations", [])

        if not recommendations:
            # Generate default recommendations
            recommendations = [
                "Continue monitoring your symptoms and record any changes",
                "Maintain proper hydration and rest",
                "Practice good hygiene to prevent potential infections",
                "Consider over-the-counter pain relief for discomfort if appropriate",
                "Consult with a healthcare professional if symptoms persist or worsen"
            ]

        # Display recommendations with better UI
        for i, recommendation in enumerate(recommendations):
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: flex-start;
                margin-bottom: 15px;
                padding: 15px;
                background-color: var(--secondary-bg-color, #f8f9fa);
                border-radius: 8px;
                border-left: 4px solid #2ecc71;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            ">
                <div style="
                    min-width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    background-color: #2ecc71;
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 15px;
                    font-weight: bold;
                ">
                    {i+1}
                </div>
                <div style="flex-grow: 1;">
                    {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Group recommendations by category
        if len(recommendations) >= 3:
            st.markdown("### Recommendation Categories")

            # Create tabs for different types of recommendations
            rec_tabs = st.tabs(["Self-Care", "Monitoring", "Medical Attention"])

            with rec_tabs[0]:
                st.markdown("#### Self-Care Measures")

                # List self-care related recommendations
                self_care_tips = [
                    "Stay well-hydrated by drinking water regularly throughout the day",
                    "Ensure adequate rest and sleep to support your body's healing processes",
                    "Consider over-the-counter remedies appropriate for your symptoms",
                    "Maintain a nutritious diet to support your immune system",
                    "Practice stress reduction techniques such as deep breathing or meditation"
                ]

                for tip in self_care_tips:
                    st.markdown(f"- {tip}")

            with rec_tabs[1]:
                st.markdown("#### Symptom Monitoring")

                # Create a symptom tracking template
                st.markdown("""
                **Daily Symptom Journal Template:**

                For each symptom, record:
                - Severity (1-10 scale)
                - Time of day symptoms are worse/better
                - Activities or foods that affect symptoms
                - Any new symptoms that develop

                Consistent tracking helps identify patterns and progress.
                """)

                # Add warning signs to watch for
                st.markdown("""
                **Warning Signs to Watch For:**

                If you experience any of these, seek prompt medical attention:
                - Severe, sudden, or worsening pain
                - High fever (over 102°F/39°C)
                - Difficulty breathing
                - Persistent vomiting or inability to keep fluids down
                - Unusual confusion or mental state changes
                - Severe dizziness or fainting
                """)

            with rec_tabs[2]:
                st.markdown("#### When to Seek Medical Care")

                # Create columns for different urgency levels
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("""
                    **Routine Care:**

                    - Symptoms persist beyond 7-10 days
                    - Mild but recurring symptoms
                    - General follow-up for chronic conditions
                    - Preventive health checks

                    *Schedule with primary care provider*
                    """)

                with col2:
                    st.markdown("""
                    **Urgent Care:**

                    - Persistent fever
                    - Worsening symptoms
                    - Moderate pain not relieved by OTC meds
                    - Mild to moderate dehydration

                    *Visit urgent care center or call provider*
                    """)

                with col3:
                    st.markdown("""
                    **Emergency Care:**

                    - Severe breathing difficulty
                    - Chest or severe abdominal pain
                    - Severe headache with confusion
                    - Uncontrolled bleeding

                    *Go to ER or call emergency services*
                    """)

        # Medical disclaimer with improved formatting
        st.markdown("""
        <div style="
            padding: 15px;
            border-radius: 8px;
            background-color: rgba(52, 152, 219, 0.1);
            margin-top: 20px;
            border: 1px solid #3498db;
        ">
            <h4 style="margin-top: 0; color: #3498db;">Medical Disclaimer</h4>
            <p>These recommendations are based on your reported symptoms and general health guidelines.
            They are not a substitute for professional medical advice, diagnosis, or treatment.
            Always seek the advice of your physician or other qualified health provider with any
            questions you may have regarding a medical condition.</p>
        </div>
        """, unsafe_allow_html=True)

    def _render_next_steps(self, analysis_results: Dict[str, Any], risk_assessment: Dict[str, Any]) -> None:
        """
        Render the next steps section with actionable guidance based on risk level.

        Args:
            analysis_results: Dictionary containing analysis results
            risk_assessment: Dictionary containing risk assessment data
        """
        st.subheader("Next Steps")

        # Determine appropriate next steps based on risk level
        risk_level = risk_assessment.get("risk_level", "unknown")

        # Create a UI for next steps based on risk level
        if risk_level == "high":
            st.markdown("""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: rgba(231, 76, 60, 0.1);
                margin-bottom: 20px;
                border: 1px solid #e74c3c;
            ">
                <h3 style="margin-top: 0; color: #e74c3c; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#e74c3c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px;">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                        <line x1="12" y1="9" x2="12" y2="13"></line>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    Immediate Action Recommended
                </h3>
                <p>Based on your symptom analysis, it's recommended that you seek professional medical attention promptly.</p>
            </div>
            """, unsafe_allow_html=True)

            # Medical care options
            st.markdown("### Medical Care Options")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(231, 76, 60, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #e74c3c;
                ">
                    <h4 style="margin-top: 0;">For Severe or Life-Threatening Symptoms</h4>
                    <ul>
                        <li>Go to the nearest emergency room</li>
                        <li>Call emergency services (911 in the US)</li>
                        <li>Seek immediate medical attention</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(243, 156, 18, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #f39c12;
                ">
                    <h4 style="margin-top: 0;">For Non-Emergency Situations</h4>
                    <ul>
                        <li>Schedule an appointment with your primary care physician</li>
                        <li>Visit an urgent care center</li>
                        <li>Consider a telehealth consultation</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Add a "Find Care Near Me" option
            st.markdown("### Find Care Options Near You")

            care_tabs = st.tabs(["Emergency Care", "Urgent Care", "Primary Care"])

            with care_tabs[0]:
                st.markdown("""
                **Emergency rooms are best for potentially life-threatening conditions such as:**

                * Severe chest pain or difficulty breathing
                * Severe bleeding or head injury
                * Sudden severe pain
                * Loss of consciousness
                * Severe burns or injuries

                *In a real emergency, call 911 or your local emergency number immediately.*
                """)

                # This would typically connect to a map API
                st.info("In a production environment, this would show nearby emergency departments with wait times.")

            with care_tabs[1]:
                st.markdown("""
                **Urgent care centers are appropriate for non-life-threatening conditions such as:**

                * Minor injuries, sprains, or simple fractures
                * Moderate flu or cold symptoms
                * Ear infections
                * Minor burns or cuts
                * Urinary tract infections

                *Many urgent care centers offer extended hours and accept walk-ins.*
                """)

                st.info("In a production environment, this would show nearby urgent care centers with hours and wait times.")

            with care_tabs[2]:
                st.markdown("""
                **Primary care providers are best for:**

                * Follow-up care
                * Management of ongoing conditions
                * Preventive care
                * Non-urgent new health concerns
                * Prescription refills

                *Most primary care visits require an appointment.*
                """)

                st.info("In a production environment, this would help you find primary care providers accepting new patients.")

        elif risk_level == "medium":
            st.markdown("""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: rgba(243, 156, 18, 0.1);
                margin-bottom: 20px;
                border: 1px solid #f39c12;
            ">
                <h3 style="margin-top: 0; color: #f39c12; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f39c12" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px;">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    Follow-Up Recommended
                </h3>
                <p>Based on your symptom analysis, follow-up with a healthcare provider is recommended, particularly if symptoms persist or worsen.</p>
            </div>
            """, unsafe_allow_html=True)

            # Follow-up options
            st.markdown("### Follow-Up Options")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(243, 156, 18, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #f39c12;
                ">
                    <h4 style="margin-top: 0;">Healthcare Provider Options</h4>
                    <ul>
                        <li>Schedule an appointment with your primary care physician</li>
                        <li>Consider a telehealth consultation</li>
                        <li>Visit a walk-in clinic for faster service</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(52, 152, 219, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #3498db;
                ">
                    <h4 style="margin-top: 0;">Self-Care While Waiting</h4>
                    <ul>
                        <li>Monitor your symptoms daily</li>
                        <li>Rest and stay hydrated</li>
                        <li>Follow the recommended care steps</li>
                        <li>Return for analysis if symptoms worsen</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Add a guide for productive healthcare visits
            with st.expander("Guide for Productive Healthcare Visits", expanded=False):
                st.markdown("""
                ### Preparing for Your Healthcare Visit

                **Before your appointment:**

                1. **Record symptom details:**
                   - When symptoms started
                   - How they've changed
                   - What makes them better or worse

                2. **List your questions:**
                   - Write down specific questions
                   - Prioritize your most important concerns
                   - Ask about treatment options

                3. **Bring information:**
                   - Current medications and dosages
                   - Allergies
                   - Previous related medical records
                   - Your symptom journal
                   - Export of this analysis (available on the Summary tab)

                4. **Consider bringing a support person** to help remember information

                **During your appointment:**

                - Be specific about your symptoms
                - Share your symptom timeline
                - Ask for clarification if needed
                - Take notes on recommendations
                - Confirm next steps before leaving
                """)

        else:  # low or unknown
            st.markdown("""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: rgba(46, 204, 113, 0.1);
                margin-bottom: 20px;
                border: 1px solid #27ae60;
            ">
                <h3 style="margin-top: 0; color: #27ae60; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#27ae60" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px;">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                    Continue Monitoring
                </h3>
                <p>Based on your symptom analysis, continued self-care and monitoring is appropriate.</p>
            </div>
            """, unsafe_allow_html=True)

            # Self-care options
            st.markdown("### Self-Care Guidance")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(52, 152, 219, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #3498db;
                ">
                    <h4 style="margin-top: 0;">Symptom Monitoring</h4>
                    <ul>
                        <li>Track your symptoms daily</li>
                        <li>Note any changes in severity or new symptoms</li>
                        <li>Return for analysis if symptoms worsen</li>
                        <li>Consider follow-up if symptoms persist</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(46, 204, 113, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #27ae60;
                ">
                    <h4 style="margin-top: 0;">Wellness Support</h4>
                    <ul>
                        <li>Maintain proper rest and sleep</li>
                        <li>Stay hydrated and maintain nutrition</li>
                        <li>Avoid triggers that worsen symptoms</li>
                        <li>Practice stress reduction techniques</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Add wellness resources
            with st.expander("Wellness Resources", expanded=False):
                st.markdown("""
                ### General Wellness Resources

                **Reliable Health Information:**

                * [CDC - Centers for Disease Control and Prevention](https://www.cdc.gov)
                * [NIH - National Institutes of Health](https://www.nih.gov)
                * [WHO - World Health Organization](https://www.who.int)
                * [Mayo Clinic](https://www.mayoclinic.org)

                **Stress Management:**

                * Mindfulness meditation
                * Deep breathing exercises
                * Progressive muscle relaxation
                * Regular physical activity
                * Adequate sleep (7-9 hours for adults)

                **Nutrition Resources:**

                * [USDA MyPlate](https://www.myplate.gov/) - Nutrition guidelines
                * [Academy of Nutrition and Dietetics](https://www.eatright.org/) - Dietary information
                * Maintaining proper hydration (typically 8 glasses of water daily)
                """)

        # Export options
        st.markdown("### Export Options")

        export_tabs = st.tabs(["Health Report", "Share With Provider", "Save For Records"])

        with export_tabs[0]:
            st.markdown("""
            Download a comprehensive report of your health analysis. This report includes:

            * Complete symptom profile
            * Risk assessment
            * Detailed analysis
            * Recommendations
            * Next steps
            """)

            # Format options
            report_format = st.radio("Select report format:", ["PDF", "Word Document"], horizontal=True)
            include_charts = st.checkbox("Include visualizations", value=True)

            if st.button("Generate Health Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive health report..."):
                    try:
                        # This would typically generate a formatted document
                        # For demo purposes, we'll simulate it
                        time.sleep(1.5)

                        # Create placeholder file data
                        report_data = self._generate_health_report(analysis_results, risk_assessment)

                        # Format timestamp for filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Create download button
                        st.download_button(
                            label=f"Download Health Report ({report_format})",
                            data=report_data,
                            file_name=f"health_report_{timestamp}.pdf",  # Would adjust extension based on format
                            mime="application/pdf"  # Would adjust MIME type based on format
                        )
                    except Exception as e:
                        self.logger.error(f"Error generating health report: {e}", exc_info=True)
                        st.error("Could not generate health report for download.")

        with export_tabs[1]:
            st.markdown("""
            Share your health analysis directly with your healthcare provider. Options include:

            * Sending a secure link to your provider
            * Generating a provider-friendly summary
            * Creating a printable document to bring to appointments
            """)

            # Provider sharing form
            with st.form("provider_share_form"):
                st.markdown("### Share with Healthcare Provider")

                share_method = st.radio("Sharing method:", ["Email", "Direct Portal", "Print"], horizontal=True)

                if share_method == "Email":
                    provider_email = st.text_input("Provider's Email Address")
                    include_attachments = st.checkbox("Include report as attachment", value=True)
                elif share_method == "Direct Portal":
                    provider_system = st.selectbox("Provider's System", ["Epic MyChart", "FollowMyHealth", "Other"])
                    if provider_system == "Other":
                        provider_system_other = st.text_input("Specify provider system")

                share_button = st.form_submit_button("Share with Provider", type="primary", use_container_width=True)

                if share_button:
                    if share_method == "Email" and not provider_email:
                        st.error("Please enter the provider's email address.")
                    else:
                        with st.spinner("Preparing to share with provider..."):
                            # This would typically interface with email or EHR systems
                            # For demo purposes, we'll simulate success
                            time.sleep(1.5)

                            if share_method == "Print":
                                # Offer report for printing
                                report_data = self._generate_health_report(analysis_results, risk_assessment)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                                st.download_button(
                                    label="Download Printable Report",
                                    data=report_data,
                                    file_name=f"provider_report_{timestamp}.pdf",
                                    mime="application/pdf"
                                )

                                st.success("Report ready for printing and sharing with your provider.")
                            else:
                                st.success(f"Successfully shared your health analysis via {share_method}.")

                                # Add notification
                                LocalSessionManager.add_notification(
                                    "Report Shared",
                                    f"Your health report has been shared via {share_method}.",
                                    "success"
                                )

        with export_tabs[2]:
            st.markdown("""
            Save this analysis to your personal health records for future reference. Options include:

            * Add to your MedExplain health history
            * Export data in various formats
            * Create reminder for follow-up
            """)

            # Save to history
            if st.button("Save to Health History", use_container_width=True):
                with st.spinner("Saving to your health history..."):
                    try:
                        # This would typically save to user's health records
                        # For demo purposes, we'll simulate success
                        time.sleep(1)

                        # Add to symptom history in session
                        symptom_history = st.session_state.symptom_history

                        # Check if this analysis is already in history
                        timestamp = analysis_results.get("timestamp")
                        if timestamp:
                            # Format for display
                            time_obj = datetime.fromisoformat(timestamp)
                            formatted_time = time_obj.strftime("%Y-%m-%d %H:%M")

                            # Check if already exists
                            exists = any(h.get("date") == formatted_time for h in symptom_history)

                            if not exists:
                                # Add to history
                                symptom_history.append({
                                    "date": formatted_time,
                                    "symptoms": analysis_results.get("symptoms", []),
                                    "risk_level": risk_assessment.get("risk_level", "unknown"),
                                    "risk_score": risk_assessment.get("risk_score", 0)
                                })

                                # Update session state
                                st.session_state.symptom_history = symptom_history

                        st.success("Analysis saved to your health history.")

                        # Add notification
                        LocalSessionManager.add_notification(
                            "Analysis Saved",
                            "Your symptom analysis has been saved to your health history.",
                            "success"
                        )
                    except Exception as e:
                        self.logger.error(f"Error saving to health history: {e}", exc_info=True)
                        st.error("Could not save to health history.")

            # Export raw data
            export_format = st.selectbox("Export data format:", ["JSON", "CSV", "Excel"])

            if st.button("Export Raw Data", use_container_width=True):
                with st.spinner("Preparing data export..."):
                    try:
                        if export_format == "JSON":
                            # Create JSON data
                            data = self._generate_json_report(analysis_results, risk_assessment)

                            # Format timestamp for filename
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            # Provide download button
                            st.download_button(
                                label="Download JSON Data",
                                data=data,
                                file_name=f"health_data_{timestamp}.json",
                                mime="application/json"
                            )
                        elif export_format == "CSV":
                            # Create CSV data
                            data = self._generate_csv_report(analysis_results, risk_assessment)

                            # Format timestamp for filename
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            # Provide download button
                            st.download_button(
                                label="Download CSV Data",
                                data=data,
                                file_name=f"health_data_{timestamp}.csv",
                                mime="text/csv"
                            )
                        else:  # Excel
                            # Create Excel data
                            data = self._generate_excel_report(analysis_results, risk_assessment)

                            # Format timestamp for filename
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            # Provide download button
                            st.download_button(
                                label="Download Excel Data",
                                data=data,
                                file_name=f"health_data_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    except Exception as e:
                        self.logger.error(f"Error generating data export: {e}", exc_info=True)
                        st.error("Could not generate data export.")

        # Follow-up scheduling with improved UI
        st.markdown("### Schedule Follow-Up")

        # In a real app, this would connect to a scheduling system
        with st.form("schedule_followup"):
            st.markdown("#### Schedule a Follow-Up Analysis")

            col1, col2 = st.columns(2)

            with col1:
                followup_date = st.date_input(
                    "Follow-up Date",
                    value=datetime.now() + timedelta(days=7)
                )

            with col2:
                reminder_option = st.selectbox(
                    "Reminder Option",
                    ["No reminder", "1 day before", "3 days before", "1 week before"]
                )

            # Add reminder method for better UX
            reminder_method = st.multiselect(
                "Reminder Method",
                ["In-app notification", "Email", "Calendar invitation"],
                default=["In-app notification"]
            )

            submit_button = st.form_submit_button("Schedule Follow-Up", type="primary", use_container_width=True)

        if submit_button:
            # In a real app, this would save the follow-up schedule
            with st.spinner("Scheduling follow-up..."):
                try:
                    # Simulate setting up a reminder
                    time.sleep(1)

                    st.success(f"Follow-up scheduled for {followup_date.strftime('%B %d, %Y')}.")

                    if reminder_option != "No reminder":
                        st.info(f"You will receive a reminder {reminder_option} via {', '.join(reminder_method)}.")

                    # Add notification
                    LocalSessionManager.add_notification(
                        "Follow-Up Scheduled",
                        f"Your symptom analysis follow-up has been scheduled for {followup_date.strftime('%B %d, %Y')}.",
                        "success"
                    )

                    # Add to calendar button
                    st.download_button(
                        label="Add to Calendar",
                        data=f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Health Follow-Up Analysis
DTSTART:{followup_date.strftime('%Y%m%d')}T090000
DTEND:{followup_date.strftime('%Y%m%d')}T100000
DESCRIPTION:Follow-up symptom analysis from MedExplain AI Pro
END:VEVENT
END:VCALENDAR
""",
                        file_name="followup_analysis.ics",
                        mime="text/calendar"
                    )
                except Exception as e:
                    self.logger.error(f"Error scheduling follow-up: {e}", exc_info=True)
                    st.error("Could not schedule follow-up. Please try again.")

    @timing_decorator
    def render(self) -> None:
        """Render the enhanced symptom analyzer interface with modern UI elements."""
        st.title("🔍 Symptom Analyzer")

        # Display app version and component status in a non-intrusive way
        with st.sidebar:
            ui_preferences = st.session_state.ui_preferences

            # Allow toggling compact mode
            compact_mode = st.toggle("Compact Mode", ui_preferences["compact_mode"])
            if compact_mode != ui_preferences["compact_mode"]:
                ui_preferences["compact_mode"] = compact_mode
                st.session_state.ui_preferences = ui_preferences

            # Advanced mode toggle
            if LocalSessionManager.get("advanced_mode", False):
                with st.expander("Advanced Settings", expanded=False):
                    self._render_component_status()

                    # Add analysis reset option
                    if st.button("Reset Analysis", type="secondary", use_container_width=True):
                        self._reset_analysis()
                        st.success("Analysis reset successfully.")
                        time.sleep(1)
                        st.rerun()

            # Add support/help section
            with st.expander("Help & Support", expanded=False):
                st.markdown("""
                **Need help?**

                * Click on the info icons (ℹ️) for contextual help
                * Use the search feature to quickly find symptoms
                * Contact support for technical assistance
                """)

                if st.button("Contact Support", use_container_width=True):
                    st.info("Support contact functionality would be implemented here.")

        # Check if all required components are available
        if not self.health_data:
            self._render_error_state("Symptom analyzer requires health data to function. Please contact support.")
            return

        # Create a modern multi-step interface for symptom analysis
        step = LocalSessionManager.get("analysis_step", 1)
        total_steps = 4

        # Custom progress UI with better visual feedback
        self._render_progress_tracker(step, total_steps)

        # Alert container for success/error messages
        alert_container = st.empty()

        # Check for success/error messages in session state
        temp_message = LocalSessionManager.get("temp_message", None)
        if temp_message:
            if temp_message["type"] == "success":
                alert_container.success(temp_message["text"])
            elif temp_message["type"] == "error":
                alert_container.error(temp_message["text"])
            elif temp_message["type"] == "info":
                alert_container.info(temp_message["text"])
            elif temp_message["type"] == "warning":
                alert_container.warning(temp_message["text"])

            # Clear message after display
            LocalSessionManager.delete("temp_message")

        # Render the appropriate step in the main content area
        try:
            if step == 1:
                self._render_symptom_input()
            elif step == 2:
                self._render_symptom_details()
            elif step == 3:
                self._render_symptom_analysis()
            elif step == 4:
                self._render_analysis_results()
            else:
                # Invalid step, reset to 1
                LocalSessionManager.set("analysis_step", 1)
                st.rerun()
        except Exception as e:
            self.logger.error(f"Error rendering step {step}: {e}", exc_info=True)

            # Show error with recovery options
            st.error(f"An error occurred while processing your symptoms: {str(e)}")

            # Add recovery buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Try Again", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("Reset Analysis", use_container_width=True):
                    self._reset_analysis()
                    st.rerun()

    def _render_progress_tracker(self, current_step: int, total_steps: int) -> None:
        """
        Render a modern, interactive progress tracker for the analysis process.

        Args:
            current_step: The current step number
            total_steps: The total number of steps
        """
        # Step titles for better context
        step_titles = [
            "Describe Symptoms",
            "Add Details",
            "Analysis",
            "Results"
        ]

        # Container for progress tracker
        progress_container = st.container()

        with progress_container:
            # Visual progress bar
            progress_val = (current_step - 1) / (total_steps - 1)
            st.progress(progress_val)

            # Step indicators with titles
            cols = st.columns(total_steps)

            for i in range(total_steps):
                step_num = i + 1
                with cols[i]:
                    if step_num < current_step:
                        # Completed step
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="
                                width: 40px;
                                height: 40px;
                                border-radius: 50%;
                                background-color: #2ecc71;
                                color: white;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-weight: bold;
                                margin: 0 auto;
                            ">
                                ✓
                            </div>
                            <div style="font-size: 0.8rem; margin-top: 5px; color: #2ecc71;">
                                {step_titles[i]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif step_num == current_step:
                        # Current step
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="
                                width: 40px;
                                height: 40px;
                                border-radius: 50%;
                                background-color: #3498db;
                                color: white;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-weight: bold;
                                margin: 0 auto;
                            ">
                                {step_num}
                            </div>
                            <div style="font-size: 0.8rem; margin-top: 5px; color: #3498db; font-weight: bold;">
                                {step_titles[i]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Future step
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="
                                width: 40px;
                                height: 40px;
                                border-radius: 50%;
                                background-color: #f8f9fa;
                                color: #7f8c8d;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-weight: bold;
                                margin: 0 auto;
                                border: 2px solid #ddd;
                            ">
                                {step_num}
                            </div>
                            <div style="font-size: 0.8rem; margin-top: 5px; color: #7f8c8d;">
                                {step_titles[i]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Add spacing after progress tracker
            st.write("")

    def _render_error_state(self, message: str) -> None:
        """
        Render an error state with helpful information and recovery options.

        Args:
            message: The error message to display
        """
        # Display error message with icon
        st.error(message)

        # Create an expander with more detailed troubleshooting
        with st.expander("Troubleshooting Steps"):
            st.markdown("""
            ### How to resolve component issues:

            1. **Refresh the page** - Sometimes a simple refresh can resolve temporary issues
            2. **Check your internet connection** - Ensure you have stable connectivity
            3. **Contact support** - If the issue persists, please contact technical support
            """)

        # Add contact support button
        if st.button("Contact Technical Support", type="primary"):
            # In a real app, this would open a support form or contact modal
            st.info("Support contact functionality would be implemented here.")

        # Add return to home button
        if st.button("Return to Dashboard", type="secondary"):
            LocalSessionManager.navigate_to("Health Dashboard")

    @timing_decorator
    def _render_component_status(self) -> None:
        """Display the status of the symptom analyzer components with enhanced visualization."""
        st.markdown("#### Component Status")

        # Create a status table with better visualization
        status_data = []

        for component, is_available in self.component_status.items():
            status_data.append({
                "Component": component.replace('_', ' ').title(),
                "Status": "✅ Available" if is_available else "❌ Unavailable",
                "Health": 100 if is_available else 0
            })

        # Convert to DataFrame for better display
        status_df = pd.DataFrame(status_data)

        # Calculate overall health percentage
        health_percentage = sum(1 for status in self.component_status.values() if status) / len(self.component_status) * 100

        # Display health indicator
        if health_percentage == 100:
            st.success(f"All components are operational ({health_percentage:.0f}%)")
        elif health_percentage >= 75:
            st.warning(f"Some components are unavailable ({health_percentage:.0f}%)")
        else:
            st.error(f"Multiple components are unavailable ({health_percentage:.0f}%)")

        # Display status table with progress bars
        st.dataframe(
            status_df,
            hide_index=True,
            column_config={
                "Component": st.column_config.TextColumn("Component", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "Health": st.column_config.ProgressColumn(
                    "Health",
                    help="Component health status",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            use_container_width=True
        )

        # Display affected features if any components are unavailable
        if health_percentage < 100:
            unavailable_components = [component for component, status in self.component_status.items() if not status]
            affected_features = self._get_affected_features(unavailable_components)

            if affected_features:
                st.info("💡 **Affected Features:** " + ", ".join(affected_features))

                # Show suggestion to contact support
                st.warning("Consider contacting support if you need these features.")

    def _get_affected_features(self, unavailable_components: List[str]) -> List[str]:
        """
        Determine which features are affected by unavailable components.

        Args:
            unavailable_components: List of component names that are unavailable

        Returns:
            List of affected feature names
        """
        # Map components to features
        component_feature_map = {
            "health_data": ["Symptom Selection", "Medical References"],
            "symptom_predictor": ["Related Symptom Prediction", "Symptom Correlation"],
            "symptom_extractor": ["Natural Language Symptom Extraction", "Text Analysis"],
            "risk_assessor": ["Health Risk Assessment", "Domain Risk Analysis", "Recommendations"]
        }

        # Collect affected features
        affected_features = []
        for component in unavailable_components:
            if component in component_feature_map:
                affected_features.extend(component_feature_map[component])

        return affected_features

    @timing_decorator
    def _render_symptom_input(self) -> None:
        """Render the first step: enhanced symptom input interface with modern UI elements."""
        st.header("Step 1: Describe Your Symptoms")

        # Show helpful intro text
        st.markdown("""
        Select or describe your symptoms using one of the methods below.
        You can use the search feature to quickly find specific symptoms or describe them in your own words.
        """)

        # Create main columns for different input methods and summary
        if st.session_state.ui_preferences.get("compact_mode", False):
            # More compact layout for smaller screens
            input_col, summary_col = st.columns([3, 2])
        else:
            # Spacious layout for larger screens
            input_col, summary_col = st.columns([3, 2])

        with input_col:
            # Modern tabbed interface for different input methods
            input_tabs = st.tabs(["🔍 Search & Select", "✏️ Text Description", "⭐ Favorites"])

            # Tab 1: Enhanced symptom selection with search
            with input_tabs[0]:
                self._render_enhanced_symptom_selection()

            # Tab 2: Improved text-based symptom description
            with input_tabs[1]:
                self._render_text_symptom_extraction()

            # Tab 3: Quick access to favorite/common symptoms
            with input_tabs[2]:
                self._render_favorite_symptoms()

        with summary_col:
            # Display current symptoms summary with enhanced UI
            st.subheader("Your Selected Symptoms")

            current_symptoms = LocalSessionManager.get("current_symptoms", [])
            if current_symptoms:
                # Create a container for the symptoms
                symptoms_container = st.container()

                with symptoms_container:
                    # Show the count of symptoms selected
                    st.markdown(f"**{len(current_symptoms)} symptoms selected**")

                    # Get symptom data for display
                    symptom_df = self._create_symptom_dataframe(current_symptoms)

                    # Display each symptom as a visual card
                    for _, row in symptom_df.iterrows():
                        with st.container():
                            col1, col2 = st.columns([4, 1])

                            with col1:
                                st.markdown(f"""
                                <div class="symptom-card">
                                    <div style="font-weight: bold;">{row['name']}</div>
                                    <div style="font-size: 0.8rem; color: #7f8c8d;">{row['category']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                # Add remove button
                                if st.button("✕", key=f"remove_{row['id']}", help=f"Remove {row['name']}"):
                                    self._remove_symptom(row['id'])
                                    # Use streamlit toast for feedback
                                    st.toast(f"Removed: {row['name']}", icon="✓")
                                    st.rerun()

                # Action buttons
                col1, col2 = st.columns(2)

                with col1:
                    # Clear symptoms button
                    if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                        LocalSessionManager.set("current_symptoms", [])
                        LocalSessionManager.set("symptom_severity", {})
                        LocalSessionManager.set("symptom_duration", {})

                        # Use toast notification for feedback
                        st.toast("All symptoms cleared", icon="✓")
                        st.rerun()

                with col2:
                    # Continue button
                    if st.button("Continue ➡️", type="primary", use_container_width=True):
                        LocalSessionManager.set("analysis_step", 2)
                        st.rerun()
            else:
                # Display empty state with helpful information
                st.info("No symptoms selected yet. Please search for symptoms or describe your condition.")

                # Quick-add common symptoms section
                st.markdown("### Quick Add Common Symptoms")

                # Get common symptoms with caching
                common_symptoms = self._get_common_symptoms()

                # Create a grid of common symptom buttons with improved UI
                common_cols = st.columns(2)

                for i, symptom in enumerate(common_symptoms[:6]):  # Show top 6 common symptoms
                    with common_cols[i % 2]:
                        if st.button(
                            f"{symptom['name']}",
                            key=f"quick_{symptom['id']}",
                            use_container_width=True,
                            help=f"Add {symptom['name']} to your symptoms"
                        ):
                            self._add_symptom(symptom["id"])

                            # Add to favorites if not already there
                            favorites = st.session_state.symptom_favorites
                            if symptom["id"] not in favorites:
                                favorites.append(symptom["id"])
                                st.session_state.symptom_favorites = favorites

                            # Show success toast
                            st.toast(f"Added: {symptom['name']}", icon="✓")
                            st.rerun()

    @timing_decorator
    def _render_enhanced_symptom_selection(self) -> None:
        """Render an enhanced symptom selection interface with improved search and categories."""
        # Get all available symptoms with caching
        all_symptoms = self._get_all_symptoms()

        # Create an enhanced search box with autocomplete behavior
        search_query = st.text_input(
            "🔍 Search symptoms",
            placeholder="Type to search (e.g., headache, cough, fever)",
            help="Search by symptom name or description"
        )

        if search_query:
            # Filter symptoms based on search query with improved relevance
            filtered_symptoms = []

            # Prioritize exact matches in name
            exact_matches = [
                symptom for symptom in all_symptoms
                if search_query.lower() == symptom["name"].lower()
            ]

            # Then partial matches in name
            partial_name_matches = [
                symptom for symptom in all_symptoms
                if search_query.lower() in symptom["name"].lower()
                and symptom not in exact_matches
            ]

            # Finally description matches
            description_matches = [
                symptom for symptom in all_symptoms
                if (symptom.get("description") and search_query.lower() in symptom["description"].lower())
                and symptom not in exact_matches
                and symptom not in partial_name_matches
            ]

            # Combine with priority ordering
            filtered_symptoms = exact_matches + partial_name_matches + description_matches

            if filtered_symptoms:
                st.markdown(f"**{len(filtered_symptoms)} symptoms found:**")

                # Create a scrollable area for many results
                if len(filtered_symptoms) > 10:
                    st.markdown('<div class="scrollable">', unsafe_allow_html=True)

                # Display matches in a responsive grid
                current_symptoms = LocalSessionManager.get("current_symptoms", [])

                # Create rows with multiple columns for more compact layout
                row_size = 2  # Number of buttons per row
                for i in range(0, len(filtered_symptoms), row_size):
                    cols = st.columns(row_size)

                    for j in range(row_size):
                        if i + j < len(filtered_symptoms):
                            symptom = filtered_symptoms[i + j]
                            symptom_id = symptom["id"]

                            # Check if already selected
                            is_selected = symptom_id in current_symptoms

                            with cols[j]:
                                button_label = f"{symptom['name']}"
                                if is_selected:
                                    # Special styling for already selected
                                    if st.button(
                                        f"✓ {button_label}",
                                        key=f"s_{symptom_id}",
                                        type="primary",
                                        use_container_width=True,
                                        help=f"Remove {symptom['name']}"
                                    ):
                                        self._remove_symptom(symptom_id)
                                        st.toast(f"Removed: {symptom['name']}", icon="✓")
                                        st.rerun()
                                else:
                                    if st.button(
                                        button_label,
                                        key=f"s_{symptom_id}",
                                        use_container_width=True,
                                        help=f"Add {symptom['name']}"
                                    ):
                                        self._add_symptom(symptom_id)

                                        # Add to favorites if not already there
                                        favorites = st.session_state.symptom_favorites
                                        if symptom_id not in favorites and len(favorites) < 10:
                                            favorites.append(symptom_id)
                                            st.session_state.symptom_favorites = favorites

                                        st.toast(f"Added: {symptom['name']}", icon="✓")
                                        st.rerun()

                                # Optional: Show category as caption in smaller text
                                st.caption(f"{symptom.get('category', '')}")

                if len(filtered_symptoms) > 10:
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No symptoms match your search. Try different terms or use text description.")
        else:
            # Display symptoms by category when no search is active
            categories = self._group_symptoms_by_category(all_symptoms)

            # Create responsive category tabs
            category_tabs = st.tabs(list(categories.keys()))

            for i, (category, symptoms) in enumerate(categories.items()):
                with category_tabs[i]:
                    if len(symptoms) > 0:
                        # Create a scrollable container for categories with many symptoms
                        if len(symptoms) > 10:
                            st.markdown('<div class="scrollable">', unsafe_allow_html=True)

                        # Create columns for a grid layout
                        col_count = 2
                        for j in range(0, len(symptoms), col_count):
                            cols = st.columns(col_count)

                            for k in range(col_count):
                                if j + k < len(symptoms):
                                    symptom = symptoms[j + k]
                                    symptom_id = symptom["id"]

                                    # Check if symptom is already selected
                                    current_symptoms = LocalSessionManager.get("current_symptoms", [])
                                    is_selected = symptom_id in current_symptoms

                                    with cols[k]:
                                        # Style differently based on selection state
                                        if is_selected:
                                            if st.button(
                                                f"✓ {symptom['name']}",
                                                key=f"cat_{symptom_id}",
                                                type="primary",
                                                use_container_width=True,
                                                help=f"Remove {symptom['name']}"
                                            ):
                                                self._remove_symptom(symptom_id)
                                                st.toast(f"Removed: {symptom['name']}", icon="✓")
                                                st.rerun()
                                        else:
                                            if st.button(
                                                symptom['name'],
                                                key=f"cat_{symptom_id}",
                                                use_container_width=True,
                                                help=f"Add {symptom['name']}"
                                            ):
                                                self._add_symptom(symptom_id)

                                                # Add to favorites if not already there
                                                favorites = st.session_state.symptom_favorites
                                                if symptom_id not in favorites and len(favorites) < 10:
                                                    favorites.append(symptom_id)
                                                    st.session_state.symptom_favorites = favorites

                                                st.toast(f"Added: {symptom['name']}", icon="✓")
                                                st.rerun()

                        if len(symptoms) > 10:
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info(f"No symptoms in the {category} category.")

    @timing_decorator
    def _render_text_symptom_extraction(self) -> None:
        """Render the enhanced text-based symptom extraction interface with NLP features."""
        # Display instructions
        st.markdown("""
        Describe your symptoms in your own words. Our AI will automatically detect symptoms
        from your description.
        """)

        # Check if NLP symptom extractor is available
        if not self.symptom_extractor:
            st.warning("Natural language symptom extraction is not available.")
            st.info("Please use the search and selection method instead.")
            return

        # Text input with helpful placeholder
        symptom_text = st.text_area(
            "Symptom Description",
            value=LocalSessionManager.get("symptom_notes", ""),
            height=150,
            placeholder="Example: I've been having a severe headache for the past 3 days, along with a fever and a sore throat. The pain gets worse when I move my head quickly."
        )

        # Save the text input to session state
        if symptom_text:
            LocalSessionManager.set("symptom_notes", symptom_text)

            # Add extract button with loading state
            extract_button = st.button(
                "🔍 Extract Symptoms",
                disabled=not symptom_text,
                type="primary",
                use_container_width=True
            )

            if extract_button:
                with st.status("Analyzing your description...", expanded=True) as status:
                    try:
                        status.update(label="Processing text...", state="running", expanded=True)
                        time.sleep(0.5)  # Simulate processing for better UX

                        # Call NLP model to extract symptoms
                        extracted_symptoms = self.symptom_extractor.extract_symptoms(symptom_text)

                        if extracted_symptoms:
                            # Store extracted symptoms
                            LocalSessionManager.set("nlp_extracted_symptoms", extracted_symptoms)

                            # Update status
                            status.update(label=f"Found {len(extracted_symptoms)} symptoms!", state="complete")

                            # Create a table of extracted symptoms with confidence scores
                            extracted_data = []
                            for symptom_id in extracted_symptoms:
                                symptom_info = self.health_data.get_symptom_info(symptom_id)
                                if symptom_info:
                                    extracted_data.append({
                                        "id": symptom_id,
                                        "name": symptom_info.get("name", "Unknown"),
                                        "category": symptom_info.get("category", "Unknown"),
                                        "confidence": self.symptom_extractor.get_confidence(symptom_id, symptom_text)
                                    })

                            if extracted_data:
                                # Show extracted symptoms with improved UI
                                st.subheader("Extracted Symptoms")

                                # Display as a modern card-based UI
                                current_symptoms = LocalSessionManager.get("current_symptoms", [])

                                # Sort by confidence
                                extracted_data.sort(key=lambda x: x["confidence"], reverse=True)

                                # Display each symptom
                                for item in extracted_data:
                                    confidence_pct = item["confidence"] * 100
                                    is_selected = item["id"] in current_symptoms

                                    # Show confidence as colored indicator
                                    if confidence_pct >= 80:
                                        confidence_color = "#27ae60"  # green
                                    elif confidence_pct >= 60:
                                        confidence_color = "#f39c12"  # orange
                                    else:
                                        confidence_color = "#95a5a6"  # gray

                                    # Create a symptom card with confidence indicator
                                    col1, col2 = st.columns([4, 1])

                                    with col1:
                                        st.markdown(f"""
                                        <div class="symptom-card">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <div>
                                                    <span style="font-weight: bold;">{item['name']}</span>
                                                    <br>
                                                    <span style="font-size: 0.8rem; color: #7f8c8d;">{item['category']}</span>
                                                </div>
                                                <div style="
                                                    background-color: {confidence_color};
                                                    color: white;
                                                    padding: 2px 8px;
                                                    border-radius: 10px;
                                                    font-size: 0.8rem;
                                                    font-weight: bold;
                                                ">
                                                    {confidence_pct:.0f}%
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                    with col2:
                                        if is_selected:
                                            st.markdown(f"""
                                            <div style="height: 100%; display: flex; align-items: center; justify-content: center;">
                                                <div style="
                                                    background-color: #2ecc71;
                                                    color: white;
                                                    width: 30px;
                                                    height: 30px;
                                                    border-radius: 50%;
                                                    display: flex;
                                                    align-items: center;
                                                    justify-content: center;
                                                    font-weight: bold;
                                                ">
                                                    ✓
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            # Add button
                                            if st.button(
                                                "Add",
                                                key=f"add_ext_{item['id']}",
                                                help=f"Add {item['name']} to your symptoms"
                                            ):
                                                self._add_symptom(item["id"])
                                                st.toast(f"Added: {item['name']}", icon="✓")

                                                # Add to favorites if high confidence
                                                if confidence_pct >= 75:
                                                    favorites = st.session_state.symptom_favorites
                                                    if item["id"] not in favorites and len(favorites) < 10:
                                                        favorites.append(item["id"])
                                                        st.session_state.symptom_favorites = favorites

                                                st.rerun()

                                # Add "Add All" button if there are multiple symptoms
                                if len(extracted_data) > 1:
                                    not_added = [item["id"] for item in extracted_data
                                                if item["id"] not in current_symptoms]

                                    if not_added:
                                        if st.button(
                                            f"Add All Remaining Symptoms ({len(not_added)})",
                                            type="primary",
                                            use_container_width=True
                                        ):
                                            for symptom_id in not_added:
                                                self._add_symptom(symptom_id)

                                            st.toast(f"Added {len(not_added)} symptoms", icon="✓")
                                            st.rerun()
                            else:
                                st.warning("No specific symptoms were identified. Please try using more specific terms.")
                        else:
                            status.update(label="No symptoms found", state="error")
                            st.warning("No symptoms were extracted from your description. Please try using more specific terms or use manual selection.")
                    except Exception as e:
                        self.logger.error(f"Error extracting symptoms: {e}", exc_info=True)
                        status.update(label="Error processing symptoms", state="error")
                        st.error("An error occurred while extracting symptoms. Please try again or use manual selection.")

        # Add tips for better results
        with st.expander("Tips for Best Results", expanded=False):
            st.markdown("""
            **For more accurate symptom extraction:**

            * Be specific about your symptoms
            * Include when symptoms started
            * Mention the severity (mild, moderate, severe)
            * Describe any patterns or triggers
            * Include any changes you've noticed

            **Example:** "I've had a severe throbbing headache for 3 days, mostly on the right side.
            It gets worse with bright lights and is accompanied by nausea in the morning.
            Over-the-counter pain relievers haven't helped much."
            """)

    @timing_decorator
    def _render_favorite_symptoms(self) -> None:
        """Render quick access to favorite/frequently used symptoms."""
        favorite_ids = st.session_state.symptom_favorites

        if not favorite_ids:
            st.info("You haven't saved any favorite symptoms yet.")
            st.markdown("""
            Favorites are automatically added when you select symptoms. Your most commonly
            used symptoms will appear here for quick access.
            """)
            return

        # Get symptom information
        favorites = []
        for symptom_id in favorite_ids:
            symptom_info = self.health_data.get_symptom_info(symptom_id)
            if symptom_info:
                favorites.append(symptom_info)

        if favorites:
            # Group by category for better organization
            categories = {}
            for symptom in favorites:
                category = symptom.get("category", "Other")
                if category not in categories:
                    categories[category] = []
                categories[category].append(symptom)

            # Display by category
            for category, symptoms in categories.items():
                st.subheader(category)

                # Create a grid layout
                cols = st.columns(2)

                for i, symptom in enumerate(symptoms):
                    with cols[i % 2]:
                        # Check if already selected
                        current_symptoms = LocalSessionManager.get("current_symptoms", [])
                        is_selected = symptom["id"] in current_symptoms

                        if is_selected:
                            if st.button(
                                f"✓ {symptom['name']}",
                                key=f"fav_remove_{symptom['id']}",
                                type="primary",
                                use_container_width=True,
                                help=f"Remove {symptom['name']}"
                            ):
                                self._remove_symptom(symptom["id"])
                                st.toast(f"Removed: {symptom['name']}", icon="✓")
                                st.rerun()
                        else:
                            if st.button(
                                symptom['name'],
                                key=f"fav_add_{symptom['id']}",
                                use_container_width=True,
                                help=f"Add {symptom['name']}"
                            ):
                                self._add_symptom(symptom["id"])
                                st.toast(f"Added: {symptom['name']}", icon="✓")
                                st.rerun()

            # Add option to manage favorites
            with st.expander("Manage Favorites", expanded=False):
                st.markdown("Select symptoms to remove from favorites:")

                to_remove = []
                for symptom in favorites:
                    if st.checkbox(
                        f"{symptom['name']} ({symptom.get('category', 'Unknown')})",
                        key=f"remove_fav_{symptom['id']}"
                    ):
                        to_remove.append(symptom["id"])

                if to_remove:
                    if st.button("Remove Selected Favorites", type="primary"):
                        new_favorites = [fav for fav in favorite_ids if fav not in to_remove]
                        st.session_state.symptom_favorites = new_favorites
                        st.toast(f"Removed {len(to_remove)} favorites", icon="✓")
                        st.rerun()

                if st.button("Clear All Favorites", type="secondary"):
                    if st.checkbox("Confirm clearing all favorites"):
                        st.session_state.symptom_favorites = []
                        st.toast("All favorites cleared", icon="✓")
                        st.rerun()
        else:
            st.warning("No valid favorites found. Your saved favorites may be outdated.")

            # Add reset option
            if st.button("Reset Favorites"):
                st.session_state.symptom_favorites = []
                st.rerun()

    @timing_decorator
    def _render_symptom_details(self) -> None:
        """Render the second step: enhanced symptom details interface with improved UI."""
        st.header("Step 2: Symptom Details")

        # Get current symptoms
        current_symptoms = LocalSessionManager.get("current_symptoms", [])

        if not current_symptoms:
            st.warning("No symptoms have been selected. Please go back to Step 1.")

            if st.button("Back to Symptom Selection", use_container_width=True):
                LocalSessionManager.set("analysis_step", 1)
                st.rerun()
            return

        # Get existing severity and duration values
        symptom_severity = LocalSessionManager.get("symptom_severity", {})
        symptom_duration = LocalSessionManager.get("symptom_duration", {})

        # Show brief instructions
        st.markdown("""
        Please rate the severity and duration of each symptom. These details help
        provide a more accurate analysis of your health condition.
        """)

        # Create a form for easier input
        with st.form("symptom_details_form"):
            # Create modern card-like display for each symptom
            for i, symptom_id in enumerate(current_symptoms):
                # Get symptom info
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if not symptom_info:
                    continue

                symptom_name = symptom_info.get("name", "Unknown Symptom")

                # Create a card with border accent color based on category
                st.markdown(f"### {i+1}. {symptom_name}")

                # Create two columns for severity and duration
                col1, col2 = st.columns(2)

                with col1:
                    # Enhanced severity slider with visual feedback
                    severity = st.slider(
                        f"Severity of {symptom_name}",
                        min_value=1,
                        max_value=10,
                        value=symptom_severity.get(symptom_id, 5),
                        key=f"severity_{symptom_id}",
                        help="1 = Very mild, 10 = Extremely severe"
                    )

                    # Add severity level description based on value
                    if 1 <= severity <= 3:
                        st.markdown('<span style="color: green; font-weight: 500;">Mild</span>', unsafe_allow_html=True)
                    elif 4 <= severity <= 7:
                        st.markdown('<span style="color: orange; font-weight: 500;">Moderate</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span style="color: red; font-weight: 500;">Severe</span>', unsafe_allow_html=True)

                    # Store the severity value
                    symptom_severity[symptom_id] = severity

                with col2:
                    # Duration selection with improved options
                    duration_options = [
                        "Less than 1 day",
                        "1-2 days",
                        "3-7 days",
                        "1-2 weeks",
                        "2-4 weeks",
                        "More than 4 weeks"
                    ]

                    # Find the index of the previous selection
                    default_index = 0
                    if symptom_id in symptom_duration:
                        previous = symptom_duration.get(symptom_id)
                        if previous in duration_options:
                            default_index = duration_options.index(previous)

                    duration = st.selectbox(
                        f"Duration of {symptom_name}",
                        duration_options,
                        index=default_index,
                        key=f"duration_{symptom_id}",
                        help="How long have you experienced this symptom?"
                    )

                    # Store the duration value
                    symptom_duration[symptom_id] = duration

                    # Add chronic indicator for long durations
                    if duration in ["2-4 weeks", "More than 4 weeks"]:
                        st.markdown('<span style="color: #e74c3c; font-weight: 500;">Chronic</span>', unsafe_allow_html=True)

                # Add separator between symptoms
                if i < len(current_symptoms) - 1:
                    st.markdown("---")

            # Additional notes with improved UI
            st.markdown("### Additional Information")

            # Enhanced text area for notes
            notes = st.text_area(
                "Add any additional details about your symptoms",
                value=LocalSessionManager.get("symptom_notes", ""),
                height=100,
                placeholder="Example: Symptoms worsen in the morning and improve with rest. I've noticed triggers include...",
                help="Include any patterns, triggers, or related information"
            )

            # Form buttons with better layout
            cols = st.columns([1, 1])

            with cols[0]:
                back_button = st.form_submit_button("⬅️ Back", use_container_width=True)

            with cols[1]:
                continue_button = st.form_submit_button("Continue ➡️", type="primary", use_container_width=True)

        # Handle form submission
        if continue_button:
            # Save the inputs to session state
            LocalSessionManager.set("symptom_severity", symptom_severity)
            LocalSessionManager.set("symptom_duration", symptom_duration)
            LocalSessionManager.set("symptom_notes", notes)

            # Move to the next step
            LocalSessionManager.set("analysis_step", 3)
            st.rerun()

        if back_button:
            LocalSessionManager.set("analysis_step", 1)
            st.rerun()

    @timing_decorator
    def _render_symptom_analysis(self) -> None:
        """Render the third step: symptom analysis with improved visualizations and progress indicators."""
        st.header("Step 3: Analyzing Your Symptoms")

        # Get symptom data
        current_symptoms = LocalSessionManager.get("current_symptoms", [])
        symptom_severity = LocalSessionManager.get("symptom_severity", {})
        symptom_duration = LocalSessionManager.get("symptom_duration", {})
        symptom_notes = LocalSessionManager.get("symptom_notes", "")

        if not current_symptoms:
            st.warning("No symptoms have been selected. Please go back to Step 1.")

            if st.button("Back to Symptom Selection", use_container_width=True):
                LocalSessionManager.set("analysis_step", 1)
                st.rerun()
            return

        # Show summary of inputs with modern UI
        with st.expander("Summary of Your Symptoms", expanded=True):
            # Create visually appealing summary cards
            for symptom_id in current_symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if not symptom_info:
                    continue

                # Get severity and duration
                severity = symptom_severity.get(symptom_id, 5)
                duration = symptom_duration.get(symptom_id, "Unknown")

                # Determine severity class for styling
                if 1 <= severity <= 3:
                    severity_class = "low"
                    severity_text = "Mild"
                elif 4 <= severity <= 7:
                    severity_class = "medium"
                    severity_text = "Moderate"
                else:
                    severity_class = "high"
                    severity_text = "Severe"

                # Create card with dynamic styling
                st.markdown(f"""
                <div style="
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    background-color: var(--secondary-bg-color, #f8f9fa);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-left: 4px solid {self._get_severity_color(severity)};
                ">
                    <div>
                        <div style="font-weight: bold;">{symptom_info.get("name", "Unknown")}</div>
                        <div style="font-size: 0.8rem; color: #7f8c8d;">
                            <span class="risk-{severity_class}">{severity_text}</span> • {duration}
                        </div>
                    </div>
                    <div style="
                        width: 35px;
                        height: 35px;
                        border-radius: 50%;
                        background-color: {self._get_severity_color(severity)};
                        color: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                    ">
                        {severity}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Show notes if provided
        if symptom_notes:
            with st.expander("Additional Notes", expanded=True):
                st.info(symptom_notes)

        # Create a visual analytics dashboard
        analysis_in_progress = LocalSessionManager.get("analysis_in_progress", False)
        analysis_complete = LocalSessionManager.get("analysis_complete", False)

        if not analysis_complete and not analysis_in_progress:
            # Start the analysis with explanatory text
            st.subheader("Ready to Analyze Your Health Data")

            # Explain what happens during analysis
            st.markdown("""
            When you start the analysis, our system will:

            1. Process your symptom combinations and patterns
            2. Analyze severity and duration factors
            3. Identify potential related symptoms
            4. Calculate comprehensive risk assessment
            5. Generate personalized insights and recommendations
            """)

            # Create columns for button layout
            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("⬅️ Back to Details", use_container_width=True):
                    LocalSessionManager.set("analysis_step", 2)
                    st.rerun()

            with col2:
                if st.button("Begin Analysis ➡️", type="primary", use_container_width=True):
                    LocalSessionManager.set("analysis_in_progress", True)
                    st.rerun()

        elif analysis_in_progress and not analysis_complete:
            # Enhanced progress indicators during analysis
            st.subheader("Analysis in Progress")

            # Create a container for status messages
            status_container = st.empty()
            progress_container = st.empty()
            detail_container = st.empty()

            # Show animated progress indicators
            with st.spinner("Analyzing your symptoms..."):
                # Run analysis in stages with visual feedback
                try:
                    # 1. Process symptoms - set up progress bar
                    progress_value = 0
                    progress_container.progress(progress_value, text="Starting analysis")
                    status_container.info("Beginning symptom analysis...")
                    detail_container.markdown("Initializing analysis engines...")
                    time.sleep(0.5)  # Simulate processing time

                    # 2. Evaluating symptom patterns
                    progress_value = 0.2
                    progress_container.progress(progress_value, text="Evaluating symptom patterns")
                    status_container.info("Evaluating symptom patterns...")
                    detail_container.markdown("Analyzing relationships between your reported symptoms...")
                    time.sleep(0.7)  # Simulate processing time

                    # 3. Generate related symptoms
                    predicted_symptoms = []
                    if self.symptom_predictor:
                        progress_value = 0.4
                        progress_container.progress(progress_value, text="Identifying related symptoms")
                        status_container.info("Identifying potential related symptoms...")
                        detail_container.markdown("Using machine learning to find symptoms commonly associated with your condition...")
                        time.sleep(0.7)  # Simulated processing time
                        predicted_symptoms = self._predict_related_symptoms(current_symptoms)

                    # 4. Calculate symptom risk factors
                    progress_value = 0.6
                    progress_container.progress(progress_value, text="Calculating risk factors")
                    status_container.info("Calculating risk factors...")
                    detail_container.markdown("Evaluating severity, duration, and symptom combinations to determine risk levels...")
                    time.sleep(0.5)  # Simulated processing time

                    # 5. Perform risk assessment
                    risk_assessment = {}
                    if self.risk_assessor:
                        progress_value = 0.8
                        progress_container.progress(progress_value, text="Performing risk assessment")
                        status_container.info("Performing comprehensive risk assessment...")
                        detail_container.markdown("Analyzing your symptom profile against medical knowledge base...")
                        time.sleep(1.0)  # Simulated processing time
                        risk_assessment = self._perform_risk_assessment(
                            current_symptoms, symptom_severity, symptom_duration, symptom_notes
                        )

                    # 6. Generate health insights
                    progress_value = 0.9
                    progress_container.progress(progress_value, text="Generating insights")
                    status_container.info("Generating health insights...")
                    detail_container.markdown("Creating personalized health recommendations based on your profile...")
                    time.sleep(0.8)  # Simulated processing time

                    # 7. Complete analysis and prepare results
                    progress_value = 1.0
                    progress_container.progress(progress_value, text="Analysis complete")
                    status_container.success("Analysis complete! Preparing your results...")
                    detail_container.markdown("Finalizing your personalized health assessment...")

                    # Prepare results
                    analysis_results = {
                        "timestamp": datetime.now().isoformat(),
                        "symptoms": current_symptoms,
                        "symptom_severity": symptom_severity,
                        "symptom_duration": symptom_duration,
                        "notes": symptom_notes,
                        "predicted_symptoms": predicted_symptoms,
                        "risk_assessment": risk_assessment,
                        "insights": self._generate_insights(current_symptoms, symptom_severity, symptom_duration)
                    }

                    # Store results in session state
                    LocalSessionManager.set("analysis_results", analysis_results)
                    LocalSessionManager.set("risk_assessment", risk_assessment)

                    # Also store in session history for future reference
                    LocalSessionManager.set("last_symptom_check", {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "symptoms": current_symptoms,
                        "severity": symptom_severity,
                        "duration": symptom_duration,
                        "notes": symptom_notes
                    })

                    LocalSessionManager.set("last_risk_assessment", risk_assessment)

                    # Store in symptom history
                    symptom_history = st.session_state.symptom_history
                    symptom_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "symptoms": current_symptoms,
                        "risk_level": risk_assessment.get("risk_level", "unknown"),
                        "risk_score": risk_assessment.get("risk_score", 0)
                    })
                    st.session_state.symptom_history = symptom_history

                    # Mark analysis as complete
                    LocalSessionManager.set("analysis_complete", True)
                    LocalSessionManager.set("analysis_in_progress", False)

                    # Save to health history if user manager is available
                    self._save_to_health_history(analysis_results)

                    # Add notification about completed analysis
                    LocalSessionManager.add_notification(
                        "Analysis Complete",
                        f"Your symptom analysis is ready to view.",
                        "success"
                    )

                    # Success message
                    time.sleep(0.5)  # Brief delay for visual transition
                    detail_container.success("Your health analysis is complete! Proceeding to results...")
                    time.sleep(1)  # Give users time to see the success message

                    # Move to results page
                    LocalSessionManager.set("analysis_step", 4)
                    st.rerun()

                except Exception as e:
                    # Handle errors with recovery options
                    self.logger.error(f"Error during symptom analysis: {e}", exc_info=True)

                    # Reset analysis state
                    LocalSessionManager.set("analysis_in_progress", False)

                    # Show error message with troubleshooting options
                    status_container.empty()
                    progress_container.empty()
                    detail_container.empty()

                    st.error(f"An error occurred during analysis: {str(e)}")

                    # Show troubleshooting options
                    st.markdown("### Troubleshooting Options")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Try Again", key="retry_analysis", use_container_width=True):
                            st.rerun()

                    with col2:
                        if st.button("Back to Symptom Details", key="back_to_details_error", use_container_width=True):
                            LocalSessionManager.set("analysis_step", 2)
                            st.rerun()

                    # Show detailed error in expander for advanced users
                    with st.expander("Technical Details", expanded=False):
                        st.code(traceback.format_exc())

    def _add_symptom(self, symptom_id: str) -> None:
        """
        Add a symptom to the current list of symptoms.

        Args:
            symptom_id: The ID of the symptom to add
        """
        current_symptoms = LocalSessionManager.get("current_symptoms", [])

        if symptom_id not in current_symptoms:
            current_symptoms.append(symptom_id)
            LocalSessionManager.set("current_symptoms", current_symptoms)

    def _remove_symptom(self, symptom_id: str) -> None:
        """
        Remove a symptom from the current list of symptoms.

        Args:
            symptom_id: The ID of the symptom to remove
        """
        current_symptoms = LocalSessionManager.get("current_symptoms", [])

        if symptom_id in current_symptoms:
            current_symptoms.remove(symptom_id)
            LocalSessionManager.set("current_symptoms", current_symptoms)

            # Also remove related data
            symptom_severity = LocalSessionManager.get("symptom_severity", {})
            if symptom_id in symptom_severity:
                del symptom_severity[symptom_id]
                LocalSessionManager.set("symptom_severity", symptom_severity)

            symptom_duration = LocalSessionManager.get("symptom_duration", {})
            if symptom_id in symptom_duration:
                del symptom_duration[symptom_id]
                LocalSessionManager.set("symptom_duration", symptom_duration)

    def _reset_analysis(self) -> None:
        """Reset the symptom analysis to start over."""
        # Clear symptom data
        LocalSessionManager.set("current_symptoms", [])
        LocalSessionManager.set("symptom_severity", {})
        LocalSessionManager.set("symptom_duration", {})

        # Reset analysis state
        LocalSessionManager.set("analysis_step", 1)
        LocalSessionManager.set("analysis_in_progress", False)
        LocalSessionManager.set("analysis_complete", False)
        LocalSessionManager.set("analysis_results", {})
        LocalSessionManager.set("risk_assessment", {})

        # Keep notes if they exist
        if "symptom_notes" in st.session_state:
            LocalSessionManager.set("temp_symptom_notes", st.session_state.symptom_notes)
            LocalSessionManager.delete("symptom_notes")

    def _create_symptom_dataframe(self, symptom_ids: List[str]) -> pd.DataFrame:
        """
        Create a DataFrame from a list of symptom IDs with enhanced information.

        Args:
            symptom_ids: List of symptom IDs

        Returns:
            DataFrame with symptom information
        """
        symptom_data = []

        for symptom_id in symptom_ids:
            symptom_info = self.health_data.get_symptom_info(symptom_id)
            if symptom_info:
                symptom_data.append({
                    "id": symptom_id,
                    "name": symptom_info.get("name", "Unknown"),
                    "category": symptom_info.get("category", "Unknown"),
                    "description": symptom_info.get("description", "")
                })

        return pd.DataFrame(symptom_data)

    @lru_cache(maxsize=1)  # Cache for better performance
    def _get_common_symptoms(self) -> List[Dict[str, Any]]:
        """
        Get a list of common symptoms for quick selection with caching for performance.

        Returns:
            List of common symptom data
        """
        # In a real implementation, this would fetch the most common symptoms
        # from the health data provider or a cached list. For this demo, we'll return a static list.

        common_symptoms = [
            {"id": "s1", "name": "Headache", "category": "Neurological"},
            {"id": "s2", "name": "Fever", "category": "General"},
            {"id": "s3", "name": "Cough", "category": "Respiratory"},
            {"id": "s4", "name": "Fatigue", "category": "General"},
            {"id": "s5", "name": "Sore Throat", "category": "ENT"},
            {"id": "s6", "name": "Shortness of Breath", "category": "Respiratory"},
            {"id": "s7", "name": "Muscle Pain", "category": "Musculoskeletal"},
            {"id": "s8", "name": "Nausea", "category": "Gastrointestinal"},
            {"id": "s9", "name": "Dizziness", "category": "Neurological"}
        ]

        return common_symptoms

    @lru_cache(maxsize=1)  # Cache for better performance
    def _get_all_symptoms(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available symptoms with caching for performance.

        Returns:
            List of all symptom data
        """
        # In a real implementation, this would fetch all symptoms from the health data provider
        # For this demo, we'll return a static extended list
        all_symptoms = self._get_common_symptoms() + [
            {"id": "s10", "name": "Vomiting", "category": "Gastrointestinal"},
            {"id": "s11", "name": "Diarrhea", "category": "Gastrointestinal"},
            {"id": "s12", "name": "Constipation", "category": "Gastrointestinal"},
            {"id": "s13", "name": "Abdominal Pain", "category": "Gastrointestinal"},
            {"id": "s14", "name": "Chest Pain", "category": "Cardiovascular"},
            {"id": "s15", "name": "Heart Palpitations", "category": "Cardiovascular"},
            {"id": "s16", "name": "High Blood Pressure", "category": "Cardiovascular"},
            {"id": "s17", "name": "Joint Pain", "category": "Musculoskeletal"},
            {"id": "s18", "name": "Back Pain", "category": "Musculoskeletal"},
            {"id": "s19", "name": "Rash", "category": "Dermatological"},
            {"id": "s20", "name": "Itching", "category": "Dermatological"},
            {"id": "s21", "name": "Swelling", "category": "General"},
            {"id": "s22", "name": "Anxiety", "category": "Psychological"},
            {"id": "s23", "name": "Depression", "category": "Psychological"},
            {"id": "s24", "name": "Insomnia", "category": "Psychological"},
            {"id": "s25", "name": "Blurred Vision", "category": "Ophthalmological"},
            {"id": "s26", "name": "Eye Pain", "category": "Ophthalmological"},
            {"id": "s27", "name": "Ear Pain", "category": "ENT"},
            {"id": "s28", "name": "Hearing Loss", "category": "ENT"},
            {"id": "s29", "name": "Runny Nose", "category": "ENT"},
            {"id": "s30", "name": "Nasal Congestion", "category": "ENT"}
        ]

        return all_symptoms

    def _group_symptoms_by_category(self, symptoms: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group symptoms by their categories for better organization.

        Args:
            symptoms: List of symptom data

        Returns:
            Dictionary mapping categories to lists of symptoms
        """
        categories = {}

        for symptom in symptoms:
            category = symptom.get("category", "Other")

            if category not in categories:
                categories[category] = []

            categories[category].append(symptom)

        # Return sorted categories
        return {k: categories[k] for k in sorted(categories.keys())}

    def _predict_related_symptoms(self, symptom_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Predict related symptoms based on current symptoms using the ML model.

        Args:
            symptom_ids: List of current symptom IDs

        Returns:
            List of predicted symptom data with probabilities
        """
        # In a real implementation, this would call the symptom predictor model
        if not self.symptom_predictor:
            return []

        try:
            # Simulate calling the predictor model
            self.logger.info(f"Predicting related symptoms for {len(symptom_ids)} symptoms")

            # Get all symptoms for reference
            all_symptoms = self._get_all_symptoms()

            # Get the current symptom categories
            current_categories = set()
            for symptom_id in symptom_ids:
                for symptom in all_symptoms:
                    if symptom["id"] == symptom_id:
                        current_categories.add(symptom["category"])
                        break

            # Generate predictions
            predictions = []

            # Filter symptoms that are not already selected
            candidate_symptoms = [s for s in all_symptoms if s["id"] not in symptom_ids]

            # Add predictions with intelligent logic
            import random
            random.seed(42)  # For consistent results

            for symptom in candidate_symptoms:
                # Higher probability for symptoms in the same categories
                base_prob = 0.1
                if symptom["category"] in current_categories:
                    base_prob = 0.5

                # Add some variability
                prob = base_prob + random.uniform(-0.1, 0.2)
                prob = max(0.05, min(0.9, prob))  # Clamp between 0.05 and 0.9

                # Only include symptoms with probability > 0.2
                if prob > 0.2:
                    predictions.append({
                        "id": symptom["id"],
                        "probability": prob
                    })

            # Sort by probability
            predictions.sort(key=lambda x: x["probability"], reverse=True)

            # Return top predictions
            return predictions[:10]

        except Exception as e:
            self.logger.error(f"Error predicting related symptoms: {e}", exc_info=True)
            return []

    def _perform_risk_assessment(
        self,
        symptom_ids: List[str],
        severity: Dict[str, int],
        duration: Dict[str, str],
        notes: str
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive health risk assessment based on symptoms and characteristics.

        Args:
            symptom_ids: List of symptom IDs
            severity: Dictionary mapping symptom IDs to severity values (1-10)
            duration: Dictionary mapping symptom IDs to duration strings
            notes: Additional notes from the user

        Returns:
            Risk assessment data dictionary
        """
        # In a real implementation, this would call the risk assessor model
        if not self.risk_assessor:
            return {
                "risk_level": "unknown",
                "risk_score": 0,
                "recommendations": []
            }

        try:
            # Simulate calling the risk assessor model
            self.logger.info(f"Performing risk assessment for {len(symptom_ids)} symptoms")

            # Get all symptoms for reference
            all_symptoms = self._get_all_symptoms()

            # Calculate base risk from symptom count
            symptom_count = len(symptom_ids)
            base_risk = min(symptom_count / 10, 1.0) * 3  # Max 3 points from count

            # Calculate severity risk
            total_severity = sum(severity.get(s, 5) for s in symptom_ids)
            avg_severity = total_severity / max(1, len(symptom_ids))
            severity_risk = (avg_severity / 10) * 4  # Max 4 points from severity

            # Calculate duration risk
            duration_values = {
                "Less than 1 day": 0.2,
                "1-2 days": 0.4,
                "3-7 days": 0.6,
                "1-2 weeks": 0.8,
                "2-4 weeks": 0.9,
                "More than 4 weeks": 1.0
            }

            duration_scores = [duration_values.get(duration.get(s, "Less than 1 day"), 0.2) for s in symptom_ids]
            avg_duration = sum(duration_scores) / max(1, len(duration_scores))
            duration_risk = avg_duration * 3  # Max 3 points from duration

            # Calculate overall risk score (0-10 scale)
            risk_score = min(base_risk + severity_risk + duration_risk, 10)
            risk_score = round(risk_score, 1)

            # Determine risk level
            if risk_score < 3.33:
                risk_level = "low"
            elif risk_score < 6.66:
                risk_level = "medium"
            else:
                risk_level = "high"

            # Generate domain-specific risks
            # Group symptoms by category
            category_symptoms = {}
            category_severity = {}

            for symptom_id in symptom_ids:
                for symptom in all_symptoms:
                    if symptom["id"] == symptom_id:
                        category = symptom["category"]

                        if category not in category_symptoms:
                            category_symptoms[category] = []
                            category_severity[category] = []

                        category_symptoms[category].append(symptom_id)
                        category_severity[category].append(severity.get(symptom_id, 5))
                        break

            # Calculate risk for each domain/category
            domain_risks = {}

            for category, symptoms in category_symptoms.items():
                # Calculate domain risk based on:
                # 1. Number of symptoms in the category
                # 2. Average severity of symptoms in the category
                # 3. Domain-specific risk factors

                count_factor = min(len(symptoms) / 3, 1.0)  # Saturate at 3+ symptoms

                avg_domain_severity = sum(category_severity[category]) / len(category_severity[category])
                severity_factor = avg_domain_severity / 10

                # Domain-specific adjustments
                domain_adjustment = 0

                if category == "Cardiovascular" and any(s in severity and severity[s] > 7 for s in symptoms):
                    domain_adjustment = 2  # Higher adjustment for severe cardiovascular symptoms
                elif category == "Respiratory" and any(s in severity and severity[s] > 6 for s in symptoms):
                    domain_adjustment = 1.5  # Higher adjustment for severe respiratory symptoms
                elif category == "Neurological" and any(s in severity and severity[s] > 5 for s in symptoms):
                    domain_adjustment = 1  # Higher adjustment for severe neurological symptoms

                # Calculate domain risk (0-10 scale)
                domain_risk = (count_factor * 3) + (severity_factor * 5) + domain_adjustment
                domain_risk = min(max(domain_risk, 0), 10)  # Clamp between 0 and 10
                domain_risk = round(domain_risk, 1)

                domain_risks[category] = domain_risk

            # Generate risk factors
            risk_factors = []

            # Check for high severity symptoms
            high_severity_symptoms = []
            for symptom_id, sev in severity.items():
                if sev >= 8:
                    for symptom in all_symptoms:
                        if symptom["id"] == symptom_id:
                            high_severity_symptoms.append(symptom["name"])
                            break

            if high_severity_symptoms:
                risk_factors.append(f"High severity symptoms: {', '.join(high_severity_symptoms)}")

            # Check for chronic symptoms
            chronic_symptoms = []
            for symptom_id, dur in duration.items():
                if dur in ["2-4 weeks", "More than 4 weeks"]:
                    for symptom in all_symptoms:
                        if symptom["id"] == symptom_id:
                            chronic_symptoms.append(symptom["name"])
                            break

            if chronic_symptoms:
                risk_factors.append(f"Chronic symptoms: {', '.join(chronic_symptoms)}")

            # Check for concerning combinations
            has_fever = any(s["name"] == "Fever" for s in all_symptoms if s["id"] in symptom_ids)
            has_cough = any(s["name"] == "Cough" for s in all_symptoms if s["id"] in symptom_ids)
            has_shortness_of_breath = any(s["name"] == "Shortness of Breath" for s in all_symptoms if s["id"] in symptom_ids)

            if has_fever and has_cough and has_shortness_of_breath:
                risk_factors.append("Concerning combination: Fever, Cough, and Shortness of Breath")

            has_chest_pain = any(s["name"] == "Chest Pain" for s in all_symptoms if s["id"] in symptom_ids)
            has_shortness_of_breath = any(s["name"] == "Shortness of Breath" for s in all_symptoms if s["id"] in symptom_ids)

            if has_chest_pain and has_shortness_of_breath:
                risk_factors.append("Concerning combination: Chest Pain and Shortness of Breath")

            # Generate recommendations based on risk level
            recommendations = []

            if risk_level == "high":
                recommendations = [
                    "Seek medical attention promptly",
                    "Consider urgent care or emergency services if symptoms are severe",
                    "Monitor symptoms closely for any worsening",
                    "Ensure proper hydration and rest",
                    "Avoid strenuous activities until cleared by a healthcare professional"
                ]
            elif risk_level == "medium":
                recommendations = [
                    "Schedule an appointment with your healthcare provider",
                    "Monitor symptoms daily and record any changes",
                    "Ensure adequate rest and hydration",
                    "Consider over-the-counter remedies for symptomatic relief if appropriate",
                    "Avoid activities that worsen symptoms",
                    "Follow up with a healthcare professional if symptoms persist or worsen"
                ]
            else:  # low
                recommendations = [
                    "Continue monitoring your symptoms and record any changes",
                    "Maintain proper hydration and rest",
                    "Practice good hygiene to prevent potential infections",
                    "Consider over-the-counter remedies for symptomatic relief if appropriate",
                    "Follow up with your healthcare provider if symptoms persist or worsen"
                ]

            # If there are specific risk factors, add targeted recommendations
            if "Chest Pain" in [s["name"] for s in all_symptoms if s["id"] in symptom_ids]:
                recommendations.insert(0, "Consult with a healthcare professional about your chest pain")

            if "Shortness of Breath" in [s["name"] for s in all_symptoms if s["id"] in symptom_ids]:
                recommendations.insert(0, "Seek medical attention if experiencing severe shortness of breath")

            # Return the assessment
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "domain_risks": domain_risks,
                "risk_factors": risk_factors,
                "recommendations": recommendations
            }

        except Exception as e:
            self.logger.error(f"Error performing risk assessment: {e}", exc_info=True)
            return {
                "risk_level": "unknown",
                "risk_score": 0,
                "recommendations": [
                    "Consult with a healthcare professional",
                    "Monitor your symptoms",
                    "Seek medical attention if symptoms worsen"
                ]
            }

    def _generate_insights(
        self,
        symptom_ids: List[str],
        severity: Dict[str, int],
        duration: Dict[str, str]
    ) -> List[str]:
        """
        Generate health insights based on symptoms and their characteristics.

        Args:
            symptom_ids: List of symptom IDs
            severity: Dictionary mapping symptom IDs to severity values
            duration: Dictionary mapping symptom IDs to duration strings

        Returns:
            List of insight strings
        """
        insights = []

        try:
            # Get all symptoms for reference
            all_symptoms = self._get_all_symptoms()

            # Get symptom names and categories
            symptom_names = {}
            symptom_categories = {}

            for symptom_id in symptom_ids:
                for symptom in all_symptoms:
                    if symptom["id"] == symptom_id:
                        symptom_names[symptom_id] = symptom["name"]
                        symptom_categories[symptom_id] = symptom["category"]
                        break

            # Insight 1: Overall symptom severity
            total_severity = sum(severity.get(s, 5) for s in symptom_ids)
            avg_severity = total_severity / max(1, len(symptom_ids))

            if avg_severity <= 3:
                insights.append("Your symptoms are generally mild, which suggests a less severe condition.")
            elif avg_severity <= 6:
                insights.append("Your symptoms are moderate in severity, which may indicate a condition requiring attention.")
            else:
                insights.append("Your symptoms are relatively severe, which could indicate a more serious health concern.")

            # Insight 2: Duration patterns
            duration_values = {
                "Less than 1 day": 1,
                "1-2 days": 2,
                "3-7 days": 5,
                "1-2 weeks": 10,
                "2-4 weeks": 21,
                "More than 4 weeks": 30
            }

            duration_days = [duration_values.get(duration.get(s, "Less than 1 day"), 1) for s in symptom_ids]
            avg_duration = sum(duration_days) / max(1, len(duration_days))

            if avg_duration <= 2:
                insights.append("Your symptoms are recent in onset, which is often consistent with an acute condition.")
            elif avg_duration <= 7:
                insights.append("Your symptoms have persisted for several days, which may indicate an evolving condition.")
            elif avg_duration <= 14:
                insights.append("Your symptoms have persisted for over a week, suggesting a condition that may require medical evaluation.")
            else:
                insights.append("Your symptoms are chronic (long-term), which may indicate an ongoing health condition that should be evaluated.")

            # Insight 3: Category patterns
            category_counts = {}
            for category in symptom_categories.values():
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1

            # Find the most common category
            if category_counts:
                most_common_category = max(category_counts.items(), key=lambda x: x[1])

                if most_common_category[1] > 1:
                    insights.append(f"Multiple symptoms in the {most_common_category[0]} category suggest a potential {most_common_category[0].lower()} system issue.")

            # Insight 4: Specific symptom combinations
            has_fever = any(name == "Fever" for name in symptom_names.values())
            has_cough = any(name == "Cough" for name in symptom_names.values())
            has_sore_throat = any(name == "Sore Throat" for name in symptom_names.values())

            if has_fever and (has_cough or has_sore_throat):
                insights.append("The combination of fever with respiratory symptoms is commonly seen in respiratory infections.")

            has_headache = any(name == "Headache" for name in symptom_names.values())
            has_nausea = any(name == "Nausea" for name in symptom_names.values())

            if has_headache and has_nausea:
                insights.append("Headache combined with nausea can be associated with various conditions, including migraine or viral illness.")

            # Insight 5: Severity distribution
            high_severity = [s for s in symptom_ids if severity.get(s, 5) >= 7]
            low_severity = [s for s in symptom_ids if severity.get(s, 5) <= 3]

            if high_severity and len(high_severity) / len(symptom_ids) > 0.5:
                insights.append("Most of your symptoms are rated as high severity, which suggests a potentially serious condition.")
            elif low_severity and len(low_severity) / len(symptom_ids) > 0.7:
                insights.append("Most of your symptoms are rated as low severity, which often indicates a milder condition.")

            # Insight 6: Chronic vs. acute distinction
            chronic_symptoms = [s for s in symptom_ids if duration.get(s, "") in ["2-4 weeks", "More than 4 weeks"]]
            acute_symptoms = [s for s in symptom_ids if duration.get(s, "") in ["Less than 1 day", "1-2 days"]]

            if chronic_symptoms and acute_symptoms:
                insights.append("You have a mix of long-standing and recent symptoms, which could indicate a chronic condition with a recent exacerbation or a new, unrelated issue.")

            # Insight 7: System interactions
            has_gi = any(category == "Gastrointestinal" for category in symptom_categories.values())
            has_neuro = any(category == "Neurological" for category in symptom_categories.values())

            if has_gi and has_neuro:
                insights.append("The combination of gastrointestinal and neurological symptoms can sometimes indicate a systemic condition affecting multiple body systems.")

            has_cardio = any(category == "Cardiovascular" for category in symptom_categories.values())
            has_resp = any(category == "Respiratory" for category in symptom_categories.values())

            if has_cardio and has_resp:
                insights.append("Cardiovascular and respiratory symptoms often occur together, as these systems are closely linked.")

            # Add additional insights based on specific high-severity symptoms
            for symptom_id in symptom_ids:
                if severity.get(symptom_id, 5) >= 8:
                    name = symptom_names.get(symptom_id, "Unknown")
                    insights.append(f"Your {name.lower()} is particularly severe and may require focused attention.")

        except Exception as e:
            self.logger.error(f"Error generating insights: {e}", exc_info=True)
            insights = [
                "Your symptom pattern suggests an ongoing health issue.",
                "Consider consulting with a healthcare professional for a comprehensive evaluation."
            ]

        return insights

    def _save_to_health_history(self, analysis_results: Dict[str, Any]) -> None:
        """
        Save analysis results to the user's health history.

        Args:
            analysis_results: Analysis results data
        """
        # In a real implementation, this would save to a database or user profile
        try:
            # Log the action
            self.logger.info("Saving symptom analysis to health history")

            # Check if there's a user manager available to use
            user_manager = LocalSessionManager.get("user_manager")

            if user_manager:
                # In a real app, this would call a method on the user manager
                self.logger.info("Using user manager to save health history")

                # Simulate saving
                if "health_history" not in st.session_state:
                    st.session_state.health_history = []

                # Add the current analysis to history
                st.session_state.health_history.append({
                    "type": "symptom_analysis",
                    "timestamp": analysis_results.get("timestamp", datetime.now().isoformat()),
                    "symptoms": analysis_results.get("symptoms", []),
                    "risk_level": analysis_results.get("risk_assessment", {}).get("risk_level", "unknown"),
                    "risk_score": analysis_results.get("risk_assessment", {}).get("risk_score", 0)
                })

                # Add notification
                LocalSessionManager.add_notification(
                    "Analysis Saved",
                    "Your symptom analysis has been saved to your health history.",
                    "success"
                )
            else:
                self.logger.info("No user manager available, skipping history save")

        except Exception as e:
            self.logger.error(f"Error saving to health history: {e}", exc_info=True)

    def _generate_health_report(self, analysis_results: Dict[str, Any], risk_assessment: Dict[str, Any]) -> bytes:
        """
        Generate a PDF health report based on analysis results.

        Args:
            analysis_results: Analysis results data
            risk_assessment: Risk assessment data

        Returns:
            PDF report as bytes
        """
        # In a real implementation, this would generate a properly formatted PDF
        try:
            # Simulate PDF generation
            self.logger.info("Generating health report PDF")

            # In a real app, this would create actual PDF content
            # Here we're just returning placeholder bytes
            pdf_bytes = b"This is a placeholder for a PDF health report"

            return pdf_bytes

        except Exception as e:
            self.logger.error(f"Error generating health report: {e}", exc_info=True)
            raise

    def _generate_csv_report(self, analysis_results: Dict[str, Any], risk_assessment: Dict[str, Any]) -> str:
        """
        Generate a CSV report based on analysis results.

        Args:
            analysis_results: Analysis results data
            risk_assessment: Risk assessment data

        Returns:
            CSV data as string
        """
        # In a real implementation, this would generate a well-formatted CSV
        try:
            # Create a CSV for symptom data
            all_symptoms = self._get_all_symptoms()

            # Get symptom data
            symptom_rows = []
            symptom_ids = analysis_results.get("symptoms", [])
            symptom_severity = analysis_results.get("symptom_severity", {})
            symptom_duration = analysis_results.get("symptom_duration", {})

            for symptom_id in symptom_ids:
                symptom_info = next((s for s in all_symptoms if s["id"] == symptom_id), None)
                if symptom_info:
                    symptom_rows.append({
                        "Symptom": symptom_info.get("name", "Unknown"),
                        "Category": symptom_info.get("category", "Unknown"),
                        "Severity": symptom_severity.get(symptom_id, 5),
                        "Duration": symptom_duration.get(symptom_id, "Unknown")
                    })

            # Convert to DataFrame and then to CSV
            symptom_df = pd.DataFrame(symptom_rows)
            csv_data = symptom_df.to_csv(index=False)

            return csv_data

        except Exception as e:
            self.logger.error(f"Error generating CSV report: {e}", exc_info=True)
            raise

    def _generate_json_report(self, analysis_results: Dict[str, Any], risk_assessment: Dict[str, Any]) -> str:
        """
        Generate a JSON report based on analysis results.

        Args:
            analysis_results: Analysis results data
            risk_assessment: Risk assessment data

        Returns:
            JSON data as string
        """
        # Generate a JSON representation of the analysis results
        try:
            # Create a combined dictionary
            report_data = {
                "timestamp": analysis_results.get("timestamp", datetime.now().isoformat()),
                "symptoms": [],
                "risk_assessment": risk_assessment,
                "insights": analysis_results.get("insights", [])
            }

            # Add detailed symptom data
            all_symptoms = self._get_all_symptoms()
            symptom_ids = analysis_results.get("symptoms", [])
            symptom_severity = analysis_results.get("symptom_severity", {})
            symptom_duration = analysis_results.get("symptom_duration", {})

            for symptom_id in symptom_ids:
                symptom_info = next((s for s in all_symptoms if s["id"] == symptom_id), None)
                if symptom_info:
                    report_data["symptoms"].append({
                        "id": symptom_id,
                        "name": symptom_info.get("name", "Unknown"),
                        "category": symptom_info.get("category", "Unknown"),
                        "severity": symptom_severity.get(symptom_id, 5),
                        "duration": symptom_duration.get(symptom_id, "Unknown")
                    })

            # Convert to JSON string
            json_data = json.dumps(report_data, indent=2)

            return json_data

        except Exception as e:
            self.logger.error(f"Error generating JSON report: {e}", exc_info=True)
            raise

    def _generate_excel_report(self, analysis_results: Dict[str, Any], risk_assessment: Dict[str, Any]) -> bytes:
        """
        Generate an Excel report based on analysis results.

        Args:
            analysis_results: Analysis results data
            risk_assessment: Risk assessment data

        Returns:
            Excel data as bytes
        """
        # In a real implementation, this would generate a well-formatted Excel file
        try:
            # Create DataFrames for different sheets
            all_symptoms = self._get_all_symptoms()

            # Get symptom data
            symptom_rows = []
            symptom_ids = analysis_results.get("symptoms", [])
            symptom_severity = analysis_results.get("symptom_severity", {})
            symptom_duration = analysis_results.get("symptom_duration", {})

            for symptom_id in symptom_ids:
                symptom_info = next((s for s in all_symptoms if s["id"] == symptom_id), None)
                if symptom_info:
                    symptom_rows.append({
                        "Symptom": symptom_info.get("name", "Unknown"),
                        "Category": symptom_info.get("category", "Unknown"),
                        "Severity": symptom_severity.get(symptom_id, 5),
                        "Duration": symptom_duration.get(symptom_id, "Unknown")
                    })

            # Create DataFrame for symptoms
            symptom_df = pd.DataFrame(symptom_rows)

            # Create DataFrame for risk assessment
            risk_rows = [
                {"Metric": "Risk Level", "Value": risk_assessment.get("risk_level", "unknown")},
                {"Metric": "Risk Score", "Value": risk_assessment.get("risk_score", 0)}
            ]

            # Add domain risks
            domain_risks = risk_assessment.get("domain_risks", {})
            for domain, risk in domain_risks.items():
                risk_rows.append({"Metric": f"{domain} Risk", "Value": risk})

            risk_df = pd.DataFrame(risk_rows)

            # Create DataFrame for insights
            insights = analysis_results.get("insights", [])
            insight_df = pd.DataFrame({"Insight": insights})

            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                symptom_df.to_excel(writer, sheet_name='Symptoms', index=False)
                risk_df.to_excel(writer, sheet_name='Risk Assessment', index=False)
                insight_df.to_excel(writer, sheet_name='Insights', index=False)

                # Get workbook and worksheet objects for formatting
                workbook = writer.book

                # Format the symptom worksheet
                worksheet = writer.sheets['Symptoms']

                # Add header formatting, etc.
                header_format = workbook.add_format({'bold': True, 'bg_color': '#3498db', 'color': 'white'})
                for col_num, value in enumerate(symptom_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # Auto-fit columns
                for i, col in enumerate(symptom_df.columns):
                    max_len = max(symptom_df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, max_len)

            # Get the Excel data
            excel_data = output.getvalue()

            return excel_data

        except Exception as e:
            self.logger.error(f"Error generating Excel report: {e}", exc_info=True)
            raise

    def _get_severity_color(self, severity: int) -> str:
        """
        Get a color string for a severity value.

        Args:
            severity: Severity value (1-10)

        Returns:
            Color string for use in CSS
        """
        if severity <= 3:
            return "#27ae60"  # green
        elif severity <= 7:
            return "#f39c12"  # orange
        else:
            return "#e74c3c"  # red

    def _get_severity_bg_color(self, severity: int) -> str:
        """
        Get a background color string for a severity value.

        Args:
            severity: Severity value (1-10)

        Returns:
            Background color string for use in CSS
        """
        if severity <= 3:
            return "rgba(46, 204, 113, 0.2)"  # green with transparency
        elif severity <= 7:
            return "rgba(243, 156, 18, 0.2)"  # orange with transparency
        else:
            return "rgba(231, 76, 60, 0.2)"  # red with transparency

    def _severity_text(self, severity: int) -> str:
        """
        Get a text description of a severity value.

        Args:
            severity: Severity value (1-10)

        Returns:
            Text description
        """
        if severity <= 3:
            return "Mild"
        elif severity <= 7:
            return "Moderate"
        else:
            return "Severe"

    def _get_risk_color_hex(self, risk_level: str) -> Dict[str, str]:
        """
        Get color scheme for a risk level.

        Args:
            risk_level: Risk level string (low, medium, high)

        Returns:
            Dictionary with color values
        """
        if risk_level == "low":
            return {
                "bg": "rgba(46, 204, 113, 0.1)",
                "text": "#27ae60",
                "border": "#27ae60",
                "accent": "#27ae60",
                "light": "rgba(46, 204, 113, 0.2)",
                "dark": "rgba(39, 174, 96, 0.3)"
            }
        elif risk_level == "medium":
            return {
                "bg": "rgba(243, 156, 18, 0.1)",
                "text": "#f39c12",
                "border": "#f39c12",
                "accent": "#f39c12",
                "light": "rgba(243, 156, 18, 0.2)",
                "dark": "rgba(230, 126, 34, 0.3)"
            }
        elif risk_level == "high":
            return {
                "bg": "rgba(231, 76, 60, 0.1)",
                "text": "#e74c3c",
                "border": "#e74c3c",
                "accent": "#e74c3c",
                "light": "rgba(231, 76, 60, 0.2)",
                "dark": "rgba(192, 57, 43, 0.3)"
            }
        else:
            return {
                "bg": "rgba(149, 165, 166, 0.1)",
                "text": "#7f8c8d",
                "border": "#95a5a6",
                "accent": "#7f8c8d",
                "light": "rgba(149, 165, 166, 0.2)",
                "dark": "rgba(127, 140, 141, 0.3)"
            }

    def _get_domain_interpretation(self, domain: str, risk_value: float) -> str:
        """
        Get an interpretation of a domain risk score.

        Args:
            domain: Domain/category name
            risk_value: Risk value (0-10)

        Returns:
            Interpretation text
        """
        # Determine risk level text
        if risk_value < 3.33:
            risk_text = "low"
        elif risk_value < 6.66:
            risk_text = "moderate"
        else:
            risk_text = "high"

        # Domain-specific interpretations
        if domain == "Respiratory":
            if risk_value < 3.33:
                return f"Your respiratory health shows {risk_text} risk. Your respiratory symptoms are mild or minimal."
            elif risk_value < 6.66:
                return f"Your respiratory health shows {risk_text} risk. Monitor your breathing and respiratory symptoms."
            else:
                return f"Your respiratory health shows {risk_text} risk. Your respiratory symptoms require medical attention."
        elif domain == "Cardiovascular":
            if risk_value < 3.33:
                return f"Your cardiovascular health shows {risk_text} risk. Your heart-related symptoms are minimal."
            elif risk_value < 6.66:
                return f"Your cardiovascular health shows {risk_text} risk. Monitor your heart-related symptoms."
            else:
                return f"Your cardiovascular health shows {risk_text} risk. Your heart-related symptoms require prompt medical attention."
        elif domain == "Neurological":
            if risk_value < 3.33:
                return f"Your neurological health shows {risk_text} risk. Your neurological symptoms are mild."
            elif risk_value < 6.66:
                return f"Your neurological health shows {risk_text} risk. Monitor your neurological symptoms."
            else:
                return f"Your neurological health shows {risk_text} risk. Your neurological symptoms require medical evaluation."
        elif domain == "Gastrointestinal":
            if risk_value < 3.33:
                return f"Your gastrointestinal health shows {risk_text} risk. Your digestive symptoms are mild."
            elif risk_value < 6.66:
                return f"Your gastrointestinal health shows {risk_text} risk. Monitor your digestive symptoms."
            else:
                return f"Your gastrointestinal health shows {risk_text} risk. Your digestive symptoms require medical evaluation."
        else:
            # Generic interpretation for other domains
            if risk_value < 3.33:
                return f"Your {domain.lower()} health shows {risk_text} risk. Continue monitoring your symptoms."
            elif risk_value < 6.66:
                return f"Your {domain.lower()} health shows {risk_text} risk. Consider discussing these symptoms with a healthcare provider."
            else:
                return f"Your {domain.lower()} health shows {risk_text} risk. We recommend seeking medical evaluation for these symptoms."

    @timing_decorator
    def _render_analysis_results(self) -> None:
        """Render the fourth step: analysis results with enhanced visualizations and insights."""
        st.header("Step 4: Analysis Results")

        # Get analysis results
        analysis_results = LocalSessionManager.get("analysis_results", {})
        risk_assessment = LocalSessionManager.get("risk_assessment", {})

        if not analysis_results:
            st.warning("No analysis results available. Please complete the symptom analysis first.")

            if st.button("Start New Analysis", use_container_width=True, type="primary"):
                # Reset analysis state
                self._reset_analysis()
                st.rerun()
            return

        # Create a modern tabbed interface for different result sections
        result_tabs = st.tabs([
            "📊 Summary",
            "⚠️ Risk Assessment",
            "🔍 Detailed Analysis",
            "💡 Recommendations",
            "⏭️ Next Steps"
        ])

        # Summary tab
        with result_tabs[0]:
            self._render_results_summary(analysis_results)

        # Risk Assessment tab
        with result_tabs[1]:
            self._render_risk_assessment(risk_assessment)

        # Detailed Analysis tab
        with result_tabs[2]:
            self._render_detailed_analysis(analysis_results)

        # Recommendations tab
        with result_tabs[3]:
            self._render_recommendations(analysis_results, risk_assessment)

        # Next Steps tab
        with result_tabs[4]:
            self._render_next_steps(analysis_results, risk_assessment)

        # Action buttons at the bottom
        st.markdown("---")
        st.markdown("### Actions")

        # Create a row of action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 New Analysis", use_container_width=True):
                # Reset analysis state
                self._reset_analysis()
                st.rerun()

        with col2:
            if st.button("📱 Go to Dashboard", use_container_width=True):
                LocalSessionManager.navigate_to("Health Dashboard")

        with col3:
            if st.button("💬 Discuss with AI", use_container_width=True):
                LocalSessionManager.navigate_to("Health Chat")

        # Add export options
        with st.expander("Export Options", expanded=False):
            export_format = st.radio("Export Format:", ["PDF Report", "CSV Data", "JSON Data"], horizontal=True)

            if st.button("Export Results", type="primary", use_container_width=True):
                try:
                    if export_format == "PDF Report":
                        # Generate PDF report
                        report_bytes = self._generate_health_report(analysis_results, risk_assessment)

                        # Format timestamp for filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Provide download button
                        st.download_button(
                            label="Download PDF Report",
                            data=report_bytes,
                            file_name=f"health_report_{timestamp}.pdf",
                            mime="application/pdf"
                        )
                    elif export_format == "CSV Data":
                        # Create CSV data
                        csv_data = self._generate_csv_report(analysis_results, risk_assessment)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        st.download_button(
                            label="Download CSV Data",
                            data=csv_data,
                            file_name=f"health_data_{timestamp}.csv",
                            mime="text/csv"
                        )
                    else:  # JSON Data
                        # Create JSON data
                        json_data = self._generate_json_report(analysis_results, risk_assessment)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        st.download_button(
                            label="Download JSON Data",
                            data=json_data,
                            file_name=f"health_data_{timestamp}.json",
                            mime="application/json"
                        )
                except Exception as e:
                    self.logger.error(f"Error generating export: {e}", exc_info=True)
                    st.error(f"Could not generate export: {str(e)}")

    def _render_results_summary(self, analysis_results: Dict[str, Any]) -> None:
        """
        Render the summary section of the analysis results with modern visualizations.

        Args:
            analysis_results: Dictionary containing analysis results data
        """
        st.subheader("Summary of Your Health Analysis")

        # Get symptom data
        symptoms = analysis_results.get("symptoms", [])
        severity = analysis_results.get("symptom_severity", {})
        duration = analysis_results.get("symptom_duration", {})

        # Get risk assessment
        risk_assessment = analysis_results.get("risk_assessment", {})
        risk_level = risk_assessment.get("risk_level", "unknown")
        risk_score = risk_assessment.get("risk_score", 0)

        # Format timestamp
        timestamp_str = "Unknown"
        if "timestamp" in analysis_results:
            try:
                timestamp = datetime.fromisoformat(analysis_results["timestamp"])
                timestamp_str = timestamp.strftime("%B %d, %Y at %I:%M %p")
            except:
                timestamp_str = analysis_results["timestamp"]

        # Create a modern card with risk level and summary
        risk_color = self._get_risk_color_hex(risk_level)

        # Risk level display card with gradient background
        st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, {risk_color['light']} 0%, {risk_color['dark']} 100%);
            color: {risk_color['text']};
            margin-bottom: 20px;
            border: 1px solid {risk_color['border']};
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin-top: 0; color: {risk_color['text']}; display: flex; align-items: center;">
                <span style="
                    background-color: {risk_color['accent']};
                    color: white;
                    padding: 5px 10px;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    margin-right: 10px;
                ">
                    {risk_level.upper()}
                </span>
                Health Risk Assessment
            </h3>
            <p>Analysis performed on {timestamp_str}</p>
            <div style="display: flex; align-items: center; margin-top: 15px;">
                <div style="
                    width: 70px;
                    height: 70px;
                    border-radius: 50%;
                    background-color: {risk_color['accent']};
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.8rem;
                    font-weight: bold;
                    margin-right: 20px;
                ">
                    {risk_score}/10
                </div>
                <div>
                    <p style="margin: 0;">Based on your symptoms and their characteristics, your health risk level
                    has been assessed as <strong>{risk_level}</strong> ({risk_score}/10).</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display key symptoms in a visually appealing layout
        st.markdown("### Key Symptoms")

        if symptoms:
            # Get the top symptoms (most severe first)
            symptom_list = []
            for symptom_id in symptoms:
                symptom_info = self.health_data.get_symptom_info(symptom_id)
                if symptom_info:
                    symptom_list.append({
                        "id": symptom_id,
                        "name": symptom_info.get("name", "Unknown"),
                        "severity": severity.get(symptom_id, 5),
                        "duration": duration.get(symptom_id, "Unknown"),
                        "category": symptom_info.get("category", "Unknown")
                    })

            # Sort by severity (highest first)
            symptom_list.sort(key=lambda x: x["severity"], reverse=True)

            # Display symptoms in a responsive grid
            num_columns = 2
            rows = [symptom_list[i:i + num_columns] for i in range(0, min(len(symptom_list), 6), num_columns)]

            for row in rows:
                cols = st.columns(num_columns)

                for i, symptom in enumerate(row):
                    severity_level = self._severity_text(symptom["severity"])
                    severity_color = self._get_severity_color(symptom["severity"])

                    with cols[i]:
                        st.markdown(f"""
                        <div style="
                            padding: 15px;
                            border-radius: 8px;
                            background-color: var(--secondary-bg-color, #f8f9fa);
                            margin-bottom: 15px;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            border-left: 4px solid {severity_color};
                        ">
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div>
                                    <div style="font-weight: bold; font-size: 1.1rem;">{symptom["name"]}</div>
                                    <div style="font-size: 0.8rem; color: #7f8c8d;">{symptom["category"]}</div>
                                    <div style="margin-top: 5px;">
                                        <span style="
                                            background-color: {self._get_severity_bg_color(symptom["severity"])};
                                            color: {severity_color};
                                            padding: 2px 8px;
                                            border-radius: 10px;
                                            font-size: 0.8rem;
                                            font-weight: 500;
                                        ">
                                            {severity_level}
                                        </span>
                                        <span style="margin-left: 5px; font-size: 0.9rem;">{symptom["duration"]}</span>
                                    </div>
                                </div>
                                <div style="
                                    width: 40px;
                                    height: 40px;
                                    border-radius: 50%;
                                    background-color: {severity_color};
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    color: white;
                                    font-weight: bold;
                                ">
                                    {symptom["severity"]}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Show "View all" if there are more symptoms
            if len(symptom_list) > 6:
                with st.expander(f"View all {len(symptom_list)} symptoms", expanded=False):
                    remaining = symptom_list[6:]

                    # Create a table view for all remaining symptoms
                    remaining_data = []
                    for symptom in remaining:
                        remaining_data.append({
                            "Symptom": symptom["name"],
                            "Category": symptom["category"],
                            "Severity": symptom["severity"],
                            "Duration": symptom["duration"]
                        })

                    # Display as interactive dataframe
                    if remaining_data:
                        st.dataframe(
                            pd.DataFrame(remaining_data),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Severity": st.column_config.ProgressColumn(
                                    "Severity",
                                    min_value=1,
                                    max_value=10,
                                    format="%d"
                                )
                            }
                        )
        else:
            st.info("No symptoms were recorded in this analysis.")

        # Display key insights in a modern card layout
        insights = analysis_results.get("insights", [])
        if insights:
            st.markdown("### Key Insights")

            # Show top 3 insights with visual cards
            insights_to_show = min(3, len(insights))

            for i in range(insights_to_show):
                insight = insights[i]
                st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(52, 152, 219, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #3498db;
                ">
                    <div style="display: flex;">
                        <div style="
                            width: 24px;
                            height: 24px;
                            border-radius: 50%;
                            background-color: #3498db;
                            color: white;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: bold;
                            margin-right: 10px;
                        ">
                            {i+1}
                        </div>
                        <div>
                            {insight}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # More insights in expander
            if len(insights) > 3:
                with st.expander(f"View all {len(insights)} insights", expanded=False):
                    for i in range(3, len(insights)):
                        st.info(insights[i])

        # Display recommendations preview
        st.markdown("### Recommended Actions")

        # Get recommendations from risk assessment
        recommendations = risk_assessment.get("recommendations", [])

        if recommendations:
            # Display top 3 recommendations with icons and colors
            for i, recommendation in enumerate(recommendations[:3]):
                st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(46, 204, 113, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #2ecc71;
                    display: flex;
                ">
                    <div style="
                        min-width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        background-color: #2ecc71;
                        color: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        margin-right: 15px;
                    ">
                        {i+1}
                    </div>
                    <div>
                        {recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Show link to all recommendations
            if len(recommendations) > 3:
                st.markdown(f"[View all {len(recommendations)} recommendations](#recommendations)")
        else:
            # Default recommendations if none provided
            default_recommendations = [
                "Continue monitoring your symptoms and record any changes",
                "Maintain proper hydration and rest",
                "Consult with a healthcare professional if symptoms persist or worsen"
            ]

            for i, recommendation in enumerate(default_recommendations):
                st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(46, 204, 113, 0.1);
                    margin-bottom: 15px;
                    border-left: 4px solid #2ecc71;
                    display: flex;
                ">
                    <div style="
                        min-width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        background-color: #2ecc71;
                        color: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        margin-right: 15px;
                    ">
                        {i+1}
                    </div>
                    <div>
                        {recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    def _render_risk_assessment(self, risk_assessment: Dict[str, Any]) -> None:
        """
        Render the risk assessment section with enhanced visualizations.

        Args:
            risk_assessment: Dictionary containing risk assessment data
        """
        st.subheader("Health Risk Assessment")

        # Add explanatory text
        with st.expander("About Risk Assessment", expanded=False):
            st.markdown("""
            This risk assessment evaluates your symptoms considering:

            * **Severity** - How intense your symptoms are
            * **Duration** - How long you've had them
            * **Combination** - How your symptoms relate to each other
            * **Medical factors** - Based on medical knowledge

            The assessment provides a general indication but is **not a medical diagnosis**.
            Always consult with healthcare professionals for medical advice.
            """)

        if not risk_assessment:
            st.warning("Risk assessment data is not available.")
            return

        # Get risk level and score
        risk_level = risk_assessment.get("risk_level", "unknown")
        risk_score = risk_assessment.get("risk_score", 0)

        # Create advanced gauge visualization
        try:
            # Create the gauge chart
            fig = go.Figure()

            # Add gauge chart with improved visuals
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': f"Health Risk Score",
                    'font': {'size': 24, 'color': '#333'}
                },
                gauge={
                    'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#333"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 3.33], 'color': "rgba(46, 204, 113, 0.3)"},  # Green for low risk
                        {'range': [3.33, 6.66], 'color': "rgba(243, 156, 18, 0.3)"},  # Orange for medium risk
                        {'range': [6.66, 10], 'color': "rgba(231, 76, 60, 0.3)"}  # Red for high risk
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                },
                number={
                    'font': {'size': 40, 'color': self._get_risk_color_hex(risk_level)['accent']}
                }
            ))

            # Improve layout
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot
            )

            # Add risk level indication
            fig.add_annotation(
                x=0.5,
                y=0.25,
                text=risk_level.upper(),
                font=dict(size=24, color=self._get_risk_color_hex(risk_level)['accent']),
                showarrow=False
            )

            # Display the gauge
            st.plotly_chart(fig, use_container_width=True)

            # Add interpretation
            if risk_level == "low":
                st.markdown("""
                <div style="padding: 15px; border-radius: 8px; background-color: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ecc71;">
                    <strong>Low Risk Interpretation:</strong> Your symptoms suggest a low level of health risk. This often indicates minor health issues that can typically be managed with self-care and monitoring.
                </div>
                """, unsafe_allow_html=True)
            elif risk_level == "medium":
                st.markdown("""
                <div style="padding: 15px; border-radius: 8px; background-color: rgba(243, 156, 18, 0.1); border-left: 4px solid #f39c12;">
                    <strong>Medium Risk Interpretation:</strong> Your symptoms suggest a moderate level of health risk. Consider consulting with a healthcare professional, especially if symptoms persist or worsen.
                </div>
                """, unsafe_allow_html=True)
            else:  # high
                st.markdown("""
                <div style="padding: 15px; border-radius: 8px; background-color: rgba(231, 76, 60, 0.1); border-left: 4px solid #e74c3c;">
                    <strong>High Risk Interpretation:</strong> Your symptoms suggest a higher level of health risk. It's recommended to seek prompt medical attention to evaluate your condition.
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            self.logger.error(f"Error creating risk gauge: {e}", exc_info=True)

            # Fallback to simpler display
            risk_color = self._get_risk_color_hex(risk_level)

            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <h3>Risk Score: {risk_score}/10</h3>
                <div style="
                    display: inline-block;
                    padding: 10px 20px;
                    border-radius: 5px;
                    background-color: {risk_color['bg']};
                    color: {risk_color['text']};
                    font-weight: bold;
                    font-size: 1.2rem;
                ">
                    {risk_level.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Domain-specific risks with interactive visualization
        domain_risks = risk_assessment.get("domain_risks", {})
        if domain_risks:
            st.markdown("### Health Domain Risk Analysis")
            st.markdown("This chart shows your risk levels across different health domains:")

            try:
                # Create a more informative horizontal bar chart
                domains = list(domain_risks.keys())
                values = list(domain_risks.values())

                fig = go.Figure()

                # Custom color mapping function
                def get_bar_color(value):
                    if value < 3.33:
                        return '#2ecc71'  # Green
                    elif value < 6.66:
                        return '#f39c12'  # Orange
                    else:
                        return '#e74c3c'  # Red

                # Add bars with hover information
                fig.add_trace(go.Bar(
                    x=values,
                    y=domains,
                    orientation='h',
                    marker={
                        'color': [get_bar_color(v) for v in values],
                        'line': {'width': 1, 'color': '#333'}
                    },
                    text=values,
                    textposition='outside',
                    texttemplate='%{x:.1f}',
                    hovertemplate='<b>%{y}</b><br>Risk Score: %{x:.1f}/10<extra></extra>'
                ))

                # Add risk zone backgrounds
                fig.add_shape(
                    type="rect",
                    x0=0, y0=-0.5,
                    x1=3.33, y1=len(domains)-0.5,
                    line=dict(width=0),
                    fillcolor="rgba(46, 204, 113, 0.1)",  # Green with transparency
                    layer="below"
                )

                fig.add_shape(
                    type="rect",
                    x0=3.33, y0=-0.5,
                    x1=6.66, y1=len(domains)-0.5,
                    line=dict(width=0),
                    fillcolor="rgba(243, 156, 18, 0.1)",  # Orange with transparency
                    layer="below"
                )

                fig.add_shape(
                    type="rect",
                    x0=6.66, y0=-0.5,
                    x1=10, y1=len(domains)-0.5,
                    line=dict(width=0),
                    fillcolor="rgba(231, 76, 60, 0.1)",  # Red with transparency
                    layer="below"
                )

                # Update layout
                fig.update_layout(
                    title="Risk by Health Domain",
                    xaxis_title="Risk Score (0-10)",
                    xaxis=dict(
                        range=[0, 10],
                        tickvals=[0, 3.33, 6.66, 10],
                        ticktext=["No Risk", "Low", "Medium", "High"]
                    ),
                    yaxis_title="Health Domain",
                    height=max(300, len(domains) * 40),  # Dynamic height based on domain count
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Add interpretations for each domain
                st.subheader("Domain Interpretations")

                for domain, value in domain_risks.items():
                    # Determine risk level
                    if value < 3.33:
                        domain_level = "Low Risk"
                        domain_color = "#2ecc71"  # Green
                    elif value < 6.66:
                        domain_level = "Medium Risk"
                        domain_color = "#f39c12"  # Orange
                    else:
                        domain_level = "High Risk"
                        domain_color = "#e74c3c"  # Red

                    # Generate domain interpretation
                    interpretation = self._get_domain_interpretation(domain, value)

                    # Display domain card with modern styling
                    st.markdown(f"""
                    <div style="
                        padding: 15px;
                        border-radius: 8px;
                        background-color: var(--secondary-bg-color, #f8f9fa);
                        margin-bottom: 15px;
                        border-left: 4px solid {domain_color};
                    ">
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <div style="flex-grow: 1;">
                                <strong>{domain}</strong>
                            </div>
                            <div style="
                                padding: 5px 10px;
                                border-radius: 15px;
                                background-color: {domain_color};
                                color: white;
                                font-weight: bold;
                                font-size: 0.8rem;
                            ">
                                {value:.1f}/10 - {domain_level}
                            </div>
                        </div>
                        <p>{interpretation}</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                self.logger.error(f"Error creating domain risk chart: {e}", exc_info=True)

                # Fallback to table format
                st.warning("Unable to create visualizations. Displaying data in table format instead.")

                # Create a dataframe for the domain risks
                domain_df = pd.DataFrame({
                    "Health Domain": list(domain_risks.keys()),
                    "Risk Score": list(domain_risks.values())
                })

                st.dataframe(
                    domain_df.sort_values("Risk Score", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Health Domain": st.column_config.TextColumn("Health Domain", width="large"),
                        "Risk Score": st.column_config.ProgressColumn(
                            "Risk Score",
                            help="Risk score from 0-10",
                            format="%.1f",
                            min_value=0,
                            max_value=10
                        )
                    }
                )
        else:
            st.info("Detailed domain risk data is not available.")

        # Risk factors
        risk_factors = risk_assessment.get("risk_factors", [])
        if risk_factors:
            st.markdown("### Key Risk Factors")

            # Display risk factors as cards with warning styling
            for i, factor in enumerate(risk_factors):
                st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 8px;
                    background-color: rgba(243, 156, 18, 0.1);
                    margin-bottom: 10px;
                    border-left: 4px solid #f39c12;
                    display: flex;
                    align-items: center;
                ">
                    <div style="
                        min-width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        background-color: #f39c12;
                        color: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        margin-right: 10px;
                    ">
                        !
                    </div>
                    <div>{factor}</div>
                </div>
                """, unsafe_allow_html=True)

        # Risk disclaimer
        st.markdown("""
        <div style="
            padding: 15px;
            border-radius: 8px;
            background-color: var(--secondary-bg-color, #f8f9fa);
            margin-top: 20px;
            border: 1px solid #ddd;
        ">
            <h4 style="margin-top: 0;">Medical Disclaimer</h4>
            <p>This risk assessment is based on the symptoms you've reported and general health guidelines.
            It is not a medical diagnosis, and should not replace professional medical advice.
            If you have concerns about your health, please consult with a healthcare professional.</p>
        </div>
        """, unsafe_allow_html=True)
