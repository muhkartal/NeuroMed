"""
Medical Literature UI for MedExplain AI Pro.

This module provides the user interface for browsing and searching
medical literature related to various health conditions.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

def render_medical_literature() -> None:
    """Render the medical literature interface."""
    st.title("Medical Literature")
    st.markdown("Explore summaries of medical research relevant to various health conditions.")

    # Check if health data manager is in session state
    if "health_data_manager" not in st.session_state:
        st.error("Application not properly initialized. Please refresh the page.")
        return

    health_data_manager = st.session_state.health_data_manager
    user_manager = st.session_state.get("user_manager")

    # Create tabs for different categories
    tabs = st.tabs(["Common Conditions", "Recent Research", "Search", "Personalized Recommendations"])

    with tabs[0]:
        _render_common_conditions(health_data_manager)

    with tabs[1]:
        _render_recent_research()

    with tabs[2]:
        _render_search_interface(health_data_manager)

    with tabs[3]:
        _render_personalized_recommendations(health_data_manager, user_manager)

def _render_common_conditions(health_data_manager) -> None:
    """
    Render the common conditions tab.

    Args:
        health_data_manager: The health data manager instance
    """
    st.subheader("Common Health Conditions")

    try:
        # Get all conditions
        conditions = health_data_manager.get_all_conditions()

        if not conditions:
            st.info("No condition information available.")
            return

        # Create a grid of condition cards
        cols = st.columns(3)

        for i, condition in enumerate(conditions):
            col_idx = i % 3
            condition_id = condition["id"]
            condition_data = health_data_manager.get_condition_info(condition_id)

            if not condition_data:
                continue

            with cols[col_idx]:
                st.markdown(f"""
                <div class="symptom-card">
                    <h3>{condition_data["name"]}</h3>
                    <p>{condition_data["description"]}</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Learn more about {condition_data['name']}", key=f"learn_{condition_id}"):
                    _display_condition_details(health_data_manager, condition_id, condition_data)
    except Exception as e:
        logger.error("Error rendering common conditions: %s", str(e))
        st.error(f"Error loading condition information: {str(e)}")

def _display_condition_details(health_data_manager, condition_id, condition_data) -> None:
    """
    Display detailed information about a condition.

    Args:
        health_data_manager: The health data manager instance
        condition_id: ID of the condition to display
        condition_data: Condition data dictionary
    """
    st.markdown(f"### {condition_data['name']}")
    st.markdown(f"**Description:** {condition_data['description']}")

    if "typical_duration" in condition_data:
        st.markdown(f"**Typical Duration:** {condition_data['typical_duration']}")

    if "treatment" in condition_data:
        st.markdown(f"**Treatment:** {condition_data['treatment']}")

    if "when_to_see_doctor" in condition_data:
        st.markdown(f"**When to See a Doctor:** {condition_data['when_to_see_doctor']}")

    # Display associated symptoms
    if "symptoms" in condition_data:
        st.markdown("#### Common Symptoms")
        for symptom_id in condition_data["symptoms"]:
            symptom_info = health_data_manager.get_symptom_info(symptom_id)
            if symptom_info:
                st.markdown(f"- **{symptom_info['name']}**: {symptom_info['description']}")

    # Display related literature
    literature = health_data_manager.get_medical_literature(condition_id)
    if literature:
        st.markdown("#### Related Medical Literature")
        for article in literature:
            st.markdown(f"""
            <div class="citation">
                <strong>{article["title"]}</strong><br>
                {article["journal"]} ({article["year"]})<br>
                <em>Summary:</em> {article["summary"]}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific medical literature available for this condition in our database.")

def _render_recent_research() -> None:
    """Render the recent research tab."""
    st.subheader("Recent Medical Research")
    st.markdown("This section displays summaries of recent medical research papers.")

    # Mock recent research papers (in a real app, this would come from an API)
    recent_papers = [
        {
            "title": "Advances in Treatment Options for Migraines",
            "journal": "Neurology Today",
            "year": 2023,
            "summary": "This review examines recent developments in migraine treatments, including CGRP antagonists and neuromodulation devices."
        },
        {
            "title": "Long-term Effects of COVID-19: A Systematic Review",
            "journal": "Journal of Infectious Diseases",
            "year": 2023,
            "summary": "This systematic review analyzes the growing body of evidence regarding long-term effects of COVID-19, including neurological, cardiovascular, and respiratory sequelae."
        },
        {
            "title": "Gut Microbiome and Its Impact on Mental Health",
            "journal": "Psychological Medicine",
            "year": 2023,
            "summary": "This study investigates the bidirectional relationship between gut microbiota and mental health conditions, finding significant associations with depression and anxiety disorders."
        },
        {
            "title": "Artificial Intelligence in Early Cancer Detection",
            "journal": "Nature Medicine",
            "year": 2023,
            "summary": "This paper reviews the current state of AI applications in cancer screening, showing promising results for improving early detection rates while reducing false positives."
        }
    ]

    for paper in recent_papers:
        st.markdown(f"""
        <div class="citation">
            <strong>{paper["title"]}</strong><br>
            {paper["journal"]} ({paper["year"]})<br>
            <em>Summary:</em> {paper["summary"]}
        </div>
        """, unsafe_allow_html=True)

        # Add option to get full information
        if st.button(f"Get more details on {paper['title']}", key=f"details_{paper['title'][:20]}"):
            st.info("In a production version, this would connect to a medical literature database API to retrieve the full article details and generate a more comprehensive summary.")

def _render_search_interface(health_data_manager) -> None:
    """
    Render the search interface tab.

    Args:
        health_data_manager: The health data manager instance
    """
    st.subheader("Search Medical Literature")

    search_query = st.text_input("Enter keywords to search:", placeholder="e.g., migraine treatment")

    advanced_options = st.expander("Advanced Search Options")
    with advanced_options:
        col1, col2 = st.columns(2)
        with col1:
            year_range = st.slider("Publication Year", 2000, 2023, (2018, 2023))
            study_types = st.multiselect(
                "Study Types",
                ["Randomized Controlled Trial", "Meta-Analysis", "Systematic Review", "Cohort Study", "Case Report", "Clinical Trial"],
                ["Randomized Controlled Trial", "Meta-Analysis", "Systematic Review"]
            )
        with col2:
            journals = st.multiselect(
                "Journals",
                ["New England Journal of Medicine", "JAMA", "The Lancet", "BMJ", "Nature Medicine", "All Journals"],
                ["All Journals"]
            )
            sort_by = st.selectbox(
                "Sort Results By",
                ["Relevance", "Publication Date (Newest First)", "Citation Count", "Impact Factor"]
            )

    if st.button("Search", key="search_literature"):
        _process_literature_search(health_data_manager, search_query, year_range, study_types, journals, sort_by)

def _process_literature_search(health_data_manager, search_query, year_range, study_types, journals, sort_by) -> None:
    """
    Process a medical literature search query.

    Args:
        health_data_manager: The health data manager instance
        search_query: The search query string
        year_range: Range of publication years to include
        study_types: Types of studies to include
        journals: Journals to include
        sort_by: How to sort the results
    """
    try:
        if not search_query:
            st.warning("Please enter search keywords.")
            return

        st.info("In a production version, this would search through medical databases. For this demo, we'll simulate results.")

        st.markdown(f"### Search Results for '{search_query}'")

        # Use search method from health data manager if available
        if hasattr(health_data_manager, 'search_medical_database'):
            search_results = health_data_manager.search_medical_database(search_query)

            if search_results:
                # Display literature results
                if "literature" in search_results and search_results["literature"]:
                    for article in search_results["literature"]:
                        st.markdown(f"""
                        <div class="citation">
                            <strong>{article["title"]}</strong><br>
                            {article["journal"]} ({article["year"]})<br>
                            <em>Summary:</em> {article["summary"]}
                        </div>
                        """, unsafe_allow_html=True)

                # Display condition results
                if "conditions" in search_results and search_results["conditions"]:
                    st.markdown("### Related Conditions")
                    for condition in search_results["conditions"]:
                        st.markdown(f"""
                        <div class="symptom-card">
                            <h4>{condition["name"]}</h4>
                            <p>{condition["description"]}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # If no results in any category
                if not any(search_results.values()):
                    st.info("No results found for your search query. Try different keywords.")
            else:
                st.info("No results found for your search query. Try different keywords.")
        else:
            # Fallback for demo purposes
            if "migraine" in search_query.lower():
                literature = health_data_manager.get_medical_literature("migraine")
                if literature:
                    for article in literature:
                        st.markdown(f"""
                        <div class="citation">
                            <strong>{article["title"]}</strong><br>
                            {article["journal"]} ({article["year"]})<br>
                            <em>Summary:</em> {article["summary"]}
                        </div>
                        """, unsafe_allow_html=True)
            elif "covid" in search_query.lower():
                literature = health_data_manager.get_medical_literature("covid19")
                if literature:
                    for article in literature:
                        st.markdown(f"""
                        <div class="citation">
                            <strong>{article["title"]}</strong><br>
                            {article["journal"]} ({article["year"]})<br>
                            <em>Summary:</em> {article["summary"]}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No results found for your search query. Try different keywords.")
    except Exception as e:
        logger.error("Error processing literature search: %s", str(e))
        st.error(f"Error processing search: {str(e)}")

def _render_personalized_recommendations(health_data_manager, user_manager) -> None:
    """
    Render personalized literature recommendations.

    Args:
        health_data_manager: The health data manager instance
        user_manager: The user profile manager instance
    """
    st.subheader("Personalized Literature Recommendations")

    st.markdown("""
    This feature provides medical literature recommendations based on your health profile and symptom history.
    The AI analyzes your health data to suggest relevant research that may be of interest to you.
    """)

    if not user_manager or not user_manager.health_history:
        st.info("Add some health history data first to get personalized recommendations.")
        return

    try:
        # Get the most common symptoms from history
        symptom_counts = {}
        for entry in user_manager.health_history:
            for symptom in entry.get("symptoms", []):
                if symptom not in symptom_counts:
                    symptom_counts[symptom] = 0
                symptom_counts[symptom] += 1

        if not symptom_counts:
            st.info("No symptom history found. Please use the Symptom Analyzer to track symptoms.")
            return

        # Sort by frequency
        sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
        top_symptoms = [symptom for symptom, _ in sorted_symptoms[:3]]

        # Get conditions related to top symptoms
        related_conditions = set()
        for symptom in top_symptoms:
            related = health_data_manager.find_conditions_by_symptom(symptom)
            for condition in related:
                related_conditions.add(condition["id"])

        if related_conditions:
            st.markdown("### Literature Recommendations Based on Your Health History")

            for condition_id in related_conditions:
                condition_info = health_data_manager.get_condition_info(condition_id)
                if not condition_info:
                    continue

                condition_name = condition_info["name"]
                st.markdown(f"#### Research Related to {condition_name}")

                literature = health_data_manager.get_medical_literature(condition_id)
                if literature:
                    for article in literature:
                        st.markdown(f"""
                        <div class="citation">
                            <strong>{article["title"]}</strong><br>
                            {article["journal"]} ({article["year"]})<br>
                            <em>Summary:</em> {article["summary"]}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"No literature available for {condition_name} in our database.")
        else:
            st.info("Not enough health data to generate personalized recommendations.")
    except Exception as e:
        logger.error("Error generating personalized recommendations: %s", str(e))
        st.error(f"Error generating recommendations: {str(e)}")
