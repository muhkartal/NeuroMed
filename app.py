"""
Main Application Module for MedExplain AI
Integrates all components into a complete application
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import time
from pathlib import Path
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Import custom modules
from machine_learning import SymptomPredictor, NLPSymptomExtractor, PatientRiskAssessor
from dashboard import HealthDashboard, render_predictive_insights
from nlp_interface import NaturalLanguageInterface, ChatInterface
from data_analytics import HealthDataAnalyzer, render_advanced_analytics

# Import original app components
from app import HealthDataManager, UserProfileManager, OpenAIClient

# Constants
APP_VERSION = "2.0.0"

# Configure page settings
st.set_page_config(
    page_title="MedExplain AI Pro - Personal Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
def local_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Try to load the CSS file, create if it doesn't exist
try:
    local_css()
except FileNotFoundError:
    # Write a basic CSS if the file doesn't exist
    with open("style.css", "w") as f:
        f.write("""
        /* Main Styling */
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8f9fa;
        }

        /* Header Styling */
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }

        /* Component Styling */
        .main-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .symptom-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 1.2rem;
            margin-bottom: 1rem;
            border-left: 4px solid #3498db;
        }
        """)
    local_css()

class MedExplainApp:
    """Main application class for MedExplain AI."""

    def __init__(self):
        """Initialize the application."""
        # Initialize the core components
        self.openai_client = OpenAIClient()
        self.health_data = HealthDataManager()
        self.user_manager = UserProfileManager()

        # Initialize additional components
        self.nlp_interface = NaturalLanguageInterface()
        self.chat_interface = ChatInterface(self.nlp_interface)

        # Initialize ML components
        self.symptom_predictor = SymptomPredictor()
        self.symptom_extractor = NLPSymptomExtractor()
        self.risk_assessor = PatientRiskAssessor()

        # Initialize analysis components
        self.health_analyzer = HealthDataAnalyzer(
            self.user_manager.health_history,
            self.user_manager.profile
        )
        self.health_dashboard = HealthDashboard(
            self.user_manager.health_history,
            self.user_manager.profile
        )

        # Set up API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            self.openai_client.set_api_key(api_key)
            self.nlp_interface.set_api_key(api_key)

    def render_sidebar(self):
        """Render the application sidebar."""
        # Display logo and title
        st.sidebar.image("https://via.placeholder.com/150x150.png?text=MedExplain+AI", width=150)
        st.sidebar.title("MedExplain AI Pro")
        st.sidebar.caption(f"Version {APP_VERSION}")

        # Main navigation
        st.sidebar.subheader("Navigation")
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

        choice = st.sidebar.radio("Go to", menu_options)

        # Additional sidebar components
        st.sidebar.markdown("---")

        # Quick health check button
        if st.sidebar.button("🔍 Quick Symptom Check", key="quick_check"):
            st.session_state.page = "Symptom Analyzer"
            return "Symptom Analyzer"

        # Display user profile summary if available
        if any(self.user_manager.profile.values()):
            st.sidebar.markdown("---")
            st.sidebar.subheader("User Profile")

            if self.user_manager.profile.get("name"):
                st.sidebar.markdown(f"**Name:** {self.user_manager.profile['name']}")

            if self.user_manager.profile.get("age"):
                st.sidebar.markdown(f"**Age:** {self.user_manager.profile['age']}")

            if self.user_manager.profile.get("gender"):
                st.sidebar.markdown(f"**Gender:** {self.user_manager.profile['gender']}")

        # Display disclaimer
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**MEDICAL DISCLAIMER:** This application is for educational purposes only. "
            "It does not provide medical advice. Always consult a healthcare professional "
            "for medical concerns."
        )

        return choice

    def run(self):
        """Run the main application."""
        # Configure the page based on the selected menu option
        selected_page = self.render_sidebar()

        # Check session state for page overrides
        if hasattr(st.session_state, 'page') and st.session_state.page != selected_page:
            selected_page = st.session_state.page
            st.session_state.page = None  # Reset for next time

        # Render the selected page
        self.render_page(selected_page)

    def render_page(self, page):
        """Render the selected page."""
        if page == "Home":
            self.render_home()
        elif page == "Symptom Analyzer":
            self.render_symptom_analyzer()
        elif page == "Health Dashboard":
            self.render_health_dashboard()
        elif page == "Advanced Analytics":
            self.render_advanced_analytics()
        elif page == "Medical Literature":
            self.render_medical_literature()
        elif page == "Health Chat":
            self.render_health_chat()
        elif page == "Health History":
            self.render_health_history()
        elif page == "Settings":
            self.render_settings()

    def render_home(self):
        """Render the home page."""
        st.title("Welcome to MedExplain AI Pro")
        st.subheader("Your advanced personal health assistant powered by artificial intelligence")

        # Display main features in columns
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("""
            ### Advanced Healthcare Analytics at Your Fingertips

            MedExplain AI Pro combines medical knowledge with artificial intelligence to help you:

            - **Analyze your symptoms** with machine learning models
            - **Track your health** over time with interactive visualizations
            - **Uncover patterns** in your symptoms and health data
            - **Identify potential risks** based on your health profile
            - **Chat naturally** about medical topics with our AI assistant
            - **Access medical literature** summarized in plain language
            """)

            # Feature highlights
            st.markdown("### Key Advanced Features")

            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:
                st.markdown("""
                🧠 **AI-Powered Analysis**
                Machine learning models analyze your health data

                📊 **Health Dashboard**
                Interactive visualizations of your health patterns

                🔍 **Pattern Recognition**
                Identify trends and correlations in your symptoms
                """)

            with feature_col2:
                st.markdown("""
                💬 **Natural Language Interface**
                Discuss your health in plain language

                📈 **Predictive Insights**
                Anticipate potential health trends

                📱 **Comprehensive Tracking**
                Monitor your health journey over time
                """)

        with col2:
            # Hero image
            st.image("https://via.placeholder.com/500x300.png?text=Advanced+Health+Analytics", width=500)

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

        recent_checks = self.user_manager.get_recent_symptom_checks(limit=3)

        if not recent_checks:
            st.info("No recent health activity. Start by analyzing your symptoms or setting up your health profile.")
        else:
            for check in recent_checks:
                date = check.get("date", "")
                symptoms = check.get("symptoms", [])

                st.markdown(f"""
                <div class="symptom-card">
                    <h4>Check from {date}</h4>
                    <p><strong>Symptoms:</strong> {", ".join(symptoms)}</p>
                </div>
                """, unsafe_allow_html=True)

        # Call to action and medical disclaimer
        st.markdown("---")
        st.warning("""
        **Medical Disclaimer:** MedExplain AI Pro is not a substitute for professional medical advice,
        diagnosis, or treatment. Always seek the advice of your physician or other qualified health
        provider with any questions you may have regarding a medical condition.
        """)

    def render_symptom_analyzer(self):
        """Render the symptom analyzer page with enhanced ML capabilities."""
        st.title("AI-Enhanced Symptom Analyzer")
        st.markdown("Describe your symptoms to get AI-powered analysis of possible conditions. Remember, this is not a diagnosis.")

        # Create tabs for different input methods
        tabs = st.tabs(["Selection Method", "Text Description", "Voice Input (Beta)"])

        with tabs[0]:
            # This is the original selection-based method from app.py
            # Create a list of symptoms from our database
            symptom_options = []
            for symptom_id, symptom_data in self.health_data.symptoms_db.items():
                symptom_options.append({"id": symptom_id, "name": symptom_data["name"]})

            # Sort alphabetically by name
            symptom_options = sorted(symptom_options, key=lambda x: x["name"])

            # Multi-select for symptoms
            selected_symptoms = st.multiselect(
                "Select your symptoms:",
                options=[s["id"] for s in symptom_options],
                format_func=lambda x: next(s["name"] for s in symptom_options if s["id"] == x)
            )

            # Additional information
            st.markdown("### Additional Information")
            cols = st.columns(3)

            with cols[0]:
                symptom_duration = st.selectbox(
                    "How long have you had these symptoms?",
                    ["Less than 24 hours", "1-3 days", "3-7 days", "1-2 weeks", "More than 2 weeks"]
                )

            with cols[1]:
                symptom_severity = st.select_slider(
                    "Rate the severity of your symptoms:",
                    options=["Mild", "Moderate", "Severe"]
                )

            with cols[2]:
                had_fever = st.checkbox("Do you have a fever?")
                if had_fever:
                    fever_temp = st.number_input("Temperature (°F):", min_value=96.0, max_value=107.0, value=99.5, step=0.1)

            # Analyze button
            if st.button("Analyze Symptoms", key="analyze_symptoms_selection"):
                if not selected_symptoms:
                    st.warning("Please select at least one symptom.")
                else:
                    with st.spinner("Analyzing your symptoms with advanced AI models..."):
                        # Get analysis results from original method
                        analysis_results = self.health_data.analyze_symptoms(selected_symptoms)

                        # Enhance with ML predictions if available
                        try:
                            # Prepare data for ML model
                            # This is a simplified version, in production this would use a trained model
                            ml_input = {
                                "symptoms": {symptom: 1 for symptom in selected_symptoms}
                            }

                            # Add demographic data if available
                            if self.user_manager.profile:
                                ml_input["user_data"] = {
                                    "age": self.user_manager.profile.get("age", 35),  # Default to 35 if missing
                                    "gender_male": 1 if self.user_manager.profile.get("gender") == "Male" else 0
                                }

                            # Get risk assessment
                            risk_assessment = self.risk_assessor.assess_risk({
                                "symptoms": selected_symptoms,
                                "symptom_duration": symptom_duration,
                                "symptom_severity": symptom_severity,
                                "age": self.user_manager.profile.get("age", 35),
                                "chronic_conditions": self.user_manager.profile.get("chronic_conditions", [])
                            })
                        except Exception as e:
                            st.error(f"Error in ML analysis: {str(e)}")
                            risk_assessment = None

                        # Save to history with enhanced data
                        history_entry = {
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symptoms": selected_symptoms,
                            "analysis_results": analysis_results,
                            "symptom_duration": symptom_duration,
                            "symptom_severity": symptom_severity,
                            "had_fever": had_fever,
                            "fever_temp": fever_temp if had_fever else None
                        }

                        # Add risk assessment if available
                        if risk_assessment:
                            history_entry["risk_assessment"] = risk_assessment

                        self.user_manager.health_history.append(history_entry)
                        self.user_manager.save_health_history()

                        # Display results
                        self._display_symptom_analysis_results(analysis_results, risk_assessment)

                        # Add AI-enhanced explanation (using OpenAI)
                        if self.openai_client.api_key and selected_symptoms:
                            self._display_ai_explanation(
                                selected_symptoms,
                                symptom_options,
                                symptom_duration,
                                symptom_severity,
                                had_fever,
                                fever_temp if had_fever else None
                            )

        with tabs[1]:
            # Text-based input using NLP
            st.markdown("### Describe Your Symptoms in Natural Language")
            st.markdown("""
            Tell us about your symptoms in your own words. Our AI will analyze your description
            and identify potential symptoms and conditions.
            """)

            symptom_text = st.text_area(
                "Describe your symptoms:",
                placeholder="Example: I've been having a headache for the past 3 days, along with a sore throat and runny nose...",
                height=150
            )

            if st.button("Analyze Description", key="analyze_symptoms_text"):
                if not symptom_text or len(symptom_text) < 10:
                    st.warning("Please provide a more detailed description of your symptoms.")
                else:
                    with st.spinner("Analyzing your symptom description with NLP..."):
                        try:
                            # Extract symptoms using NLP
                            symptom_extractor = NLPSymptomExtractor()
                            symptom_extractor.load_symptom_dictionary("data/medical_data.json")
                            extracted_symptoms = symptom_extractor.extract_symptoms(symptom_text)

                            if not extracted_symptoms:
                                st.warning("No specific symptoms could be identified in your description. Please try again with more details or use the selection method.")
                            else:
                                # Display extracted symptoms
                                st.markdown("### Identified Symptoms")

                                for symptom in extracted_symptoms:
                                    confidence = symptom.get("confidence", 0) * 100
                                    st.markdown(f"- **{symptom['name']}** (Confidence: {confidence:.1f}%)")

                                # Get symptom IDs for analysis
                                symptom_ids = [symptom["symptom_id"] for symptom in extracted_symptoms]

                                # Analyze symptoms
                                analysis_results = self.health_data.analyze_symptoms(symptom_ids)

                                # Save to history
                                history_entry = {
                                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "symptoms": symptom_ids,
                                    "symptom_text": symptom_text,
                                    "analysis_results": analysis_results
                                }

                                self.user_manager.health_history.append(history_entry)
                                self.user_manager.save_health_history()

                                # Display results
                                self._display_symptom_analysis_results(analysis_results, None)

                                # Add AI-enhanced explanation (using OpenAI)
                                if self.openai_client.api_key:
                                    try:
                                        # Create the prompt
                                        prompt = f"""
                                        A user has described the following symptoms:
                                        "{symptom_text}"

                                        Based on this description, provide a clear, concise explanation of:
                                        1. What could potentially be causing these symptoms (2-3 possibilities)
                                        2. What these symptoms typically mean in the body
                                        3. General self-care recommendations
                                        4. When they should definitely see a doctor

                                        Keep your answer factual, educational, and concise. Include a disclaimer about consulting a healthcare professional.
                                        """

                                        # Get response from OpenAI
                                        ai_response = self.openai_client.generate_response(prompt)

                                        if ai_response:
                                            st.markdown("## AI-Enhanced Explanation")
                                            st.markdown(ai_response)
                                    except Exception as e:
                                        st.error(f"Error generating AI explanation: {str(e)}")
                        except Exception as e:
                            st.error(f"Error analyzing symptom description: {str(e)}")

        with tabs[2]:
            # Voice input (placeholder, would require additional libraries in production)
            st.markdown("### Voice Input for Symptom Description")
            st.markdown("""
            This feature allows you to describe your symptoms using your voice.
            Click the button below and speak clearly about your symptoms.
            """)

            st.info("🚧 Voice input feature is currently in beta. In a production version, this would use speech-to-text technology to capture your symptom description.")

            if st.button("Start Voice Recording (Demo)", key="voice_record"):
                with st.spinner("Recording... (simulated)"):
                    # Simulate a delay for recording
                    time.sleep(2)

                st.success("Recording complete! (simulated)")

                # Simulated transcription
                sample_transcription = "I've been having a headache and sore throat for the past two days, and I'm feeling very tired."

                st.markdown("### Transcription")
                st.text_area("Your spoken description:", value=sample_transcription, height=100, disabled=True)

                if st.button("Analyze Voice Description", key="analyze_voice"):
                    st.info("This would analyze the transcribed text using the same NLP methods as the text description tab.")
                    st.markdown("For now, please use the Text Description tab to enter your symptoms.")

    def _display_symptom_analysis_results(self, analysis_results, risk_assessment=None):
        """Display the results of the symptom analysis."""
        st.markdown("## Analysis Results")

        if not analysis_results:
            st.info("No specific conditions match your symptoms in our database. Please consult a healthcare professional for proper evaluation.")
        else:
            # Display disclaimer
            st.markdown("""
            <div class="disclaimer">
                <strong>Medical Disclaimer:</strong> This information is for educational purposes only and
                is not a substitute for professional medical advice. The confidence levels shown are
                based on symptom matching and do not constitute a diagnosis. Please consult a healthcare
                professional for proper evaluation.
            </div>
            """, unsafe_allow_html=True)

            # Show risk level if available
            if risk_assessment:
                risk_level = risk_assessment.get("risk_level", "unknown")
                risk_score = risk_assessment.get("risk_score", 0)

                risk_class = "severity-low"
                if risk_level == "medium":
                    risk_class = "severity-medium"
                elif risk_level == "high":
                    risk_class = "severity-high"

                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h3>Risk Assessment</h3>
                    <p><strong>Risk Level:</strong> <span class="{risk_class}">{risk_level.capitalize()}</span></p>
                    <p><strong>Risk Score:</strong> {risk_score}</p>
                </div>
                """, unsafe_allow_html=True)

                # Show risk factors
                if "risk_factors" in risk_assessment and risk_assessment["risk_factors"]:
                    st.markdown("### Risk Factors")
                    for factor in risk_assessment["risk_factors"]:
                        st.markdown(f"- {factor}")

                # Show recommendations
                if "recommendations" in risk_assessment and risk_assessment["recommendations"]:
                    st.markdown("### Recommendations")
                    for rec in risk_assessment["recommendations"]:
                        st.markdown(f"- {rec}")

            # Show each potential condition
            st.markdown("### Potential Conditions")

            for condition_id, condition_data in analysis_results.items():
                condition_name = condition_data["name"]
                confidence = condition_data["confidence"]
                severity = condition_data["severity"]

                # Determine severity class for styling
                severity_class = "severity-low"
                if severity == "medium":
                    severity_class = "severity-medium"
                elif severity == "high":
                    severity_class = "severity-high"

                st.markdown(f"""
                <div class="symptom-card">
                    <h3>{condition_name}</h3>
                    <p><strong>Match confidence:</strong> {confidence}%</p>
                    <p><strong>Severity level:</strong> <span class="{severity_class}">{severity.capitalize()}</span></p>
                    <p><strong>Description:</strong> {condition_data["description"]}</p>
                    <p><strong>Treatment:</strong> {condition_data["treatment"]}</p>
                    <p><strong>When to see a doctor:</strong> {condition_data["when_to_see_doctor"]}</p>
                </div>
                """, unsafe_allow_html=True)

                # Get and display medical literature
                literature = self.health_data.get_medical_literature(condition_id)
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

    def _display_ai_explanation(self, selected_symptoms, symptom_options, duration, severity, had_fever, fever_temp):
        """Display an AI-enhanced explanation of symptoms."""
        try:
            symptoms_text = ", ".join([
                next(s["name"] for s in symptom_options if s["id"] == symptom_id)
                for symptom_id in selected_symptoms
            ])

            # Create the prompt
            prompt = f"""
            I'm experiencing the following symptoms: {symptoms_text}.
            Duration: {duration}
            Severity: {severity}
            {'Fever: Yes, ' + str(fever_temp) + '°F' if had_fever else 'Fever: No'}

            Based on these symptoms, provide a clear, concise explanation of:
            1. What could potentially be causing these symptoms (2-3 possibilities)
            2. What these symptoms typically mean in the body
            3. General self-care recommendations
            4. When I should definitely see a doctor

            Keep your answer factual, educational, and concise. Include a disclaimer about consulting a healthcare professional.
            """

            # Get response from OpenAI
            ai_response = self.openai_client.generate_response(prompt)

            if ai_response:
                st.markdown("## AI-Enhanced Explanation")
                st.markdown(ai_response)
        except Exception as e:
            st.error(f"Error generating AI explanation: {str(e)}")

    def render_health_dashboard(self):
        """Render the health dashboard page."""
        # Use the dashboard module to render the dashboard
        self.health_dashboard = HealthDashboard(
            self.user_manager.health_history,
            self.user_manager.profile
        )
        self.health_dashboard.render_dashboard()

    def render_advanced_analytics(self):
        """Render the advanced analytics page."""
        # Update the health analyzer with the latest data
        self.health_analyzer = HealthDataAnalyzer(
            self.user_manager.health_history,
            self.user_manager.profile
        )

        # Use the data analytics module to render the analytics
        render_advanced_analytics(self.health_analyzer)

    def render_medical_literature(self):
        """Render the medical literature page."""
        st.title("Medical Literature")
        st.markdown("Explore summaries of medical research relevant to various health conditions.")

        # Create tabs for different categories
        tabs = st.tabs(["Common Conditions", "Recent Research", "Search", "Personalized Recommendations"])

        with tabs[0]:
            st.subheader("Common Health Conditions")

            # Create a grid of condition cards
            cols = st.columns(3)

            for i, (condition_id, condition_data) in enumerate(self.health_data.conditions_db.items()):
                col_idx = i % 3

                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="symptom-card">
                        <h3>{condition_data["name"]}</h3>
                        <p>{condition_data["description"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Learn more about {condition_data['name']}", key=f"learn_{condition_id}"):
                        # Display condition details
                        st.markdown(f"### {condition_data['name']}")
                        st.markdown(f"**Description:** {condition_data['description']}")
                        st.markdown(f"**Typical Duration:** {condition_data['typical_duration']}")
                        st.markdown(f"**Treatment:** {condition_data['treatment']}")
                        st.markdown(f"**When to See a Doctor:** {condition_data['when_to_see_doctor']}")

                        # Display associated symptoms
                        st.markdown("#### Common Symptoms")
                        for symptom_id in condition_data["symptoms"]:
                            symptom_info = self.health_data.get_symptom_info(symptom_id)
                            if symptom_info:
                                st.markdown(f"- **{symptom_info['name']}**: {symptom_info['description']}")

                        # Display related literature
                        literature = self.health_data.get_medical_literature(condition_id)
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

        with tabs[1]:
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

        with tabs[2]:
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
                st.info("In a production version, this would search through medical databases. For this demo, we'll simulate results.")

                if search_query:
                    st.markdown(f"### Search Results for '{search_query}'")

                    if "migraine" in search_query.lower():
                        literature = self.health_data.get_medical_literature("migraine")
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
                        literature = self.health_data.get_medical_literature("covid19")
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
                else:
                    st.warning("Please enter search keywords.")

        with tabs[3]:
            st.subheader("Personalized Literature Recommendations")

            st.markdown("""
            This feature provides medical literature recommendations based on your health profile and symptom history.
            The AI analyzes your health data to suggest relevant research that may be of interest to you.
            """)

            if not self.user_manager.health_history:
                st.info("Add some health history data first to get personalized recommendations.")
            else:
                # Get the most common symptoms from history
                symptom_counts = {}
                for entry in self.user_manager.health_history:
                    for symptom in entry.get("symptoms", []):
                        if symptom not in symptom_counts:
                            symptom_counts[symptom] = 0
                        symptom_counts[symptom] += 1

                # Sort by frequency
                sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
                top_symptoms = [symptom for symptom, _ in sorted_symptoms[:3]]

                # Get conditions related to top symptoms
                related_conditions = set()
                for symptom in top_symptoms:
                    for condition_id, condition_data in self.health_data.conditions_db.items():
                        if symptom in condition_data.get("symptoms", []):
                            related_conditions.add(condition_id)

                if related_conditions:
                    st.markdown("### Literature Recommendations Based on Your Health History")

                    for condition_id in related_conditions:
                        condition_name = self.health_data.conditions_db[condition_id]["name"]
                        st.markdown(f"#### Research Related to {condition_name}")

                        literature = self.health_data.get_medical_literature(condition_id)
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

    def render_health_chat(self):
        """Render the health chat page."""
        # Update the OpenAI API key for the NLP interface
        if self.openai_client.api_key:
            self.nlp_interface.set_api_key(self.openai_client.api_key)

        # Use the chat interface module to render the chat
        self.chat_interface.render(
            self.user_manager.profile,
            self.health_data
        )

    def render_health_history(self):
        """Render the health history page."""
        st.title("Health History")
        st.markdown("View and track your symptom history over time.")

        # Get health history
        history = self.user_manager.health_history

        if not history:
            st.info("No health history available yet. Use the Symptom Analyzer to start tracking your symptoms.")
        else:
            # Create tabs for different views
            tabs = st.tabs(["Timeline", "Symptom Trends", "Detailed History", "Export Data"])

            with tabs[0]:
                st.subheader("Timeline View")

                # Display recent entries in reverse chronological order
                for entry in reversed(history):
                    date = entry.get("date", "")
                    symptoms = entry.get("symptoms", [])
                    results = entry.get("analysis_results", {})

                    # Get symptom names
                    symptom_names = []
                    for symptom_id in symptoms:
                        symptom_info = self.health_data.get_symptom_info(symptom_id)
                        if symptom_info:
                            symptom_names.append(symptom_info["name"])

                    st.markdown(f"""
                    <div class="symptom-card">
                        <h4>Check from {date}</h4>
                        <p><strong>Symptoms:</strong> {", ".join(symptom_names)}</p>
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

            with tabs[1]:
                # Use the health dashboard to render symptom trends
                self.health_dashboard = HealthDashboard(history, self.user_manager.profile)

                # Prepare data for trends chart
                symptom_counts = {}
                dates = []

                for entry in history:
                    dates.append(entry["date"])
                    for symptom_id in entry["symptoms"]:
                        symptom_info = self.health_data.get_symptom_info(symptom_id)
                        if symptom_info:
                            symptom_name = symptom_info["name"]
                            if symptom_name not in symptom_counts:
                                symptom_counts[symptom_name] = 0
                            symptom_counts[symptom_name] += 1

                # Create a bar chart of symptom frequency
                symptom_df = pd.DataFrame({
                    "Symptom": list(symptom_counts.keys()),
                    "Count": list(symptom_counts.values())
                })

                if not symptom_df.empty:
                    # Sort by count descending
                    symptom_df = symptom_df.sort_values("Count", ascending=False)

                    fig = px.bar(
                        symptom_df,
                        x="Symptom",
                        y="Count",
                        title="Symptom Frequency",
                        color="Count",
                        color_continuous_scale="Blues",
                    )

                    fig.update_layout(
                        xaxis_title="Symptom",
                        yaxis_title="Number of Occurrences",
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Create a time-based visualization
                    st.subheader("Symptoms Over Time")

                    # This would ideally use the more sophisticated visualizations from the dashboard component
                    st.info("For a more detailed breakdown of your symptom patterns over time, visit the Health Dashboard page.")
                else:
                    st.info("Not enough data to display trends yet.")

            with tabs[2]:
                st.subheader("Detailed History")

                # Create expandable sections for each entry
                for i, entry in enumerate(reversed(history)):
                    date = entry.get("date", "")
                    symptoms = entry.get("symptoms", [])
                    results = entry.get("analysis_results", {})

                    # Get symptom names
                    symptom_names = []
                    for symptom_id in symptoms:
                        symptom_info = self.health_data.get_symptom_info(symptom_id)
                        if symptom_info:
                            symptom_names.append(symptom_info["name"])

                    with st.expander(f"Check from {date}"):
                        st.markdown(f"**Symptoms:** {', '.join(symptom_names)}")

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

            with tabs[3]:
                st.subheader("Export Health Data")

                st.markdown("""
                You can export your health history data in various formats for your records or to share with healthcare providers.
                """)

                export_format = st.selectbox(
                    "Export Format",
                    ["JSON", "CSV", "PDF Report"]
                )

                if st.button("Export Data"):
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

                    elif export_format == "CSV":
                        # Create a simplified DataFrame for CSV export
                        csv_data = []

                        for entry in history:
                            date = entry.get("date", "")
                            symptoms = entry.get("symptoms", [])

                            # Get symptom names
                            symptom_names = []
                            for symptom_id in symptoms:
                                symptom_info = self.health_data.get_symptom_info(symptom_id)
                                if symptom_info:
                                    symptom_names.append(symptom_info["name"])

                            # Get top condition if available
                            top_condition = "None"
                            top_confidence = 0

                            if "analysis_results" in entry and entry["analysis_results"]:
                                results = entry["analysis_results"]
                                top_condition_id = next(iter(results))
                                top_condition = results[top_condition_id]["name"]
                                top_confidence = results[top_condition_id]["confidence"]

                            csv_data.append({
                                "Date": date,
                                "Symptoms": ", ".join(symptom_names),
                                "Duration": entry.get("symptom_duration", ""),
                                "Severity": entry.get("symptom_severity", ""),
                                "Top Condition": top_condition,
                                "Confidence": top_confidence,
                                "Risk Level": entry.get("risk_assessment", {}).get("risk_level", "").capitalize()
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

                    elif export_format == "PDF Report":
                        st.info("In a production version, this would generate a comprehensive PDF report of your health history that you could share with healthcare providers.")

                # Option to delete history
                st.markdown("---")
                with st.expander("Delete History Data"):
                    st.warning("This action will permanently delete your health history. This cannot be undone.")
                    if st.button("Delete All Health History", key="delete_history"):
                        # Clear history
                        self.user_manager.health_history = []
                        self.user_manager.save_health_history()
                        st.success("Health history has been deleted.")
                        st.experimental_rerun()

    def render_settings(self):
        """Render the settings page."""
        st.title("Settings")
        st.markdown("Configure your personal information and application settings.")

        # Create tabs for different settings sections
        tabs = st.tabs(["Profile", "API Configuration", "Preferences", "Data Management"])

        with tabs[0]:
            st.subheader("Personal Profile")
            st.markdown("This information helps provide more personalized health information.")

            col1, col2 = st.columns(2)

            with col1:
                self.user_manager.profile["name"] = st.text_input("Name", value=self.user_manager.profile.get("name", ""))
                self.user_manager.profile["age"] = st.number_input("Age", min_value=0, max_value=120, value=int(self.user_manager.profile.get("age", 0) or 0))
                self.user_manager.profile["gender"] = st.selectbox("Gender", ["", "Male", "Female", "Non-binary", "Prefer not to say"], index=0 if not self.user_manager.profile.get("gender") else ["", "Male", "Female", "Non-binary", "Prefer not to say"].index(self.user_manager.profile.get("gender")))

            with col2:
                self.user_manager.profile["height"] = st.text_input("Height", value=self.user_manager.profile.get("height", ""), placeholder="e.g., 5'10\" or 178 cm")
                self.user_manager.profile["weight"] = st.text_input("Weight", value=self.user_manager.profile.get("weight", ""), placeholder="e.g., 165 lbs or 75 kg")
                self.user_manager.profile["blood_type"] = st.selectbox("Blood Type", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], index=0 if not self.user_manager.profile.get("blood_type") else ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].index(self.user_manager.profile.get("blood_type", "")))

            st.subheader("Additional Health Information")

            # Add fields for additional information
            self.user_manager.profile["exercise_frequency"] = st.selectbox(
                "Exercise Frequency",
                ["", "Sedentary", "Light (1-2 days/week)", "Moderate (3-4 days/week)", "Active (5+ days/week)"],
                index=0 if not self.user_manager.profile.get("exercise_frequency") else ["", "Sedentary", "Light (1-2 days/week)", "Moderate (3-4 days/week)", "Active (5+ days/week)"].index(self.user_manager.profile.get("exercise_frequency", ""))
            )

            self.user_manager.profile["smoking_status"] = st.selectbox(
                "Smoking Status",
                ["", "Never smoked", "Former smoker", "Current smoker"],
                index=0 if not self.user_manager.profile.get("smoking_status") else ["", "Never smoked", "Former smoker", "Current smoker"].index(self.user_manager.profile.get("smoking_status", ""))
            )

            # Allergies
            st.subheader("Allergies")
            allergy_input = st.text_input("Add Allergy", key="add_allergy")
            if st.button("Add", key="add_allergy_btn") and allergy_input:
                if "allergies" not in self.user_manager.profile:
                    self.user_manager.profile["allergies"] = []
                if allergy_input not in self.user_manager.profile["allergies"]:
                    self.user_manager.profile["allergies"].append(allergy_input)

            if "allergies" in self.user_manager.profile and self.user_manager.profile["allergies"]:
                st.markdown("**Current Allergies:**")
                for i, allergy in enumerate(self.user_manager.profile["allergies"]):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"- {allergy}")
                    with col2:
                        if st.button("Remove", key=f"remove_allergy_{i}"):
                            self.user_manager.profile["allergies"].pop(i)
                            st.experimental_rerun()

            # Chronic conditions
            st.subheader("Chronic Conditions")
            condition_input = st.text_input("Add Chronic Condition", key="add_condition")
            if st.button("Add", key="add_condition_btn") and condition_input:
                if "chronic_conditions" not in self.user_manager.profile:
                    self.user_manager.profile["chronic_conditions"] = []
                if condition_input not in self.user_manager.profile["chronic_conditions"]:
                    self.user_manager.profile["chronic_conditions"].append(condition_input)

            if "chronic_conditions" in self.user_manager.profile and self.user_manager.profile["chronic_conditions"]:
                st.markdown("**Current Chronic Conditions:**")
                for i, condition in enumerate(self.user_manager.profile["chronic_conditions"]):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"- {condition}")
                    with col2:
                        if st.button("Remove", key=f"remove_condition_{i}"):
                            self.user_manager.profile["chronic_conditions"].pop(i)
                            st.experimental_rerun()

            # Medications
            st.subheader("Medications")
            medication_input = st.text_input("Add Medication", key="add_medication")
            if st.button("Add", key="add_medication_btn") and medication_input:
                if "medications" not in self.user_manager.profile:
                    self.user_manager.profile["medications"] = []
                if medication_input not in self.user_manager.profile["medications"]:
                    self.user_manager.profile["medications"].append(medication_input)

            if "medications" in self.user_manager.profile and self.user_manager.profile["medications"]:
                st.markdown("**Current Medications:**")
                for i, medication in enumerate(self.user_manager.profile["medications"]):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"- {medication}")
                    with col2:
                        if st.button("Remove", key=f"remove_medication_{i}"):
                            self.user_manager.profile["medications"].pop(i)
                            st.experimental_rerun()

            if st.button("Save Profile", key="save_profile"):
                self.user_manager.save_profile()
                st.success("Profile saved successfully!")

        with tabs[1]:
            st.subheader("API Configuration")
            st.markdown("Configure API keys for enhanced functionality.")

            # OpenAI API key
            openai_key = st.text_input(
                "OpenAI API Key",
                value=self.openai_client.api_key if self.openai_client.api_key else "",
                type="password",
                help="Enter your OpenAI API key to enable AI-powered symptom analysis."
            )

            if st.button("Save API Key", key="save_api_key"):
                if openai_key:
                    if self.openai_client.set_api_key(openai_key):
                        # Also update the NLP interface
                        self.nlp_interface.set_api_key(openai_key)
                        st.success("API key saved successfully!")
                    else:
                        st.error("Failed to save API key. Please try again.")
                else:
                    st.warning("Please enter an API key.")

            st.markdown("""
            **Note:** The OpenAI API key is used for advanced symptom analysis and explanation.
            If you don't have a key, the application will still function, but with limited capabilities.
            Your API key is stored locally and is never sent to our servers.
            """)

        with tabs[2]:
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

            # Set the dashboard theme
            if dashboard_theme != "Default":
                self.health_dashboard.set_theme(dashboard_theme.lower())

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
                st.success("Preferences saved successfully!")
                st.info("Note: Some preferences may require restarting the application to take effect.")

        with tabs[3]:
            st.subheader("Data Management")
            st.markdown("Manage your data stored in the application.")

            # Data backup
            st.markdown("### Backup Your Data")
            st.markdown("Create a backup of all your health data and settings.")

            if st.button("Create Backup", key="create_backup"):
                # Prepare data for backup
                backup_data = {
                    "profile": self.user_manager.profile,
                    "health_history": self.user_manager.health_history,
                    "app_version": APP_VERSION,
                    "backup_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Convert to JSON
                backup_json = json.dumps(backup_data, indent=4)

                # Create download button
                st.download_button(
                    label="Download Backup",
                    data=backup_json,
                    file_name=f"medexplain_backup_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

            # Data restore
            st.markdown("### Restore From Backup")
            st.markdown("Restore your data from a previous backup.")

            backup_file = st.file_uploader("Upload Backup File", type=["json"])

            if backup_file and st.button("Restore Data", key="restore_data"):
                try:
                    # Read the backup file
                    backup_data = json.load(backup_file)

                    # Validate backup data
                    if "profile" in backup_data and "health_history" in backup_data:
                        # Restore data
                        self.user_manager.profile = backup_data["profile"]
                        self.user_manager.health_history = backup_data["health_history"]

                        # Save restored data
                        self.user_manager.save_profile()
                        self.user_manager.save_health_history()

                        st.success("Data restored successfully!")
                        st.info("Please refresh the application to see the restored data.")
                    else:
                        st.error("Invalid backup file format. Could not restore data.")
                except Exception as e:
                    st.error(f"Error restoring data: {str(e)}")

            # Data deletion
            st.markdown("### Delete All Data")
            st.markdown("Delete all your data from the application. This action cannot be undone.")

            with st.expander("Delete All Data"):
                st.warning("This will permanently delete all your data, including your profile and health history. This action cannot be undone.")

                confirmation = st.text_input("Type 'DELETE' to confirm deletion:", key="delete_confirmation")

                if st.button("Delete All Data", key="delete_all_data") and confirmation == "DELETE":
                    # Clear all data
                    self.user_manager.profile = self.user_manager._create_default_profile()
                    self.user_manager.health_history = []

                    # Save empty data
                    self.user_manager.save_profile()
                    self.user_manager.save_health_history()

                    st.success("All data has been deleted successfully.")
                    st.experimental_rerun()

# Main function to run the application
def main():
    """Main entry point for the application."""
    # Initialize and run the application
    app = MedExplainApp()
    app.run()

if __name__ == "__main__":
    main()
