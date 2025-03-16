"""
Symptom Analyzer UI for MedExplain AI Pro.

This module provides the user interface for the symptom analyzer,
allowing users to input and analyze their health symptoms.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

class SymptomAnalyzerUI:
    """
    Class for the Symptom Analyzer UI, providing a streamlined interface
    for entering and analyzing health symptoms.
    """

    def __init__(self, health_data_manager=None, user_manager=None, openai_client=None):
        """
        Initialize the SymptomAnalyzerUI.

        Args:
            health_data_manager: The health data manager instance
            user_manager: The user profile manager instance
            openai_client: Optional OpenAI client for AI-enhanced explanations
        """
        self.health_data_manager = health_data_manager
        self.user_manager = user_manager
        self.openai_client = openai_client

    def render(self) -> None:
        """Render the symptom analyzer interface."""
        st.title("AI-Enhanced Symptom Analyzer")
        st.markdown("Describe your symptoms to get AI-powered analysis of possible conditions. Remember, this is not a diagnosis.")

        # Use session state components if not explicitly provided
        health_data_manager = self.health_data_manager or st.session_state.get("health_data_manager")
        user_manager = self.user_manager or st.session_state.get("user_manager")
        openai_client = self.openai_client or st.session_state.get("openai_client")

        # Check if required components are available
        if not health_data_manager or not user_manager:
            st.error("Application not properly initialized. Please refresh the page.")
            return

        # Create tabs for different input methods
        tabs = st.tabs(["Selection Method", "Text Description", "Voice Input (Beta)"])

        with tabs[0]:
            self._render_selection_method(health_data_manager, user_manager, openai_client)

        with tabs[1]:
            self._render_text_description(health_data_manager, user_manager, openai_client)

        with tabs[2]:
            self._render_voice_input(health_data_manager, user_manager, openai_client)

    def _render_selection_method(self, health_data_manager, user_manager, openai_client) -> None:
        """
        Render the selection-based symptom input method.

        Args:
            health_data_manager: The health data manager instance
            user_manager: The user profile manager instance
            openai_client: Optional OpenAI client for AI-enhanced explanations
        """
        # Create a list of symptoms from the database
        try:
            symptom_options = health_data_manager.get_all_symptoms()

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
                else:
                    fever_temp = None

            # Analyze button
            if st.button("Analyze Symptoms", key="analyze_symptoms_selection"):
                if not selected_symptoms:
                    st.warning("Please select at least one symptom.")
                else:
                    self._process_symptom_analysis(
                        health_data_manager,
                        user_manager,
                        openai_client,
                        selected_symptoms,
                        {
                            "symptom_duration": symptom_duration,
                            "symptom_severity": symptom_severity,
                            "had_fever": had_fever,
                            "fever_temp": fever_temp
                        }
                    )
        except Exception as e:
            logger.error("Error in selection method: %s", str(e))
            st.error(f"An error occurred: {str(e)}")

    def _render_text_description(self, health_data_manager, user_manager, openai_client) -> None:
        """
        Render the text-based symptom input method.

        Args:
            health_data_manager: The health data manager instance
            user_manager: The user profile manager instance
            openai_client: Optional OpenAI client for AI-enhanced explanations
        """
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
                self._process_text_analysis(
                    health_data_manager,
                    user_manager,
                    openai_client,
                    symptom_text
                )

    def _render_voice_input(self, health_data_manager, user_manager, openai_client) -> None:
        """
        Render the voice input method for symptom description.

        Args:
            health_data_manager: The health data manager instance
            user_manager: The user profile manager instance
            openai_client: Optional OpenAI client for AI-enhanced explanations
        """
        st.markdown("### Voice Input for Symptom Description")
        st.markdown("""
        This feature allows you to describe your symptoms using your voice.
        Click the button below and speak clearly about your symptoms.
        """)

        st.info("🚧 Voice input feature is currently in beta. In a production version, this would use speech-to-text technology to capture your symptom description.")

        if st.button("Start Voice Recording (Demo)", key="voice_record"):
            with st.spinner("Recording... (simulated)"):
                import time
                time.sleep(2)

            st.success("Recording complete! (simulated)")

            # Simulated transcription
            sample_transcription = "I've been having a headache and sore throat for the past two days, and I'm feeling very tired."

            st.markdown("### Transcription")
            text_transcription = st.text_area(
                "Your spoken description:",
                value=sample_transcription,
                height=100
            )

            if st.button("Analyze Voice Description", key="analyze_voice"):
                self._process_text_analysis(
                    health_data_manager,
                    user_manager,
                    openai_client,
                    text_transcription
                )

    def _process_symptom_analysis(self, health_data_manager, user_manager, openai_client,
                                selected_symptoms, additional_data) -> None:
        """
        Process and display the results of symptom analysis.

        Args:
            health_data_manager: The health data manager instance
            user_manager: The user profile manager instance
            openai_client: Optional OpenAI client for AI-enhanced explanations
            selected_symptoms: List of selected symptom IDs
            additional_data: Additional symptom data like duration, severity, etc.
        """
        with st.spinner("Analyzing your symptoms with advanced AI models..."):
            try:
                # Get symptom analyzer from session state if available
                symptom_predictor = st.session_state.get("symptom_predictor")
                risk_assessor = st.session_state.get("risk_assessor")

                # Get analysis results from health data manager
                analysis_results = health_data_manager.analyze_symptoms(selected_symptoms)

                # Get risk assessment if risk assessor is available
                risk_assessment = None
                if risk_assessor:
                    risk_assessment = risk_assessor.assess_risk({
                        "symptoms": selected_symptoms,
                        "symptom_duration": additional_data.get("symptom_duration", ""),
                        "symptom_severity": additional_data.get("symptom_severity", ""),
                        "age": user_manager.profile.get("age", 35),
                        "chronic_conditions": user_manager.profile.get("chronic_conditions", [])
                    })

                # Save to health history
                history_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symptoms": selected_symptoms,
                    "analysis_results": analysis_results,
                    "symptom_duration": additional_data.get("symptom_duration", ""),
                    "symptom_severity": additional_data.get("symptom_severity", ""),
                    "had_fever": additional_data.get("had_fever", False),
                    "fever_temp": additional_data.get("fever_temp")
                }

                # Add risk assessment if available
                if risk_assessment:
                    history_entry["risk_assessment"] = risk_assessment

                user_manager.health_history.append(history_entry)
                user_manager.save_health_history()

                # Display results
                self._display_symptom_analysis_results(analysis_results, risk_assessment)

                # Add AI-enhanced explanation if OpenAI client is available
                if openai_client and openai_client.is_available() and selected_symptoms:
                    self._display_ai_explanation(
                        openai_client,
                        health_data_manager,
                        selected_symptoms,
                        additional_data
                    )
            except Exception as e:
                logger.error("Error in symptom analysis: %s", str(e))
                st.error(f"An error occurred during analysis: {str(e)}")

    def _process_text_analysis(self, health_data_manager, user_manager, openai_client, symptom_text) -> None:
        """
        Process and display the results of text-based symptom analysis.

        Args:
            health_data_manager: The health data manager instance
            user_manager: The user profile manager instance
            openai_client: Optional OpenAI client for AI-enhanced explanations
            symptom_text: Text description of symptoms
        """
        with st.spinner("Analyzing your symptom description with NLP..."):
            try:
                # Get symptom extractor from session state if available
                symptom_extractor = st.session_state.get("symptom_extractor")

                if not symptom_extractor:
                    st.warning("Symptom extraction not available. Using basic text analysis.")

                    # Basic fallback extraction (would be more sophisticated in production)
                    extracted_symptoms = []
                    symptom_options = health_data_manager.get_all_symptoms()

                    for symptom in symptom_options:
                        if symptom["name"].lower() in symptom_text.lower():
                            extracted_symptoms.append({
                                "symptom_id": symptom["id"],
                                "name": symptom["name"],
                                "confidence": 1.0
                            })
                else:
                    # Use NLP symptom extractor
                    extracted_symptoms = symptom_extractor.extract_symptoms(symptom_text)

                if not extracted_symptoms:
                    st.warning("No specific symptoms could be identified in your description. Please try again with more details or use the selection method.")
                    return

                # Display extracted symptoms
                st.markdown("### Identified Symptoms")

                for symptom in extracted_symptoms:
                    confidence = symptom.get("confidence", 0) * 100
                    st.markdown(f"- **{symptom['name']}** (Confidence: {confidence:.1f}%)")

                # Get symptom IDs for analysis
                symptom_ids = [symptom["symptom_id"] for symptom in extracted_symptoms]

                # Analyze symptoms
                analysis_results = health_data_manager.analyze_symptoms(symptom_ids)

                # Save to history
                history_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symptoms": symptom_ids,
                    "symptom_text": symptom_text,
                    "analysis_results": analysis_results
                }

                user_manager.health_history.append(history_entry)
                user_manager.save_health_history()

                # Display results
                self._display_symptom_analysis_results(analysis_results, None)

                # Add AI-enhanced explanation if OpenAI client is available
                if openai_client and openai_client.is_available():
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
                        ai_response = openai_client.generate_response(prompt)

                        if ai_response:
                            st.markdown("## AI-Enhanced Explanation")
                            st.markdown(ai_response)
                    except Exception as e:
                        logger.error("Error generating AI explanation: %s", str(e))
                        st.error(f"Error generating AI explanation: {str(e)}")
            except Exception as e:
                logger.error("Error in text analysis: %s", str(e))
                st.error(f"An error occurred during analysis: {str(e)}")

    def _display_symptom_analysis_results(self, analysis_results, risk_assessment=None) -> None:
        """
        Display the results of the symptom analysis.

        Args:
            analysis_results: Results from symptom analysis
            risk_assessment: Optional risk assessment data
        """
        st.markdown("## Analysis Results")

        if not analysis_results:
            st.info("No specific conditions match your symptoms in our database. Please consult a healthcare professional for proper evaluation.")
            return

        # Display disclaimer
        st.markdown("""
        <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
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
            if risk_level == "medium" or risk_level == "moderate":
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
            health_data_manager = st.session_state.get("health_data_manager")
            if health_data_manager:
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

    def _display_ai_explanation(self, openai_client, health_data_manager, selected_symptoms, additional_data) -> None:
        """
        Display an AI-enhanced explanation of symptoms.

        Args:
            openai_client: The OpenAI client
            health_data_manager: The health data manager instance
            selected_symptoms: List of selected symptom IDs
            additional_data: Additional symptom data
        """
        try:
            # Get symptom names
            symptom_names = []
            for symptom_id in selected_symptoms:
                symptom_info = health_data_manager.get_symptom_info(symptom_id)
                if symptom_info and "name" in symptom_info:
                    symptom_names.append(symptom_info["name"])
                else:
                    symptom_names.append(symptom_id)

            symptoms_text = ", ".join(symptom_names)

            # Create the prompt
            prompt = f"""
            I'm experiencing the following symptoms: {symptoms_text}.
            Duration: {additional_data.get('symptom_duration', 'Unknown')}
            Severity: {additional_data.get('symptom_severity', 'Unknown')}
            """

            if additional_data.get('had_fever', False):
                prompt += f"Fever: Yes, {additional_data.get('fever_temp', '')}°F\n"
            else:
                prompt += "Fever: No\n"

            prompt += """
            Based on these symptoms, provide a clear, concise explanation of:
            1. What could potentially be causing these symptoms (2-3 possibilities)
            2. What these symptoms typically mean in the body
            3. General self-care recommendations
            4. When I should definitely see a doctor

            Keep your answer factual, educational, and concise. Include a disclaimer about consulting a healthcare professional.
            """

            # Get response from OpenAI
            ai_response = openai_client.generate_response(prompt)

            if ai_response:
                st.markdown("## AI-Enhanced Explanation")
                st.markdown(ai_response)
        except Exception as e:
            logger.error("Error generating AI explanation: %s", str(e))
            st.error(f"Error generating AI explanation: {str(e)}")


# For backward compatibility with the functional approach
def render_symptom_analyzer() -> None:
    """Render the symptom analyzer interface using the class-based implementation."""
    analyzer_ui = SymptomAnalyzerUI()
    analyzer_ui.render()
