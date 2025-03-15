import os
import json
import logging
import pickle
import random
import math
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class PatientRiskAssessor:
    """
    Risk assessment model for evaluating patient health risks based on symptoms and profile.
    Uses rule-based algorithms and risk models to generate assessments.
    """

    def __init__(self, model_dir: str):
        """
        Initialize the patient risk assessor.

        Args:
            model_dir: Directory where ML models are stored
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "risk_assessor.pkl")
        self.risk_factors = {}
        self.symptom_severity = {}
        self.condition_risks = {}
        self.age_risk_modifiers = {}
        self.comorbidity_risks = {}

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Load model or create default
        self._load_model()

        logger.info("PatientRiskAssessor initialized")

    def _load_model(self):
        """Load the risk assessment model from disk or create default."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.risk_factors = model_data.get('risk_factors', {})
                self.symptom_severity = model_data.get('symptom_severity', {})
                self.condition_risks = model_data.get('condition_risks', {})
                self.age_risk_modifiers = model_data.get('age_risk_modifiers', {})
                self.comorbidity_risks = model_data.get('comorbidity_risks', {})

                logger.info("Loaded risk assessment model from disk")
            else:
                # Create default model
                self._create_default_model()
                logger.info("Created default risk assessment model")
        except Exception as e:
            logger.error(f"Error loading risk assessment model: {e}", exc_info=True)
            # Create default model on error
            self._create_default_model()

    def _create_default_model(self):
        """Create a default risk assessment model with common risk factors and weights."""
        # Define symptom severity weights (0-10 scale, 10 being most severe)
        self.symptom_severity = {
            "fever": 3,
            "high fever": 6,
            "cough": 2,
            "severe cough": 5,
            "dry cough": 3,
            "productive cough": 4,
            "chest pain": 7,
            "severe chest pain": 9,
            "headache": 2,
            "severe headache": 6,
            "fatigue": 2,
            "extreme fatigue": 5,
            "shortness of breath": 6,
            "difficulty breathing": 8,
            "nausea": 2,
            "vomiting": 4,
            "diarrhea": 3,
            "abdominal pain": 4,
            "severe abdominal pain": 7,
            "dizziness": 3,
            "fainting": 7,
            "sore throat": 1,
            "loss of taste": 2,
            "loss of smell": 2,
            "back pain": 2,
            "severe back pain": 5,
            "joint pain": 2,
            "muscle pain": 2,
            "rash": 2,
            "confusion": 7,
            "drowsiness": 4,
            "difficulty walking": 6,
            "difficulty speaking": 8,
            "numbness": 5,
            "weakness": 3
        }

        # Define condition risk levels (0-10 scale)
        self.condition_risks = {
            "common cold": 1,
            "influenza": 3,
            "pneumonia": 7,
            "bronchitis": 4,
            "asthma attack": 6,
            "hypertension": 5,
            "anxiety": 3,
            "migraine": 3,
            "gastroenteritis": 4,
            "appendicitis": 8,
            "urinary tract infection": 3,
            "kidney infection": 6,
            "heart attack": 10,
            "stroke": 10,
            "sepsis": 9,
            "meningitis": 9,
            "allergic reaction": 5,
            "severe allergic reaction": 9,
            "diabetes complications": 7,
            "viral infection": 2,
            "bacterial infection": 4
        }

        # Define age-related risk modifiers
        self.age_risk_modifiers = {
            "0-2": 1.5,     # Infants have higher risk
            "3-12": 1.0,
            "13-18": 0.9,
            "19-40": 0.8,   # Young adults generally lower risk
            "41-60": 1.0,   # Middle age baseline
            "61-75": 1.3,   # Senior increased risk
            "76+": 1.7      # Elderly substantially higher risk
        }

        # Define comorbidity risk factors (additional risk per condition)
        self.comorbidity_risks = {
            "diabetes": 1.5,
            "hypertension": 1.3,
            "heart disease": 1.7,
            "copd": 1.8,
            "asthma": 1.3,
            "immunosuppression": 2.0,
            "cancer": 1.8,
            "obesity": 1.4,
            "liver disease": 1.6,
            "kidney disease": 1.7,
            "autoimmune disease": 1.5,
            "stroke history": 1.5,
            "smoking": 1.3,
            "pregnancy": 1.4
        }

        # Define risk factor categories and weights for comprehensive assessment
        self.risk_factors = {
            "symptom_risk": 0.35,        # Weight for symptom severity
            "condition_risk": 0.25,      # Weight for identified conditions
            "age_risk": 0.15,            # Weight for age factor
            "comorbidity_risk": 0.25     # Weight for comorbidities/medical history
        }

        # Save the default model
        self._save_model()

    def _save_model(self):
        """Save the risk assessment model to disk."""
        try:
            model_data = {
                'risk_factors': self.risk_factors,
                'symptom_severity': self.symptom_severity,
                'condition_risks': self.condition_risks,
                'age_risk_modifiers': self.age_risk_modifiers,
                'comorbidity_risks': self.comorbidity_risks,
                'version': '1.0',
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info("Saved risk assessment model to disk")
        except Exception as e:
            logger.error(f"Error saving risk assessment model: {e}", exc_info=True)

    def assess_risk(self, symptoms: List[str], conditions: List[str] = None, user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess patient risk based on symptoms, conditions, and profile.

        Args:
            symptoms: List of reported symptom names
            conditions: Optional list of identified conditions
            user_profile: Optional user profile with age, gender, medical history

        Returns:
            Risk assessment results including overall risk level and scores
        """
        try:
            # Default result structure
            assessment = {
                'risk_level': 'unknown',
                'risk_score': 0.0,
                'symptom_risk': 0.0,
                'condition_risk': 0.0,
                'age_risk': 0.0,
                'comorbidity_risk': 0.0,
                'high_risk_factors': [],
                'moderate_risk_factors': [],
                'domain_risks': {},
                'symptoms': symptoms  # Store original symptoms for reference
            }

            # 1. Calculate symptom risk
            symptom_risk_score = self._calculate_symptom_risk(symptoms)
            assessment['symptom_risk'] = symptom_risk_score

            # 2. Calculate condition risk if conditions provided
            condition_risk_score = 0.0
            if conditions:
                condition_risk_score = self._calculate_condition_risk(conditions)
            assessment['condition_risk'] = condition_risk_score

            # 3. Calculate age risk if profile provided
            age_risk_score = 0.0
            if user_profile and 'age' in user_profile:
                age_risk_score = self._calculate_age_risk(user_profile['age'])
            assessment['age_risk'] = age_risk_score

            # 4. Calculate comorbidity risk if profile provided with medical history
            comorbidity_risk_score = 0.0
            if user_profile and 'medical_history' in user_profile:
                comorbidity_risk_score = self._calculate_comorbidity_risk(user_profile['medical_history'])
            assessment['comorbidity_risk'] = comorbidity_risk_score

            # 5. Calculate overall risk score (weighted sum of component risks)
            overall_risk = (
                symptom_risk_score * self.risk_factors['symptom_risk'] +
                condition_risk_score * self.risk_factors['condition_risk'] +
                age_risk_score * self.risk_factors['age_risk'] +
                comorbidity_risk_score * self.risk_factors['comorbidity_risk']
            )

            # Normalize to 0-10 scale
            overall_risk = min(10, max(0, overall_risk))
            assessment['risk_score'] = round(overall_risk, 1)

            # 6. Determine risk level
            if overall_risk < 3.5:
                assessment['risk_level'] = 'low'
            elif overall_risk < 7.0:
                assessment['risk_level'] = 'medium'
            else:
                assessment['risk_level'] = 'high'

            # 7. Identify high and moderate risk factors
            # High risk symptoms
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                if symptom_lower in self.symptom_severity:
                    if self.symptom_severity[symptom_lower] >= 7:
                        assessment['high_risk_factors'].append(f"High-risk symptom: {symptom}")
                    elif self.symptom_severity[symptom_lower] >= 5:
                        assessment['moderate_risk_factors'].append(f"Moderate-risk symptom: {symptom}")

            # High risk conditions
            if conditions:
                for condition in conditions:
                    condition_lower = condition.lower()
                    if condition_lower in self.condition_risks:
                        if self.condition_risks[condition_lower] >= 7:
                            assessment['high_risk_factors'].append(f"High-risk condition: {condition}")
                        elif self.condition_risks[condition_lower] >= 5:
                            assessment['moderate_risk_factors'].append(f"Moderate-risk condition: {condition}")

            # High risk comorbidities
            if user_profile and 'medical_history' in user_profile:
                for condition in user_profile['medical_history']:
                    condition_lower = condition.lower()
                    if condition_lower in self.comorbidity_risks:
                        if self.comorbidity_risks[condition_lower] >= 1.7:
                            assessment['high_risk_factors'].append(f"High-risk comorbidity: {condition}")
                        elif self.comorbidity_risks[condition_lower] >= 1.4:
                            assessment['moderate_risk_factors'].append(f"Moderate-risk comorbidity: {condition}")

            # Age risk
            if user_profile and 'age' in user_profile:
                age = user_profile['age']
                if age >= 76:
                    assessment['high_risk_factors'].append(f"High-risk age group: {age} years")
                elif age >= 61:
                    assessment['moderate_risk_factors'].append(f"Moderate-risk age group: {age} years")

            # 8. Calculate domain-specific risks
            assessment['domain_risks'] = self._calculate_domain_risks(symptoms, conditions, user_profile)

            return assessment
        except Exception as e:
            logger.error(f"Error assessing risk: {e}", exc_info=True)
            return {'risk_level': 'unknown', 'risk_score': 0.0, 'error': str(e)}

    def _calculate_symptom_risk(self, symptoms: List[str]) -> float:
        """
        Calculate risk score based on symptom severity.

        Args:
            symptoms: List of symptom names

        Returns:
            Normalized risk score (0-10)
        """
        if not symptoms:
            return 0.0

        # Convert symptoms to lowercase for matching
        symptoms_lower = [s.lower() for s in symptoms]

        # Calculate total severity
        total_severity = 0.0
        matched_symptoms = 0

        for symptom in symptoms_lower:
            if symptom in self.symptom_severity:
                total_severity += self.symptom_severity[symptom]
                matched_symptoms += 1
            else:
                # For unknown symptoms, use a default moderate severity
                total_severity += 2.0
                matched_symptoms += 1

        # Check for certain high-risk symptom combinations
        high_risk_combinations = [
            (["chest pain", "shortness of breath"], 3.0),  # Potential cardiac issues
            (["high fever", "stiff neck", "headache"], 3.0),  # Potential meningitis
            (["shortness of breath", "cough", "fever"], 2.0),  # Potential respiratory infection
            (["abdominal pain", "vomiting", "fever"], 2.0),  # Potential abdominal infection
            (["confusion", "fever", "headache"], 2.5)  # Potential neurological infection
        ]

        combination_bonus = 0.0
        for combo, bonus in high_risk_combinations:
            if all(c in symptoms_lower for c in combo):
                combination_bonus += bonus

        # Calculate average severity with combination bonus
        if matched_symptoms > 0:
            avg_severity = (total_severity / matched_symptoms) + combination_bonus

            # Normalize to 0-10 scale
            return min(10, max(0, avg_severity))

        return 0.0

    def _calculate_condition_risk(self, conditions: List[str]) -> float:
        """
        Calculate risk score based on identified conditions.

        Args:
            conditions: List of condition names

        Returns:
            Normalized risk score (0-10)
        """
        if not conditions:
            return 0.0

        # Convert conditions to lowercase for matching
        conditions_lower = [c.lower() for c in conditions]

        # Calculate total risk
        total_risk = 0.0
        matched_conditions = 0

        for condition in conditions_lower:
            if condition in self.condition_risks:
                total_risk += self.condition_risks[condition]
                matched_conditions += 1
            else:
                # For unknown conditions, use a default moderate risk
                total_risk += 3.0
                matched_conditions += 1

        # Check for certain high-risk condition combinations
        high_risk_combinations = [
            (["pneumonia", "influenza"], 2.0),  # Combined respiratory infections
            (["hypertension", "heart disease"], 2.5),  # Cardiovascular comorbidities
            (["diabetes", "kidney infection"], 2.0)  # Diabetes with infection risk
        ]

        combination_bonus = 0.0
        for combo, bonus in high_risk_combinations:
            if all(c in conditions_lower for c in combo):
                combination_bonus += bonus

        # Calculate average risk with combination bonus
        if matched_conditions > 0:
            avg_risk = (total_risk / matched_conditions) + combination_bonus

            # Normalize to 0-10 scale
            return min(10, max(0, avg_risk))

        return 0.0

    def _calculate_age_risk(self, age: int) -> float:
        """
        Calculate risk score based on patient age.

        Args:
            age: Patient age

        Returns:
            Normalized risk score (0-10)
        """
        # Determine age group
        if age <= 2:
            age_group = "0-2"
        elif age <= 12:
            age_group = "3-12"
        elif age <= 18:
            age_group = "13-18"
        elif age <= 40:
            age_group = "19-40"
        elif age <= 60:
            age_group = "41-60"
        elif age <= 75:
            age_group = "61-75"
        else:
            age_group = "76+"

        # Get risk modifier for age group
        risk_modifier = self.age_risk_modifiers.get(age_group, 1.0)

        # Calculate age risk (baseline of 5.0 multiplied by modifier)
        age_risk = 5.0 * risk_modifier

        # Normalize to 0-10 scale
        return min(10, max(0, age_risk))

    def _calculate_comorbidity_risk(self, medical_history: List[str]) -> float:
        """
        Calculate risk score based on comorbidities in medical history.

        Args:
            medical_history: List of medical conditions

        Returns:
            Normalized risk score (0-10)
        """
        if not medical_history:
            return 0.0

        # Convert medical history to lowercase for matching
        medical_history_lower = [c.lower() for c in medical_history]

        # Calculate total risk modifier
        risk_modifier = 1.0  # Baseline (no change)

        for condition in medical_history_lower:
            if condition in self.comorbidity_risks:
                # Cumulative multiplication of risk modifiers
                risk_modifier *= self.comorbidity_risks[condition]

        # Calculate comorbidity risk (baseline of 4.0 multiplied by modifier)
        comorbidity_risk = 4.0 * risk_modifier

        # Normalize to 0-10 scale
        return min(10, max(0, comorbidity_risk))

    def _calculate_domain_risks(self, symptoms: List[str], conditions: List[str] = None, user_profile: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Calculate risk scores for different health domains.

        Args:
            symptoms: List of symptom names
            conditions: Optional list of identified conditions
            user_profile: Optional user profile with age, gender, medical history

        Returns:
            Dictionary mapping health domains to risk scores
        """
        # Define domain categories and their associated symptoms/conditions
        domains = {
            "respiratory": {
                "symptoms": ["cough", "shortness of breath", "wheezing", "chest congestion", "difficulty breathing"],
                "conditions": ["pneumonia", "bronchitis", "asthma", "copd", "influenza"],
                "comorbidities": ["asthma", "copd", "smoking"]
            },
            "cardiovascular": {
                "symptoms": ["chest pain", "palpitations", "shortness of breath", "dizziness", "fainting", "swelling"],
                "conditions": ["hypertension", "heart attack", "heart failure", "arrhythmia"],
                "comorbidities": ["hypertension", "heart disease", "diabetes", "obesity"]
            },
            "gastrointestinal": {
                "symptoms": ["abdominal pain", "nausea", "vomiting", "diarrhea", "constipation", "bloating"],
                "conditions": ["gastroenteritis", "appendicitis", "ibs", "gastritis"],
                "comorbidities": ["ibs", "crohn's disease", "ulcerative colitis", "liver disease"]
            },
            "neurological": {
                "symptoms": ["headache", "dizziness", "confusion", "numbness", "weakness", "difficulty speaking"],
                "conditions": ["migraine", "stroke", "meningitis", "seizure"],
                "comorbidities": ["stroke history", "epilepsy", "multiple sclerosis"]
            },
            "musculoskeletal": {
                "symptoms": ["joint pain", "muscle pain", "back pain", "stiffness", "swelling", "difficulty walking"],
                "conditions": ["arthritis", "fracture", "sprain", "strain"],
                "comorbidities": ["arthritis", "osteoporosis", "fibromyalgia"]
            }
        }

        # Initialize domain risks
        domain_risks = {domain: 0.0 for domain in domains}

        # Convert symptoms and conditions to lowercase
        symptoms_lower = [s.lower() for s in symptoms]
        conditions_lower = [c.lower() for c in conditions] if conditions else []
        medical_history_lower = []
        if user_profile and 'medical_history' in user_profile:
            medical_history_lower = [c.lower() for c in user_profile['medical_history']]

        # Calculate risk for each domain
        for domain, category in domains.items():
            domain_score = 0.0

            # Check for matching symptoms
            matching_symptoms = [s for s in symptoms_lower if any(variant in s for variant in category["symptoms"])]
            if matching_symptoms:
                # Calculate severity of matching symptoms
                severity_sum = 0.0
                for symptom in matching_symptoms:
                    severity = self.symptom_severity.get(symptom, 2.0)  # Default moderate severity
                    severity_sum += severity

                # Average symptom severity normalized to 0-10
                symptom_score = min(10, severity_sum / max(1, len(matching_symptoms)))
                domain_score += symptom_score * 0.5  # Symptom component (50%)

            # Check for matching conditions
            matching_conditions = [c for c in conditions_lower if any(variant in c for variant in category["conditions"])]
            if matching_conditions:
                # Calculate risk of matching conditions
                risk_sum = 0.0
                for condition in matching_conditions:
                    risk = self.condition_risks.get(condition, 3.0)  # Default moderate risk
                    risk_sum += risk

                # Average condition risk normalized to 0-10
                condition_score = min(10, risk_sum / max(1, len(matching_conditions)))
                domain_score += condition_score * 0.3  # Condition component (30%)

            # Check for matching comorbidities
            matching_comorbidities = [c for c in medical_history_lower if any(variant in c for variant in category["comorbidities"])]
            if matching_comorbidities:
                # Calculate impact of matching comorbidities
                modifier_product = 1.0
                for comorbidity in matching_comorbidities:
                    modifier = self.comorbidity_risks.get(comorbidity, 1.2)  # Default moderate modifier
                    modifier_product *= modifier

                # Comorbidity score normalized to 0-10
                comorbidity_score = min(10, 4.0 * modifier_product)
                domain_score += comorbidity_score * 0.2  # Comorbidity component (20%)

            # Store normalized domain risk score
            domain_risks[domain] = round(min(10, max(0, domain_score)), 1)

        return domain_risks

    def generate_health_report(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive health report based on risk assessment.

        Args:
            assessment: Risk assessment results

        Returns:
            Health report with detailed analysis and recommendations
        """
        try:
            report = {
                'summary': '',
                'risk_level': assessment.get('risk_level', 'unknown'),
                'risk_factors': [],
                'domain_analysis': {},
                'recommendations': [],
                'follow_up': None
            }

            # Generate summary based on risk level
            risk_level = assessment.get('risk_level', 'unknown')
            risk_score = assessment.get('risk_score', 0.0)

            if risk_level == 'high':
                report['summary'] = (
                    f"Based on the assessment, your current health status shows a high risk level ({risk_score}/10). "
                    "Several significant risk factors have been identified that may require prompt medical attention."
                )
                report['follow_up'] = "Recommended follow-up: Consult with a healthcare provider within 24-48 hours."
            elif risk_level == 'medium':
                report['summary'] = (
                    f"Based on the assessment, your current health status shows a moderate risk level ({risk_score}/10). "
                    "Some concerning factors have been identified that may benefit from medical evaluation."
                )
                report['follow_up'] = "Recommended follow-up: Schedule an appointment with your healthcare provider within the next week."
            elif risk_level == 'low':
                report['summary'] = (
                    f"Based on the assessment, your current health status shows a low risk level ({risk_score}/10). "
                    "Few concerning factors have been identified, suggesting your symptoms may be associated with minor, self-limiting conditions."
                )
                report['follow_up'] = "Recommended follow-up: Monitor your symptoms and seek care if they persist beyond 7-10 days or worsen."
            else:
                report['summary'] = (
                    "Based on the limited information available, a definitive risk assessment could not be determined. "
                    "Consider providing more detailed health information for a more accurate assessment."
                )
                report['follow_up'] = "Recommended follow-up: Consider consulting with a healthcare provider to evaluate your symptoms."

            # Compile risk factors
            report['risk_factors'] = (
                assessment.get('high_risk_factors', []) +
                assessment.get('moderate_risk_factors', [])
            )

            # Domain analysis
            for domain, score in assessment.get('domain_risks', {}).items():
                domain_analysis = {
                    'risk_score': score,
                    'interpretation': '',
                    'recommendations': []
                }

                # Generate domain interpretation
                if score >= 7.0:
                    domain_analysis['interpretation'] = f"High concern level in {domain} health domain."

                    if domain == "respiratory":
                        domain_analysis['recommendations'] = [
                            "Monitor breathing rate and difficulty",
                            "Record any changes in cough or sputum",
                            "Avoid respiratory irritants",
                            "Consider evaluation by a healthcare provider"
                        ]
                    elif domain == "cardiovascular":
                        domain_analysis['recommendations'] = [
                            "Monitor heart rate and any chest discomfort",
                            "Limit strenuous physical activity",
                            "Follow low-sodium diet if appropriate",
                            "Seek prompt medical care for any chest pain"
                        ]
                    elif domain == "gastrointestinal":
                        domain_analysis['recommendations'] = [
                            "Stay well-hydrated",
                            "Follow bland diet until symptoms improve",
                            "Record frequency and characteristics of bowel movements",
                            "Seek care for severe or persistent pain"
                        ]
                    elif domain == "neurological":
                        domain_analysis['recommendations'] = [
                            "Monitor for changes in level of consciousness",
                            "Avoid activities requiring full alertness if experiencing symptoms",
                            "Record any progression of symptoms",
                            "Seek prompt medical care for severe or worsening symptoms"
                        ]
                    elif domain == "musculoskeletal":
                        domain_analysis['recommendations'] = [
                            "Apply ice to reduce inflammation if appropriate",
                            "Minimize activities that exacerbate pain",
                            "Consider over-the-counter pain relievers if appropriate",
                            "Seek care if mobility is significantly affected"
                        ]

                elif score >= 4.0:
                    domain_analysis['interpretation'] = f"Moderate concern level in {domain} health domain."

                    if domain == "respiratory":
                        domain_analysis['recommendations'] = [
                            "Monitor breathing during rest and activity",
                            "Stay well-hydrated",
                            "Consider over-the-counter remedies appropriate for symptoms"
                        ]
                    elif domain == "cardiovascular":
                        domain_analysis['recommendations'] = [
                            "Monitor blood pressure if equipment is available",
                            "Maintain adequate hydration",
                            "Follow heart-healthy diet"
                        ]
                    elif domain == "gastrointestinal":
                        domain_analysis['recommendations'] = [
                            "Stay well-hydrated",
                            "Eat smaller, more frequent meals",
                            "Avoid foods that may irritate the digestive system"
                        ]
                    elif domain == "neurological":
                        domain_analysis['recommendations'] = [
                            "Ensure adequate rest",
                            "Avoid potential triggers for symptoms",
                            "Monitor for changes in symptom intensity"
                        ]
                    elif domain == "musculoskeletal":
                        domain_analysis['recommendations'] = [
                            "Balance rest and gentle movement",
                            "Apply heat or cold as appropriate",
                            "Maintain proper posture and body mechanics"
                        ]

                else:
                    domain_analysis['interpretation'] = f"Low concern level in {domain} health domain."

                    domain_analysis['recommendations'] = [
                        "Continue normal activities as tolerated",
                        "Monitor for any new or worsening symptoms",
                        "Follow general health maintenance practices"
                    ]

                report['domain_analysis'][domain] = domain_analysis

            # General recommendations based on risk level
            if risk_level == 'high':
                report['recommendations'] = [
                    "Schedule a healthcare appointment within 24-48 hours",
                    "Keep a detailed log of symptoms, including timing and severity",
                    "Avoid activities that might exacerbate your condition",
                    "Have someone stay with you if symptoms are severe",
                    "Prepare a list of current medications and allergies for your healthcare visit"
                ]
            elif risk_level == 'medium':
                report['recommendations'] = [
                    "Schedule a healthcare appointment within the next week",
                    "Monitor symptoms and note any changes in intensity or frequency",
                    "Continue medications as prescribed",
                    "Ensure adequate rest and hydration",
                    "Avoid activities that worsen symptoms"
                ]
            else:  # low or unknown
                report['recommendations'] = [
                    "Monitor symptoms over the next several days",
                    "Rest and stay well-hydrated",
                    "Use over-the-counter remedies as appropriate for symptom relief",
                    "Maintain a healthy diet and adequate sleep",
                    "Seek care if symptoms persist beyond 7-10 days or worsen"
                ]

            return report
        except Exception as e:
            logger.error(f"Error generating health report: {e}", exc_info=True)
            return {'summary': 'Error generating health report. Please try again.', 'error': str(e)}

    def assess_symptom_progression(self, previous_symptoms: List[str], current_symptoms: List[str]) -> Dict[str, Any]:
        """
        Assess the progression of symptoms over time.

        Args:
            previous_symptoms: List of previously reported symptom names
            current_symptoms: List of currently reported symptom names

        Returns:
            Assessment of symptom progression
        """
        try:
            # Convert symptoms to lowercase for matching
            previous_lower = [s.lower() for s in previous_symptoms]
            current_lower = [s.lower() for s in current_symptoms]

            # Compare symptom sets
            improved_symptoms = [s for s in previous_lower if s not in current_lower]
            new_symptoms = [s for s in current_lower if s not in previous_lower]
            persistent_symptoms = [s for s in current_lower if s in previous_lower]

            # Calculate severity scores
            previous_severity = sum(self.symptom_severity.get(s, 2.0) for s in previous_lower) / max(1, len(previous_lower))
            current_severity = sum(self.symptom_severity.get(s, 2.0) for s in current_lower) / max(1, len(current_lower))

            # Determine progression status
            if len(current_lower) == 0:
                progression = "resolved"
            elif current_severity < previous_severity * 0.7:
                progression = "significantly_improved"
            elif current_severity < previous_severity * 0.9:
                progression = "slightly_improved"
            elif current_severity > previous_severity * 1.3:
                progression = "worsened"
            elif current_severity > previous_severity * 1.1:
                progression = "slightly_worsened"
            else:
                progression = "stable"

            # Check for concerning progression patterns
            concerning_patterns = []

            # New high-severity symptoms
            for symptom in new_symptoms:
                severity = self.symptom_severity.get(symptom, 2.0)
                if severity >= 7.0:
                    concerning_patterns.append(f"New high-severity symptom: {symptom}")

            # Persistent high-severity symptoms
            for symptom in persistent_symptoms:
                severity = self.symptom_severity.get(symptom, 2.0)
                if severity >= 7.0:
                    concerning_patterns.append(f"Persistent high-severity symptom: {symptom}")

            # Increasing number of symptoms
            if len(current_lower) > len(previous_lower) * 1.5:
                concerning_patterns.append("Significant increase in number of symptoms")

            # Create progression report
            progression_report = {
                'status': progression,
                'improved_symptoms': improved_symptoms,
                'new_symptoms': new_symptoms,
                'persistent_symptoms': persistent_symptoms,
                'previous_severity': round(previous_severity, 1),
                'current_severity': round(current_severity, 1),
                'concerning_patterns': concerning_patterns,
                'recommendations': []
            }

            # Add recommendations based on progression
            if progression == "resolved":
                progression_report['recommendations'] = [
                    "Continue monitoring for any return of symptoms",
                    "Complete any prescribed treatment courses",
                    "Schedule follow-up if recommended by healthcare provider"
                ]
            elif progression in ["significantly_improved", "slightly_improved"]:
                progression_report['recommendations'] = [
                    "Continue current treatment approach",
                    "Monitor for continued improvement",
                    "Be alert for any return of resolved symptoms"
                ]
            elif progression == "stable":
                progression_report['recommendations'] = [
                    "Continue monitoring symptoms",
                    "Follow up with healthcare provider if symptoms persist beyond expected timeframe",
                    "Note any subtle changes in symptom characteristics"
                ]
            elif progression in ["slightly_worsened", "worsened"]:
                progression_report['recommendations'] = [
                    "Consider scheduling healthcare appointment to reassess",
                    "Document specific changes in symptoms",
                    "Review and strictly adhere to treatment recommendations",
                    "Be vigilant for any further deterioration"
                ]

            # Add specific recommendations for concerning patterns
            if concerning_patterns:
                progression_report['recommendations'].append("Discuss concerning symptoms with healthcare provider")

                if any("high-severity" in pattern for pattern in concerning_patterns):
                    progression_report['recommendations'].append("Consider expedited medical evaluation")

            return progression_report
        except Exception as e:
            logger.error(f"Error assessing symptom progression: {e}", exc_info=True)
            return {'status': 'unknown', 'error': str(e)}

    def estimate_recovery_time(self, symptoms: List[str], conditions: List[str] = None, user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Estimate expected recovery time based on symptoms and conditions.

        Args:
            symptoms: List of symptom names
            conditions: Optional list of condition names
            user_profile: Optional user profile with age, medical history

        Returns:
            Recovery time estimate with ranges and factors
        """
        try:
            # Default recovery estimates for common conditions (in days)
            condition_recovery_times = {
                "common cold": {"min": 7, "max": 10, "typical": 7},
                "influenza": {"min": 7, "max": 14, "typical": 10},
                "bronchitis": {"min": 10, "max": 21, "typical": 14},
                "pneumonia": {"min": 14, "max": 30, "typical": 21},
                "sinusitis": {"min": 7, "max": 14, "typical": 10},
                "gastroenteritis": {"min": 2, "max": 7, "typical": 3},
                "urinary tract infection": {"min": 3, "max": 10, "typical": 7},
                "migraine": {"min": 1, "max": 3, "typical": 2},
                "sprain": {"min": 7, "max": 42, "typical": 14},
                "strain": {"min": 7, "max": 21, "typical": 10}
            }

            # Default recovery estimates for common symptoms (in days)
            symptom_recovery_times = {
                "fever": {"min": 1, "max": 5, "typical": 3},
                "cough": {"min": 7, "max": 21, "typical": 14},
                "sore throat": {"min": 3, "max": 7, "typical": 5},
                "headache": {"min": 1, "max": 3, "typical": 1},
                "fatigue": {"min": 7, "max": 21, "typical": 10},
                "nausea": {"min": 1, "max": 3, "typical": 2},
                "diarrhea": {"min": 2, "max": 5, "typical": 3},
                "rash": {"min": 3, "max": 14, "typical": 7}
            }

            # Age-based recovery modifiers
            age_recovery_modifiers = {
                "0-12": 0.8,    # Children often recover faster
                "13-18": 0.9,
                "19-40": 1.0,   # Baseline
                "41-60": 1.1,
                "61-75": 1.3,   # Seniors recover more slowly
                "76+": 1.5      # Elderly recover much more slowly
            }

            # Comorbidity recovery modifiers
            comorbidity_recovery_modifiers = {
                "diabetes": 1.3,
                "hypertension": 1.1,
                "heart disease": 1.3,
                "copd": 1.4,
                "asthma": 1.2,
                "immunosuppression": 1.5,
                "obesity": 1.2
            }

            # Initialize estimate
            estimate = {
                'min_days': 0,
                'max_days': 0,
                'typical_days': 0,
                'modifying_factors': [],
                'confidence': 'low'
            }

            # Base recovery time on conditions if available
            base_min = 0
            base_max = 0
            base_typical = 0

            if conditions and len(conditions) > 0:
                # Use condition-based estimates
                condition_times = []

                for condition in conditions:
                    condition_lower = condition.lower()
                    if condition_lower in condition_recovery_times:
                        condition_times.append(condition_recovery_times[condition_lower])

                if condition_times:
                    # Take the maximum of all conditions for conservative estimate
                    base_min = max(time["min"] for time in condition_times)
                    base_max = max(time["max"] for time in condition_times)
                    base_typical = max(time["typical"] for time in condition_times)

                    estimate['confidence'] = 'medium'

            # If no condition-based estimates, use symptom-based
            if base_typical == 0 and symptoms and len(symptoms) > 0:
                symptom_times = []

                for symptom in symptoms:
                    symptom_lower = symptom.lower()
                    if symptom_lower in symptom_recovery_times:
                        symptom_times.append(symptom_recovery_times[symptom_lower])

                if symptom_times:
                    # Take the maximum of all symptoms for conservative estimate
                    base_min = max(time["min"] for time in symptom_times)
                    base_max = max(time["max"] for time in symptom_times)
                    base_typical = max(time["typical"] for time in symptom_times)

            # Apply age modifier if available
            age_modifier = 1.0
            if user_profile and 'age' in user_profile:
                age = user_profile['age']

                if age <= 12:
                    age_group = "0-12"
                elif age <= 18:
                    age_group = "13-18"
                elif age <= 40:
                    age_group = "19-40"
                elif age <= 60:
                    age_group = "41-60"
                elif age <= 75:
                    age_group = "61-75"
                else:
                    age_group = "76+"

                age_modifier = age_recovery_modifiers.get(age_group, 1.0)

                if age_modifier != 1.0:
                    estimate['modifying_factors'].append(f"Age group ({age} years): {'+' if age_modifier > 1.0 else '-'}{abs(age_modifier - 1.0) * 100:.0f}% recovery time")

            # Apply comorbidity modifiers if available
            comorbidity_modifier = 1.0
            if user_profile and 'medical_history' in user_profile:
                for condition in user_profile['medical_history']:
                    condition_lower = condition.lower()
                    if condition_lower in comorbidity_recovery_modifiers:
                        modifier = comorbidity_recovery_modifiers[condition_lower]
                        comorbidity_modifier *= modifier

                        estimate['modifying_factors'].append(f"{condition}: +{(modifier - 1.0) * 100:.0f}% recovery time")

            # Calculate final estimates
            total_modifier = age_modifier * comorbidity_modifier

            estimate['min_days'] = max(1, int(base_min * total_modifier))
            estimate['max_days'] = max(1, int(base_max * total_modifier))
            estimate['typical_days'] = max(1, int(base_typical * total_modifier))

            # Add total modifier if different from 1.0
            if abs(total_modifier - 1.0) > 0.05:
                estimate['modifying_factors'].insert(0, f"Total recovery modifier: {'+' if total_modifier > 1.0 else '-'}{abs(total_modifier - 1.0) * 100:.0f}%")

            # Set confidence if not already set
            if estimate['confidence'] == 'low' and base_typical > 0:
                estimate['confidence'] = 'medium'

            return estimate
        except Exception as e:
            logger.error(f"Error estimating recovery time: {e}", exc_info=True)
            return {'min_days': 0, 'max_days': 0, 'typical_days': 0, 'modifying_factors': [], 'confidence': 'low', 'error': str(e)}

    def update_model_with_feedback(self, assessment_result: Dict[str, Any], actual_outcome: Dict[str, Any]) -> bool:
        """
        Update the risk assessment model based on feedback and actual outcomes.

        Args:
            assessment_result: Original risk assessment
            actual_outcome: Actual health outcome data

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Extract key data for comparison
            predicted_risk_level = assessment_result.get('risk_level', 'unknown')
            actual_risk_level = actual_outcome.get('actual_risk_level', 'unknown')

            # Skip update if insufficient data
            if predicted_risk_level == 'unknown' or actual_risk_level == 'unknown':
                logger.warning("Insufficient data for model update")
                return False

            # Determine if prediction was accurate
            accurate = predicted_risk_level == actual_risk_level

            # Adjustment direction
            adjustment_factor = 1.05 if not accurate else 0.98

            # Get symptoms from original assessment
            symptoms = assessment_result.get('symptoms', [])

            # Adjust symptom severity weights based on accuracy
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                if symptom_lower in self.symptom_severity:
                    if not accurate:
                        # If prediction was too low, increase severity weights
                        if predicted_risk_level in ['low', 'medium'] and actual_risk_level in ['medium', 'high']:
                            self.symptom_severity[symptom_lower] = min(10, self.symptom_severity[symptom_lower] * adjustment_factor)
                        # If prediction was too high, decrease severity weights
                        elif predicted_risk_level in ['medium', 'high'] and actual_risk_level in ['low', 'medium']:
                            self.symptom_severity[symptom_lower] = max(1, self.symptom_severity[symptom_lower] / adjustment_factor)

            # Save updated model
            self._save_model()

            logger.info(f"Updated risk assessment model based on feedback (accurate: {accurate})")

            return True
        except Exception as e:
            logger.error(f"Error updating model with feedback: {e}", exc_info=True)
            return False

    def simulate_risk_assessment(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a risk assessment for demonstration purposes.

        Args:
            input_data: Dictionary containing input parameters

        Returns:
            Dictionary containing simulated risk assessment results
        """
        try:
            # Extract input parameters
            symptoms = input_data.get('symptoms', [])
            conditions = input_data.get('conditions', [])
            user_profile = input_data.get('user_profile', {})

            # Perform risk assessment
            assessment = self.assess_risk(symptoms, conditions, user_profile)

            # Add simulation metadata
            assessment['simulation'] = True
            assessment['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            assessment['model_version'] = '1.0'

            logger.info(f"Generated simulated risk assessment for {len(symptoms)} symptoms and {len(conditions) if conditions else 0} conditions")

            return assessment
        except Exception as e:
            logger.error(f"Error simulating risk assessment: {e}", exc_info=True)
            return {'risk_level': 'unknown', 'risk_score': 0.0, 'error': str(e), 'simulation': True}

    def get_symptom_list(self) -> List[str]:
        """
        Get a list of all known symptoms in the model.

        Returns:
            List of symptom names
        """
        return list(self.symptom_severity.keys())

    def get_condition_list(self) -> List[str]:
        """
        Get a list of all known medical conditions in the model.

        Returns:
            List of condition names
        """
        return list(self.condition_risks.keys())

    def add_custom_symptom(self, symptom: str, severity: float) -> bool:
        """
        Add a custom symptom to the model with a specified severity.

        Args:
            symptom: Symptom name
            severity: Severity score (0-10)

        Returns:
            True if addition was successful, False otherwise
        """
        try:
            symptom_lower = symptom.lower()
            self.symptom_severity[symptom_lower] = min(10, max(0, severity))
            self._save_model()
            logger.info(f"Added custom symptom: {symptom} with severity {severity}")
            return True
        except Exception as e:
            logger.error(f"Error adding custom symptom: {e}", exc_info=True)
            return False

    def add_custom_condition(self, condition: str, risk: float) -> bool:
        """
        Add a custom medical condition to the model with a specified risk level.

        Args:
            condition: Condition name
            risk: Risk level (0-10)

        Returns:
            True if addition was successful, False otherwise
        """
        try:
            condition_lower = condition.lower()
            self.condition_risks[condition_lower] = min(10, max(0, risk))
            self._save_model()
            logger.info(f"Added custom condition: {condition} with risk {risk}")
            return True
        except Exception as e:
            logger.error(f"Error adding custom condition: {e}", exc_info=True)
            return False

    def reset_to_defaults(self) -> bool:
        """
        Reset the model to default values.

        Returns:
            True if reset was successful, False otherwise
        """
        try:
            self._create_default_model()
            logger.info("Reset risk assessment model to defaults")
            return True
        except Exception as e:
            logger.error(f"Error resetting model to defaults: {e}", exc_info=True)
            return False

    def export_model(self, export_path: str = None) -> str:
        """
        Export the risk assessment model to a JSON file.

        Args:
            export_path: Optional path for export file

        Returns:
            Path to the exported model file
        """
        try:
            if export_path is None:
                export_path = os.path.join(self.model_dir, "risk_model_export.json")

            model_data = {
                'risk_factors': self.risk_factors,
                'symptom_severity': self.symptom_severity,
                'condition_risks': self.condition_risks,
                'age_risk_modifiers': self.age_risk_modifiers,
                'comorbidity_risks': self.comorbidity_risks,
                'version': '1.0',
                'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(export_path, 'w') as f:
                json.dump(model_data, f, indent=4)

            logger.info(f"Exported risk assessment model to {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"Error exporting risk assessment model: {e}", exc_info=True)
            return ""

    def import_model(self, import_path: str) -> bool:
        """
        Import a risk assessment model from a JSON file.

        Args:
            import_path: Path to the model file

        Returns:
            True if import was successful, False otherwise
        """
        try:
            with open(import_path, 'r') as f:
                model_data = json.load(f)

            # Validate required keys
            required_keys = ['risk_factors', 'symptom_severity', 'condition_risks',
                            'age_risk_modifiers', 'comorbidity_risks']

            if not all(key in model_data for key in required_keys):
                logger.error(f"Invalid model file format: missing required keys")
                return False

            # Update model data
            self.risk_factors = model_data['risk_factors']
            self.symptom_severity = model_data['symptom_severity']
            self.condition_risks = model_data['condition_risks']
            self.age_risk_modifiers = model_data['age_risk_modifiers']
            self.comorbidity_risks = model_data['comorbidity_risks']

            # Save to disk
            self._save_model()

            logger.info(f"Imported risk assessment model from {import_path}")
            return True
        except Exception as e:
            logger.error(f"Error importing risk assessment model: {e}", exc_info=True)
            return False
