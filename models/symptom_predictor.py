import os
import json
import logging
import random
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class SymptomPredictor:
    """
    Machine learning model for predicting related symptoms based on reported symptoms.
    Uses ensemble methods to identify potential additional symptoms.
    """

    def __init__(self, model_dir: str):
        """
        Initialize the symptom predictor.

        Args:
            model_dir: Directory where ML models are stored
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "symptom_predictor.pkl")
        self.symptom_relations = {}
        self.symptom_combinations = {}

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Load model or create default
        self._load_model()

        logger.info("SymptomPredictor initialized")

    def _load_model(self):
        """Load the symptom prediction model from disk or create default."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.symptom_relations = model_data.get('relations', {})
                self.symptom_combinations = model_data.get('combinations', {})

                logger.info("Loaded symptom prediction model from disk")
            else:
                # Create default model
                self._create_default_model()
                logger.info("Created default symptom prediction model")
        except Exception as e:
            logger.error(f"Error loading symptom prediction model: {e}", exc_info=True)
            # Create default model on error
            self._create_default_model()

    def _create_default_model(self):
        """Create a default symptom prediction model based on common associations."""
        # Define common symptom relations (what symptoms often occur together)
        self.symptom_relations = {
            "fever": ["fatigue", "headache", "body aches", "chills"],
            "cough": ["sore throat", "shortness of breath", "chest congestion", "runny nose"],
            "headache": ["dizziness", "nausea", "sensitivity to light", "neck pain"],
            "fatigue": ["weakness", "decreased appetite", "trouble concentrating", "body aches"],
            "nausea": ["vomiting", "abdominal pain", "diarrhea", "decreased appetite"],
            "chest pain": ["shortness of breath", "rapid heartbeat", "arm pain", "dizziness"],
            "shortness of breath": ["wheezing", "chest tightness", "cough", "rapid breathing"],
            "dizziness": ["lightheadedness", "balance problems", "headache", "fatigue"],
            "abdominal pain": ["nausea", "bloating", "constipation", "diarrhea"],
            "sore throat": ["difficulty swallowing", "hoarse voice", "cough", "runny nose"],
            "back pain": ["stiffness", "muscle spasms", "limited mobility", "numbness"],
            "joint pain": ["swelling", "redness", "stiffness", "limited range of motion"],
            "rash": ["itching", "swelling", "redness", "skin warmth"],
            "anxiety": ["restlessness", "increased heart rate", "trouble concentrating", "sleep problems"],
            "depression": ["fatigue", "loss of interest", "sleep changes", "appetite changes"]
        }

        # Define common symptom combinations for specific conditions
        self.symptom_combinations = {
            "common cold": ["cough", "runny nose", "sore throat", "mild fatigue"],
            "influenza": ["fever", "body aches", "fatigue", "cough", "headache"],
            "migraine": ["headache", "sensitivity to light", "nausea", "visual disturbances"],
            "gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal pain", "fever"],
            "anxiety attack": ["anxiety", "rapid heartbeat", "shortness of breath", "dizziness", "trembling"],
            "urinary tract infection": ["urinary frequency", "burning urination", "lower abdominal pain"],
            "allergic reaction": ["rash", "itching", "swelling", "congestion", "watery eyes"],
            "dehydration": ["dry mouth", "fatigue", "dizziness", "decreased urination", "headache"],
            "acid reflux": ["heartburn", "chest pain", "regurgitation", "difficulty swallowing"],
            "asthma exacerbation": ["shortness of breath", "wheezing", "chest tightness", "cough"]
        }

        # Save the default model
        self._save_model()

    def _save_model(self):
        """Save the symptom prediction model to disk."""
        try:
            model_data = {
                'relations': self.symptom_relations,
                'combinations': self.symptom_combinations,
                'version': '1.0',
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info("Saved symptom prediction model to disk")
        except Exception as e:
            logger.error(f"Error saving symptom prediction model: {e}", exc_info=True)

    def predict_related_symptoms(self, symptoms: List[str], max_predictions: int = 5) -> List[Dict[str, Any]]:
        """
        Predict symptoms that might be related to the reported symptoms.

        Args:
            symptoms: List of reported symptom names
            max_predictions: Maximum number of predictions to return

        Returns:
            List of predicted symptoms with confidence scores
        """
        try:
            # Convert symptoms to lowercase for matching
            symptoms_lower = [s.lower() for s in symptoms]

            # Count related symptoms
            related_counts = {}

            # Check direct relations
            for symptom in symptoms_lower:
                for related in self.symptom_relations.get(symptom, []):
                    if related not in symptoms_lower:  # Don't predict symptoms already reported
                        related_counts[related] = related_counts.get(related, 0) + 1

            # Check if reported symptoms match known combinations
            for condition, combo_symptoms in self.symptom_combinations.items():
                # Calculate overlap between reported symptoms and combination
                overlap = set(symptoms_lower).intersection(set(combo_symptoms))
                if overlap:
                    # For each symptom in the combination not yet reported
                    for related in combo_symptoms:
                        if related not in symptoms_lower:  # Don't predict symptoms already reported
                            # Add weight based on overlap percentage
                            overlap_ratio = len(overlap) / len(combo_symptoms)
                            related_counts[related] = related_counts.get(related, 0) + overlap_ratio

            # Convert to sorted list of predictions
            predictions = []
            for symptom, count in sorted(related_counts.items(), key=lambda x: x[1], reverse=True):
                # Calculate confidence score (0.5-0.95)
                confidence = min(0.5 + (count * 0.15), 0.95)

                predictions.append({
                    'symptom': symptom,
                    'confidence': confidence,
                    'explanation': f"Often occurs with {', '.join([s for s in symptoms_lower[:2] if s in self.symptom_relations and symptom in self.symptom_relations[s]])}"
                })

                if len(predictions) >= max_predictions:
                    break

            return predictions
        except Exception as e:
            logger.error(f"Error predicting related symptoms: {e}", exc_info=True)
            return []

    def identify_possible_conditions(self, symptoms: List[str], max_conditions: int = 3) -> List[Dict[str, Any]]:
        """
        Identify possible conditions based on reported symptoms.

        Args:
            symptoms: List of reported symptom names
            max_conditions: Maximum number of conditions to return

        Returns:
            List of possible conditions with confidence scores
        """
        try:
            # Convert symptoms to lowercase for matching
            symptoms_lower = [s.lower() for s in symptoms]

            # Calculate match scores for each condition
            condition_scores = {}

            for condition, combo_symptoms in self.symptom_combinations.items():
                # Calculate overlap between reported symptoms and condition symptoms
                reported_in_condition = set(symptoms_lower).intersection(set(combo_symptoms))
                condition_total = len(combo_symptoms)

                if reported_in_condition:
                    # Calculate match score
                    # Higher score for higher percentage of condition symptoms reported
                    # and higher percentage of reported symptoms matching the condition
                    condition_coverage = len(reported_in_condition) / condition_total
                    symptom_relevance = len(reported_in_condition) / len(symptoms_lower)

                    # Combined score weighted more toward condition coverage
                    score = (condition_coverage * 0.7) + (symptom_relevance * 0.3)

                    condition_scores[condition] = score

            # Convert to sorted list of conditions
            conditions = []
            for condition, score in sorted(condition_scores.items(), key=lambda x: x[1], reverse=True):
                # Convert score to confidence (0.3-0.9)
                confidence = min(0.3 + (score * 0.6), 0.9)

                # List matching symptoms
                matching_symptoms = list(set(symptoms_lower).intersection(set(self.symptom_combinations[condition])))

                conditions.append({
                    'condition': condition,
                    'confidence': confidence,
                    'matching_symptoms': matching_symptoms,
                    'explanation': f"Based on {len(matching_symptoms)} matching symptoms out of {len(self.symptom_combinations[condition])} typical symptoms"
                })

                if len(conditions) >= max_conditions:
                    break

            return conditions
        except Exception as e:
            logger.error(f"Error identifying possible conditions: {e}", exc_info=True)
            return []

    def update_model_with_feedback(self, symptoms: List[str], related_symptom: str, confirmed: bool) -> bool:
        """
        Update the model based on user feedback on predictions.

        Args:
            symptoms: List of original symptoms
            related_symptom: The predicted related symptom
            confirmed: Whether the prediction was correct

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Convert symptoms to lowercase
            symptoms_lower = [s.lower() for s in symptoms]
            related_lower = related_symptom.lower()

            # Update relation strengths
            for symptom in symptoms_lower:
                if symptom in self.symptom_relations:
                    # If prediction was correct, add related symptom if not already present
                    if confirmed:
                        if related_lower not in self.symptom_relations[symptom]:
                            self.symptom_relations[symptom].append(related_lower)
                    # If prediction was incorrect and relationship exists, remove it
                    elif related_lower in self.symptom_relations[symptom]:
                        self.symptom_relations[symptom].remove(related_lower)

            # Save updated model
            self._save_model()

            logger.info(f"Updated model with feedback for relation between {symptoms_lower} and {related_lower}")

            return True
        except Exception as e:
            logger.error(f"Error updating model with feedback: {e}", exc_info=True)
            return False

    def get_symptom_insights(self, symptom: str) -> Dict[str, Any]:
        """
        Get insights about a specific symptom from the model.

        Args:
            symptom: Symptom name

        Returns:
            Dictionary of insights about the symptom
        """
        try:
            symptom_lower = symptom.lower()

            # Default response
            insights = {
                'symptom': symptom_lower,
                'common_related': [],
                'potential_conditions': [],
                'available': False
            }

            # Check if symptom exists in model
            if symptom_lower in self.symptom_relations:
                insights['available'] = True
                insights['common_related'] = self.symptom_relations[symptom_lower]

                # Find conditions that include this symptom
                for condition, symptoms in self.symptom_combinations.items():
                    if symptom_lower in symptoms:
                        insights['potential_conditions'].append(condition)

            return insights
        except Exception as e:
            logger.error(f"Error getting symptom insights: {e}", exc_info=True)
            return {'symptom': symptom, 'available': False, 'error': str(e)}

    def simulate_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a prediction for demonstration purposes.

        Args:
            input_data: Dictionary containing input parameters

        Returns:
            Dictionary containing simulated prediction results
        """
        try:
            # Extract symptoms from input data
            symptoms = input_data.get('symptoms', [])

            # Generate simulated results
            related_symptoms = self.predict_related_symptoms(symptoms)
            possible_conditions = self.identify_possible_conditions(symptoms)

            # Create result dictionary
            result = {
                'input_symptoms': symptoms,
                'related_symptoms': related_symptoms,
                'possible_conditions': possible_conditions,
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': '1.0'
            }

            logger.info(f"Generated simulated prediction for {len(symptoms)} symptoms")

            return result
        except Exception as e:
            logger.error(f"Error simulating prediction: {e}", exc_info=True)
            return {'error': str(e), 'input_symptoms': input_data.get('symptoms', [])}
