import os
import json
import logging
import re
import pickle
import random
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class NLPSymptomExtractor:
    """
    Natural Language Processing model for extracting symptoms from text descriptions.
    Uses keyword matching, pattern recognition, and context understanding.
    """

    def __init__(self, model_dir: str):
        """
        Initialize the NLP symptom extractor.

        Args:
            model_dir: Directory where ML models are stored
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "symptom_extractor.pkl")
        self.symptoms_dict = {}
        self.symptom_patterns = {}
        self.negation_terms = []
        self.intensity_modifiers = {}
        self.anatomical_locations = {}

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Load model or create default
        self._load_model()

        logger.info("NLPSymptomExtractor initialized")

    def _load_model(self):
        """Load the symptom extraction model from disk or create default."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.symptoms_dict = model_data.get('symptoms', {})
                self.symptom_patterns = model_data.get('patterns', {})
                self.negation_terms = model_data.get('negation_terms', [])
                self.intensity_modifiers = model_data.get('intensity_modifiers', {})
                self.anatomical_locations = model_data.get('anatomical_locations', {})

                logger.info("Loaded symptom extraction model from disk")
            else:
                # Create default model
                self._create_default_model()
                logger.info("Created default symptom extraction model")
        except Exception as e:
            logger.error(f"Error loading symptom extraction model: {e}", exc_info=True)
            # Create default model on error
            self._create_default_model()

    def _create_default_model(self):
        """Create a default symptom extraction model with common patterns and terms."""
        # Define common symptoms with variations and synonyms
        self.symptoms_dict = {
            "fever": ["fever", "temperature", "hot", "burning up", "feverish", "high temperature"],
            "cough": ["cough", "coughing", "hacking", "hack"],
            "headache": ["headache", "head ache", "head pain", "migraine", "head pounding", "throbbing head"],
            "fatigue": ["fatigue", "tired", "exhausted", "no energy", "weakness", "lethargic", "exhaustion"],
            "nausea": ["nausea", "nauseated", "feel sick", "queasy", "sick to my stomach"],
            "chest pain": ["chest pain", "pain in chest", "chest discomfort", "chest tightness", "pressure in chest"],
            "shortness of breath": ["shortness of breath", "sob", "difficulty breathing", "hard to breathe", "breathless", "can't catch breath"],
            "dizziness": ["dizziness", "dizzy", "lightheaded", "vertigo", "room spinning", "unsteady"],
            "abdominal pain": ["abdominal pain", "stomach pain", "tummy ache", "belly pain", "stomachache", "pain in abdomen"],
            "sore throat": ["sore throat", "throat pain", "painful throat", "throat irritation", "scratchy throat"]
        }

        # Define common symptom patterns in natural language
        self.symptom_patterns = {
            r"\b(my|having|have|had|experiencing|with)\s+(\w+\s*)+(pain|ache)\b": "pain",
            r"\b(not|cannot|can't|couldn't)\s+(\w+\s*)+(sleep|rest|relax)\b": "insomnia",
            r"\b(feel(ing)?|am|been)\s+(\w+\s*)+(sick|ill|unwell)\b": "general illness",
            r"\b(have|having|had|with)\s+(\w+\s*)+(rash|hives|bumps)\b": "skin condition",
            r"\b(my|the)\s+(\w+\s*)+(is|are|feels?)\s+(\w+\s*)+(swollen|inflamed|puffy)\b": "swelling"
        }

        # Define negation terms to avoid false positives
        self.negation_terms = [
            "no", "not", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't",
            "haven't", "hasn't", "hadn't", "never", "none", "without", "deny", "denies",
            "denied", "doesn't have", "don't have", "free of", "negative for"
        ]

        # Define intensity modifiers
        self.intensity_modifiers = {
            "mild": 0.3,
            "slight": 0.3,
            "little": 0.3,
            "minor": 0.3,
            "moderate": 0.6,
            "significant": 0.6,
            "considerable": 0.6,
            "severe": 0.9,
            "extreme": 0.9,
            "intense": 0.9,
            "terrible": 0.9,
            "worst": 1.0,
            "excruciating": 1.0
        }

        # Define anatomical locations
        self.anatomical_locations = {
            "head": ["head", "cranial", "skull", "brain"],
            "chest": ["chest", "thoracic", "thorax", "breast", "ribcage"],
            "abdomen": ["abdomen", "stomach", "belly", "gut", "tummy"],
            "back": ["back", "spine", "spinal", "lumbar", "thoracic spine"],
            "arm": ["arm", "upper limb", "forearm", "elbow", "wrist", "hand"],
            "leg": ["leg", "lower limb", "thigh", "knee", "calf", "ankle", "foot"]
        }

        # Save the default model
        self._save_model()

    def _save_model(self):
        """Save the symptom extraction model to disk."""
        try:
            model_data = {
                'symptoms': self.symptoms_dict,
                'patterns': self.symptom_patterns,
                'negation_terms': self.negation_terms,
                'intensity_modifiers': self.intensity_modifiers,
                'anatomical_locations': self.anatomical_locations,
                'version': '1.0',
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info("Saved symptom extraction model to disk")
        except Exception as e:
            logger.error(f"Error saving symptom extraction model: {e}", exc_info=True)

    def extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract symptoms from natural language text.

        Args:
            text: Natural language text describing symptoms

        Returns:
            List of extracted symptoms with confidence scores
        """
        try:
            # Convert text to lowercase for matching
            text_lower = text.lower()

            # Initialize results
            extracted_symptoms = []
            seen_symptoms = set()

            # 1. Direct symptom matching
            for symptom, variants in self.symptoms_dict.items():
                for variant in variants:
                    if variant in text_lower:
                        # Check for negation context
                        if not self._is_negated(text_lower, variant):
                            if symptom not in seen_symptoms:
                                # Extract intensity modifiers
                                intensity = self._extract_intensity(text_lower, variant)

                                # Extract anatomical location if applicable
                                location = self._extract_location(text_lower, variant)

                                extracted_symptoms.append({
                                    'symptom': symptom,
                                    'confidence': 0.8 * intensity,
                                    'match': variant,
                                    'intensity': intensity,
                                    'location': location
                                })
                                seen_symptoms.add(symptom)

            # 2. Pattern-based extraction
            for pattern, symptom_type in self.symptom_patterns.items():
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    matched_text = match.group(0)
                    if not self._is_negated(text_lower, matched_text):
                        if symptom_type not in seen_symptoms:
                            # Extract intensity modifiers
                            intensity = self._extract_intensity(text_lower, matched_text)

                            # Extract anatomical location if applicable
                            location = self._extract_location(text_lower, matched_text)

                            extracted_symptoms.append({
                                'symptom': symptom_type,
                                'confidence': 0.6 * intensity,  # Lower confidence for pattern matches
                                'match': matched_text,
                                'intensity': intensity,
                                'location': location
                            })
                            seen_symptoms.add(symptom_type)

            # Sort by confidence
            extracted_symptoms.sort(key=lambda x: x['confidence'], reverse=True)

            return extracted_symptoms
        except Exception as e:
            logger.error(f"Error extracting symptoms: {e}", exc_info=True)
            return []

    def _is_negated(self, text: str, symptom: str) -> bool:
        """
        Check if a symptom is negated in the text.

        Args:
            text: Full text
            symptom: Symptom to check for negation

        Returns:
            True if negated, False otherwise
        """
        # Find symptom position
        pos = text.find(symptom)
        if pos == -1:
            return False

        # Check for negation terms before the symptom (within 5 words)
        context_before = text[:pos].split()[-5:]
        for term in self.negation_terms:
            if term in context_before:
                return True

        return False

    def _extract_intensity(self, text: str, symptom: str) -> float:
        """
        Extract intensity modifier for a symptom.

        Args:
            text: Full text
            symptom: Symptom to check for intensity

        Returns:
            Intensity score (0.0-1.0), default is 0.7
        """
        # Default intensity if no modifier found
        default_intensity = 0.7

        # Find symptom position
        pos = text.find(symptom)
        if pos == -1:
            return default_intensity

        # Check for intensity modifiers before the symptom (within 3 words)
        context_before = text[:pos].split()[-3:]
        for word in context_before:
            if word in self.intensity_modifiers:
                return self.intensity_modifiers[word]

        return default_intensity

    def _extract_location(self, text: str, symptom: str) -> Optional[str]:
        """
        Extract anatomical location for a symptom if available.

        Args:
            text: Full text
            symptom: Symptom to check for location

        Returns:
            Anatomical location or None if not found
        """
        # Check for locations in the text
        for location, terms in self.anatomical_locations.items():
            for term in terms:
                if term in text:
                    return location

        return None

    def get_symptom_summary(self, text: str) -> Dict[str, Any]:
        """
        Generate a summary of extracted symptoms.

        Args:
            text: Natural language text describing symptoms

        Returns:
            Summary of extracted symptoms
        """
        try:
            # Extract symptoms
            symptoms = self.extract_symptoms(text)

            # Count symptoms by category
            categories = {}
            for symptom in symptoms:
                location = symptom.get('location')
                if location:
                    if location not in categories:
                        categories[location] = []
                    categories[location].append(symptom['symptom'])

            # Generate summary
            summary = {
                'total_symptoms': len(symptoms),
                'primary_symptoms': [s['symptom'] for s in symptoms[:3]] if symptoms else [],
                'symptom_categories': categories,
                'confidence': sum(s['confidence'] for s in symptoms) / max(1, len(symptoms)),
                'extracted_symptoms': symptoms
            }

            return summary
        except Exception as e:
            logger.error(f"Error generating symptom summary: {e}", exc_info=True)
            return {'total_symptoms': 0, 'primary_symptoms': [], 'symptom_categories': {}, 'confidence': 0.0, 'error': str(e)}

    def analyze_symptom_text(self, text: str, extract_severity: bool = True, extract_duration: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of symptom text including severity and duration.

        Args:
            text: Natural language text describing symptoms
            extract_severity: Whether to extract symptom severity
            extract_duration: Whether to extract symptom duration

        Returns:
            Comprehensive analysis result
        """
        try:
            # Extract basic symptoms
            symptoms = self.extract_symptoms(text)

            # Initialize analysis
            analysis = {
                'symptoms': symptoms,
                'severity': None,
                'duration': None,
                'onset': None,
                'triggers': [],
                'relieving_factors': []
            }

            # Extract severity if requested
            if extract_severity:
                severity_terms = {
                    'mild': ['mild', 'slight', 'minor', 'light', 'barely'],
                    'moderate': ['moderate', 'medium', 'average', 'somewhat'],
                    'severe': ['severe', 'intense', 'extreme', 'worst', 'terrible', 'unbearable', 'excruciating']
                }

                text_lower = text.lower()

                for level, terms in severity_terms.items():
                    for term in terms:
                        if re.search(r'\b' + term + r'\b', text_lower):
                            analysis['severity'] = level
                            break
                    if analysis['severity']:
                        break

            # Extract duration if requested
            if extract_duration:
                duration_patterns = [
                    (r'\b(\d+)\s*days?\b', lambda x: f"{x[1]} days"),
                    (r'\b(\d+)\s*weeks?\b', lambda x: f"{x[1]} weeks"),
                    (r'\b(\d+)\s*months?\b', lambda x: f"{x[1]} months"),
                    (r'\bsince\s+(\w+)\b', lambda x: f"since {x[1]}"),
                    (r'\bfor\s+(\w+\s+\w+)\b', lambda x: f"for {x[1]}"),
                    (r'\bstarted\s+(\w+\s+\w+)\s+ago\b', lambda x: f"{x[1]} ago")
                ]

                for pattern, formatter in duration_patterns:
                    match = re.search(pattern, text.lower())
                    if match:
                        analysis['duration'] = formatter(match)
                        break

            # Extract potential triggers
            trigger_patterns = [
                r'(?:after|when|from|during)\s+(\w+\s+\w+)',
                r'(?:triggered|worsened)\s+by\s+(\w+\s+\w+)',
                r'(?:happened|started)\s+(?:after|when|from)\s+(\w+\s+\w+)'
            ]

            for pattern in trigger_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    trigger = match.group(1)
                    if trigger and trigger not in analysis['triggers']:
                        analysis['triggers'].append(trigger)

            # Extract relieving factors
            relief_patterns = [
                r'(?:better|improved|relieved)\s+(?:with|by|from)\s+(\w+\s+\w+)',
                r'(\w+\s+\w+)\s+(?:helps|helped|relieves|relieved)',
                r'(?:taking|using)\s+(\w+\s+\w+)\s+(?:for|to help)'
            ]

            for pattern in relief_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    factor = match.group(1)
                    if factor and factor not in analysis['relieving_factors']:
                        analysis['relieving_factors'].append(factor)

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing symptom text: {e}", exc_info=True)
            return {'symptoms': [], 'error': str(e)}

    def extract_medications(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mentioned medications from text.

        Args:
            text: Natural language text

        Returns:
            List of extracted medications with confidence scores
        """
        try:
            # Common medication names and classes (simplified for demo)
            medications = {
                "antibiotic": ["antibiotic", "amoxicillin", "penicillin", "azithromycin", "ciprofloxacin", "doxycycline"],
                "painkiller": ["painkiller", "aspirin", "ibuprofen", "tylenol", "advil", "motrin", "acetaminophen", "naproxen", "aleve"],
                "antihistamine": ["antihistamine", "benadryl", "claritin", "zyrtec", "allegra", "cetirizine", "loratadine"],
                "decongestant": ["decongestant", "sudafed", "pseudoephedrine", "phenylephrine"],
                "antacid": ["antacid", "tums", "rolaids", "maalox", "pepto-bismol"],
                "antihypertensive": ["blood pressure medication", "lisinopril", "metoprolol", "amlodipine", "losartan"],
                "antidepressant": ["antidepressant", "prozac", "zoloft", "lexapro", "fluoxetine", "sertraline", "escitalopram"],
                "steroid": ["steroid", "prednisone", "cortisone", "hydrocortisone", "dexamethasone"]
            }

            # Convert text to lowercase for matching
            text_lower = text.lower()

            # Extract medications
            extracted_meds = []
            seen_meds = set()

            for med_class, terms in medications.items():
                for term in terms:
                    if term in text_lower:
                        # Check if not negated
                        if not self._is_negated(text_lower, term):
                            # Find the specific term matched
                            specific_med = term

                            # Avoid duplicates within the same class
                            if med_class not in seen_meds:
                                extracted_meds.append({
                                    'medication_class': med_class,
                                    'specific_medication': specific_med,
                                    'confidence': 0.85 if specific_med != med_class else 0.7,
                                    'match': term
                                })
                                seen_meds.add(med_class)

            return extracted_meds
        except Exception as e:
            logger.error(f"Error extracting medications: {e}", exc_info=True)
            return []

    def update_model_with_text(self, text: str, confirmed_symptoms: List[str]) -> bool:
        """
        Update the extraction model based on confirmed symptoms from text.

        Args:
            text: Original text
            confirmed_symptoms: List of confirmed symptom names

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Extract current symptoms
            extracted = self.extract_symptoms(text)
            extracted_symptom_names = [s['symptom'] for s in extracted]

            # Find missed symptoms (in confirmed but not extracted)
            missed_symptoms = [s for s in confirmed_symptoms if s not in extracted_symptom_names]

            # Find false positives (in extracted but not confirmed)
            false_positives = [s for s in extracted_symptom_names if s not in confirmed_symptoms]

            # Update model based on feedback
            if missed_symptoms:
                # For missed symptoms, try to find what terms we should have matched
                text_lower = text.lower()
                for symptom in missed_symptoms:
                    # If symptom exists in dictionary, expand variants
                    if symptom in self.symptoms_dict:
                        # Look for new variants in text that aren't in our dictionary
                        words = text_lower.split()
                        for i in range(len(words)):
                            # Check 1-3 word combinations
                            for j in range(1, 4):
                                if i + j <= len(words):
                                    phrase = " ".join(words[i:i+j])
                                    # Check if phrase might be a variant (contains key part of symptom name)
                                    if symptom in phrase and phrase not in self.symptoms_dict[symptom]:
                                        self.symptoms_dict[symptom].append(phrase)
                    else:
                        # New symptom - create entry
                        self.symptoms_dict[symptom] = [symptom]

            # Save updated model
            self._save_model()

            logger.info(f"Updated model with {len(missed_symptoms)} missed symptoms and {len(false_positives)} false positives")

            return True
        except Exception as e:
            logger.error(f"Error updating model with text: {e}", exc_info=True)
            return False

    def extract_temporal_information(self, text: str) -> Dict[str, Any]:
        """
        Extract temporal information about symptoms.

        Args:
            text: Natural language text

        Returns:
            Dictionary of temporal information
        """
        try:
            temporal_info = {
                'onset': None,
                'duration': None,
                'frequency': None,
                'progression': None,
                'time_patterns': []
            }

            text_lower = text.lower()

            # Extract onset
            onset_patterns = [
                (r'started\s+(\w+)', lambda x: x[1]),
                (r'began\s+(\w+)', lambda x: x[1]),
                (r'noticed\s+(\w+)', lambda x: x[1]),
                (r'since\s+(\w+)', lambda x: x[1])
            ]

            for pattern, formatter in onset_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    temporal_info['onset'] = formatter(match)
                    break

            # Extract duration (already implemented in analyze_symptom_text)
            duration_patterns = [
                (r'(\d+)\s*days?', lambda x: f"{x[1]} days"),
                (r'(\d+)\s*weeks?', lambda x: f"{x[1]} weeks"),
                (r'(\d+)\s*months?', lambda x: f"{x[1]} months"),
                (r'for\s+(\w+\s+\w+)', lambda x: f"for {x[1]}")
            ]

            for pattern, formatter in duration_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    temporal_info['duration'] = formatter(match)
                    break

            # Extract frequency
            frequency_patterns = [
                (r'(\d+)\s*times\s+a\s+day', lambda x: f"{x[1]} times a day"),
                (r'(\d+)\s*times\s+a\s+week', lambda x: f"{x[1]} times a week"),
                (r'(\w+)\s+day', lambda x: f"{x[1]} day"),
                (r'(\w+)\s+week', lambda x: f"{x[1]} week"),
                (r'(\w+)\s+month', lambda x: f"{x[1]} month")
            ]

            for pattern, formatter in frequency_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    temporal_info['frequency'] = formatter(match)
                    break

            # Extract progression
            progression_terms = {
                'improving': ['better', 'improving', 'improved', 'getting better', 'subsiding', 'decreasing'],
                'worsening': ['worse', 'worsening', 'getting worse', 'increasing', 'intensifying'],
                'stable': ['same', 'unchanged', 'stable', 'steady', 'constant', 'not changing']
            }

            for progression, terms in progression_terms.items():
                for term in terms:
                    if term in text_lower:
                        temporal_info['progression'] = progression
                        break
                if temporal_info['progression']:
                    break

            # Extract time patterns
            time_patterns = [
                r'(in the morning)',
                r'(at night)',
                r'(in the evening)',
                r'(after meals)',
                r'(before meals)',
                r'(after exercise)',
                r'(during exercise)',
                r'(when lying down)',
                r'(when standing)',
                r'(when sitting)'
            ]

            for pattern in time_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    temporal_info['time_patterns'].append(match.group(1))

            return temporal_info
        except Exception as e:
            logger.error(f"Error extracting temporal information: {e}", exc_info=True)
            return {'onset': None, 'duration': None, 'frequency': None, 'progression': None, 'time_patterns': [], 'error': str(e)}

    def extract_health_report(self, text: str) -> Dict[str, Any]:
        """
        Generate a comprehensive health report from user text.

        Args:
            text: Natural language text describing health issues

        Returns:
            Comprehensive health report
        """
        try:
            report = {
                'symptoms': self.extract_symptoms(text),
                'medications': self.extract_medications(text),
                'temporal_info': self.extract_temporal_information(text),
                'summary': '',
                'potential_concerns': []
            }

            # Generate summary based on extracted information
            symptom_count = len(report['symptoms'])
            primary_symptoms = [s['symptom'] for s in report['symptoms'][:3]] if report['symptoms'] else []

            summary_parts = []

            if symptom_count > 0:
                if symptom_count == 1:
                    summary_parts.append(f"Reported {symptom_count} symptom: {primary_symptoms[0]}.")
                else:
                    symptoms_text = ", ".join(primary_symptoms[:-1]) + " and " + primary_symptoms[-1]
                    summary_parts.append(f"Reported {symptom_count} symptoms, primarily {symptoms_text}.")

            if report['temporal_info']['onset']:
                summary_parts.append(f"Symptoms began {report['temporal_info']['onset']}.")

            if report['temporal_info']['duration']:
                summary_parts.append(f"Symptoms have been present {report['temporal_info']['duration']}.")

            if report['temporal_info']['progression']:
                summary_parts.append(f"Condition appears to be {report['temporal_info']['progression']}.")

            if report['medications']:
                med_names = [m['specific_medication'] for m in report['medications']]
                if len(med_names) == 1:
                    summary_parts.append(f"Currently taking {med_names[0]}.")
                else:
                    meds_text = ", ".join(med_names[:-1]) + " and " + med_names[-1]
                    summary_parts.append(f"Currently taking {meds_text}.")

            report['summary'] = " ".join(summary_parts)

            # Add potential concerns based on symptom combinations
            high_concern_combinations = [
                (["chest pain", "shortness of breath"], "Possible cardiopulmonary concern"),
                (["fever", "stiff neck", "headache"], "Possible meningeal irritation"),
                (["headache", "visual changes", "confusion"], "Possible neurological concern"),
                (["chest pain", "arm pain", "sweating"], "Possible cardiac concern")
            ]

            extracted_symptom_names = [s['symptom'] for s in report['symptoms']]

            for symptoms, concern in high_concern_combinations:
                if all(symptom in extracted_symptom_names for symptom in symptoms):
                    report['potential_concerns'].append(concern)

            return report
        except Exception as e:
            logger.error(f"Error generating health report: {e}", exc_info=True)
            return {'symptoms': [], 'medications': [], 'temporal_info': {}, 'summary': '', 'potential_concerns': [], 'error': str(e)}

    def extract_relevant_medical_data(self, text: str) -> Dict[str, Any]:
        """
        Extract all relevant medical data from text, including symptoms, medications,
        temporal information, and potential concerns.

        Args:
            text: Natural language text

        Returns:
            Comprehensive dictionary of medical data
        """
        try:
            # Initialize result dictionary
            medical_data = {
                'symptoms': self.extract_symptoms(text),
                'medications': self.extract_medications(text),
                'temporal_info': self.extract_temporal_information(text),
                'concerns': [],
                'allergies': [],
                'vitals': {},
                'family_history': []
            }

            # Extract allergies
            allergy_patterns = [
                r'(?:allergic|allergy) to (\w+(?:\s+\w+){0,2})',
                r'(\w+(?:\s+\w+){0,2}) allergy'
            ]

            for pattern in allergy_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    allergy = match.group(1)
                    if allergy and allergy not in medical_data['allergies']:
                        medical_data['allergies'].append(allergy)

            # Extract vitals
            vitals_patterns = {
                'temperature': r'(?:temperature|temp)[:\s]+(\d+\.?\d*)',
                'heart_rate': r'(?:heart rate|pulse|hr)[:\s]+(\d+)',
                'blood_pressure': r'(?:blood pressure|bp)[:\s]+(\d+\/\d+)',
                'respiratory_rate': r'(?:respiratory rate|breathing rate|rr)[:\s]+(\d+)',
                'oxygen_saturation': r'(?:oxygen|o2|saturation|sat)[:\s]+(\d+)'
            }

            for vital, pattern in vitals_patterns.items():
                match = re.search(pattern, text.lower())
                if match:
                    medical_data['vitals'][vital] = match.group(1)

            # Extract family history
            family_history_patterns = [
                r'(?:family|family history) of (\w+(?:\s+\w+){0,2})',
                r'(?:mother|father|parent|sibling|brother|sister) (?:has|had) (\w+(?:\s+\w+){0,2})'
            ]

            for pattern in family_history_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    condition = match.group(1)
                    if condition and condition not in medical_data['family_history']:
                        medical_data['family_history'].append(condition)

            # Identify potential concerns
            high_concern_combinations = [
                (["chest pain", "shortness of breath"], "Possible cardiopulmonary concern"),
                (["fever", "stiff neck", "headache"], "Possible meningeal irritation"),
                (["headache", "visual changes", "confusion"], "Possible neurological concern"),
                (["chest pain", "arm pain", "sweating"], "Possible cardiac concern"),
                (["abdominal pain", "fever", "vomiting"], "Possible abdominal infection")
            ]

            extracted_symptom_names = [s['symptom'] for s in medical_data['symptoms']]

            for symptoms, concern in high_concern_combinations:
                if all(any(symptom in s.lower() for s in extracted_symptom_names) for symptom in symptoms):
                    medical_data['concerns'].append(concern)

            return medical_data
        except Exception as e:
            logger.error(f"Error extracting relevant medical data: {e}", exc_info=True)
            return {'symptoms': [], 'medications': [], 'temporal_info': {}, 'concerns': [], 'allergies': [], 'vitals': {}, 'family_history': [], 'error': str(e)}

    def process_chat_message(self, message: str) -> Dict[str, Any]:
        """
        Process a chat message for health-related information.

        Args:
            message: User chat message

        Returns:
            Extracted health information with response suggestions
        """
        try:
            # Extract all medical data
            medical_data = self.extract_relevant_medical_data(message)

            # Initialize response
            response = {
                'extracted_data': medical_data,
                'has_medical_content': len(medical_data['symptoms']) > 0 or len(medical_data['medications']) > 0,
                'response_suggestions': [],
                'follow_up_questions': []
            }

            # Generate response suggestions based on extracted data
            if response['has_medical_content']:
                # Add general acknowledgment
                if medical_data['symptoms']:
                    symptom_names = [s['symptom'] for s in medical_data['symptoms'][:3]]
                    symptom_text = ", ".join(symptom_names[:-1]) + (" and " + symptom_names[-1] if len(symptom_names) > 1 else symptom_names[0])
                    response['response_suggestions'].append(f"I see you're experiencing {symptom_text}.")

                # Add medication acknowledgment
                if medical_data['medications']:
                    med_names = [m['specific_medication'] for m in medical_data['medications'][:2]]
                    med_text = " and ".join(med_names)
                    response['response_suggestions'].append(f"I notice you mentioned taking {med_text}.")

                # Add concern response if applicable
                if medical_data['concerns']:
                    response['response_suggestions'].append("Based on the symptoms you've described, it would be advisable to speak with a healthcare provider.")

                # Add temporal information if available
                if medical_data['temporal_info'].get('duration'):
                    response['response_suggestions'].append(f"You mentioned these symptoms have been present {medical_data['temporal_info']['duration']}.")

                # Generate follow-up questions
                if medical_data['symptoms']:
                    # Ask about duration if not provided
                    if not medical_data['temporal_info'].get('duration'):
                        response['follow_up_questions'].append("How long have you been experiencing these symptoms?")

                    # Ask about severity
                    response['follow_up_questions'].append("On a scale of 1-10, how would you rate the severity of your symptoms?")

                    # Ask about alleviating or aggravating factors
                    response['follow_up_questions'].append("Is there anything that makes your symptoms better or worse?")

                # Add disclaimer
                response['response_suggestions'].append("Please remember that I can provide general information, but cannot diagnose conditions or provide personalized medical advice.")
            else:
                # No medical content detected
                response['response_suggestions'].append("I don't see any specific health concerns in your message. How can I assist you with your health questions today?")

            return response
        except Exception as e:
            logger.error(f"Error processing chat message: {e}", exc_info=True)
            return {
                'extracted_data': {'symptoms': []},
                'has_medical_content': False,
                'response_suggestions': ["I'm having trouble processing your message. Could you rephrase your health concern?"],
                'follow_up_questions': [],
                'error': str(e)
            }

    def create_symptom_summary(self, text: str) -> str:
        """
        Create a concise natural language summary of symptoms from text.

        Args:
            text: Natural language text describing symptoms

        Returns:
            Concise natural language summary
        """
        try:
            # Extract data
            medical_data = self.extract_relevant_medical_data(text)
            symptoms = medical_data['symptoms']
            temporal_info = medical_data['temporal_info']

            if not symptoms:
                return "No specific symptoms were identified in the description."

            # Create summary sections
            sections = []

            # Primary symptoms
            primary_symptoms = [s['symptom'] for s in symptoms[:3]]
            if primary_symptoms:
                primary_text = ", ".join(primary_symptoms[:-1]) + (" and " + primary_symptoms[-1] if len(primary_symptoms) > 1 else primary_symptoms[0])
                sections.append(f"Primary symptoms include {primary_text}.")

            # Symptom duration
            if temporal_info.get('duration'):
                sections.append(f"Symptoms have been present {temporal_info['duration']}.")

            # Symptom progression
            if temporal_info.get('progression'):
                sections.append(f"The condition appears to be {temporal_info['progression']}.")

            # Timing patterns
            if temporal_info.get('time_patterns'):
                patterns = temporal_info['time_patterns']
                if patterns:
                    pattern_text = ", ".join(patterns[:2])
                    sections.append(f"Symptoms occur primarily {pattern_text}.")

            # Medications
            medications = medical_data['medications']
            if medications:
                med_names = [m['specific_medication'] for m in medications[:2]]
                med_text = " and ".join(med_names)
                sections.append(f"Currently taking {med_text}.")

            # Concerns
            if medical_data['concerns']:
                concern = medical_data['concerns'][0]
                sections.append(f"Note: {concern}.")

            # Join sections into paragraph
            summary = " ".join(sections)

            return summary
        except Exception as e:
            logger.error(f"Error creating symptom summary: {e}", exc_info=True)
            return "Unable to create symptom summary due to an error."

    def extract_structured_symptom_report(self, text: str) -> Dict[str, Any]:
        """
        Extract a structured symptom report from text for database storage.

        Args:
            text: Natural language text describing symptoms

        Returns:
            Structured symptom report
        """
        try:
            # Extract all medical data
            medical_data = self.extract_relevant_medical_data(text)

            # Create structured report
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symptoms': [],
                'metadata': {
                    'source': 'nlp_extraction',
                    'confidence': 0.0,
                    'version': '1.0'
                },
                'context': {}
            }

            # Process symptoms
            for symptom in medical_data['symptoms']:
                structured_symptom = {
                    'name': symptom['symptom'],
                    'confidence': symptom['confidence'],
                    'severity': None,  # To be filled if available
                    'duration': None,  # To be filled if available
                    'location': symptom.get('location')
                }

                # Add to symptoms list
                report['symptoms'].append(structured_symptom)

            # Calculate average confidence
            if report['symptoms']:
                avg_confidence = sum(s['confidence'] for s in report['symptoms']) / len(report['symptoms'])
                report['metadata']['confidence'] = round(avg_confidence, 2)

            # Add temporal context
            if medical_data['temporal_info']:
                report['context']['temporal'] = medical_data['temporal_info']

            # Add medication context
            if medical_data['medications']:
                report['context']['medications'] = [m['specific_medication'] for m in medical_data['medications']]

            # Add concerns
            if medical_data['concerns']:
                report['context']['concerns'] = medical_data['concerns']

            # Add vitals if available
            if medical_data['vitals']:
                report['context']['vitals'] = medical_data['vitals']

            return report
        except Exception as e:
            logger.error(f"Error extracting structured symptom report: {e}", exc_info=True)
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symptoms': [],
                'metadata': {
                    'source': 'nlp_extraction',
                    'confidence': 0.0,
                    'version': '1.0',
                    'error': str(e)
                },
                'context': {}
            }

    def extract_symptom_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract relationships between symptoms from text.

        Args:
            text: Natural language text describing symptoms and their relationships

        Returns:
            Dictionary of symptom relationships
        """
        try:
            # Extract symptoms
            symptoms = self.extract_symptoms(text)
            symptom_names = [s['symptom'] for s in symptoms]

            # Initialize relationship matrix
            relationships = {
                'temporal': [],  # Which symptoms came first
                'causal': [],    # Which symptoms might cause others
                'severity': {},  # Comparative severity
                'clusters': []   # Symptoms that appear related
            }

            # Look for temporal relationships
            temporal_patterns = [
                r'([\w\s]+) (before|after|following|preceded|prior to) ([\w\s]+)',
                r'([\w\s]+) (then|subsequently) ([\w\s]+)',
                r'(first|initially|originally) ([\w\s]+) (then|later|subsequently|afterwards) ([\w\s]+)'
            ]

            for pattern in temporal_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    # Extract the symptoms in the relationship
                    groups = match.groups()

                    if len(groups) == 3:  # Pattern 1 or 2
                        symptom1 = groups[0].strip()
                        relation = groups[1].strip()
                        symptom2 = groups[2].strip()

                        # Check if extracted text contains known symptoms
                        s1_match = next((s for s in symptom_names if s in symptom1), None)
                        s2_match = next((s for s in symptom_names if s in symptom2), None)

                        if s1_match and s2_match:
                            if relation in ['before', 'prior to', 'preceded']:
                                relationships['temporal'].append({'first': s1_match, 'then': s2_match})
                            elif relation in ['after', 'following', 'then', 'subsequently']:
                                relationships['temporal'].append({'first': s2_match, 'then': s1_match})

                    elif len(groups) == 4:  # Pattern 3
                        initial_marker = groups[0].strip()
                        symptom1 = groups[1].strip()
                        later_marker = groups[2].strip()
                        symptom2 = groups[3].strip()

                        # Check if extracted text contains known symptoms
                        s1_match = next((s for s in symptom_names if s in symptom1), None)
                        s2_match = next((s for s in symptom_names if s in symptom2), None)

                        if s1_match and s2_match:
                            relationships['temporal'].append({'first': s1_match, 'then': s2_match})

            # Look for causal relationships
            causal_patterns = [
                r'([\w\s]+) (caused|triggered|led to|resulted in|made) ([\w\s]+)',
                r'([\w\s]+) (because of|due to|from) ([\w\s]+)'
            ]

            for pattern in causal_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    groups = match.groups()

                    symptom1 = groups[0].strip()
                    relation = groups[1].strip()
                    symptom2 = groups[2].strip()

                    # Check if extracted text contains known symptoms
                    s1_match = next((s for s in symptom_names if s in symptom1), None)
                    s2_match = next((s for s in symptom_names if s in symptom2), None)

                    if s1_match and s2_match:
                        if relation in ['caused', 'triggered', 'led to', 'resulted in', 'made']:
                            relationships['causal'].append({'cause': s1_match, 'effect': s2_match})
                        elif relation in ['because of', 'due to', 'from']:
                            relationships['causal'].append({'cause': s2_match, 'effect': s1_match})

            # Look for severity comparisons
            severity_patterns = [
                r'([\w\s]+) (worse|better|more severe|less severe|stronger|weaker) than ([\w\s]+)'
            ]

            for pattern in severity_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    groups = match.groups()

                    symptom1 = groups[0].strip()
                    comparison = groups[1].strip()
                    symptom2 = groups[2].strip()

                    # Check if extracted text contains known symptoms
                    s1_match = next((s for s in symptom_names if s in symptom1), None)
                    s2_match = next((s for s in symptom_names if s in symptom2), None)

                    if s1_match and s2_match:
                        if comparison in ['worse', 'more severe', 'stronger']:
                            relationships['severity'][s1_match] = 'higher'
                            relationships['severity'][s2_match] = 'lower'
                        elif comparison in ['better', 'less severe', 'weaker']:
                            relationships['severity'][s1_match] = 'lower'
                            relationships['severity'][s2_match] = 'higher'

            # Look for symptom clusters
            if len(symptom_names) >= 3:
                # Use co-occurrence in the same sentence as a simple clustering approach
                sentences = text.lower().split('.')

                for sentence in sentences:
                    cluster = []
                    for symptom in symptom_names:
                        if symptom in sentence:
                            cluster.append(symptom)

                    if len(cluster) >= 2:
                        cluster_exists = False
                        for existing_cluster in relationships['clusters']:
                            if set(cluster).issubset(set(existing_cluster)):
                                cluster_exists = True
                                break

                        if not cluster_exists:
                            relationships['clusters'].append(cluster)

            return relationships
        except Exception as e:
            logger.error(f"Error extracting symptom relations: {e}", exc_info=True)
            return {'temporal': [], 'causal': [], 'severity': {}, 'clusters': [], 'error': str(e)}

    def generate_symptom_questions(self, extracted_symptoms: List[Dict[str, Any]], max_questions: int = 5) -> List[str]:
        """
        Generate follow-up questions based on extracted symptoms.

        Args:
            extracted_symptoms: List of extracted symptom dictionaries
            max_questions: Maximum number of questions to generate

        Returns:
            List of follow-up questions
        """
        try:
            questions = []

            # Questions about temporal aspects
            questions.append("When did your symptoms first begin?")

            # Questions about specific symptoms
            for symptom in extracted_symptoms[:3]:  # Focus on top 3 symptoms
                symptom_name = symptom['symptom']

                # Severity question
                questions.append(f"On a scale of 1-10, how severe is your {symptom_name}?")

                # Specific characteristic questions based on symptom type
                if symptom_name == 'pain' or 'pain' in symptom_name:
                    questions.append(f"How would you describe the {symptom_name}? (e.g., sharp, dull, burning)")

                if symptom_name == 'cough':
                    questions.append("Is your cough productive (bringing up phlegm or mucus)?")

                if symptom_name in ['fever', 'temperature']:
                    questions.append("Have you measured your temperature, and if so, what was the reading?")

                if symptom_name in ['rash', 'skin changes']:
                    questions.append("Is the rash itchy, painful, or neither?")

                # Location questions
                if not symptom.get('location') and symptom_name not in ['fatigue', 'fever', 'nausea']:
                    questions.append(f"Where exactly do you experience the {symptom_name}?")

            # General questions
            questions.append("What makes your symptoms better or worse?")
            questions.append("Have you taken any medications for these symptoms?")

            # Limit to max_questions
            return questions[:max_questions]
        except Exception as e:
            logger.error(f"Error generating symptom questions: {e}", exc_info=True)
            return ["When did your symptoms begin?", "How severe are your symptoms?"]

    def simulate_symptom_extraction(self, text: str) -> Dict[str, Any]:
        """
        Simulate symptom extraction for demonstration purposes.

        Args:
            text: Natural language text

        Returns:
            Dictionary containing extracted symptoms and analysis
        """
        try:
            # Extract all medical data
            medical_data = self.extract_relevant_medical_data(text)

            # Create basic summary
            summary = self.create_symptom_summary(text)

            # Generate follow-up questions
            questions = self.generate_symptom_questions(medical_data['symptoms'])

            # Create simulation result
            result = {
                'extracted_data': medical_data,
                'summary': summary,
                'follow_up_questions': questions,
                'simulation': True,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': '1.0'
            }

            logger.info(f"Generated simulated symptom extraction for text: {text[:50]}...")

            return result
        except Exception as e:
            logger.error(f"Error simulating symptom extraction: {e}", exc_info=True)
            return {
                'extracted_data': {'symptoms': []},
                'summary': "Unable to extract symptoms due to an error.",
                'follow_up_questions': [],
                'simulation': True,
                'error': str(e)
            }

    def enhance_symptom_descriptions(self, symptoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance symptom descriptions with additional contextual information.

        Args:
            symptoms: List of extracted symptom dictionaries

        Returns:
            Enhanced symptom descriptions
        """
        try:
            enhanced_symptoms = []

            # Common symptom descriptions and characteristics
            symptom_info = {
                'fever': {
                    'description': 'Elevated body temperature above the normal range',
                    'common_causes': ['infections', 'inflammation', 'immune responses'],
                    'related_symptoms': ['chills', 'sweating', 'fatigue']
                },
                'cough': {
                    'description': 'Sudden expulsion of air from the lungs',
                    'common_causes': ['respiratory infections', 'allergies', 'irritants'],
                    'related_symptoms': ['sore throat', 'runny nose', 'shortness of breath']
                },
                'headache': {
                    'description': 'Pain or discomfort in the head or scalp',
                    'common_causes': ['tension', 'migraines', 'dehydration', 'stress'],
                    'related_symptoms': ['sensitivity to light', 'nausea', 'dizziness']
                },
                'fatigue': {
                    'description': 'Feeling of tiredness, lack of energy or exhaustion',
                    'common_causes': ['lack of sleep', 'stress', 'anemia', 'infections'],
                    'related_symptoms': ['weakness', 'decreased concentration', 'lack of motivation']
                },
                'nausea': {
                    'description': 'Sensation of unease and discomfort in the stomach with an urge to vomit',
                    'common_causes': ['food poisoning', 'motion sickness', 'migraines', 'medications'],
                    'related_symptoms': ['vomiting', 'dizziness', 'abdominal discomfort']
                },
                'chest pain': {
                    'description': 'Discomfort or pain in the chest area',
                    'common_causes': ['muscle strain', 'anxiety', 'heart problems', 'digestive issues'],
                    'related_symptoms': ['shortness of breath', 'sweating', 'lightheadedness']
                },
                'shortness of breath': {
                    'description': 'Feeling of not getting enough air or difficulty breathing',
                    'common_causes': ['physical exertion', 'anxiety', 'respiratory infections', 'heart problems'],
                    'related_symptoms': ['chest tightness', 'rapid breathing', 'fatigue']
                },
                'dizziness': {
                    'description': 'Sensation of lightheadedness, unsteadiness, or feeling faint',
                    'common_causes': ['inner ear problems', 'low blood pressure', 'dehydration', 'anxiety'],
                    'related_symptoms': ['nausea', 'headache', 'balance problems']
                },
                'abdominal pain': {
                    'description': 'Pain or discomfort in the area between the chest and groin',
                    'common_causes': ['digestive issues', 'menstrual cramps', 'gas', 'inflammation'],
                    'related_symptoms': ['nausea', 'bloating', 'changes in bowel habits']
                },
                'sore throat': {
                    'description': 'Pain, irritation or scratchiness in the throat',
                    'common_causes': ['viral infections', 'bacterial infections', 'allergies', 'dry air'],
                    'related_symptoms': ['difficulty swallowing', 'hoarseness', 'cough']
                }
            }

            # Enhance each symptom
            for symptom in symptoms:
                symptom_name = symptom['symptom']
                enhanced = symptom.copy()

                # Add standard info if available
                if symptom_name in symptom_info:
                    enhanced['description'] = symptom_info[symptom_name]['description']
                    enhanced['common_causes'] = symptom_info[symptom_name]['common_causes']
                    enhanced['related_symptoms'] = symptom_info[symptom_name]['related_symptoms']

                # Add location-specific context
                if symptom.get('location'):
                    location = symptom['location']
                    if location == 'head':
                        enhanced['location_context'] = 'Head symptoms may relate to neurological, vascular, or sinus issues'
                    elif location == 'chest':
                        enhanced['location_context'] = 'Chest symptoms could indicate respiratory, cardiac, or musculoskeletal issues'
                    elif location == 'abdomen':
                        enhanced['location_context'] = 'Abdominal symptoms often relate to digestive, urinary, or reproductive systems'
                    elif location == 'back':
                        enhanced['location_context'] = 'Back symptoms typically involve musculoskeletal, spinal, or renal issues'
                    elif location in ['arm', 'leg']:
                        enhanced['location_context'] = f'{location.capitalize()} symptoms may indicate musculoskeletal, vascular, or neurological issues'

                # Add severity context based on intensity
                if symptom.get('intensity'):
                    intensity = symptom['intensity']
                    if intensity >= 0.8:
                        enhanced['severity_note'] = 'This symptom appears to be severe and may require prompt attention'
                    elif intensity >= 0.5:
                        enhanced['severity_note'] = 'This symptom appears to be moderate in intensity'
                    else:
                        enhanced['severity_note'] = 'This symptom appears to be mild in intensity'

                enhanced_symptoms.append(enhanced)

            return enhanced_symptoms
        except Exception as e:
            logger.error(f"Error enhancing symptom descriptions: {e}", exc_info=True)
            return symptoms  # Return original symptoms on error

    def extract_symptom_timeline(self, text: str) -> Dict[str, Any]:
        """
        Extract a timeline of symptom development from text.

        Args:
            text: Natural language text describing symptom history

        Returns:
            Timeline of symptom development
        """
        try:
            # Extract basic information
            symptoms = self.extract_symptoms(text)
            temporal_info = self.extract_temporal_information(text)

            # Initialize timeline
            timeline = {
                'events': [],
                'duration': temporal_info.get('duration'),
                'progression': temporal_info.get('progression'),
                'current_state': 'unknown'
            }

            # Look for temporal markers in text
            text_lower = text.lower()

            # Common time marker patterns
            time_markers = [
                (r'(yesterday|today|last night|this morning|this afternoon)', 1),  # Very recent (1 day)
                (r'(\d+)\s+days?\s+ago', lambda m: int(m.group(1))),  # Days ago
                (r'last\s+week', 7),  # Last week
                (r'(\d+)\s+weeks?\s+ago', lambda m: int(m.group(1)) * 7),  # Weeks ago
                (r'last\s+month', 30),  # Last month
                (r'(\d+)\s+months?\s+ago', lambda m: int(m.group(1)) * 30),  # Months ago
                (r'(since|for the past|for)\s+(\d+)\s+days?', lambda m: int(m.group(2))),  # Duration in days
                (r'(since|for the past|for)\s+(\d+)\s+weeks?', lambda m: int(m.group(2)) * 7),  # Duration in weeks
                (r'(since|for the past|for)\s+(\d+)\s+months?', lambda m: int(m.group(2)) * 30)  # Duration in months
            ]

            # Extract time-based events
            text_segments = re.split(r'[.;!?]', text_lower)

            for segment in text_segments:
                segment = segment.strip()
                if not segment:
                    continue

                # Check for time markers in this segment
                time_value = None
                for pattern, value in time_markers:
                    match = re.search(pattern, segment)
                    if match:
                        time_value = value() if callable(value) else value
                        break

                # Extract symptoms from this segment
                segment_symptoms = []
                for symptom in symptoms:
                    symp_name = symptom['symptom']
                    if symp_name.lower() in segment:
                        segment_symptoms.append(symp_name)

                # If we have both a time marker and symptoms, add an event
                if time_value is not None and segment_symptoms:
                    timeline['events'].append({
                        'time': time_value,
                        'time_unit': 'days',
                        'symptoms': segment_symptoms,
                        'description': segment
                    })

            # Sort events by time (earliest first)
            timeline['events'].sort(key=lambda x: x['time'])

            # Determine current state based on most recent events and progression
            if timeline['events']:
                recent_symptoms = timeline['events'][-1]['symptoms']
                if temporal_info.get('progression') == 'improving':
                    timeline['current_state'] = 'improving'
                elif temporal_info.get('progression') == 'worsening':
                    timeline['current_state'] = 'worsening'
                elif len(recent_symptoms) >= 3:
                    timeline['current_state'] = 'active with multiple symptoms'
                else:
                    timeline['current_state'] = 'active'

            return timeline
        except Exception as e:
            logger.error(f"Error extracting symptom timeline: {e}", exc_info=True)
            return {'events': [], 'duration': None, 'progression': None, 'current_state': 'unknown', 'error': str(e)}

    def categorize_symptoms_by_system(self, symptoms: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Categorize symptoms by body system.

        Args:
            symptoms: List of extracted symptom dictionaries

        Returns:
            Dictionary mapping body systems to symptom lists
        """
        try:
            # Body system categories and associated symptoms
            system_categories = {
                "respiratory": ["cough", "shortness of breath", "wheezing", "chest congestion",
                               "sore throat", "runny nose", "nasal congestion", "breathing difficulty"],

                "cardiovascular": ["chest pain", "palpitations", "irregular heartbeat", "shortness of breath",
                                  "leg swelling", "fatigue", "dizziness when standing"],

                "neurological": ["headache", "dizziness", "confusion", "memory problems", "numbness",
                                "tingling", "seizure", "tremor", "balance problems", "fainting"],

                "gastrointestinal": ["abdominal pain", "nausea", "vomiting", "diarrhea", "constipation",
                                    "bloating", "heartburn", "indigestion", "loss of appetite"],

                "musculoskeletal": ["joint pain", "muscle pain", "back pain", "neck pain", "stiffness",
                                   "swelling", "reduced range of motion", "weakness"],

                "dermatological": ["rash", "itching", "hives", "skin dryness", "skin lesions", "skin discoloration",
                                  "bruising", "excessive sweating"],

                "ophthalmological": ["eye pain", "vision changes", "blurred vision", "double vision",
                                    "redness", "discharge", "light sensitivity"],

                "ear_nose_throat": ["ear pain", "hearing loss", "tinnitus", "vertigo", "sore throat",
                                   "hoarseness", "difficulty swallowing", "nasal congestion"],

                "psychological": ["anxiety", "depression", "mood changes", "sleep problems", "irritability",
                                 "confusion", "hallucinations", "delusions"],

                "general": ["fever", "fatigue", "weakness", "weight loss", "weight gain", "night sweats",
                           "malaise", "chills"]
            }

            # Initialize result dictionary
            categorized = {system: [] for system in system_categories}
            uncategorized = []

            # Categorize each symptom
            for symptom in symptoms:
                symptom_name = symptom['symptom'].lower()
                categorized_flag = False

                # Check each system
                for system, system_symptoms in system_categories.items():
                    # Check if the symptom or any part of it matches a system symptom
                    if any(system_symptom in symptom_name for system_symptom in system_symptoms):
                        categorized[system].append(symptom['symptom'])
                        categorized_flag = True
                        break

                # Add to uncategorized if not matched
                if not categorized_flag:
                    uncategorized.append(symptom['symptom'])

            # Add uncategorized symptoms to result
            categorized["uncategorized"] = uncategorized

            # Remove empty categories
            categorized = {system: symptoms for system, symptoms in categorized.items() if symptoms}

            return categorized
        except Exception as e:
            logger.error(f"Error categorizing symptoms by system: {e}", exc_info=True)
            return {"general": [s['symptom'] for s in symptoms], "error": str(e)}

    def identify_symptom_patterns(self, health_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patterns in symptom history.

        Args:
            health_history: List of health records with symptoms and dates

        Returns:
            Dictionary of identified patterns
        """
        try:
            # Initialize patterns
            patterns = {
                'recurring_symptoms': {},
                'seasonal_patterns': {},
                'symptom_progressions': [],
                'common_combinations': [],
                'symptom_frequencies': {}
            }

            # Extract all symptoms from history
            all_symptoms = []
            symptom_dates = {}

            for entry in health_history:
                date = entry.get('date', '')
                symptoms = entry.get('symptoms', [])

                for symptom in symptoms:
                    all_symptoms.append(symptom)

                    # Track dates for each symptom
                    if symptom not in symptom_dates:
                        symptom_dates[symptom] = []
                    symptom_dates[symptom].append(date)

            # Count symptom frequencies
            symptom_counts = {}
            for symptom in all_symptoms:
                if symptom not in symptom_counts:
                    symptom_counts[symptom] = 0
                symptom_counts[symptom] += 1

            # Sort by frequency
            sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
            patterns['symptom_frequencies'] = {symptom: count for symptom, count in sorted_symptoms}

            # Identify recurring symptoms (appearing more than once)
            patterns['recurring_symptoms'] = {symptom: count for symptom, count in symptom_counts.items() if count > 1}

            # Identify common combinations (co-occurring symptoms)
            symptom_combinations = {}
            for entry in health_history:
                symptoms = entry.get('symptoms', [])
                if len(symptoms) >= 2:
                    # Sort symptoms to normalize combinations
                    sorted_symptoms = sorted(symptoms)
                    combo_key = ", ".join(sorted_symptoms)

                    if combo_key not in symptom_combinations:
                        symptom_combinations[combo_key] = 0
                    symptom_combinations[combo_key] += 1

            # Get top combinations
            sorted_combos = sorted(symptom_combinations.items(), key=lambda x: x[1], reverse=True)
            for combo, count in sorted_combos[:5]:  # Top 5 combinations
                patterns['common_combinations'].append({
                    'symptoms': combo.split(", "),
                    'occurrences': count
                })

            # Identify seasonal patterns
            # This is a simplified approach - would need actual seasonal analysis for real application
            for symptom, dates in symptom_dates.items():
                if len(dates) >= 3:  # Need at least 3 occurrences for a pattern
                    patterns['seasonal_patterns'][symptom] = {
                        'occurrences': len(dates),
                        'dates': dates
                    }

            # Identify symptom progressions (symptoms that tend to follow others)
            # This requires more sophisticated time series analysis in a real application
            # This is just a placeholder for the concept
            if len(health_history) >= 3:
                patterns['symptom_progressions'].append({
                    'description': "Simplified progression analysis requires at least 3 health records with clear timeline information"
                })

            return patterns
        except Exception as e:
            logger.error(f"Error identifying symptom patterns: {e}", exc_info=True)
            return {'error': str(e)}

    def generate_medical_query(self, symptoms: List[Dict[str, Any]], patient_profile: Dict[str, Any] = None) -> str:
        """
        Generate a formal medical query based on extracted symptoms and patient profile.

        Args:
            symptoms: List of extracted symptom dictionaries
            patient_profile: Optional patient demographic and medical history information

        Returns:
            Formatted medical query string
        """
        try:
            query_parts = []

            # Add patient demographics if available
            if patient_profile:
                demographics = []

                if 'age' in patient_profile:
                    demographics.append(f"{patient_profile['age']}-year-old")

                if 'gender' in patient_profile:
                    demographics.append(patient_profile['gender'])

                if demographics:
                    patient_desc = " ".join(demographics)
                    query_parts.append(f"Patient: {patient_desc}")

                # Add medical history if available
                if 'medical_history' in patient_profile and patient_profile['medical_history']:
                    history = ", ".join(patient_profile['medical_history'])
                    query_parts.append(f"History: {history}")

            # Add presenting symptoms
            if symptoms:
                symptom_names = [s['symptom'] for s in symptoms]

                # Add symptom modifiers where available
                detailed_symptoms = []
                for symptom in symptoms:
                    symptom_text = symptom['symptom']

                    # Add intensity if available
                    if 'intensity' in symptom:
                        intensity = symptom['intensity']
                        if intensity >= 0.8:
                            symptom_text = f"severe {symptom_text}"
                        elif intensity >= 0.5:
                            symptom_text = f"moderate {symptom_text}"
                        else:
                            symptom_text = f"mild {symptom_text}"

                    # Add location if available
                    if 'location' in symptom and symptom['location']:
                        symptom_text = f"{symptom_text} in {symptom['location']}"

                    detailed_symptoms.append(symptom_text)

                symptoms_text = ", ".join(detailed_symptoms)
                query_parts.append(f"Presenting with: {symptoms_text}")

            # Add any temporal information
            duration_info = []
            for symptom in symptoms:
                if 'duration' in symptom and symptom['duration']:
                    duration_info.append(f"{symptom['symptom']} for {symptom['duration']}")

            if duration_info:
                duration_text = "; ".join(duration_info)
                query_parts.append(f"Duration: {duration_text}")

            # Combine into complete query
            query = "\n".join(query_parts)

            return query
        except Exception as e:
            logger.error(f"Error generating medical query: {e}", exc_info=True)

            # Provide a basic query as fallback
            basic_symptoms = ", ".join([s['symptom'] for s in symptoms][:5])
            return f"Patient presenting with: {basic_symptoms}"
