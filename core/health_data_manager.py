import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthDataManager:
    """
    Manages all health-related data, including symptoms, conditions, and user health records.
    Provides methods for data access, storage, and analysis.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the health data manager.

        Args:
            data_dir: Directory for storing health data files
        """
        self.data_dir = data_dir
        self.medical_data_path = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), "data", "medical_data.json")

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Initialize medical data containers
        self.symptoms = {}
        self.conditions = {}
        self.symptoms_to_conditions = {}
        self.body_systems = {}

        # Load medical database
        self._load_medical_database()

        logger.info(f"HealthDataManager initialized with {len(self.symptoms)} symptoms and {len(self.conditions)} conditions")

    def _load_medical_database(self):
        """Load medical database from JSON file."""
        try:
            if os.path.exists(self.medical_data_path):
                with open(self.medical_data_path, 'r') as f:
                    medical_data = json.load(f)

                self.symptoms = medical_data.get('symptoms', {})
                self.conditions = medical_data.get('conditions', {})
                self.symptoms_to_conditions = medical_data.get('symptoms_to_conditions', {})
                self.body_systems = medical_data.get('body_systems', {})

                logger.info(f"Medical database loaded with {len(self.symptoms)} symptoms and {len(self.conditions)} conditions")
            else:
                # Create a minimal database if file doesn't exist
                self._create_minimal_database()
                logger.warning("Medical data file not found, created minimal database")
        except Exception as e:
            logger.error(f"Error loading medical database: {e}", exc_info=True)
            # Create a minimal database on error
            self._create_minimal_database()

    def _create_minimal_database(self):
        """Create a minimal database with essential symptoms and conditions."""
        # Create some basic symptoms
        self.symptoms = {
            "s1": {"id": "s1", "name": "Fever", "description": "Elevated body temperature", "system": "general"},
            "s2": {"id": "s2", "name": "Cough", "description": "Sudden expulsion of air", "system": "respiratory"},
            "s3": {"id": "s3", "name": "Headache", "description": "Pain in the head", "system": "neurological"},
            "s4": {"id": "s4", "name": "Fatigue", "description": "Feeling of tiredness", "system": "general"},
            "s5": {"id": "s5", "name": "Nausea", "description": "Feeling of sickness with an urge to vomit", "system": "digestive"},
            "s6": {"id": "s6", "name": "Chest Pain", "description": "Discomfort or pain in the chest area", "system": "cardiovascular"},
            "s7": {"id": "s7", "name": "Shortness of Breath", "description": "Difficult or labored breathing", "system": "respiratory"},
            "s8": {"id": "s8", "name": "Dizziness", "description": "Lightheadedness or feeling faint", "system": "neurological"},
            "s9": {"id": "s9", "name": "Abdominal Pain", "description": "Pain in the stomach or belly area", "system": "digestive"},
            "s10": {"id": "s10", "name": "Sore Throat", "description": "Pain or irritation in the throat", "system": "respiratory"}
        }

        # Create some basic conditions
        self.conditions = {
            "c1": {"id": "c1", "name": "Common Cold", "description": "A viral infectious disease of the upper respiratory tract", "severity": "low"},
            "c2": {"id": "c2", "name": "Influenza", "description": "A viral infection that attacks the respiratory system", "severity": "medium"},
            "c3": {"id": "c3", "name": "Migraine", "description": "A headache of varying intensity, often accompanied by nausea and sensitivity to light", "severity": "medium"},
            "c4": {"id": "c4", "name": "Gastroenteritis", "description": "Inflammation of the stomach and intestines, typically resulting from bacterial toxins or viral infection", "severity": "medium"},
            "c5": {"id": "c5", "name": "Hypertension", "description": "High blood pressure", "severity": "medium"},
            "c6": {"id": "c6", "name": "Bronchitis", "description": "Inflammation of the lining of bronchial tubes", "severity": "medium"},
            "c7": {"id": "c7", "name": "Pneumonia", "description": "Infection that inflames the air sacs in one or both lungs", "severity": "high"},
            "c8": {"id": "c8", "name": "Anxiety Disorder", "description": "Mental health disorder characterized by feelings of worry, anxiety or fear", "severity": "medium"}
        }

        # Map symptoms to conditions
        self.symptoms_to_conditions = {
            "s1": ["c1", "c2", "c7"], # Fever
            "s2": ["c1", "c2", "c6", "c7"], # Cough
            "s3": ["c1", "c2", "c3"], # Headache
            "s4": ["c1", "c2", "c3", "c4", "c8"], # Fatigue
            "s5": ["c2", "c3", "c4"], # Nausea
            "s6": ["c5", "c7"], # Chest Pain
            "s7": ["c6", "c7"], # Shortness of Breath
            "s8": ["c3", "c5", "c8"], # Dizziness
            "s9": ["c4"], # Abdominal Pain
            "s10": ["c1", "c6"] # Sore Throat
        }

        # Basic body systems
        self.body_systems = {
            "general": {"name": "General", "description": "General body symptoms"},
            "respiratory": {"name": "Respiratory", "description": "Related to breathing and lungs"},
            "neurological": {"name": "Neurological", "description": "Related to brain and nervous system"},
            "digestive": {"name": "Digestive", "description": "Related to digestive tract"},
            "cardiovascular": {"name": "Cardiovascular", "description": "Related to heart and blood vessels"}
        }

        # Save to file
        self._save_medical_database()

    def _save_medical_database(self):
        """Save medical database to JSON file."""
        try:
            medical_data = {
                'symptoms': self.symptoms,
                'conditions': self.conditions,
                'symptoms_to_conditions': self.symptoms_to_conditions,
                'body_systems': self.body_systems
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.medical_data_path), exist_ok=True)

            with open(self.medical_data_path, 'w') as f:
                json.dump(medical_data, f, indent=2)

            logger.info("Medical database saved successfully")
        except Exception as e:
            logger.error(f"Error saving medical database: {e}", exc_info=True)

    def get_all_symptoms(self) -> List[Dict[str, Any]]:
        """
        Get a list of all symptoms.

        Returns:
            List of symptom dictionaries
        """
        return list(self.symptoms.values())

    def get_symptom_info(self, symptom_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a symptom.

        Args:
            symptom_id: ID of the symptom

        Returns:
            Symptom information dictionary or None if not found
        """
        return self.symptoms.get(symptom_id)

    def get_conditions_for_symptoms(self, symptom_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get conditions that could be associated with a set of symptoms.

        Args:
            symptom_ids: List of symptom IDs

        Returns:
            List of condition dictionaries
        """
        condition_ids = set()

        # Collect all condition IDs associated with the symptoms
        for symptom_id in symptom_ids:
            if symptom_id in self.symptoms_to_conditions:
                condition_ids.update(self.symptoms_to_conditions[symptom_id])

        # Get condition details
        conditions = []
        for cond_id in condition_ids:
            if cond_id in self.conditions:
                conditions.append(self.conditions[cond_id])

        return conditions

    def add_symptom(self, symptom_data: Dict[str, Any]) -> str:
        """
        Add a new symptom to the database.

        Args:
            symptom_data: Symptom data dictionary

        Returns:
            ID of the new symptom
        """
        try:
            # Generate a new ID if not provided
            if 'id' not in symptom_data:
                max_id = 0
                for symptom_id in self.symptoms:
                    if symptom_id.startswith('s'):
                        try:
                            id_num = int(symptom_id[1:])
                            max_id = max(max_id, id_num)
                        except ValueError:
                            pass
                symptom_data['id'] = f"s{max_id + 1}"

            # Add symptom to database
            self.symptoms[symptom_data['id']] = symptom_data

            # Save changes
            self._save_medical_database()

            logger.info(f"Added new symptom: {symptom_data['name']} ({symptom_data['id']})")

            return symptom_data['id']
        except Exception as e:
            logger.error(f"Error adding symptom: {e}", exc_info=True)
            raise

    def add_condition(self, condition_data: Dict[str, Any], related_symptoms: List[str] = None) -> str:
        """
        Add a new condition to the database.

        Args:
            condition_data: Condition data dictionary
            related_symptoms: List of symptom IDs related to this condition

        Returns:
            ID of the new condition
        """
        try:
            # Generate a new ID if not provided
            if 'id' not in condition_data:
                max_id = 0
                for condition_id in self.conditions:
                    if condition_id.startswith('c'):
                        try:
                            id_num = int(condition_id[1:])
                            max_id = max(max_id, id_num)
                        except ValueError:
                            pass
                condition_data['id'] = f"c{max_id + 1}"

            # Add condition to database
            self.conditions[condition_data['id']] = condition_data

            # Update symptom-to-condition mappings
            if related_symptoms:
                for symptom_id in related_symptoms:
                    if symptom_id in self.symptoms:
                        if symptom_id not in self.symptoms_to_conditions:
                            self.symptoms_to_conditions[symptom_id] = []
                        if condition_data['id'] not in self.symptoms_to_conditions[symptom_id]:
                            self.symptoms_to_conditions[symptom_id].append(condition_data['id'])

            # Save changes
            self._save_medical_database()

            logger.info(f"Added new condition: {condition_data['name']} ({condition_data['id']})")

            return condition_data['id']
        except Exception as e:
            logger.error(f"Error adding condition: {e}", exc_info=True)
            raise

    def save_symptom_check(self, user_id: str, symptoms: List[str], notes: str = None) -> Dict[str, Any]:
        """
        Save a symptom check record for a user.

        Args:
            user_id: User ID
            symptoms: List of symptom IDs
            notes: Optional notes about the symptom check

        Returns:
            The saved symptom check record
        """
        try:
            # Create user data directory if it doesn't exist
            user_dir = os.path.join(self.data_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)

            # Create symptom record
            now = datetime.now()
            record = {
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'symptoms': symptoms,
                'notes': notes
            }

            # Determine file path for symptom checks
            file_path = os.path.join(user_dir, 'symptom_checks.json')

            # Load existing checks or create new list
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    checks = json.load(f)
            else:
                checks = []

            # Add new check
            checks.append(record)

            # Save updated checks
            with open(file_path, 'w') as f:
                json.dump(checks, f, indent=2)

            logger.info(f"Saved symptom check for user {user_id} with {len(symptoms)} symptoms")

            return record
        except Exception as e:
            logger.error(f"Error saving symptom check: {e}", exc_info=True)
            raise

    def get_symptom_checks(self, user_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get symptom check history for a user.

        Args:
            user_id: User ID
            limit: Optional limit on the number of records to return

        Returns:
            List of symptom check records
        """
        try:
            file_path = os.path.join(self.data_dir, user_id, 'symptom_checks.json')

            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    checks = json.load(f)

                # Sort by date and time (most recent first)
                checks.sort(key=lambda x: x.get('date', '') + ' ' + x.get('time', ''), reverse=True)

                # Apply limit if specified
                if limit is not None and limit > 0:
                    checks = checks[:limit]

                return checks
            else:
                logger.info(f"No symptom check history found for user {user_id}")
                return []
        except Exception as e:
            logger.error(f"Error getting symptom checks: {e}", exc_info=True)
            return []

    def get_symptom_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's symptom history.

        Args:
            user_id: User ID

        Returns:
            Dictionary of symptom statistics
        """
        try:
            checks = self.get_symptom_checks(user_id)

            if not checks:
                return {
                    'total_checks': 0,
                    'total_symptoms': 0,
                    'common_symptoms': [],
                    'first_check': None,
                    'last_check': None
                }

            # Count symptom occurrences
            symptom_counts = {}
            for check in checks:
                for symptom_id in check.get('symptoms', []):
                    if symptom_id not in symptom_counts:
                        symptom_counts[symptom_id] = 0
                    symptom_counts[symptom_id] += 1

            # Get top symptoms
            top_symptoms = []
            for symptom_id, count in sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                symptom_info = self.get_symptom_info(symptom_id)
                if symptom_info:
                    top_symptoms.append({
                        'id': symptom_id,
                        'name': symptom_info.get('name', symptom_id),
                        'count': count
                    })
                else:
                    top_symptoms.append({
                        'id': symptom_id,
                        'name': symptom_id,
                        'count': count
                    })

            # Return statistics
            return {
                'total_checks': len(checks),
                'total_symptoms': sum(len(check.get('symptoms', [])) for check in checks),
                'common_symptoms': top_symptoms,
                'first_check': checks[-1].get('date'),
                'last_check': checks[0].get('date')
            }
        except Exception as e:
            logger.error(f"Error getting symptom stats: {e}", exc_info=True)
            return {
                'total_checks': 0,
                'total_symptoms': 0,
                'common_symptoms': [],
                'first_check': None,
                'last_check': None,
                'error': str(e)
            }

    def search_symptoms(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symptoms matching a query string.

        Args:
            query: Search query string

        Returns:
            List of matching symptom dictionaries
        """
        results = []
        query = query.lower()

        for symptom in self.symptoms.values():
            # Search in name and description
            if query in symptom.get('name', '').lower() or query in symptom.get('description', '').lower():
                results.append(symptom)

        return results

    def get_symptoms_by_system(self, system: str) -> List[Dict[str, Any]]:
        """
        Get symptoms belonging to a specific body system.

        Args:
            system: Body system identifier

        Returns:
            List of symptom dictionaries
        """
        results = []

        for symptom in self.symptoms.values():
            if symptom.get('system') == system:
                results.append(symptom)

        return results

    def get_all_body_systems(self) -> List[Dict[str, Any]]:
        """
        Get a list of all body systems.

        Returns:
            List of body system dictionaries
        """
        return list(self.body_systems.values())

    def export_user_health_data(self, user_id: str, format: str = 'json') -> Any:
        """
        Export all health data for a user.

        Args:
            user_id: User ID
            format: Export format ('json' or 'csv')

        Returns:
            Exported data in the requested format
        """
        try:
            # Get symptom checks
            symptom_checks = self.get_symptom_checks(user_id)

            if format.lower() == 'json':
                return symptom_checks
            elif format.lower() == 'csv':
                # Convert to a format suitable for CSV
                csv_data = []

                for check in symptom_checks:
                    # Get symptom names
                    symptom_names = []
                    for symptom_id in check.get('symptoms', []):
                        symptom_info = self.get_symptom_info(symptom_id)
                        if symptom_info:
                            symptom_names.append(symptom_info.get('name', symptom_id))
                        else:
                            symptom_names.append(symptom_id)

                    csv_data.append({
                        'Date': check.get('date', ''),
                        'Time': check.get('time', ''),
                        'Symptoms': ', '.join(symptom_names),
                        'Notes': check.get('notes', '')
                    })

                # Create DataFrame
                return pd.DataFrame(csv_data)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            logger.error(f"Error exporting user health data: {e}", exc_info=True)
            raise
