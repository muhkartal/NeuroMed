import os
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class UserProfileManager:
    """
    Manages user profiles, authentication, and health history.
    Handles user data storage, retrieval, and session management.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the user profile manager.

        Args:
            data_dir: Directory for storing user data
        """
        self.data_dir = data_dir
        self.users_path = os.path.join(data_dir, "users.json")
        self.current_user_id = "default_user"
        self.profile = {}
        self.health_history = []

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Load users database
        self._load_users_database()

        # Load current user profile
        self._load_user_profile(self.current_user_id)

        logger.info(f"UserProfileManager initialized with user: {self.current_user_id}")

    def _load_users_database(self):
        """Load users database from JSON file."""
        try:
            if os.path.exists(self.users_path):
                with open(self.users_path, 'r') as f:
                    self.users = json.load(f)
                logger.info(f"Users database loaded with {len(self.users)} users")
            else:
                # Create default users database
                self.users = {
                    "default_user": {
                        "id": "default_user",
                        "name": "Default User",
                        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "last_login": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
                self._save_users_database()
                logger.info("Created default users database")
        except Exception as e:
            logger.error(f"Error loading users database: {e}", exc_info=True)
            # Create default users database on error
            self.users = {
                "default_user": {
                    "id": "default_user",
                    "name": "Default User",
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "last_login": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            self._save_users_database()

    def _save_users_database(self):
        """Save users database to JSON file."""
        try:
            with open(self.users_path, 'w') as f:
                json.dump(self.users, f, indent=2)
            logger.info("Users database saved successfully")
        except Exception as e:
            logger.error(f"Error saving users database: {e}", exc_info=True)

    def _load_user_profile(self, user_id: str):
        """
        Load a user's profile and health history.

        Args:
            user_id: User ID to load
        """
        try:
            # Update current user
            self.current_user_id = user_id

            # Ensure user exists in database
            if user_id not in self.users:
                logger.warning(f"User {user_id} not found, creating new user")
                self.users[user_id] = {
                    "id": user_id,
                    "name": f"User {user_id}",
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "last_login": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self._save_users_database()
            else:
                # Update last login time
                self.users[user_id]["last_login"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self._save_users_database()

            # Load profile
            profile_path = os.path.join(self.data_dir, user_id, "profile.json")
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    self.profile = json.load(f)
                logger.info(f"Loaded profile for user {user_id}")
            else:
                # Create default profile
                self.profile = {
                    "id": user_id,
                    "name": self.users[user_id].get("name", f"User {user_id}"),
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self._save_user_profile()
                logger.info(f"Created default profile for user {user_id}")

            # Load health history
            self._load_health_history()
        except Exception as e:
            logger.error(f"Error loading user profile: {e}", exc_info=True)
            # Create default profile on error
            self.profile = {
                "id": user_id,
                "name": f"User {user_id}",
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.health_history = []

    def _save_user_profile(self):
        """Save the current user's profile."""
        try:
            # Ensure user directory exists
            user_dir = os.path.join(self.data_dir, self.current_user_id)
            os.makedirs(user_dir, exist_ok=True)

            # Save profile
            profile_path = os.path.join(user_dir, "profile.json")
            with open(profile_path, 'w') as f:
                json.dump(self.profile, f, indent=2)
            logger.info(f"Saved profile for user {self.current_user_id}")
        except Exception as e:
            logger.error(f"Error saving user profile: {e}", exc_info=True)

    def _load_health_history(self):
        """Load the current user's health history."""
        try:
            # Check for symptom checks
            symptom_checks_path = os.path.join(self.data_dir, self.current_user_id, "symptom_checks.json")

            if os.path.exists(symptom_checks_path):
                with open(symptom_checks_path, 'r') as f:
                    self.health_history = json.load(f)

                # Sort by date and time (most recent first)
                self.health_history.sort(key=lambda x: x.get('date', '') + ' ' + x.get('time', ''), reverse=True)

                logger.info(f"Loaded {len(self.health_history)} health history records for user {self.current_user_id}")
            else:
                self.health_history = []
                logger.info(f"No health history found for user {self.current_user_id}")
        except Exception as e:
            logger.error(f"Error loading health history: {e}", exc_info=True)
            self.health_history = []

    def create_user(self, user_data: Dict[str, Any]) -> str:
        """
        Create a new user.

        Args:
            user_data: User data dictionary

        Returns:
            ID of the new user
        """
        try:
            # Generate new user ID if not provided
            if 'id' not in user_data:
                user_data['id'] = str(uuid.uuid4())

            # Add creation timestamp
            user_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            user_data['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Add to users database
            self.users[user_data['id']] = user_data
            self._save_users_database()

            # Create user directory
            user_dir = os.path.join(self.data_dir, user_data['id'])
            os.makedirs(user_dir, exist_ok=True)

            logger.info(f"Created new user: {user_data.get('name', user_data['id'])} ({user_data['id']})")

            return user_data['id']
        except Exception as e:
            logger.error(f"Error creating user: {e}", exc_info=True)
            raise

    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user information.

        Args:
            user_id: User ID to update
            user_data: Updated user data

        Returns:
            Updated user data
        """
        try:
            # Ensure user exists
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")

            # Update user information
            for key, value in user_data.items():
                if key != 'id' and key != 'created_at':  # Don't overwrite these fields
                    self.users[user_id][key] = value

            # Save changes
            self._save_users_database()

            # If updating current user, also update profile
            if user_id == self.current_user_id:
                self.profile['name'] = self.users[user_id].get('name', f"User {user_id}")
                self._save_user_profile()

            logger.info(f"Updated user {user_id}")

            return self.users[user_id]
        except Exception as e:
            logger.error(f"Error updating user: {e}", exc_info=True)
            raise

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.

        Args:
            user_id: User ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure user exists
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")

            # Cannot delete default user
            if user_id == "default_user":
                logger.warning("Cannot delete default user")
                return False

            # Remove from users database
            del self.users[user_id]
            self._save_users_database()

            # Remove user directory
            user_dir = os.path.join(self.data_dir, user_id)
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)

            logger.info(f"Deleted user {user_id}")

            # If deleted current user, switch to default user
            if user_id == self.current_user_id:
                self._load_user_profile("default_user")

            return True
        except Exception as e:
            logger.error(f"Error deleting user: {e}", exc_info=True)
            return False

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information.

        Args:
            user_id: User ID

        Returns:
            User data dictionary or None if not found
        """
        return self.users.get(user_id)

    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Get all users.

        Returns:
            List of user data dictionaries
        """
        return list(self.users.values())

    def switch_user(self, user_id: str) -> bool:
        """
        Switch to a different user.

        Args:
            user_id: User ID to switch to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure user exists
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")

            # Load user profile
            self._load_user_profile(user_id)

            logger.info(f"Switched to user {user_id}")

            return True
        except Exception as e:
            logger.error(f"Error switching user: {e}", exc_info=True)
            return False

    def update_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current user's profile.

        Args:
            profile_data: Updated profile data

        Returns:
            Updated profile
        """
        try:
            # Update profile
            for key, value in profile_data.items():
                if key != 'id':  # Don't overwrite ID
                    self.profile[key] = value

            # Ensure name is synchronized with users database
            if 'name' in profile_data and self.current_user_id in self.users:
                self.users[self.current_user_id]['name'] = profile_data['name']
                self._save_users_database()

            # Save profile
            self._save_user_profile()

            logger.info(f"Updated profile for user {self.current_user_id}")

            return self.profile
        except Exception as e:
            logger.error(f"Error updating profile: {e}", exc_info=True)
            raise

    def add_health_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a health record to the current user's history.

        Args:
            record: Health record data

        Returns:
            The added record
        """
        try:
            # Ensure required fields
            if 'date' not in record:
                record['date'] = datetime.now().strftime('%Y-%m-%d')
            if 'time' not in record:
                record['time'] = datetime.now().strftime('%H:%M:%S')

            # Add to health history
            self.health_history.insert(0, record)  # Add to beginning (most recent)

            # Save symptom checks
            self._save_health_records()

            logger.info(f"Added health record for user {self.current_user_id}")

            return record
        except Exception as e:
            logger.error(f"Error adding health record: {e}", exc_info=True)
            raise

    def _save_health_records(self):
        """Save the current user's health history."""
        try:
            # Ensure user directory exists
            user_dir = os.path.join(self.data_dir, self.current_user_id)
            os.makedirs(user_dir, exist_ok=True)

            # Save symptom checks
            symptom_checks_path = os.path.join(user_dir, "symptom_checks.json")
            with open(symptom_checks_path, 'w') as f:
                json.dump(self.health_history, f, indent=2)

            logger.info(f"Saved {len(self.health_history)} health history records for user {self.current_user_id}")
        except Exception as e:
            logger.error(f"Error saving health records: {e}", exc_info=True)

    def get_recent_symptom_checks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent symptom checks for the current user.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent symptom check records
        """
        # Health history is already sorted with most recent first
        return self.health_history[:limit] if limit else self.health_history

    def get_health_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current user's health.

        Returns:
            Dictionary of health statistics
        """
        try:
            if not self.health_history:
                return {
                    'total_checks': 0,
                    'total_symptoms': 0,
                    'unique_symptoms': 0,
                    'first_check': None,
                    'last_check': None
                }

            # Count unique symptoms
            all_symptoms = set()
            for check in self.health_history:
                all_symptoms.update(check.get('symptoms', []))

            # Get first and last check dates
            first_check = self.health_history[-1].get('date')
            last_check = self.health_history[0].get('date')

            return {
                'total_checks': len(self.health_history),
                'total_symptoms': sum(len(check.get('symptoms', [])) for check in self.health_history),
                'unique_symptoms': len(all_symptoms),
                'first_check': first_check,
                'last_check': last_check
            }
        except Exception as e:
            logger.error(f"Error getting health stats: {e}", exc_info=True)
            return {
                'total_checks': 0,
                'total_symptoms': 0,
                'unique_symptoms': 0,
                'first_check': None,
                'last_check': None,
                'error': str(e)
            }

    def clear_health_history(self) -> bool:
        """
        Clear the current user's health history.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.health_history = []
            self._save_health_records()

            logger.info(f"Cleared health history for user {self.current_user_id}")

            return True
        except Exception as e:
            logger.error(f"Error clearing health history: {e}", exc_info=True)
            return False

    def export_user_data(self, format: str = 'json') -> Dict[str, Any]:
        """
        Export all data for the current user.

        Args:
            format: Export format identifier (for future use)

        Returns:
            Dictionary containing all user data
        """
        try:
            # Get all data for current user
            export_data = {
                'profile': self.profile,
                'health_history': self.health_history
            }

            logger.info(f"Exported data for user {self.current_user_id}")

            return export_data
        except Exception as e:
            logger.error(f"Error exporting user data: {e}", exc_info=True)
            raise

    def import_user_data(self, import_data: Dict[str, Any]) -> bool:
        """
        Import data for the current user.

        Args:
            import_data: User data to import

        Returns:
            True if successful, False otherwise
        """
        try:
            # Import profile
            if 'profile' in import_data:
                # Keep original ID
                import_profile = import_data['profile'].copy()
                import_profile['id'] = self.current_user_id
                self.update_profile(import_profile)

            # Import health history
            if 'health_history' in import_data and isinstance(import_data['health_history'], list):
                # Clear existing history
                self.health_history = []

                # Add each record
                for record in import_data['health_history']:
                    self.add_health_record(record)

            logger.info(f"Imported data for user {self.current_user_id}")

            return True
        except Exception as e:
            logger.error(f"Error importing user data: {e}", exc_info=True)
            return False
