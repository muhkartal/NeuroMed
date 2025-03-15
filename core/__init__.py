"""
Core components for MedExplain AI Pro.

This package contains the fundamental functionality for the application,
including health data management, user profiles, and API integrations.
"""

from core.health_data_manager import HealthDataManager
from core.user_profile_manager import UserProfileManager
from core.openai_client import OpenAIClient

__all__ = ['HealthDataManager', 'UserProfileManager', 'OpenAIClient']
