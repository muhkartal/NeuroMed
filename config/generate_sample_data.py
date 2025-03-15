#!/usr/bin/env python
"""
Sample Data Generator for MedExplain AI Pro.

This script generates realistic sample data for testing and demonstration
purposes, including user profiles and health history records.

Usage:
  python generate_sample_data.py [--entries N] [--seed SEED] [--output-dir DIR]

Options:
  --entries N      Number of health history entries to generate (default: 20)
  --seed SEED      Random seed for reproducible results (default: 42)
  --output-dir DIR Output directory for generated files (default: project root)
  --overwrite      Overwrite existing files if they exist
  --help           Show this help message and exit
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sample_data_generator")

# Try to import from the medexplain package if it's in the path
try:
    from config import (
        USER_PROFILE_FILE,
        HEALTH_HISTORY_FILE,
        MEDICAL_DATA_FILE
    )
    from utils import save_json_file, load_json_file
    MEDEXPLAIN_AVAILABLE = True
except ImportError:
    logger.warning("medexplain package not found in path, using default paths")
    MEDEXPLAIN_AVAILABLE = False

    # Default paths if medexplain module is not available
    USER_PROFILE_FILE = "user_profile.json"
    HEALTH_HISTORY_FILE = "health_history.json"
    MEDICAL_DATA_FILE = "data/medical_data.json"

# Sample data for generation
SAMPLE_FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "Logan", "Amelia", "Lucas", "Mia", "Jackson", "Charlotte",
    "Aiden", "Harper", "Elijah", "Abigail", "Benjamin"
]

SAMPLE_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
]

SAMPLE_SYMPTOM_DURATIONS = [
    "Less than 24 hours",
    "1-3 days",
    "3-7 days",
    "1-2 weeks",
    "More than 2 weeks"
]

SAMPLE_SYMPTOM_SEVERITIES = ["Mild", "Moderate", "Severe"]

SAMPLE_FEVER_TEMPS = [99.5, 100.2, 101.4, 102.3, 103.1]

SAMPLE_BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

SAMPLE_EXERCISE_FREQUENCIES = [
    "Sedentary",
    "Light (1-2 days/week)",
    "Moderate (3-4 days/week)",
    "Active (5+ days/week)"
]

SAMPLE_SMOKING_STATUSES = ["Never smoked", "Former smoker", "Current smoker"]

SAMPLE_ALLERGIES = [
    "Pollen", "Dust mites", "Pet dander", "Mold", "Latex", "Insect stings",
    "Peanuts", "Tree nuts", "Shellfish", "Dairy", "Eggs", "Wheat", "Soy",
    "Penicillin", "Sulfa drugs", "NSAIDs", "Aspirin"
]

SAMPLE_CHRONIC_CONDITIONS = [
    "Asthma", "Diabetes Type 2", "Hypertension", "COPD", "Arthritis",
    "Migraine", "Depression", "Anxiety disorder", "Hypothyroidism",
    "Hyperlipidemia", "GERD", "IBS", "Eczema", "Psoriasis", "Chronic sinusitis"
]

SAMPLE_MEDICATIONS = [
    "Lisinopril", "Metformin", "Levothyroxine", "Atorvastatin", "Amlodipine",
    "Metoprolol", "Albuterol", "Omeprazole", "Losartan", "Gabapentin",
    "Hydrochlorothiazide", "Sertraline", "Fluoxetine", "Montelukast",
    "Pantoprazole", "Furosemide", "Escitalopram", "Prednisone"
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate sample data for MedExplain AI Pro")

    parser.add_argument("--entries", type=int, default=20,
                      help="Number of health history entries to generate (default: 20)")

    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducible results (default: 42)")

    parser.add_argument("--output-dir", type=str, default=".",
                      help="Output directory for generated files (default: current directory)")

    parser.add_argument("--overwrite", action="store_true",
                      help="Overwrite existing files if they exist")

    return parser.parse_args()

def load_medical_data(output_dir):
    """
    Load medical data (symptoms, conditions) for reference.

    Args:
        output_dir: Output directory for generated files

    Returns:
        Dictionary with medical data
    """
    medical_data_path = Path(output_dir) / MEDICAL_DATA_FILE

    if MEDEXPLAIN_AVAILABLE:
        medical_data = load_json_file(MEDICAL_DATA_FILE, {})
    else:
        try:
            with open(medical_data_path, 'r') as f:
                medical_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Medical data file not found or invalid at {medical_data_path}")
            medical_data = {}

    # If no data is available, use simplified default data
    if not medical_data or "symptoms" not in medical_data or "conditions" not in medical_data:
        logger.warning("Creating simplified default medical data")
        medical_data = create_default_medical_data()

    return medical_data

def create_default_medical_data():
    """
    Create default medical data with basic symptoms and conditions.

    Returns:
        Dictionary with default medical data
    """
    return {
        "symptoms": {
            "headache": {
                "name": "Headache",
                "description": "Pain in the head or upper neck."
            },
            "fever": {
                "name": "Fever",
                "description": "Elevated body temperature above the normal range."
            },
            "cough": {
                "name": "Cough",
                "description": "Sudden expulsion of air from the lungs."
            },
            "sore_throat": {
                "name": "Sore Throat",
                "description": "Pain or irritation in the throat."
            },
            "fatigue": {
                "name": "Fatigue",
                "description": "Extreme tiredness resulting from physical or mental exertion."
            },
            "nausea": {
                "name": "Nausea",
                "description": "Sensation of unease and discomfort in the stomach with urge to vomit."
            },
            "runny_nose": {
                "name": "Runny Nose",
                "description": "Excess drainage produced by nasal tissues and blood vessels."
            },
            "shortness_of_breath": {
                "name": "Shortness of Breath",
                "description": "Difficulty breathing or feeling that you cannot get enough air."
            }
        },
        "conditions": {
            "common_cold": {
                "name": "Common Cold",
                "symptoms": ["cough", "runny_nose", "sore_throat"],
                "severity": "low",
                "description": "A viral infection of the upper respiratory tract.",
                "treatment": "Rest, fluids, over-the-counter medications for symptoms."
            },
            "influenza": {
                "name": "Influenza (Flu)",
                "symptoms": ["fever", "cough", "fatigue", "headache"],
                "severity": "medium",
                "description": "A viral infection that attacks your respiratory system.",
                "treatment": "Rest, fluids, antiviral medications if prescribed."
            },
            "migraine": {
                "name": "Migraine",
                "symptoms": ["headache", "nausea"],
                "severity": "medium",
                "description": "A neurological condition that causes severe headaches.",
                "treatment": "Pain relievers, triptans, prevention medications."
            },
            "covid19": {
                "name": "COVID-19",
                "symptoms": ["fever", "cough", "fatigue", "shortness_of_breath"],
                "severity": "high",
                "description": "An infectious disease caused by the SARS-CoV-2 virus.",
                "treatment": "Rest, fluids, medications for symptoms."
            }
        }
    }

def generate_user_profile():
    """
    Generate a realistic user profile with random demographic information.

    Returns:
        Dictionary containing the user profile
    """
    gender = random.choice(["Male", "Female", "Non-binary"])

    # Adjust age range based on gender for more realistic life expectancy
    if gender == "Male":
        age = random.randint(18, 85)
    else:
        age = random.randint(18, 90)

    # Generate random name
    first_name = random.choice(SAMPLE_FIRST_NAMES)
    last_name = random.choice(SAMPLE_LAST_NAMES)
    name = f"{first_name} {last_name}"

    # Height and weight that are somewhat correlated with age and gender
    if gender == "Male":
        height = f"{random.randint(5, 6)}'{random.randint(0, 11)}\""
        weight = f"{random.randint(140, 230)} lbs"
    else:
        height = f"{random.randint(4, 5)}'{random.randint(0, 11)}\""
        weight = f"{random.randint(110, 190)} lbs"

    # Generate other profile data
    profile = {
        "name": name,
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "blood_type": random.choice(SAMPLE_BLOOD_TYPES),
        "exercise_frequency": random.choice(SAMPLE_EXERCISE_FREQUENCIES),
        "smoking_status": random.choice(SAMPLE_SMOKING_STATUSES),
        "allergies": random.sample(SAMPLE_ALLERGIES, random.randint(0, 3)),
        "chronic_conditions": random.sample(SAMPLE_CHRONIC_CONDITIONS, random.randint(0, 2)),
        "medications": random.sample(SAMPLE_MEDICATIONS, random.randint(0, 3)),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return profile

def generate_health_history_entry(medical_data, entry_date=None):
    """
    Generate a realistic health history entry.

    Args:
        medical_data: Dictionary containing symptoms and conditions data
        entry_date: Optional specific date for the entry

    Returns:
        Dictionary containing the health history entry
    """
    # Use provided date or generate a random one
    if entry_date is None:
        # Generate a random date within the last 6 months
        days_ago = random.randint(0, 180)
        entry_date = datetime.now() - timedelta(days=days_ago)

    # Format date as string
    date_str = entry_date.strftime("%Y-%m-%d %H:%M:%S")

    # Get available symptoms from medical data
    available_symptoms = list(medical_data.get("symptoms", {}).keys())
    if not available_symptoms:
        # Fallback if no symptoms found
        available_symptoms = ["headache", "fever", "cough", "fatigue"]

    # Select random number of symptoms (1-4)
    num_symptoms = random.randint(1, min(4, len(available_symptoms)))
    symptoms = random.sample(available_symptoms, num_symptoms)

    # Generate additional information
    had_fever = random.random() < 0.3  # 30% chance of fever
    fever_temp = random.choice(SAMPLE_FEVER_TEMPS) if had_fever else None
    symptom_duration = random.choice(SAMPLE_SYMPTOM_DURATIONS)
    symptom_severity = random.choice(SAMPLE_SYMPTOM_SEVERITIES)

    # Create base entry
    entry = {
        "date": date_str,
        "symptoms": symptoms,
        "symptom_duration": symptom_duration,
        "symptom_severity": symptom_severity,
        "had_fever": had_fever
    }

    # Add fever temperature if applicable
    if had_fever:
        entry["fever_temp"] = fever_temp

    # Add analysis results based on symptoms
    entry["analysis_results"] = generate_analysis_results(symptoms, medical_data)

    # Add risk assessment with 70% probability
    if random.random() < 0.7:
        entry["risk_assessment"] = generate_risk_assessment(symptoms, symptom_severity)

    return entry

def generate_analysis_results(symptoms, medical_data):
    """
    Generate analysis results based on symptoms and medical data.

    Args:
        symptoms: List of symptom IDs
        medical_data: Dictionary containing symptoms and conditions data

    Returns:
        Dictionary mapping condition IDs to condition data with confidence scores
    """
    results = {}
    conditions = medical_data.get("conditions", {})

    # For each condition, calculate match confidence
    for condition_id, condition_data in conditions.items():
        condition_symptoms = condition_data.get("symptoms", [])

        # Skip if no symptoms defined for this condition
        if not condition_symptoms:
            continue

        # Calculate matching symptoms
        matching_symptoms = set(symptoms).intersection(set(condition_symptoms))

        # Skip if no matching symptoms
        if not matching_symptoms:
            continue

        # Calculate confidence based on percentage of matching symptoms
        confidence = len(matching_symptoms) / len(condition_symptoms) * 100

        # Add random variation (+/- 10%)
        confidence += random.uniform(-10, 10)
        confidence = max(10, min(100, confidence))  # Clamp between 10 and 100

        # Only include if confidence is at least 30%
        if confidence >= 30:
            results[condition_id] = {
                "name": condition_data.get("name", condition_id),
                "confidence": round(confidence, 1),
                "matching_symptoms": list(matching_symptoms),
                "description": condition_data.get("description", ""),
                "severity": condition_data.get("severity", "medium"),
                "treatment": condition_data.get("treatment", ""),
                "when_to_see_doctor": condition_data.get("when_to_see_doctor",
                                                      "Consult a healthcare professional if symptoms persist or worsen.")
            }

    # Limit to top 3 conditions by confidence
    results = dict(sorted(results.items(), key=lambda item: item[1]["confidence"], reverse=True)[:3])

    return results

def generate_risk_assessment(symptoms, symptom_severity):
    """
    Generate a risk assessment based on symptoms and severity.

    Args:
        symptoms: List of symptom IDs
        symptom_severity: Severity level string

    Returns:
        Risk assessment dictionary
    """
    # Define high-risk symptoms
    high_risk_symptoms = ["shortness_of_breath", "chest_pain", "severe_headache", "high_fever"]

    # Calculate risk score
    risk_score = 0

    # Add points based on symptom severity
    if symptom_severity == "Mild":
        risk_score += 5
    elif symptom_severity == "Moderate":
        risk_score += 15
    elif symptom_severity == "Severe":
        risk_score += 30

    # Add points for high-risk symptoms
    for symptom in symptoms:
        if symptom in high_risk_symptoms:
            risk_score += 20

    # Add points based on number of symptoms
    risk_score += len(symptoms) * 5

    # Add random variation
    risk_score += random.randint(-10, 10)
    risk_score = max(0, risk_score)  # Ensure non-negative

    # Determine risk level
    if risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 25:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Generate risk factors
    risk_factors = []
    if symptom_severity == "Severe":
        risk_factors.append("Severe symptom intensity")

    if len(symptoms) >= 3:
        risk_factors.append("Multiple symptoms present")

    for symptom in symptoms:
        if symptom in high_risk_symptoms:
            risk_factors.append(f"High-risk symptom: {symptom}")

    # Generate protective factors
    protective_factors = []
    if random.random() < 0.3:
        protective_factors.append("Regular physical activity")

    if random.random() < 0.5:
        protective_factors.append("Non-smoker")

    # Generate recommendations
    recommendations = []

    if risk_level == "high":
        recommendations.append("Consider scheduling a prompt appointment with your healthcare provider.")
        recommendations.append("Monitor your symptoms closely and seek immediate care if they worsen.")
    elif risk_level == "moderate":
        recommendations.append("Consider discussing your symptoms with a healthcare provider.")
        recommendations.append("Monitor your symptoms for any changes in frequency or severity.")
    else:  # low risk
        recommendations.append("Continue healthy lifestyle habits and self-care measures.")
        recommendations.append("Monitor your symptoms and consult a healthcare provider if they persist or worsen.")

    # Add general recommendations
    recommendations.append("Stay hydrated and get adequate rest.")

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "recommendations": recommendations
    }

def main():
    """Main function to generate sample data."""
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define output file paths
    profile_path = output_dir / USER_PROFILE_FILE
    history_path = output_dir / HEALTH_HISTORY_FILE

    # Check if files already exist
    if profile_path.exists() and history_path.exists() and not args.overwrite:
        logger.error("Output files already exist. Use --overwrite to overwrite them.")
        sys.exit(1)

    # Load medical data
    medical_data = load_medical_data(output_dir)

    # Generate user profile
    user_profile = generate_user_profile()
    logger.info(f"Generated user profile: {user_profile['name']}, {user_profile['age']} years old")

    # Generate health history entries
    health_history = []

    # Generate some recent entries (last 30 days)
    recent_dates = []
    for _ in range(min(5, args.entries // 4)):
        days_ago = random.randint(0, 30)
        recent_dates.append(datetime.now() - timedelta(days=days_ago))

    # Sort recent dates (newest first)
    recent_dates.sort(reverse=True)

    # Generate entries with recent dates
    for date in recent_dates:
        entry = generate_health_history_entry(medical_data, date)
        health_history.append(entry)

    # Generate remaining entries with random dates
    remaining_entries = args.entries - len(recent_dates)
    for _ in range(remaining_entries):
        entry = generate_health_history_entry(medical_data)
        health_history.append(entry)

    # Sort entries by date (newest first)
    health_history.sort(key=lambda x: x["date"], reverse=True)

    # Save user profile
    if MEDEXPLAIN_AVAILABLE:
        save_json_file(user_profile, profile_path)
    else:
        with open(profile_path, 'w') as f:
            json.dump(user_profile, f, indent=2)

    logger.info(f"Saved user profile to {profile_path}")

    # Save health history
    if MEDEXPLAIN_AVAILABLE:
        save_json_file(health_history, history_path)
    else:
        with open(history_path, 'w') as f:
            json.dump(health_history, f, indent=2)

    logger.info(f"Saved {len(health_history)} health history entries to {history_path}")

    # Print summary
    print(f"\nGenerated sample data with seed: {args.seed}")
    print(f"User Profile: {user_profile['name']}, {user_profile['age']} years old, {user_profile['gender']}")
    print(f"Health History: {len(health_history)} entries")
    print(f"Output Directory: {output_dir}")
    print("\nFiles created:")
    print(f"  - {profile_path}")
    print(f"  - {history_path}")
    print("\nTo use this data in the MedExplain AI Pro application, make sure these files")
    print("are in the correct location as specified in the configuration.")

if __name__ == "__main__":
    main()
