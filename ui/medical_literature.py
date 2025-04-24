import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Union
import random

# Configure logger
logger = logging.getLogger(__name__)

class MedicalLiteratureUI:
    """
    Self-contained Medical Literature UI with built-in data management.
    Enhanced with enterprise-level UI styling.
    """

    def __init__(self, health_data_manager=None, user_manager=None):
        """
        Initialize the MedicalLiteratureUI with embedded data management.

        Args:
            health_data_manager: Optional external data manager (not used internally)
            user_manager: The user profile manager instance
        """
        self.user_manager = user_manager
        self.external_data_manager = health_data_manager  # Stored but not used

        # Initialize internal data structures
        self._conditions = None
        self._symptoms = None
        self._literature = None

        # Load data immediately
        self._initialize_data()

        # Add enterprise styling
        self._add_enterprise_styling()

    def _add_enterprise_styling(self):
        """Add enterprise-level styling to match the dark theme."""
        st.markdown("""
        <style>
        /* Enterprise dark theme styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        body {
            font-family: 'Inter', sans-serif;
        }

        /* Modern enterprise card styling */
        .condition-card {
            background: linear-gradient(145deg, #151e29 0%, #0f1620 100%);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(45, 55, 72, 0.5);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
            height: 210px;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }

        .condition-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #3498db, #2980b9);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .condition-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 25px rgba(0, 0, 0, 0.35);
            border-color: rgba(52, 152, 219, 0.4);
        }

        .condition-card:hover::before {
            opacity: 1;
        }

        .condition-card h4 {
            color: #ffffff;
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        .condition-card p {
            color: #bdc3c7;
            font-size: 0.95em;
            line-height: 1.5;
            height: 80px;
            overflow: hidden;
            margin-bottom: 25px;
        }

        .category-badge {
            display: inline-block;
            padding: 5px 12px;
            background: rgba(52, 152, 219, 0.15);
            color: #3498db;
            border-radius: 30px;
            font-size: 0.8em;
            font-weight: 500;
            position: absolute;
            bottom: 18px;
            left: 24px;
            letter-spacing: 0.02em;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(52, 152, 219, 0.2);
            text-transform: uppercase;
        }

        .category-badge.Neurological { background: rgba(155, 89, 182, 0.15); color: #9b59b6; border-color: rgba(155, 89, 182, 0.2); }
        .category-badge.Cardiovascular { background: rgba(231, 76, 60, 0.15); color: #e74c3c; border-color: rgba(231, 76, 60, 0.2); }
        .category-badge.Respiratory { background: rgba(52, 152, 219, 0.15); color: #3498db; border-color: rgba(52, 152, 219, 0.2); }
        .category-badge.Endocrine { background: rgba(46, 204, 113, 0.15); color: #2ecc71; border-color: rgba(46, 204, 113, 0.2); }
        .category-badge.Musculoskeletal { background: rgba(230, 126, 34, 0.15); color: #e67e22; border-color: rgba(230, 126, 34, 0.2); }
        .category-badge.Digestive { background: rgba(241, 196, 15, 0.15); color: #f1c40f; border-color: rgba(241, 196, 15, 0.2); }
        .category-badge.Immunological { background: rgba(52, 73, 94, 0.15); color: #34495e; border-color: rgba(52, 73, 94, 0.2); }
        .category-badge.Mental.Health { background: rgba(155, 89, 182, 0.15); color: #9b59b6; border-color: rgba(155, 89, 182, 0.2); }
        .category-badge.Sleep.Disorders { background: rgba(142, 68, 173, 0.15); color: #8e44ad; border-color: rgba(142, 68, 173, 0.2); }

        .details-button {
            position: absolute;
            bottom: 18px;
            right: 24px;
            padding: 7px 14px;
            background: linear-gradient(90deg, #3498db, #2980b9);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
            font-weight: 500;
            transition: all 0.2s;
            letter-spacing: 0.02em;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        .details-button:hover {
            background: linear-gradient(90deg, #3498db, #1d6fa5);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Enterprise styling for literature items */
        .literature-item {
            background: linear-gradient(145deg, #151e29 0%, #0f1620 100%);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(45, 55, 72, 0.5);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
            position: relative;
        }

        .literature-item::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            left: 0;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0);
            transition: box-shadow 0.3s ease;
            pointer-events: none;
        }

        .literature-item:hover {
            transform: translateY(-4px);
            border-color: rgba(52, 152, 219, 0.4);
        }

        .literature-item:hover::after {
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }

        .literature-item h4 {
            color: #ffffff;
            margin-top: 0;
            font-size: 1.15em;
            font-weight: 600;
            letter-spacing: -0.01em;
            line-height: 1.4;
        }

        .literature-item .meta {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 12px 0;
            color: #bdc3c7;
            font-size: 0.85em;
        }

        .literature-item .meta span.highlight {
            color: #3498db;
            font-weight: 500;
        }

        .literature-item .summary {
            color: #ecf0f1;
            font-size: 0.95em;
            line-height: 1.6;
            margin-top: 14px;
            border-top: 1px solid rgba(45, 55, 72, 0.5);
            padding-top: 14px;
        }

        /* Tab styling */
        div[data-baseweb="tab-list"] {
            background-color: #0e1117;
            border-radius: 8px;
            padding: 5px;
            border: 1px solid rgba(45, 55, 72, 0.5);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        button[data-baseweb="tab"] {
            color: #bdc3c7 !important;
            font-weight: 500;
            transition: all 0.2s;
            border-radius: 6px;
            margin: 0 2px;
        }

        button[data-baseweb="tab"]:hover {
            background-color: rgba(52, 152, 219, 0.1);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff !important;
            background-color: rgba(52, 152, 219, 0.15);
            border-bottom-color: #3498db !important;
            font-weight: 600;
        }

        /* Search bar styling */
        .stTextInput input {
            background-color: #151e29;
            border: 1px solid rgba(45, 55, 72, 0.8);
            color: #ffffff;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 16px;
            transition: all 0.2s;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }

        .stTextInput input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.25);
        }

        /* Filter control styling */
        .stSelectbox label, .stMultiSelect label {
            color: #bdc3c7 !important;
            font-weight: 500;
            font-size: 0.95em;
        }

        .stSelectbox [data-baseweb="select"] {
            background-color: #151e29;
            border: 1px solid rgba(45, 55, 72, 0.8);
            border-radius: 8px;
            transition: all 0.2s;
        }

        .stSelectbox [data-baseweb="select"]:hover {
            border-color: rgba(52, 152, 219, 0.5);
        }

        .stSelectbox [data-baseweb="icon"] {
            color: #bdc3c7;
        }

        /* Detail view styling */
        .detail-container {
            background: linear-gradient(145deg, #151e29 0%, #0f1620 100%);
            border: 1px solid rgba(45, 55, 72, 0.5);
            border-radius: 12px;
            padding: 28px;
            margin: 18px 0;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        }

        .detail-container h3 {
            color: #ffffff;
            border-bottom: 1px solid rgba(45, 55, 72, 0.8);
            padding-bottom: 14px;
            margin-bottom: 20px;
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        .symptom-item {
            background: linear-gradient(to right, #171f2b, #1a2433);
            border-left: 4px solid #3498db;
            padding: 16px;
            margin-bottom: 12px;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .symptom-item:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .symptom-item strong {
            color: #ffffff;
            font-weight: 600;
            font-size: 1.05em;
            display: block;
            margin-bottom: 5px;
        }

        .symptom-item span {
            color: #bdc3c7;
            font-size: 0.95em;
            line-height: 1.5;
            display: block;
        }

        .back-button {
            background-color: rgba(52, 152, 219, 0.1);
            border: 1px solid rgba(52, 152, 219, 0.4);
            color: #3498db;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.9em;
            margin-bottom: 16px;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            font-weight: 500;
        }

        .back-button::before {
            content: "←";
            font-size: 1.2em;
            margin-right: 6px;
            line-height: 0;
        }

        .back-button:hover {
            background-color: rgba(52, 152, 219, 0.2);
            transform: translateX(-4px);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        /* Custom selectbox styling */
        .category-filter {
            background-color: #151e29;
            border: 1px solid rgba(45, 55, 72, 0.8);
            border-radius: 8px;
            padding: 10px 14px;
            color: white;
            width: 100%;
            font-size: 16px;
            transition: all 0.2s;
        }

        .category-filter:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.25);
            outline: none;
        }

        /* Hide certain elements */
        .hidden-element {
            display: none;
        }

        /* Additional enterprise styling */
        h1, h2, h3, h4, h5, h6 {
            letter-spacing: -0.02em;
        }

        .stButton > button {
            border-radius: 6px;
            font-weight: 500;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: all 0.2s;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }

        /* Streamlit default button improvement */
        button[kind="secondary"] {
            border: 1px solid rgba(52, 152, 219, 0.5) !important;
            color: #3498db !important;
        }

        button[kind="primary"] {
            background: linear-gradient(90deg, #3498db, #2980b9) !important;
            border: none !important;
        }

        /* Table styling for better enterprise look */
        table {
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        thead tr th {
            background-color: #1a2433 !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px 16px !important;
            border-bottom: 2px solid rgba(52, 152, 219, 0.3) !important;
        }

        tbody tr:nth-child(even) {
            background-color: rgba(26, 36, 51, 0.4) !important;
        }

        tbody tr:hover {
            background-color: rgba(52, 152, 219, 0.1) !important;
        }

        td {
            padding: 12px 16px !important;
            border-bottom: 1px solid rgba(45, 55, 72, 0.3) !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def _initialize_data(self) -> None:
        """Initialize internal health data structures with predefined data."""
        self._initialize_conditions()
        self._initialize_symptoms()
        self._initialize_literature()
        logger.info("Medical Literature UI initialized with embedded data")

    def _initialize_conditions(self) -> None:
        """Initialize with default health conditions data."""
        self._conditions = {
            "migraine": {
                "id": "migraine",
                "name": "Migraine",
                "description": "A neurological condition characterized by recurring headaches, often with intense pain, sensitivity to light and sound, and sometimes nausea.",
                "typical_duration": "4-72 hours per episode",
                "treatment": "Pain relievers, triptans, preventive medications, lifestyle changes",
                "when_to_see_doctor": "If headaches are severe, frequent, or accompanied by neurological symptoms",
                "symptoms": ["headache", "sensitivity_to_light", "nausea", "aura"],
                "category": "Neurological"
            },
            "hypertension": {
                "id": "hypertension",
                "name": "Hypertension",
                "description": "High blood pressure that can lead to serious health problems if untreated.",
                "typical_duration": "Chronic condition requiring ongoing management",
                "treatment": "Medications, diet changes, exercise, stress reduction",
                "when_to_see_doctor": "Regular monitoring is essential; seek immediate care for very high readings",
                "symptoms": ["headache", "shortness_of_breath", "nosebleeds", "dizziness"],
                "category": "Cardiovascular"
            },
            "diabetes": {
                "id": "diabetes",
                "name": "Diabetes",
                "description": "A metabolic disorder affecting how your body processes blood sugar.",
                "typical_duration": "Chronic condition requiring lifelong management",
                "treatment": "Insulin, oral medications, lifestyle modifications, regular monitoring",
                "when_to_see_doctor": "If experiencing symptoms or for regular checkups if diagnosed",
                "symptoms": ["increased_thirst", "frequent_urination", "fatigue", "blurred_vision"],
                "category": "Endocrine"
            },
            "asthma": {
                "id": "asthma",
                "name": "Asthma",
                "description": "A condition affecting the airways, causing wheezing, shortness of breath, and coughing.",
                "typical_duration": "Chronic with periodic flare-ups",
                "treatment": "Inhalers, medications, avoiding triggers",
                "when_to_see_doctor": "If symptoms worsen or don't improve with prescribed treatment",
                "symptoms": ["wheezing", "shortness_of_breath", "chest_tightness", "coughing"],
                "category": "Respiratory"
            },
            "arthritis": {
                "id": "arthritis",
                "name": "Arthritis",
                "description": "Inflammation of one or more joints causing pain and stiffness.",
                "typical_duration": "Chronic, often progressive",
                "treatment": "Anti-inflammatory medications, physical therapy, sometimes surgery",
                "when_to_see_doctor": "If joint pain persists or worsens over time",
                "symptoms": ["joint_pain", "stiffness", "swelling", "reduced_mobility"],
                "category": "Musculoskeletal"
            },
            "anxiety": {
                "id": "anxiety",
                "name": "Anxiety Disorder",
                "description": "A mental health condition characterized by persistent, excessive worry.",
                "typical_duration": "Varies widely, can be episodic or persistent",
                "treatment": "Therapy, medications, stress management techniques",
                "when_to_see_doctor": "When anxiety interferes with daily activities or causes significant distress",
                "symptoms": ["excessive_worry", "restlessness", "fatigue", "difficulty_concentrating"],
                "category": "Mental Health"
            },
            "depression": {
                "id": "depression",
                "name": "Depression",
                "description": "A mood disorder causing persistent feelings of sadness and loss of interest.",
                "typical_duration": "Episodes typically last weeks to months if untreated",
                "treatment": "Therapy, medications, lifestyle changes",
                "when_to_see_doctor": "If symptoms persist for more than two weeks or thoughts of self-harm occur",
                "symptoms": ["persistent_sadness", "loss_of_interest", "sleep_changes", "fatigue"],
                "category": "Mental Health"
            },
            "insomnia": {
                "id": "insomnia",
                "name": "Insomnia",
                "description": "Difficulty falling asleep or staying asleep, even when given the opportunity.",
                "typical_duration": "Can be short-term (days to weeks) or chronic (months to years)",
                "treatment": "Sleep hygiene improvements, behavioral therapy, sometimes medications",
                "when_to_see_doctor": "If sleep problems persist for more than a month or significantly impact daily life",
                "symptoms": ["difficulty_falling_asleep", "waking_during_night", "daytime_fatigue", "irritability"],
                "category": "Sleep Disorders"
            },
            "gerd": {
                "id": "gerd",
                "name": "GERD",
                "description": "Gastroesophageal reflux disease, causing acid reflux and heartburn.",
                "typical_duration": "Often chronic, with varying severity",
                "treatment": "Lifestyle changes, antacids, acid blockers, sometimes surgery",
                "when_to_see_doctor": "If symptoms persist despite over-the-counter treatments or occur frequently",
                "symptoms": ["heartburn", "regurgitation", "difficulty_swallowing", "chest_pain"],
                "category": "Digestive"
            },
            "allergies": {
                "id": "allergies",
                "name": "Allergies",
                "description": "Immune system reactions to substances that are typically harmless to most people.",
                "typical_duration": "Can be seasonal or year-round depending on triggers",
                "treatment": "Avoidance of triggers, antihistamines, nasal sprays, sometimes immunotherapy",
                "when_to_see_doctor": "If symptoms significantly impact quality of life or anaphylaxis occurs",
                "symptoms": ["sneezing", "runny_nose", "itchy_eyes", "skin_rash"],
                "category": "Immunological"
            },
            "covid19": {
                "id": "covid19",
                "name": "COVID-19",
                "description": "An infectious disease caused by the SARS-CoV-2 virus affecting the respiratory system.",
                "typical_duration": "Acute phase: 1-2 weeks; long COVID can persist for months",
                "treatment": "Supportive care, antivirals for severe cases, rest and hydration for mild cases",
                "when_to_see_doctor": "If experiencing severe symptoms like difficulty breathing, persistent chest pain, or confusion",
                "symptoms": ["fever", "cough", "shortness_of_breath", "fatigue", "loss_of_taste", "sore_throat"],
                "category": "Infectious Disease"
            },
            "ibs": {
                "id": "ibs",
                "name": "Irritable Bowel Syndrome",
                "description": "A chronic disorder affecting the large intestine, causing abdominal pain, bloating, and altered bowel habits.",
                "typical_duration": "Chronic with symptom flare-ups",
                "treatment": "Dietary changes, stress management, medications for specific symptoms",
                "when_to_see_doctor": "If symptoms are severe or include weight loss, rectal bleeding, or persistent pain",
                "symptoms": ["abdominal_pain", "bloating", "diarrhea", "constipation"],
                "category": "Digestive"
            }
        }

    def _initialize_symptoms(self) -> None:
        """Initialize with default symptoms data."""
        self._symptoms = {
            # Neurological symptoms
            "headache": {"id": "headache", "name": "Headache", "description": "Pain in any region of the head.", "category": "Neurological"},
            "sensitivity_to_light": {"id": "sensitivity_to_light", "name": "Light Sensitivity", "description": "Discomfort or pain when exposed to light.", "category": "Neurological"},
            "aura": {"id": "aura", "name": "Aura", "description": "Visual disturbances that can include flashes, blind spots, or other vision changes.", "category": "Neurological"},
            "dizziness": {"id": "dizziness", "name": "Dizziness", "description": "Feeling lightheaded, unsteady, or like the room is spinning.", "category": "Neurological"},

            # Cardiovascular symptoms
            "shortness_of_breath": {"id": "shortness_of_breath", "name": "Shortness of Breath", "description": "Difficulty breathing or feeling like you can't get enough air.", "category": "Cardiovascular"},
            "chest_pain": {"id": "chest_pain", "name": "Chest Pain", "description": "Pain or discomfort in the chest area.", "category": "Cardiovascular"},
            "nosebleeds": {"id": "nosebleeds", "name": "Nosebleeds", "description": "Bleeding from the nose.", "category": "Cardiovascular"},

            # Endocrine symptoms
            "increased_thirst": {"id": "increased_thirst", "name": "Increased Thirst", "description": "Feeling more thirsty than usual.", "category": "Endocrine"},
            "frequent_urination": {"id": "frequent_urination", "name": "Frequent Urination", "description": "Needing to urinate more often than usual.", "category": "Endocrine"},
            "blurred_vision": {"id": "blurred_vision", "name": "Blurred Vision", "description": "Lack of sharpness in vision making objects appear out of focus.", "category": "Endocrine"},

            # Respiratory symptoms
            "wheezing": {"id": "wheezing", "name": "Wheezing", "description": "High-pitched whistling sound during breathing.", "category": "Respiratory"},
            "chest_tightness": {"id": "chest_tightness", "name": "Chest Tightness", "description": "Feeling of pressure or constriction in the chest.", "category": "Respiratory"},
            "coughing": {"id": "coughing", "name": "Coughing", "description": "Sudden expulsion of air from the lungs.", "category": "Respiratory"},

            # Musculoskeletal symptoms
            "joint_pain": {"id": "joint_pain", "name": "Joint Pain", "description": "Discomfort or soreness in one or more joints.", "category": "Musculoskeletal"},
            "stiffness": {"id": "stiffness", "name": "Stiffness", "description": "Reduced flexibility or difficulty moving joints.", "category": "Musculoskeletal"},
            "swelling": {"id": "swelling", "name": "Swelling", "description": "Enlargement of a body part due to fluid buildup.", "category": "Musculoskeletal"},
            "reduced_mobility": {"id": "reduced_mobility", "name": "Reduced Mobility", "description": "Difficulty moving around or performing usual activities.", "category": "Musculoskeletal"},

            # Mental health symptoms
            "excessive_worry": {"id": "excessive_worry", "name": "Excessive Worry", "description": "Persistent, uncontrollable concerns about many things.", "category": "Mental Health"},
            "restlessness": {"id": "restlessness", "name": "Restlessness", "description": "Inability to rest or relax, feeling agitated.", "category": "Mental Health"},
            "difficulty_concentrating": {"id": "difficulty_concentrating", "name": "Difficulty Concentrating", "description": "Trouble focusing on tasks or activities.", "category": "Mental Health"},
            "persistent_sadness": {"id": "persistent_sadness", "name": "Persistent Sadness", "description": "Ongoing feelings of sadness, emptiness, or hopelessness.", "category": "Mental Health"},
            "loss_of_interest": {"id": "loss_of_interest", "name": "Loss of Interest", "description": "Reduced interest or pleasure in activities once enjoyed.", "category": "Mental Health"},

            # Sleep-related symptoms
            "sleep_changes": {"id": "sleep_changes", "name": "Sleep Changes", "description": "Changes in sleep patterns, either insomnia or sleeping too much.", "category": "Sleep"},
            "difficulty_falling_asleep": {"id": "difficulty_falling_asleep", "name": "Difficulty Falling Asleep", "description": "Trouble getting to sleep when going to bed.", "category": "Sleep"},
            "waking_during_night": {"id": "waking_during_night", "name": "Waking During Night", "description": "Waking up during the night and having trouble returning to sleep.", "category": "Sleep"},
            "daytime_fatigue": {"id": "daytime_fatigue", "name": "Daytime Fatigue", "description": "Feeling tired during the day despite adequate sleep time.", "category": "Sleep"},

            # General symptoms
            "fatigue": {"id": "fatigue", "name": "Fatigue", "description": "Feeling of tiredness or exhaustion.", "category": "General"},
            "irritability": {"id": "irritability", "name": "Irritability", "description": "Being easily annoyed or agitated.", "category": "General"},
            "nausea": {"id": "nausea", "name": "Nausea", "description": "Feeling of sickness with an inclination to vomit.", "category": "Digestive"},

            # Digestive symptoms
            "heartburn": {"id": "heartburn", "name": "Heartburn", "description": "Burning sensation in the chest, usually after eating.", "category": "Digestive"},
            "regurgitation": {"id": "regurgitation", "name": "Regurgitation", "description": "Backflow of stomach contents into the throat or mouth.", "category": "Digestive"},
            "difficulty_swallowing": {"id": "difficulty_swallowing", "name": "Difficulty Swallowing", "description": "Trouble moving food or liquid from the mouth to the stomach.", "category": "Digestive"},
            "abdominal_pain": {"id": "abdominal_pain", "name": "Abdominal Pain", "description": "Pain in the area between the chest and groin.", "category": "Digestive"},
            "bloating": {"id": "bloating", "name": "Bloating", "description": "Swelling or distention of the abdomen.", "category": "Digestive"},
            "diarrhea": {"id": "diarrhea", "name": "Diarrhea", "description": "Loose, watery stools occurring more frequently than usual.", "category": "Digestive"},
            "constipation": {"id": "constipation", "name": "Constipation", "description": "Difficulty passing stools or infrequent bowel movements.", "category": "Digestive"},

            # Allergy/immunological symptoms
            "sneezing": {"id": "sneezing", "name": "Sneezing", "description": "Sudden, forceful expulsion of air through the nose and mouth.", "category": "Immunological"},
            "runny_nose": {"id": "runny_nose", "name": "Runny Nose", "description": "Excess discharge of nasal fluid.", "category": "Immunological"},
            "itchy_eyes": {"id": "itchy_eyes", "name": "Itchy Eyes", "description": "Irritation and itchiness in the eyes.", "category": "Immunological"},
            "skin_rash": {"id": "skin_rash", "name": "Skin Rash", "description": "Area of irritated or swollen skin.", "category": "Immunological"},

            # COVID-specific symptoms
            "fever": {"id": "fever", "name": "Fever", "description": "Elevated body temperature, typically above 100.4°F (38°C).", "category": "General"},
            "loss_of_taste": {"id": "loss_of_taste", "name": "Loss of Taste", "description": "Inability to detect flavors normally.", "category": "Neurological"},
            "sore_throat": {"id": "sore_throat", "name": "Sore Throat", "description": "Pain, scratchiness or irritation of the throat.", "category": "Respiratory"}
        }

    def _initialize_literature(self) -> None:
        """Initialize with default medical literature data."""
        self._literature = {
            "migraine": [
                {
                    "title": "Novel CGRP Receptor Antagonists for Migraine Treatment",
                    "journal": "Neurology Today",
                    "year": 2023,
                    "summary": "This review examines recent developments in CGRP antagonists for migraine, showing significant reduction in frequency and intensity of migraine attacks with fewer side effects than older treatments.",
                    "authors": "Johnson K, Smith P, et al."
                },
                {
                    "title": "Migraine and Sleep: Understanding the Relationship",
                    "journal": "Journal of Sleep Medicine",
                    "year": 2022,
                    "summary": "This study investigates the bidirectional relationship between migraines and sleep disorders, finding that improving sleep quality can reduce migraine frequency by up to 40% in susceptible individuals.",
                    "authors": "Martinez A, Williams T, et al."
                },
                {
                    "title": "Nutritional Interventions for Migraine Prevention",
                    "journal": "Headache: The Journal of Head and Face Pain",
                    "year": 2023,
                    "summary": "This systematic review analyzes dietary modifications for migraine prevention, with strong evidence supporting the benefits of adequate hydration, regular meal timing, and omega-3 supplementation.",
                    "authors": "Chen L, Garcia R, et al."
                }
            ],
            "hypertension": [
                {
                    "title": "Effectiveness of Combination Therapy in Resistant Hypertension",
                    "journal": "Journal of Hypertension",
                    "year": 2023,
                    "summary": "This clinical trial demonstrates improved outcomes when combining ACE inhibitors, calcium channel blockers, and diuretics for patients with resistant hypertension, achieving control in 68% of previously uncontrolled cases.",
                    "authors": "Roberts J, Patel S, et al."
                },
                {
                    "title": "Impact of Sodium Reduction on Blood Pressure: Meta-analysis",
                    "journal": "Circulation",
                    "year": 2022,
                    "summary": "This meta-analysis of 42 studies confirms that reducing sodium intake by 1,000mg per day lowers systolic blood pressure by an average of 5.4 mmHg in hypertensive individuals.",
                    "authors": "Thompson K, Nelson M, et al."
                },
                {
                    "title": "Exercise Intensity and Blood Pressure Control",
                    "journal": "American Journal of Cardiology",
                    "year": 2023,
                    "summary": "This randomized controlled trial compares different exercise intensities for hypertension management, finding that moderate-intensity exercise for 30 minutes five times weekly is optimal for most patients.",
                    "authors": "Huang Y, Fernandez J, et al."
                }
            ],
            "diabetes": [
                {
                    "title": "Continuous Glucose Monitoring and Type 2 Diabetes Management",
                    "journal": "Diabetes Care",
                    "year": 2023,
                    "summary": "This study shows that continuous glucose monitoring improves glycemic control in Type 2 diabetes patients, with a 0.8% average reduction in HbA1c compared to traditional monitoring methods.",
                    "authors": "Singh K, Wong L, et al."
                },
                {
                    "title": "Plant-Based Diets in Diabetes Prevention and Management",
                    "journal": "Journal of the American College of Nutrition",
                    "year": 2022,
                    "summary": "This review demonstrates that plant-based diets reduce the risk of Type 2 diabetes by 23% and improve insulin sensitivity in existing patients, with greatest benefits from diets rich in legumes and whole grains.",
                    "authors": "Miller A, Green T, et al."
                },
                {
                    "title": "Novel GLP-1 Receptor Agonists for Weight Management in Type 2 Diabetes",
                    "journal": "The New England Journal of Medicine",
                    "year": 2023,
                    "summary": "This landmark trial evaluates newer GLP-1 receptor agonists for diabetes management, showing both improved glycemic control and significant weight loss benefits in patients with obesity and Type 2 diabetes.",
                    "authors": "Anderson R, Collins P, et al."
                }
            ],
            "asthma": [
                {
                    "title": "Biologics for Severe Asthma: Comparative Effectiveness",
                    "journal": "European Respiratory Journal",
                    "year": 2023,
                    "summary": "This head-to-head comparison of biological therapies for severe asthma identifies specific patient phenotypes that respond best to each treatment option, offering guidance for personalized therapy selection.",
                    "authors": "Brown M, Sharma D, et al."
                },
                {
                    "title": "Environmental Triggers and Asthma Exacerbations",
                    "journal": "Journal of Allergy and Clinical Immunology",
                    "year": 2022,
                    "summary": "This longitudinal study identifies key environmental factors triggering asthma exacerbations, finding that air pollution, allergen exposure, and viral infections account for 85% of severe asthma attacks.",
                    "authors": "Reynolds K, Liu X, et al."
                }
            ],
            "covid19": [
                {
                    "title": "Long-term Effects of COVID-19: A Systematic Review",
                    "journal": "Journal of Infectious Diseases",
                    "year": 2023,
                    "summary": "This systematic review analyzes the growing body of evidence regarding long-term effects of COVID-19, including neurological, cardiovascular, and respiratory sequelae, finding persistent symptoms in approximately 30% of hospitalized patients.",
                    "authors": "Park J, Cooper A, et al."
                },
                {
                    "title": "Vaccine Effectiveness Against Emerging SARS-CoV-2 Variants",
                    "journal": "The Lancet Infectious Diseases",
                    "year": 2023,
                    "summary": "This ongoing surveillance study tracks vaccine effectiveness against new COVID-19 variants, showing maintained protection against severe disease despite decreased neutralization of newer strains.",
                    "authors": "Jones B, Wilson M, et al."
                },
                {
                    "title": "Post-COVID Cognitive Impairment: Mechanisms and Interventions",
                    "journal": "Nature Neuroscience",
                    "year": 2023,
                    "summary": "This review investigates the neurological impact of COVID-19, proposing mechanisms for 'brain fog' and cognitive symptoms, and evaluating potential therapeutic approaches for cognitive rehabilitation.",
                    "authors": "Gonzalez T, Lee S, et al."
                }
            ],
            "arthritis": [
                {
                    "title": "Novel DMARDs for Rheumatoid Arthritis Treatment",
                    "journal": "Arthritis & Rheumatology",
                    "year": 2023,
                    "summary": "This clinical trial evaluates next-generation disease-modifying antirheumatic drugs, showing superior remission rates with reduced side effects compared to conventional therapies.",
                    "authors": "Zhang H, Davies L, et al."
                },
                {
                    "title": "Physical Activity Guidelines for Osteoarthritis",
                    "journal": "Osteoarthritis and Cartilage",
                    "year": 2022,
                    "summary": "This evidence-based review establishes updated physical activity recommendations for osteoarthritis patients, emphasizing low-impact exercises and strength training for optimal joint health.",
                    "authors": "Robinson K, Walker J, et al."
                }
            ],
            "anxiety": [
                {
                    "title": "Digital Interventions for Anxiety Disorders: Systematic Review",
                    "journal": "Journal of Medical Internet Research",
                    "year": 2023,
                    "summary": "This review evaluates the effectiveness of digital mental health interventions for anxiety disorders, finding that guided mobile applications show comparable efficacy to traditional therapy for mild to moderate anxiety.",
                    "authors": "Kim Y, Evans D, et al."
                },
                {
                    "title": "Neurobiological Mechanisms of Anxiolytic Medications",
                    "journal": "Neuropsychopharmacology",
                    "year": 2022,
                    "summary": "This paper explores the molecular and neural circuit mechanisms of various anti-anxiety medications, providing insights into treatment selection based on specific anxiety subtypes.",
                    "authors": "Patel R, Thompson J, et al."
                }
            ],
            "depression": [
                {
                    "title": "Ketamine and Novel Rapid-Acting Antidepressants",
                    "journal": "American Journal of Psychiatry",
                    "year": 2023,
                    "summary": "This review examines the emerging field of rapid-acting antidepressants, finding that ketamine and related compounds can provide significant symptom relief within hours for treatment-resistant depression.",
                    "authors": "Lopez C, Stevens R, et al."
                },
                {
                    "title": "Exercise as an Adjunct Treatment for Depression",
                    "journal": "JAMA Psychiatry",
                    "year": 2022,
                    "summary": "This meta-analysis demonstrates that structured exercise programs significantly enhance the effectiveness of standard depression treatments, with greatest benefits seen from moderate-intensity aerobic exercise.",
                    "authors": "Taylor P, Morris N, et al."
                }
            ],
            "ibs": [
                {
                    "title": "Gut Microbiome Alterations in Irritable Bowel Syndrome",
                    "journal": "Gastroenterology",
                    "year": 2023,
                    "summary": "This study characterizes distinctive microbiome signatures in IBS subtypes, identifying specific bacterial populations that correlate with symptom severity and suggesting targeted probiotic approaches.",
                    "authors": "Chang L, Rodriguez K, et al."
                },
                {
                    "title": "Efficacy of the Low FODMAP Diet for IBS Management",
                    "journal": "American Journal of Gastroenterology",
                    "year": 2022,
                    "summary": "This controlled trial confirms that a properly implemented low FODMAP diet reduces symptoms in 76% of IBS patients, with greatest improvements seen in those with diarrhea-predominant IBS.",
                    "authors": "Mitchell S, Kumar V, et al."
                }
            ]
        }

    # ======================== DATA ACCESS METHODS ========================
    # These methods mimic the HealthDataManager functionality but operate on internal data

    def get_all_conditions(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all available health conditions.
        Each condition is a dictionary with at least an 'id' and 'name' field.

        Returns:
            List[Dict]: List of condition dictionaries
        """
        try:
            # Return a list of essential condition information
            return [{"id": cond_id, "name": cond_data["name"], "category": cond_data.get("category", "Uncategorized")}
                    for cond_id, cond_data in self._conditions.items()]
        except Exception as e:
            logger.error(f"Error retrieving conditions: {e}")
            return []

    def get_condition_info(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific health condition.

        Args:
            condition_id (str): The ID of the condition to retrieve

        Returns:
            Dict: Condition details including name, description, symptoms, etc.
        """
        try:
            return self._conditions.get(condition_id)
        except Exception as e:
            logger.error(f"Error retrieving condition info for {condition_id}: {e}")
            return None

    def get_symptom_info(self, symptom_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific symptom.

        Args:
            symptom_id (str): The ID of the symptom to retrieve

        Returns:
            Dict: Symptom details including name and description
        """
        try:
            return self._symptoms.get(symptom_id)
        except Exception as e:
            logger.error(f"Error retrieving symptom info for {symptom_id}: {e}")
            return None

    def get_medical_literature(self, condition_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve medical literature related to a specific condition.

        Args:
            condition_id (str): The ID of the condition to retrieve literature for

        Returns:
            List[Dict]: List of literature references
        """
        try:
            return self._literature.get(condition_id, [])
        except Exception as e:
            logger.error(f"Error retrieving medical literature for {condition_id}: {e}")
            return []

    def find_conditions_by_symptom(self, symptom_id: str) -> List[Dict[str, Any]]:
        """
        Find conditions related to a specific symptom.

        Args:
            symptom_id (str): The ID of the symptom to search for

        Returns:
            List[Dict]: List of conditions related to the symptom
        """
        try:
            related_conditions = []

            # Check each condition for the symptom
            for cond_id, condition in self._conditions.items():
                if "symptoms" in condition and symptom_id in condition["symptoms"]:
                    related_conditions.append({
                        "id": cond_id,
                        "name": condition["name"],
                        "category": condition.get("category", "Uncategorized")
                    })

            return related_conditions
        except Exception as e:
            logger.error(f"Error finding conditions for symptom {symptom_id}: {e}")
            return []

    def search_medical_database(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for medical information related to the query.

        Args:
            query (str): The search query string

        Returns:
            Dict: Dictionary with categories of search results
        """
        try:
            query = query.lower()
            results = {
                "conditions": [],
                "symptoms": [],
                "literature": []
            }

            # Search conditions
            for cond_id, condition in self._conditions.items():
                if (query in cond_id.lower() or
                    query in condition.get("name", "").lower() or
                    query in condition.get("description", "").lower()):
                    results["conditions"].append({
                        "id": cond_id,
                        "name": condition["name"],
                        "description": condition.get("description", ""),
                        "category": condition.get("category", "Uncategorized")
                    })

            # Search symptoms
            for symptom_id, symptom in self._symptoms.items():
                if (query in symptom_id.lower() or
                    query in symptom.get("name", "").lower() or
                    query in symptom.get("description", "").lower()):
                    results["symptoms"].append(symptom)

            # Search literature
            for condition_id, literature_list in self._literature.items():
                for article in literature_list:
                    if (query in article.get("title", "").lower() or
                        query in article.get("summary", "").lower() or
                        query in article.get("authors", "").lower() or
                        query in article.get("journal", "").lower()):
                        # Add the condition name to the article
                        article_copy = article.copy()
                        condition = self._conditions.get(condition_id, {})
                        article_copy["condition_name"] = condition.get("name", "Unknown")
                        article_copy["condition_id"] = condition_id
                        results["literature"].append(article_copy)

            return results
        except Exception as e:
            logger.error(f"Error searching medical database for '{query}': {e}")
            return {"conditions": [], "symptoms": [], "literature": []}

    def get_latest_literature(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent medical literature across all conditions.

        Args:
            limit (int): Maximum number of articles to return

        Returns:
            List[Dict]: List of recent literature articles
        """
        try:
            all_articles = []

            # Collect articles from all conditions
            for condition_id, articles in self._literature.items():
                for article in articles:
                    # Add condition information to each article
                    article_copy = article.copy()
                    condition = self._conditions.get(condition_id, {})
                    article_copy["condition_name"] = condition.get("name", "Unknown")
                    article_copy["condition_id"] = condition_id
                    all_articles.append(article_copy)

            # Sort by year (descending) and limit results
            sorted_articles = sorted(all_articles, key=lambda x: x.get("year", 0), reverse=True)
            return sorted_articles[:limit]
        except Exception as e:
            logger.error(f"Error getting latest literature: {e}")
            return []

    def get_related_conditions(self, condition_id: str) -> List[Dict[str, Any]]:
        """
        Find conditions that may be related to the specified condition.

        Args:
            condition_id (str): The condition ID to find related conditions for

        Returns:
            List[Dict]: List of related conditions
        """
        try:
            condition = self._conditions.get(condition_id)
            if not condition or "symptoms" not in condition:
                return []

            # Get symptoms of the target condition
            target_symptoms = set(condition.get("symptoms", []))
            if not target_symptoms:
                return []

            # Find conditions that share symptoms
            related = []
            for cond_id, cond_data in self._conditions.items():
                # Skip the target condition itself
                if cond_id == condition_id:
                    continue

                # Get symptoms of this condition
                cond_symptoms = set(cond_data.get("symptoms", []))

                # Calculate overlap
                if cond_symptoms:
                    common_symptoms = target_symptoms.intersection(cond_symptoms)
                    if common_symptoms:
                        similarity = len(common_symptoms) / len(target_symptoms.union(cond_symptoms))

                        # Only include if there's meaningful similarity
                        if similarity > 0.2:  # Arbitrary threshold
                            related.append({
                                "id": cond_id,
                                "name": cond_data["name"],
                                "similarity": similarity,
                                "common_symptoms": len(common_symptoms)
                            })

            # Sort by similarity (descending)
            return sorted(related, key=lambda x: x["similarity"], reverse=True)
        except Exception as e:
            logger.error(f"Error finding related conditions for {condition_id}: {e}")
            return []

    # ======================== UI RENDERING METHODS ========================
    # These methods handle the UI display with enterprise styling

    def render(self) -> None:
        """Render the medical literature interface with enterprise styling."""
        st.title("Medical Literature")
        st.markdown("Explore summaries of medical research relevant to various health conditions.")

        # Create tabs for different categories
        tabs = st.tabs(["Common Conditions", "Recent Research", "Search", "Personalized Recommendations"])

        with tabs[0]:
            self._render_common_conditions()

        with tabs[1]:
            self._render_recent_research()

        with tabs[2]:
            self._render_search_interface()

        with tabs[3]:
            self._render_personalized_recommendations()

    def _render_common_conditions(self) -> None:
        """
        Render the common conditions tab with improved error handling and UI.
        Fixes the ButtonMixin.button() error by ensuring proper parameters are used.
        """
        st.subheader("Common Health Conditions")

        try:
            # Get all conditions
            conditions = self.get_all_conditions()

            if not conditions:
                st.info("No condition information available.")
                return

            # Group conditions by category
            categories = {}
            for condition in conditions:
                category = condition.get("category", "Uncategorized")
                if category not in categories:
                    categories[category] = []
                categories[category].append(condition)

            # Let user filter by category if there are categories
            if len(categories) > 1:
                category_options = ["All Categories"] + sorted(categories.keys())
                selected_category = st.selectbox("Filter by category:", category_options)

                if selected_category != "All Categories":
                    filtered_conditions = categories[selected_category]
                else:
                    filtered_conditions = conditions
            else:
                filtered_conditions = conditions

            # Create a grid layout for conditions
            col_count = 3  # Number of columns

            # Calculate number of rows needed
            row_count = (len(filtered_conditions) + col_count - 1) // col_count

            # Create grid of condition cards
            for row in range(row_count):
                cols = st.columns(col_count)

                for col in range(col_count):
                    condition_idx = row * col_count + col

                    if condition_idx < len(filtered_conditions):
                        condition = filtered_conditions[condition_idx]
                        condition_id = condition["id"]

                        # Get full condition data
                        condition_data = self.get_condition_info(condition_id)

                        if condition_data:
                            with cols[col]:
                                # Create card with more engaging enterprise design
                                category = condition_data.get("category", "General")
                                st.markdown(f"""
                                <div class="condition-card">
                                    <h4>{condition_data["name"]}</h4>
                                    <p>{condition_data["description"][:120]}...</p>
                                    <div class="category-badge {category}">
                                        {category}
                                    </div>
                                    <button class="details-button" onclick="document.getElementById('view_{condition_id}').click();">
                                        View details
                                    </button>
                                </div>
                                """, unsafe_allow_html=True)

                                # FIXED: Button with correct parameters - removed label_visibility
                                if st.button("View details", key=f"view_{condition_id}"):
                                    self._display_condition_details(condition_id, condition_data)

            # If no conditions after filtering
            if not filtered_conditions:
                st.info(f"No conditions found in the {selected_category} category.")
        except Exception as e:
            logger.error(f"Error rendering common conditions: {e}", exc_info=True)
            st.error(f"Error loading condition information: {str(e)}")

    def _display_condition_details(self, condition_id, condition_data) -> None:
        """
        Display detailed information about a condition with enterprise-level UI.

        Args:
            condition_id: ID of the condition to display
            condition_data: Condition data dictionary
        """
        # Set the session state to show we're viewing this condition
        st.session_state.viewing_condition = condition_id

                        # Create a back button at the top with improved enterprise styling
        back_button_html = """
        <button class="back-button" onclick="document.getElementById('back_button').click();">
            Back to conditions list
        </button>
        """
        st.markdown(back_button_html, unsafe_allow_html=True)

        # Hidden button for navigation - FIXED: Removed label_visibility parameter
        if st.button("Back to list", key="back_button"):
            # Clear the viewing state
            if 'viewing_condition' in st.session_state:
                del st.session_state.viewing_condition
            st.rerun()

        # Condition title and basic info
        st.markdown(f"## {condition_data['name']}")

        # Display category with a badge
        category = condition_data.get('category', 'General')
        category_badge = f"""
        <div class="category-badge {category}">
            {category}
        </div>
        """
        st.markdown(category_badge, unsafe_allow_html=True)

        # Main description
        st.markdown(f"### Description")
        st.markdown(condition_data['description'])

        # Create tabs for different aspects of the condition
        detail_tabs = st.tabs(["Overview", "Symptoms", "Treatment", "Medical Literature"])

        with detail_tabs[0]:
            # Overview information in enterprise styling
            st.markdown("""
            <div class="detail-container">
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Duration")
                st.markdown(condition_data.get('typical_duration', 'Information not available'))

                st.markdown("#### When to See a Doctor")
                st.markdown(condition_data.get('when_to_see_doctor', 'Consult with a healthcare professional if you experience symptoms of this condition.'))

            with col2:
                # Try to find related conditions
                try:
                    related_conditions = self.get_related_conditions(condition_id)
                    if related_conditions:
                        st.markdown("#### Related Conditions")
                        for related in related_conditions[:3]:  # Show top 3
                            st.markdown(f"• **{related['name']}** ({related['common_symptoms']} common symptoms)")
                except Exception as e:
                    # If method fails, just skip this section
                    logger.error(f"Error getting related conditions: {e}")
                    pass

            st.markdown("""
            </div>
            """, unsafe_allow_html=True)

        with detail_tabs[1]:
            # Symptoms section with enterprise styling
            st.markdown("### Common Symptoms")

            if "symptoms" in condition_data:
                symptom_list = condition_data["symptoms"]

                # Create a grid layout for symptoms
                symptom_cols = st.columns(2)

                for i, symptom_id in enumerate(symptom_list):
                    symptom_info = self.get_symptom_info(symptom_id)
                    if symptom_info:
                        col_idx = i % 2
                        with symptom_cols[col_idx]:
                            st.markdown(f"""
                            <div class="symptom-item">
                                <strong>{symptom_info['name']}</strong><br>
                                <span>{symptom_info['description']}</span>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No symptom information available for this condition.")

        with detail_tabs[2]:
            # Treatment information with enterprise styling
            st.markdown("### Treatment Approaches")

            if "treatment" in condition_data:
                st.markdown(f"""
                <div class="detail-container">
                    {condition_data["treatment"]}

                    <div style="margin-top: 15px; background-color: rgba(52, 152, 219, 0.1); border-left: 3px solid #3498db; padding: 12px; border-radius: 0 4px 4px 0;">
                        <strong>Note:</strong> Treatment approaches should be discussed with qualified healthcare providers. This information is for educational purposes only.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No treatment information available for this condition.")

        with detail_tabs[3]:
            # Medical literature related to the condition with enterprise styling
            st.markdown("### Related Medical Literature")

            literature = self.get_medical_literature(condition_id)
            if literature:
                for i, article in enumerate(literature):
                    # Create an appealing card for each article
                    st.markdown(f"""
                    <div class="literature-item">
                        <h4>{article["title"]}</h4>
                        <div class="meta">
                            <strong>{article["journal"]}</strong> ({article["year"]})
                            {f'• {article["authors"]}' if "authors" in article else ''}
                        </div>
                        <div class="summary">
                            <strong>Summary:</strong> {article["summary"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific medical literature available for this condition in our database.")

    def _render_recent_research(self) -> None:
        """
        Render the recent research tab with enterprise-level UI.
        """
        st.subheader("Recent Medical Research")
        st.markdown("This section displays summaries of recent medical research papers.")

        try:
            # Get recent research papers
            recent_papers = self.get_latest_literature(limit=8)

            # Filter options for research
            st.markdown("### Filter Research Papers")

            col1, col2 = st.columns(2)

            with col1:
                # Get unique years
                years = sorted(set(paper.get("year", 0) for paper in recent_papers), reverse=True)
                selected_year = st.selectbox("Publication Year", ["All Years"] + [str(y) for y in years])

            with col2:
                # Get unique journals
                journals = sorted(set(paper.get("journal", "") for paper in recent_papers))
                selected_journal = st.selectbox("Journal", ["All Journals"] + journals)

            # Filter papers based on selections
            filtered_papers = recent_papers

            if selected_year != "All Years":
                year_int = int(selected_year)
                filtered_papers = [p for p in filtered_papers if p.get("year") == year_int]

            if selected_journal != "All Journals":
                filtered_papers = [p for p in filtered_papers if p.get("journal") == selected_journal]

            # Display count
            st.markdown(f"### Showing {len(filtered_papers)} Research Papers")

            if not filtered_papers:
                st.info("No research papers match your filter criteria.")

            # Display the papers with enterprise-level styling
            for paper in filtered_papers:
                # Create an appealing card for each paper using enterprise styling
                st.markdown(f"""
                <div class="literature-item">
                    <h4>{paper["title"]}</h4>
                    <div class="meta">
                        <strong>{paper["journal"]}</strong> ({paper["year"]})
                        {f'• Related to: <span class="highlight">{paper.get("condition_name", "")}</span>' if "condition_name" in paper else ''}
                        {f'• {paper.get("authors", "")}' if "authors" in paper else ''}
                    </div>
                    <div class="summary">
                        <strong>Summary:</strong> {paper["summary"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Add "Read Details" button
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button(f"Read Details", key=f"details_{paper['title'][:20]}"):
                        # Create a more detailed view when clicked
                        st.session_state.selected_paper = paper
                        st.rerun()

            # If a paper is selected, show detailed view
            if "selected_paper" in st.session_state:
                paper = st.session_state.selected_paper

                # Create a back button with enterprise styling
                back_button_html = """
                <button class="back-button" onclick="document.getElementById('back_to_list').click();">
                    ← Back to research list
                </button>
                """
                st.markdown(back_button_html, unsafe_allow_html=True)

                # Hidden button for navigation - FIXED: Removed label_visibility parameter
                if st.button("Back to list", key="back_to_list"):
                    del st.session_state.selected_paper
                    st.rerun()

                # Display paper details with enterprise styling
                st.markdown(f"""
                <div class="detail-container">
                    <h3>{paper['title']}</h3>
                    <div class="meta" style="margin-bottom: 20px;">
                        <strong>Journal:</strong> {paper['journal']} ({paper['year']})<br>
                        {f'<strong>Authors:</strong> {paper["authors"]}<br>' if "authors" in paper else ''}
                        {f'<strong>Related Condition:</strong> <span class="highlight">{paper["condition_name"]}</span><br>' if "condition_name" in paper else ''}
                    </div>

                    <h4>Summary</h4>
                    <p>{paper['summary']}</p>

                    <div style="margin-top: 20px; background-color: rgba(52, 152, 219, 0.1); border-left: 3px solid #3498db; padding: 12px; border-radius: 0 4px 4px 0;">
                        <strong>Note:</strong> In a production version, this would connect to a medical literature database API to retrieve the full article details and provide access to the complete paper.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            logger.error(f"Error rendering recent research: {e}", exc_info=True)
            st.error(f"Error loading recent research: {str(e)}")

            # Fallback to simple display
            st.info("Recent research papers are currently unavailable. Please try again later.")

    def _render_search_interface(self) -> None:
        """
        Render the search interface tab with enterprise-level UI.
        """
        st.subheader("Search Medical Literature")

        # Create a clean search interface with enterprise styling
        st.markdown("""
        <div class="detail-container">
        """, unsafe_allow_html=True)

        search_query = st.text_input("Enter keywords to search:", placeholder="e.g., migraine treatment, covid vaccine")

        advanced_options = st.expander("Advanced Search Options")
        with advanced_options:
            col1, col2 = st.columns(2)
            with col1:
                # Year range with dynamic min/max
                current_year = 2023  # This would be dynamic in production
                year_range = st.slider("Publication Year", 2000, current_year, (2018, current_year))

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

        # Search button to initiate search
        search_col1, search_col2 = st.columns([4, 1])

        with search_col2:
            search_button = st.button("Search", key="search_literature", type="primary", use_container_width=True)

        # Show search tips
        with st.expander("Search Tips"):
            st.markdown("""
            - Use specific terms for more targeted results (e.g., "migraine botox" instead of just "headache")
            - Include both medical terms and common names (e.g., "hypertension high blood pressure")
            - For treatment searches, include the condition name (e.g., "asthma inhaler therapy")
            - Use quotation marks for exact phrases (e.g., "long covid symptoms")
            """)

        st.markdown("""
        </div>
        """, unsafe_allow_html=True)

        if search_button or search_query:
            self._process_literature_search(search_query, year_range, study_types, journals, sort_by)

    def _process_literature_search(self, search_query, year_range, study_types, journals, sort_by) -> None:
        """
        Process a medical literature search query with enterprise-level UI.

        Args:
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

            # Show a progress indication
            with st.spinner("Searching medical databases..."):
                # Search medical database
                search_results = self.search_medical_database(search_query)

                # Apply filters
                if "literature" in search_results:
                    # Filter by year
                    filtered_literature = [
                        article for article in search_results["literature"]
                        if year_range[0] <= article.get("year", 0) <= year_range[1]
                    ]

                    # Apply journal filter if not "All Journals"
                    if "All Journals" not in journals:
                        filtered_literature = [
                            article for article in filtered_literature
                            if article.get("journal", "") in journals
                        ]

                    # Sort results
                    if sort_by == "Publication Date (Newest First)":
                        filtered_literature = sorted(filtered_literature, key=lambda x: x.get("year", 0), reverse=True)

                    # Update the results
                    search_results["literature"] = filtered_literature

                self._display_search_results(search_results, search_query)
        except Exception as e:
            logger.error(f"Error processing literature search: {e}", exc_info=True)
            st.error(f"Error processing search: {str(e)}")

            # Provide helpful guidance
            st.markdown("""
            ### Troubleshooting Search
            - Try simplifying your search query
            - Check for spelling errors
            - Use more general terms
            - Try searching for a specific condition name
            """)

    def _display_search_results(self, search_results, query):
        """
        Display search results with enterprise-level UI.

        Args:
            search_results: Dictionary containing search results by category
            query: The search query used
        """
        # Check if we have any results
        has_results = False
        for category, results in search_results.items():
            if results:
                has_results = True
                break

        if not has_results:
            st.info(f"No results found for '{query}'. Try different keywords.")
            return

        st.markdown(f"### Search Results for '{query}'")

        # Display literature results with enterprise styling
        if "literature" in search_results and search_results["literature"]:
            st.markdown(f"#### Medical Literature ({len(search_results['literature'])} results)")

            for article in search_results["literature"]:
                # Create an appealing card for each article
                st.markdown(f"""
                <div class="literature-item">
                    <h4>{article["title"]}</h4>
                    <div class="meta">
                        <strong>{article["journal"]}</strong> ({article["year"]})
                        {f'• {article["authors"]}' if "authors" in article else ''}
                        {f'• Related to: <span class="highlight">{article.get("condition_name", "")}</span>' if "condition_name" in article else ''}
                    </div>
                    <div class="summary">
                        <strong>Summary:</strong> {article["summary"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Display condition results with enterprise styling
        if "conditions" in search_results and search_results["conditions"]:
            st.markdown(f"#### Related Conditions ({len(search_results['conditions'])} results)")

            # Create a grid layout for conditions
            cols = st.columns(2)

            for i, condition in enumerate(search_results["conditions"]):
                col_idx = i % 2

                with cols[col_idx]:
                    # Create enterprise-styled card for each condition
                    category = condition.get("category", "General")
                    st.markdown(f"""
                    <div class="condition-card">
                        <h4>{condition["name"]}</h4>
                        <p>{condition["description"][:150]}...</p>
                        <div class="category-badge {category}">
                            {category}
                        </div>
                        <button class="details-button" onclick="document.getElementById('view_cond_{condition['id']}').click();">
                            View details
                        </button>
                    </div>
                    """, unsafe_allow_html=True)

                    # Hidden button for navigation - FIXED: Removed label_visibility parameter
                    if st.button(f"View details", key=f"view_cond_{condition['id']}"):
                        # Redirect to condition details
                        st.session_state.viewing_condition = condition["id"]
                        st.rerun()

        # Display symptom results with enterprise styling
        if "symptoms" in search_results and search_results["symptoms"]:
            st.markdown(f"#### Related Symptoms ({len(search_results['symptoms'])} results)")

            # Create a grid layout for symptoms
            cols = st.columns(2)

            for i, symptom in enumerate(search_results["symptoms"]):
                col_idx = i % 2

                with cols[col_idx]:
                    # Create enterprise-styled card for each symptom
                    st.markdown(f"""
                    <div class="symptom-item">
                        <strong>{symptom["name"]}</strong><br>
                        <span>{symptom["description"]}</span>
                    </div>
                    """, unsafe_allow_html=True)

    def _render_personalized_recommendations(self) -> None:
        """
        Render personalized literature recommendations with enterprise-level UI.
        """
        st.subheader("Personalized Literature Recommendations")

        st.markdown("""
        <div class="detail-container">
            This feature provides medical literature recommendations based on your health profile and symptom history.
            The AI analyzes your health data to suggest relevant research that may be of interest to you.
        </div>
        """, unsafe_allow_html=True)

        if not self.user_manager or not hasattr(self.user_manager, "health_history") or not self.user_manager.health_history:
            st.info("Add some health history data first to get personalized recommendations.")

            # Provide helpful instructions with enterprise styling
            st.markdown("""
            <div class="detail-container">
                <h3>How to Get Personalized Recommendations</h3>

                <ol>
                    <li>Use the <strong>Symptom Analyzer</strong> to record your symptoms</li>
                    <li>Complete your profile in the <strong>Settings</strong> page</li>
                    <li>Return to this page to see literature tailored to your health profile</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

            # Add a button to navigate to the Symptom Analyzer
            if st.button("Go to Symptom Analyzer", type="primary"):
                # This would navigate to the Symptom Analyzer page
                st.session_state.page = "Symptom Analyzer"
                st.rerun()

            return

        try:
            # Get the most common symptoms from history
            symptom_counts = {}
            for entry in self.user_manager.health_history:
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
                related = self.find_conditions_by_symptom(symptom)
                for condition in related:
                    related_conditions.add(condition["id"])

            if related_conditions:
                st.markdown("### Literature Based on Your Health History")

                # Show which symptoms were used for recommendations
                symptom_names = []
                for symptom_id in top_symptoms:
                    symptom_info = self.get_symptom_info(symptom_id)
                    if symptom_info:
                        symptom_names.append(symptom_info["name"])

                if symptom_names:
                    st.markdown(f"""
                    <div style="background-color: rgba(52, 152, 219, 0.1); border-left: 3px solid #3498db; padding: 12px; border-radius: 0 4px 4px 0; margin-bottom: 20px;">
                        <strong>Based on your reported symptoms:</strong> {", ".join(symptom_names)}
                    </div>
                    """, unsafe_allow_html=True)

                # Create a list for all literature
                all_literature = []
                condition_names = {}

                for condition_id in related_conditions:
                    condition_info = self.get_condition_info(condition_id)
                    if not condition_info:
                        continue

                    condition_name = condition_info["name"]
                    condition_names[condition_id] = condition_name

                    # Get literature
                    literature = self.get_medical_literature(condition_id)

                    # Add to combined list with condition info
                    for article in literature:
                        article_copy = article.copy()
                        article_copy["condition_id"] = condition_id
                        article_copy["condition_name"] = condition_name
                        all_literature.append(article_copy)

                # Sort by year (newest first)
                all_literature.sort(key=lambda x: x.get("year", 0), reverse=True)

                # Display literature grouped by condition
                if all_literature:
                    # Allow filtering by condition with enterprise styling
                    cond_options = ["All Conditions"] + list(condition_names.values())
                    selected_cond = st.selectbox("Filter by condition:", cond_options)

                    if selected_cond != "All Conditions":
                        # Get condition ID from name
                        selected_cond_id = None
                        for cid, cname in condition_names.items():
                            if cname == selected_cond:
                                selected_cond_id = cid
                                break

                        # Filter literature
                        if selected_cond_id:
                            filtered_literature = [a for a in all_literature if a["condition_id"] == selected_cond_id]
                        else:
                            filtered_literature = all_literature
                    else:
                        filtered_literature = all_literature

                    # Display results with enterprise styling
                    for article in filtered_literature:
                        # Create an appealing card for each article
                        st.markdown(f"""
                        <div class="literature-item">
                            <h4>{article["title"]}</h4>
                            <div class="meta">
                                <strong>{article["journal"]}</strong> ({article["year"]})
                                {f'• {article["authors"]}' if "authors" in article else ''}
                                • Related to: <span class="highlight">{article["condition_name"]}</span>
                            </div>
                            <div class="summary">
                                <strong>Summary:</strong> {article["summary"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No literature available for your health profile. Try adding more detailed symptom information.")
            else:
                st.info("Not enough health data to generate personalized recommendations.")

                # Suggest using the Symptom Analyzer with enterprise styling
                st.markdown("""
                <div class="detail-container">
                    <h3>To get personalized recommendations, try:</h3>
                    <ul>
                        <li>Recording more symptoms in the Symptom Analyzer</li>
                        <li>Adding more details to your health profile</li>
                        <li>Tracking your symptoms consistently over time</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {e}", exc_info=True)
            st.error(f"Error generating recommendations: {str(e)}")


# For backward compatibility with the functional approach
def render_medical_literature() -> None:
    """Render the medical literature interface using the class-based implementation."""
    literature_ui = MedicalLiteratureUI()
    literature_ui.render()
