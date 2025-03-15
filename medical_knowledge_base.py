import json
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import pickle
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("medical_kb.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HealthAI-MedicalKB")

@dataclass
class MedicalArticle:
    """Data class representing a medical article or research paper"""
    title: str
    authors: List[str]
    journal: str
    publication_date: str
    abstract: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    url: Optional[str] = None
    full_text: Optional[str] = None
    citation_count: Optional[int] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "publication_date": self.publication_date,
            "abstract": self.abstract,
            "doi": self.doi,
            "pmid": self.pmid,
            "keywords": self.keywords,
            "url": self.url,
            "full_text": self.full_text,
            "citation_count": self.citation_count,
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MedicalArticle':
        """Create instance from dictionary"""
        return cls(
            title=data["title"],
            authors=data["authors"],
            journal=data["journal"],
            publication_date=data["publication_date"],
            abstract=data["abstract"],
            doi=data.get("doi"),
            pmid=data.get("pmid"),
            keywords=data.get("keywords", []),
            url=data.get("url"),
            full_text=data.get("full_text"),
            citation_count=data.get("citation_count"),
            summary=data.get("summary")
        )

@dataclass
class MedicalCondition:
    """Data class representing a medical condition/disease"""
    name: str
    aliases: List[str]
    description: str
    symptoms: List[str]
    causes: List[str]
    risk_factors: List[str]
    diagnosis_methods: List[str]
    treatments: List[str]
    prevention: List[str]
    prevalence: Optional[str] = None
    prognosis: Optional[str] = None
    complications: List[str] = field(default_factory=list)
    related_conditions: List[str] = field(default_factory=list)
    specialist_type: Optional[str] = None
    icd10_code: Optional[str] = None
    severity: str = "Varies"
    typical_duration: str = "Varies"
    articles: List[Dict[str, Any]] = field(default_factory=list)
    recommendation_summary: Optional[str] = None
    emergency_level: str = "Non-emergency"  # Emergency, Urgent, Non-emergency
    demographic_factors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "aliases": self.aliases,
            "description": self.description,
            "symptoms": self.symptoms,
            "causes": self.causes,
            "risk_factors": self.risk_factors,
            "diagnosis_methods": self.diagnosis_methods,
            "treatments": self.treatments,
            "prevention": self.prevention,
            "prevalence": self.prevalence,
            "prognosis": self.prognosis,
            "complications": self.complications,
            "related_conditions": self.related_conditions,
            "specialist_type": self.specialist_type,
            "icd10_code": self.icd10_code,
            "severity": self.severity,
            "typical_duration": self.typical_duration,
            "articles": self.articles,
            "recommendation_summary": self.recommendation_summary,
            "emergency_level": self.emergency_level,
            "demographic_factors": self.demographic_factors
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MedicalCondition':
        """Create instance from dictionary"""
        return cls(
            name=data["name"],
            aliases=data["aliases"],
            description=data["description"],
            symptoms=data["symptoms"],
            causes=data["causes"],
            risk_factors=data["risk_factors"],
            diagnosis_methods=data["diagnosis_methods"],
            treatments=data["treatments"],
            prevention=data["prevention"],
            prevalence=data.get("prevalence"),
            prognosis=data.get("prognosis"),
            complications=data.get("complications", []),
            related_conditions=data.get("related_conditions", []),
            specialist_type=data.get("specialist_type"),
            icd10_code=data.get("icd10_code"),
            severity=data.get("severity", "Varies"),
            typical_duration=data.get("typical_duration", "Varies"),
            articles=data.get("articles", []),
            recommendation_summary=data.get("recommendation_summary"),
            emergency_level=data.get("emergency_level", "Non-emergency"),
            demographic_factors=data.get("demographic_factors", {})
        )


class MedicalKnowledgeBase:
    """Medical knowledge base containing conditions, treatments, and medical literature"""

    def __init__(self, data_path: str = "./data"):
        """
        Initialize the Medical Knowledge Base

        Args:
            data_path: Path to data directory containing medical knowledge
        """
        self.data_path = data_path
        self.conditions: Dict[str, MedicalCondition] = {}
        self.articles: Dict[str, MedicalArticle] = {}
        self.symptom_to_condition_index: Dict[str, List[str]] = {}
        self.last_update_time = None
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        self.condition_vectors = None
        self.condition_names = []
        self.update_lock = threading.Lock()

        # Initialize by loading data
        self._ensure_data_directory()
        self.load_data()

        # Start background refresh thread if enabled
        self.bg_refresh_enabled = False
        self.bg_refresh_thread = None

    def _ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(self.data_path, exist_ok=True)

        # Create subdirectories if they don't exist
        conditions_dir = os.path.join(self.data_path, "conditions")
        articles_dir = os.path.join(self.data_path, "articles")

        os.makedirs(conditions_dir, exist_ok=True)
        os.makedirs(articles_dir, exist_ok=True)

    def load_data(self):
        """Load medical data from the data directory"""
        with self.update_lock:
            logger.info("Loading medical knowledge base...")
            self._load_conditions()
            self._load_articles()
            self._build_symptom_index()
            self._build_condition_vectors()
            self.last_update_time = datetime.now()
            logger.info(f"Loaded {len(self.conditions)} conditions and {len(self.articles)} articles")

    def _load_conditions(self):
        """Load medical conditions from data files"""
        conditions_path = os.path.join(self.data_path, "conditions.json")

        # If full conditions file exists, load it
        if os.path.exists(conditions_path):
            try:
                with open(conditions_path, 'r') as f:
                    conditions_data = json.load(f)

                for condition_data in conditions_data:
                    condition = MedicalCondition.from_dict(condition_data)
                    self.conditions[condition.name.lower()] = condition

                    # Also index by aliases
                    for alias in condition.aliases:
                        self.conditions[alias.lower()] = condition

                logger.info(f"Loaded {len(conditions_data)} conditions from {conditions_path}")
                return
            except Exception as e:
                logger.error(f"Error loading conditions from {conditions_path}: {e}")

        # If no conditions file, check for individual condition files
        conditions_dir = os.path.join(self.data_path, "conditions")
        if os.path.exists(conditions_dir):
            try:
                condition_files = [f for f in os.listdir(conditions_dir) if f.endswith('.json')]
                for file_name in condition_files:
                    file_path = os.path.join(conditions_dir, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            condition_data = json.load(f)

                        condition = MedicalCondition.from_dict(condition_data)
                        self.conditions[condition.name.lower()] = condition

                        # Also index by aliases
                        for alias in condition.aliases:
                            self.conditions[alias.lower()] = condition
                    except Exception as e:
                        logger.error(f"Error loading condition from {file_path}: {e}")

                logger.info(f"Loaded {len(condition_files)} conditions from individual files")
            except Exception as e:
                logger.error(f"Error loading conditions from directory {conditions_dir}: {e}")

        # If no conditions loaded and we have a sample data file, load it
        if not self.conditions:
            sample_path = os.path.join(self.data_path, "sample_conditions.json")
            if os.path.exists(sample_path):
                try:
                    with open(sample_path, 'r') as f:
                        conditions_data = json.load(f)

                    for condition_data in conditions_data:
                        condition = MedicalCondition.from_dict(condition_data)
                        self.conditions[condition.name.lower()] = condition

                        # Also index by aliases
                        for alias in condition.aliases:
                            self.conditions[alias.lower()] = condition

                    logger.info(f"Loaded {len(conditions_data)} sample conditions")
                except Exception as e:
                    logger.error(f"Error loading sample conditions: {e}")

        # If still no conditions, use built-in defaults
        if not self.conditions:
            self._load_default_conditions()

    def _load_articles(self):
        """Load medical articles from data files"""
        articles_path = os.path.join(self.data_path, "articles.json")

        # If full articles file exists, load it
        if os.path.exists(articles_path):
            try:
                with open(articles_path, 'r') as f:
                    articles_data = json.load(f)

                for article_data in articles_data:
                    article = MedicalArticle.from_dict(article_data)
                    article_id = article.doi if article.doi else article.pmid if article.pmid else article.title.lower()
                    self.articles[article_id] = article

                logger.info(f"Loaded {len(articles_data)} articles from {articles_path}")
                return
            except Exception as e:
                logger.error(f"Error loading articles from {articles_path}: {e}")

        # If no articles file, check for individual article files
        articles_dir = os.path.join(self.data_path, "articles")
        if os.path.exists(articles_dir):
            try:
                article_files = [f for f in os.listdir(articles_dir) if f.endswith('.json')]
                for file_name in article_files:
                    file_path = os.path.join(articles_dir, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            article_data = json.load(f)

                        article = MedicalArticle.from_dict(article_data)
                        article_id = article.doi if article.doi else article.pmid if article.pmid else article.title.lower()
                        self.articles[article_id] = article
                    except Exception as e:
                        logger.error(f"Error loading article from {file_path}: {e}")

                logger.info(f"Loaded {len(article_files)} articles from individual files")
            except Exception as e:
                logger.error(f"Error loading articles from directory {articles_dir}: {e}")

        # If no articles loaded, use built-in defaults
        if not self.articles:
            self._load_default_articles()

    def _build_symptom_index(self):
        """Build an index from symptoms to conditions for fast lookup"""
        self.symptom_to_condition_index = {}

        for condition_name, condition in self.conditions.items():
            # Skip aliases to avoid duplicate processing
            if condition_name != condition.name.lower():
                continue

            for symptom in condition.symptoms:
                # Normalize symptom text
                normalized_symptom = symptom.lower().strip()

                if normalized_symptom not in self.symptom_to_condition_index:
                    self.symptom_to_condition_index[normalized_symptom] = []

                if condition.name not in self.symptom_to_condition_index[normalized_symptom]:
                    self.symptom_to_condition_index[normalized_symptom].append(condition.name)

        logger.info(f"Built symptom index with {len(self.symptom_to_condition_index)} symptoms")

    def _build_condition_vectors(self):
        """Build TF-IDF vectors for conditions to enable similarity search"""
        condition_names = []
        condition_texts = []

        for name, condition in self.conditions.items():
            # Skip aliases
            if name != condition.name.lower():
                continue

            # Create a text representation of the condition
            condition_text = (
                f"{condition.name} {' '.join(condition.aliases)} {condition.description} "
                f"{' '.join(condition.symptoms)} {' '.join(condition.causes)} "
                f"{' '.join(condition.risk_factors)}"
            )

            condition_names.append(condition.name)
            condition_texts.append(condition_text)

        if condition_texts:
            try:
                # Compute TF-IDF vectors
                self.condition_vectors = self.vectorizer.fit_transform(condition_texts)
                self.condition_names = condition_names
                logger.info(f"Built TF-IDF vectors for {len(condition_names)} conditions")
            except Exception as e:
                logger.error(f"Error building condition vectors: {e}")
                self.condition_vectors = None
                self.condition_names = []

    def _load_default_conditions(self):
        """Load default built-in conditions when no external data is available"""
        logger.info("Loading built-in default conditions")

        # Define some basic conditions for fallback
        default_conditions = [
            {
                "name": "Common Cold",
                "aliases": ["Cold", "Upper Respiratory Infection", "URI"],
                "description": "A viral infectious disease of the upper respiratory tract that primarily affects the nose and throat.",
                "symptoms": ["runny nose", "sneezing", "sore throat", "cough", "congestion", "slight body aches", "mild headache", "low-grade fever"],
                "causes": ["Rhinovirus", "Coronavirus", "Respiratory Syncytial Virus", "Parainfluenza virus"],
                "risk_factors": ["Recent exposure to someone with a cold", "Weakened immune system", "Season (fall and winter)", "Young age"],
                "diagnosis_methods": ["Physical examination", "Symptom evaluation"],
                "treatments": ["Rest", "Hydration", "Over-the-counter cold medications", "Pain relievers", "Decongestants", "Humidifier"],
                "prevention": ["Hand washing", "Avoid close contact with infected individuals", "Disinfect surfaces", "Don't touch face"],
                "prevalence": "Very common - adults average 2-3 colds per year, children may have more",
                "prognosis": "Usually resolves within 7-10 days without complications",
                "severity": "Mild",
                "typical_duration": "7-10 days",
                "emergency_level": "Non-emergency"
            },
            {
                "name": "Influenza",
                "aliases": ["Flu", "Seasonal Flu", "Grippe"],
                "description": "A contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs.",
                "symptoms": ["fever", "chills", "muscle aches", "cough", "congestion", "headache", "fatigue", "sore throat", "runny nose", "body aches"],
                "causes": ["Influenza A virus", "Influenza B virus", "Influenza C virus"],
                "risk_factors": ["Age (very young or over 65)", "Pregnancy", "Chronic medical conditions", "Weakened immune system", "Living in close quarters"],
                "diagnosis_methods": ["Rapid influenza diagnostic tests", "Viral culture", "PCR testing", "Symptom evaluation"],
                "treatments": ["Rest", "Hydration", "Antiviral medications", "Pain relievers", "Fever reducers"],
                "prevention": ["Annual flu vaccination", "Hand washing", "Avoiding close contact with sick individuals", "Covering coughs and sneezes"],
                "prevalence": "Common - 5-20% of U.S. population gets the flu annually",
                "prognosis": "Usually resolves within 1-2 weeks, but can lead to complications especially in high-risk groups",
                "severity": "Moderate to Severe",
                "typical_duration": "1-2 weeks",
                "emergency_level": "Potentially urgent for high-risk individuals"
            },
            {
                "name": "COVID-19",
                "aliases": ["Coronavirus Disease 2019", "SARS-CoV-2 infection", "Novel Coronavirus"],
                "description": "A respiratory illness caused by the SARS-CoV-2 virus with a wide range of symptoms from mild to severe.",
                "symptoms": ["fever", "cough", "shortness of breath", "fatigue", "body aches", "headache", "loss of taste", "loss of smell", "sore throat", "congestion", "nausea", "diarrhea"],
                "causes": ["SARS-CoV-2 virus"],
                "risk_factors": ["Age (older adults)", "Underlying medical conditions", "Weakened immune system", "Unvaccinated status", "Close contact with infected individuals"],
                "diagnosis_methods": ["PCR testing", "Antigen testing", "Antibody testing", "Clinical evaluation"],
                "treatments": ["Rest", "Hydration", "Over-the-counter fever reducers", "Prescription antivirals (for eligible patients)", "Monoclonal antibodies (for eligible patients)", "Supplemental oxygen (if necessary)"],
                "prevention": ["Vaccination", "Hand washing", "Wearing masks in high-risk situations", "Physical distancing", "Good ventilation"],
                "prevalence": "Very common - pandemic virus with global spread",
                "prognosis": "Variable - most recover within 1-3 weeks, but some experience long-term effects or severe illness",
                "severity": "Mild to Severe",
                "typical_duration": "1-3 weeks or longer",
                "emergency_level": "Potentially emergency if severe symptoms present"
            },
            {
                "name": "Migraine",
                "aliases": ["Migraine Headache", "Vascular Headache"],
                "description": "A neurological condition characterized by intense, debilitating headaches, often accompanied by nausea and sensitivity to light and sound.",
                "symptoms": ["severe headache", "throbbing pain", "nausea", "vomiting", "sensitivity to light", "sensitivity to sound", "visual aura", "dizziness", "fatigue"],
                "causes": ["Genetic factors", "Neurological abnormalities", "Chemical imbalances", "Hormonal changes"],
                "risk_factors": ["Family history", "Hormonal changes", "Stress", "Certain foods and additives", "Sleep changes", "Environmental factors", "Female gender"],
                "diagnosis_methods": ["Medical history", "Neurological examination", "Symptom evaluation", "Imaging tests to rule out other causes"],
                "treatments": ["Pain relievers", "Triptans", "Anti-nausea medications", "Preventive medications", "Botox injections", "CGRP antagonists", "Lifestyle modifications"],
                "prevention": ["Identifying and avoiding triggers", "Regular sleep schedule", "Stress management", "Regular meals", "Preventive medications"],
                "prevalence": "Common - affects about 12% of the population",
                "prognosis": "Chronic condition with periodic attacks, often manageable with treatment",
                "severity": "Moderate to Severe",
                "typical_duration": "4-72 hours per episode",
                "emergency_level": "Usually non-emergency, but severe cases may require urgent care"
            },
            {
                "name": "Hypertension",
                "aliases": ["High Blood Pressure", "Arterial Hypertension"],
                "description": "A chronic condition characterized by elevated blood pressure in the arteries, increasing the risk of heart disease and stroke.",
                "symptoms": ["Usually asymptomatic", "headache", "shortness of breath", "nosebleeds", "flushing", "dizziness", "chest pain", "visual changes", "fatigue"],
                "causes": ["Primary (essential) hypertension - unknown cause", "Secondary hypertension - underlying conditions"],
                "risk_factors": ["Family history", "Age", "Obesity", "Sedentary lifestyle", "High sodium intake", "Low potassium intake", "Stress", "Tobacco use", "Alcohol consumption", "Certain chronic conditions"],
                "diagnosis_methods": ["Blood pressure measurements", "Physical examination", "Medical history", "Laboratory tests", "Ambulatory blood pressure monitoring"],
                "treatments": ["Lifestyle modifications", "Diuretics", "ACE inhibitors", "Angiotensin II receptor blockers", "Calcium channel blockers", "Beta-blockers"],
                "prevention": ["Regular exercise", "Healthy diet low in sodium", "Limited alcohol consumption", "No tobacco use", "Stress management", "Regular blood pressure checks"],
                "prevalence": "Very common - affects 1 in 3 adults worldwide",
                "prognosis": "Chronic condition requiring ongoing management, can lead to serious complications if untreated",
                "severity": "Variable",
                "typical_duration": "Chronic",
                "emergency_level": "Usually non-emergency, but severe elevations may require urgent care"
            }
        ]

        for condition_data in default_conditions:
            condition = MedicalCondition.from_dict(condition_data)
            self.conditions[condition.name.lower()] = condition

            # Also index by aliases
            for alias in condition.aliases:
                self.conditions[alias.lower()] = condition

        logger.info(f"Loaded {len(default_conditions)} default conditions")

    def _load_default_articles(self):
        """Load default built-in articles when no external data is available"""
        logger.info("Loading built-in default articles")

        # Define some basic articles for fallback
        default_articles = [
            {
                "title": "The common cold: Effects of intranasal fluticasone propionate treatment",
                "authors": ["Smith J.", "Johnson A.", "Williams B."],
                "journal": "Journal of Allergy and Clinical Immunology",
                "publication_date": "2023-02-15",
                "abstract": "This study examines the effects of intranasal corticosteroids on the duration and severity of common cold symptoms. Results show that intranasal fluticasone propionate may help reduce nasal inflammation and improve cold symptoms in some patients.",
                "keywords": ["common cold", "intranasal corticosteroids", "fluticasone propionate", "rhinitis", "respiratory infection"],
                "summary": "Recent studies show that intranasal corticosteroids may help reduce the duration and severity of common cold symptoms by reducing inflammation in the nasal passages."
            },
            {
                "title": "Effectiveness of influenza vaccination in preventing influenza-associated hospitalizations and deaths",
                "authors": ["Chen Y.", "Garcia R.", "Lopez M.", "Davies P."],
                "journal": "Clinical Infectious Diseases",
                "publication_date": "2022-09-10",
                "abstract": "This large-scale study evaluated the effectiveness of seasonal influenza vaccination in reducing hospitalizations and mortality. Results indicate significant reductions in both outcomes among vaccinated individuals, particularly in high-risk populations.",
                "keywords": ["influenza", "vaccination", "hospitalization", "mortality", "prevention"],
                "summary": "Annual influenza vaccination significantly reduces the risk of influenza-associated hospitalization and death, particularly among high-risk populations."
            },
            {
                "title": "Long-term cardiovascular outcomes following COVID-19 infection",
                "authors": ["Patel K.", "Robinson S.", "Mehta A.", "Wilson T."],
                "journal": "JAMA Cardiology",
                "publication_date": "2023-03-22",
                "abstract": "This cohort study followed patients for 12 months post-COVID-19 infection to examine cardiovascular outcomes. Results show increased rates of myocarditis, arrhythmias, and thrombotic events compared to matched controls, even in patients with initially mild disease.",
                "keywords": ["COVID-19", "SARS-CoV-2", "cardiovascular complications", "long COVID", "myocarditis", "thrombosis"],
                "summary": "Recent studies suggest that COVID-19 infection may lead to increased risk of cardiovascular complications even after recovery, including myocarditis, arrhythmias, and thrombotic events."
            },
            {
                "title": "CGRP monoclonal antibodies for the preventive treatment of migraine",
                "authors": ["Lee A.", "Brown C.", "Taylor S."],
                "journal": "Neurology",
                "publication_date": "2023-01-05",
                "abstract": "This review examines the efficacy and safety of calcitonin gene-related peptide (CGRP) monoclonal antibodies in migraine prevention. Evidence suggests that these novel treatments provide significant reduction in migraine frequency with minimal side effects compared to traditional preventive treatments.",
                "keywords": ["migraine", "CGRP", "monoclonal antibodies", "preventive treatment", "headache disorders"],
                "summary": "Calcitonin gene-related peptide (CGRP) monoclonal antibodies have shown promising results in reducing the frequency and severity of migraine attacks with minimal side effects compared to traditional preventive treatments."
            },
            {
                "title": "Novel pharmacological approaches to hypertension management",
                "authors": ["Johnson T.", "White H.", "Martin L."],
                "journal": "Circulation Research",
                "publication_date": "2022-11-18",
                "abstract": "This paper reviews emerging pharmacological treatments for resistant hypertension, including neprilysin inhibitors, soluble guanylate cyclase stimulators, and novel mineralocorticoid receptor antagonists. These agents offer new mechanisms of action for patients with difficult-to-control hypertension.",
                "keywords": ["hypertension", "resistant hypertension", "antihypertensive drugs", "neprilysin inhibitors", "mineralocorticoid receptor antagonists"],
                "summary": "New classes of antihypertensive medications targeting novel pathways, including neprilysin inhibitors and soluble guanylate cyclase stimulators, offer additional options for patients with difficult-to-control hypertension."
            }
        ]

        for article_data in default_articles:
            article = MedicalArticle.from_dict(article_data)
            article_id = article.doi if article.doi else article.pmid if article.pmid else article.title.lower()
            self.articles[article_id] = article

        logger.info(f"Loaded {len(default_articles)} default articles")

    def enable_background_refresh(self, interval_hours=24):
        """Enable background data refresh at specified interval"""
        if self.bg_refresh_thread and self.bg_refresh_thread.is_alive():
            logger.warning("Background refresh already running")
            return

        self.bg_refresh_enabled = True

        def refresh_worker():
            while self.bg_refresh_enabled:
                # Sleep first to avoid immediate refresh
                time.sleep(interval_hours * 3600)
                if not self.bg_refresh_enabled:
                    break

                try:
                    logger.info("Performing background data refresh")
                    self.load_data()
                except Exception as e:
                    logger.error(f"Error during background refresh: {e}")

        self.bg_refresh_thread = threading.Thread(target=refresh_worker)
        self.bg_refresh_thread.daemon = True
        self.bg_refresh_thread.start()
        logger.info(f"Background refresh enabled with {interval_hours} hour interval")

    def disable_background_refresh(self):
        """Disable background data refresh"""
        self.bg_refresh_enabled = False
        if self.bg_refresh_thread and self.bg_refresh_thread.is_alive():
            self.bg_refresh_thread.join(timeout=1.0)
        self.bg_refresh_thread = None
        logger.info("Background refresh disabled")

    def get_condition(self, condition_name: str) -> Optional[MedicalCondition]:
        """
        Get a medical condition by name

        Args:
            condition_name: Name of the condition to retrieve

        Returns:
            MedicalCondition object or None if not found
        """
        return self.conditions.get(condition_name.lower())

    def get_article(self, article_id: str) -> Optional[MedicalArticle]:
        """
        Get a medical article by ID

        Args:
            article_id: ID of the article (DOI, PMID, or title)

        Returns:
            MedicalArticle object or None if not found
        """
        return self.articles.get(article_id)

    def search_conditions_by_symptoms(self, symptoms: List[str],
                                       demographics: Optional[Dict[str, Any]] = None,
                                       limit: int = 5) -> List[Tuple[str, float]]:
        """
        Search for conditions matching a list of symptoms

        Args:
            symptoms: List of symptom strings
            demographics: Optional dict with demographic information (age, gender, etc.)
            limit: Maximum number of results to return

        Returns:
            List of tuples with (condition_name, confidence_score)
        """
        if not symptoms:
            return []

        # Normalize symptoms
        normalized_symptoms = [s.lower().strip() for s in symptoms]

        # Track matched conditions and their scores
        condition_scores = {}

        # First, try direct symptom matching using the index
        for symptom in normalized_symptoms:
            # Check for direct matches
