import spacy
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nlp_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HealthAI-NLP")

@dataclass
class ExtractedSymptom:
    """Data class for extracted symptoms with metadata"""
    text: str
    confidence: float
    severity: Optional[float] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    source_text: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "severity": self.severity,
            "duration": self.duration,
            "location": self.location,
            "modifiers": self.modifiers,
            "source_text": self.source_text,
            "span_start": self.span_start,
            "span_end": self.span_end
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedSymptom':
        """Create instance from dictionary"""
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            severity=data.get("severity"),
            duration=data.get("duration"),
            location=data.get("location"),
            modifiers=data.get("modifiers", []),
            source_text=data.get("source_text"),
            span_start=data.get("span_start"),
            span_end=data.get("span_end")
        )


class MedicalNLPEngine:
    """Advanced NLP engine for processing medical text and extracting symptoms"""

    def __init__(self,
                 use_transformers: bool = True,
                 resources_path: str = "./resources",
                 model_cache_dir: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize the Medical NLP Engine

        Args:
            use_transformers: Whether to use transformer models for advanced analysis
            resources_path: Path to medical terminology resources
            model_cache_dir: Directory to cache transformer models
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.resources_path = resources_path
        self.device = device
        self.use_transformers = use_transformers
        self.model_cache_dir = model_cache_dir

        # Initialize NLP components
        logger.info("Initializing Medical NLP Engine...")
        self._init_spacy()
        self._load_medical_terminology()

        if use_transformers:
            try:
                self._init_transformers()
                logger.info("Transformer models loaded successfully")
            except Exception as e:
                logger.warning(f"Couldn't load transformer models: {e}")
                logger.warning("Falling back to rule-based extraction only")
                self.use_transformers = False

        # Initialize vectorizers for similarity computation
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english')

        logger.info("Medical NLP Engine initialized successfully")

    def _init_spacy(self):
        """Initialize spaCy NLP models"""
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded en_core_web_md spaCy model")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded en_core_web_sm spaCy model")
            except OSError:
                logger.error("Could not load spaCy model. Please install with: python -m spacy download en_core_web_md")
                raise

        # Add medical entity ruler if available
        if os.path.exists(os.path.join(self.resources_path, "medical_patterns.jsonl")):
            ruler = self.nlp.add_pipe("entity_ruler")
            ruler.from_disk(os.path.join(self.resources_path, "medical_patterns.jsonl"))
            logger.info("Added medical entity patterns to spaCy pipeline")

    def _init_transformers(self):
        """Initialize transformer models for advanced NLP tasks"""
        # Set cache directory if provided
        kwargs = {}
        if self.model_cache_dir:
            kwargs["cache_dir"] = self.model_cache_dir

        # Zero-shot classification for symptoms
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
            **kwargs
        )

        # Sentiment analysis for severity estimation
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
            **kwargs
        )

        # Named Entity Recognition specifically for medical entities
        try:
            # Try to load medical NER model if available
            self.medical_ner = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Medical NER model could not be loaded: {e}")
            self.medical_ner = None

    def _load_medical_terminology(self):
        """Load medical terminology resources"""
        # Load from JSON resources if available
        try:
            terminology_path = os.path.join(self.resources_path, "medical_terminology.json")
            if os.path.exists(terminology_path):
                with open(terminology_path, 'r') as f:
                    self.medical_terms = json.load(f)
                logger.info(f"Loaded medical terminology from {terminology_path}")
            else:
                # Fallback to built-in terminology
                self._load_builtin_terminology()
        except Exception as e:
            logger.warning(f"Error loading medical terminology: {e}")
            self._load_builtin_terminology()

    def _load_builtin_terminology(self):
        """Load built-in medical terminology"""
        logger.info("Loading built-in medical terminology")
        self.medical_terms = {
            "symptoms": [
                # Common symptoms
                "fever", "chills", "cough", "shortness of breath", "difficulty breathing",
                "fatigue", "muscle aches", "body aches", "headache", "loss of taste",
                "loss of smell", "sore throat", "congestion", "runny nose", "nausea",
                "vomiting", "diarrhea", "rash", "chest pain", "chest tightness",

                # Medical terms for symptoms
                "dyspnea", "myalgia", "arthralgia", "cephalgia", "pyrexia", "emesis",
                "rhinorrhea", "pharyngitis", "coryza", "malaise", "lethargy", "syncope",
                "vertigo", "tinnitus", "palpitations", "diaphoresis", "anorexia", "dysphagia",
                "odynophagia", "hemoptysis", "hematuria", "melena", "hematochezia",
                "dysuria", "polyuria", "oliguria", "anuria", "dyspepsia", "edema",
                "pruritus", "jaundice", "icterus", "petechiae", "ecchymosis", "paresthesia",
                "dysesthesia", "hyperesthesia", "hypoesthesia", "hematemesis", "epistaxis",
                "dysphonia", "dysarthria", "aphasia", "ataxia", "dysmetria", "dysdiadochokinesia",
                "nystagmus", "diplopia", "scotoma", "photophobia", "photopsia", "tachypnea",
                "bradypnea", "tachycardia", "bradycardia", "hypertension", "hypotension"
            ],
            "modifiers": [
                "acute", "chronic", "intermittent", "constant", "severe", "mild", "moderate",
                "worsening", "improving", "radiating", "localized", "diffuse", "bilateral",
                "unilateral", "progressive", "sudden", "gradual", "recurrent", "episodic",
                "persistent", "transient", "paroxysmal", "nocturnal", "diurnal", "postprandial",
                "exertional", "positional", "reproducible", "spontaneous", "exacerbating",
                "alleviating", "aggravating", "relieving", "debilitating", "incapacitating",
                "disabling", "significant", "minor", "major", "extreme", "excruciating"
            ],
            "locations": [
                "head", "skull", "face", "eye", "eyes", "ear", "ears", "nose", "mouth", "jaw",
                "neck", "throat", "chest", "breast", "lungs", "heart", "abdomen", "stomach",
                "liver", "spleen", "intestines", "bowel", "colon", "rectum", "bladder", "kidney",
                "groin", "genitals", "back", "spine", "shoulder", "arm", "elbow", "wrist",
                "hand", "finger", "hip", "leg", "knee", "ankle", "foot", "toe",
                "upper", "lower", "left", "right", "middle", "central", "peripheral", "lateral",
                "medial", "proximal", "distal", "anterior", "posterior", "superior", "inferior"
            ],
            "durations": [
                "seconds", "minutes", "hours", "days", "weeks", "months", "years",
                "brief", "short", "long", "extended", "prolonged", "lifelong", "recent",
                "since childhood", "since birth", "new onset", "longstanding"
            ],
            "emergency_symptoms": [
                "chest pain", "severe chest pain", "crushing chest pain", "pressure in chest",
                "difficulty breathing", "severe shortness of breath", "can't breathe",
                "face drooping", "arm weakness", "speech difficulty", "sudden numbness",
                "sudden confusion", "sudden trouble speaking", "sudden trouble seeing",
                "sudden severe headache", "loss of consciousness", "fainting", "seizure",
                "coughing up blood", "vomiting blood", "severe abdominal pain",
                "severe allergic reaction", "anaphylaxis", "swelling of face", "swelling of throat",
                "stopped breathing", "turning blue", "paralysis", "inability to move",
                "massive bleeding", "wound that won't stop bleeding", "deep cut",
                "major trauma", "head injury", "spinal injury", "poisoning", "overdose",
                "suicidal thoughts", "thoughts of harming self", "thoughts of harming others"
            ],
            "diseases": [
                "COVID-19", "influenza", "common cold", "pneumonia", "bronchitis",
                "asthma", "COPD", "hypertension", "diabetes", "heart disease", "stroke",
                "cancer", "depression", "anxiety", "arthritis", "allergies", "migraine",
                "epilepsy", "Parkinson's disease", "Alzheimer's disease", "multiple sclerosis",
                "HIV/AIDS", "hepatitis", "tuberculosis", "malaria", "dengue fever",
                "Crohn's disease", "ulcerative colitis", "celiac disease", "hypothyroidism",
                "hyperthyroidism", "osteoporosis", "fibromyalgia", "chronic fatigue syndrome",
                "lupus", "rheumatoid arthritis", "psoriasis", "eczema", "glaucoma", "cataracts",
                "sinusitis", "gastritis", "peptic ulcer", "irritable bowel syndrome",
                "kidney disease", "kidney stones", "urinary tract infection", "prostate issues",
                "endometriosis", "polycystic ovary syndrome", "menopause", "gout", "shingles"
            ]
        }

    def extract_symptoms_from_text(self, text: str) -> List[ExtractedSymptom]:
        """
        Extract symptoms from text using multiple extraction methods

        Args:
            text: User input text describing symptoms

        Returns:
            List of ExtractedSymptom objects
        """
        if not text or not text.strip():
            return []

        logger.info(f"Extracting symptoms from text of length {len(text)}")

        # Clean and normalize the text
        cleaned_text = self._preprocess_text(text)

        # Extract symptoms using different methods
        rule_based_symptoms = self._rule_based_extraction(cleaned_text)
        spacy_symptoms = self._spacy_based_extraction(cleaned_text)

        # Use transformer-based extraction if enabled
        if self.use_transformers:
            transformer_symptoms = self._transformer_based_extraction(cleaned_text)
        else:
            transformer_symptoms = []

        # Combine and deduplicate symptoms
        all_symptoms = self._merge_symptoms(rule_based_symptoms, spacy_symptoms, transformer_symptoms)

        # Enrich symptoms with additional information
        enriched_symptoms = self._enrich_symptoms(all_symptoms, cleaned_text)

        logger.info(f"Extracted {len(enriched_symptoms)} symptoms")
        return enriched_symptoms

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better extraction"""
        # Convert to lowercase
        text = text.lower()

        # Replace common non-medical contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "i'm": "i am",
            "i've": "i have",
            "i'll": "i will",
            "it's": "it is",
            "that's": "that is",
            "he's": "he is",
            "she's": "she is",
            "there's": "there is",
            "we're": "we are",
            "they're": "they are"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        # Normalize certain medical terms
        medical_normalizations = {
            "breathe": "breathing",
            "breath": "breathing",
            "feel sick": "nausea",
            "feel nauseous": "nausea",
            "throw up": "vomiting",
            "throwing up": "vomiting",
            "threw up": "vomiting",
            "feel dizzy": "dizziness",
            "stomach ache": "abdominal pain",
            "tummy ache": "abdominal pain",
            "belly ache": "abdominal pain",
            "can't sleep": "insomnia",
            "trouble sleeping": "insomnia",
            "stuffy nose": "nasal congestion",
            "blocked nose": "nasal congestion",
            "running nose": "runny nose",
            "short of breath": "shortness of breath",
            "hard to breathe": "difficulty breathing",
            "tired": "fatigue",
            "exhausted": "fatigue",
            "no energy": "fatigue",
            "feel weak": "weakness",
            "heart racing": "heart palpitations",
            "irregular heartbeat": "heart palpitations"
        }

        for term, normalized in medical_normalizations.items():
            text = re.sub(r'\b' + re.escape(term) + r'\b', normalized, text)

        return text

    def _rule_based_extraction(self, text: str) -> List[ExtractedSymptom]:
        """Extract symptoms using rule-based methods"""
        extracted = []

        # Simple term matching
        for symptom in self.medical_terms["symptoms"]:
            matches = list(re.finditer(r'\b' + re.escape(symptom) + r'\b', text))
            for match in matches:
                # Get context (20 chars before and after)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                # Initial confidence based on exact match
                confidence = 0.7

                # Check for negation in context
                negation_terms = ["no", "not", "don't", "doesn't", "didn't", "haven't",
                                 "hasn't", "hadn't", "never", "none", "deny", "denies",
                                 "denied", "absent", "negative for"]

                is_negated = any(term in context.split() for term in negation_terms)

                if not is_negated:
                    # Look for modifiers
                    modifiers = []
                    for modifier in self.medical_terms["modifiers"]:
                        if modifier in context:
                            modifiers.append(modifier)
                            # Increase confidence if modifiers are present
                            confidence = min(confidence + 0.05, 0.9)

                    # Look for locations
                    location = None
                    for loc in self.medical_terms["locations"]:
                        if loc in context:
                            location = loc
                            # Location specificity increases confidence
                            confidence = min(confidence + 0.05, 0.9)
                            break

                    # Look for duration
                    duration = None
                    for dur in self.medical_terms["durations"]:
                        if dur in context:
                            duration = dur
                            confidence = min(confidence + 0.05, 0.9)
                            break

                    extracted.append(ExtractedSymptom(
                        text=symptom,
                        confidence=confidence,
                        severity=None,
                        duration=duration,
                        location=location,
                        modifiers=modifiers,
                        source_text=context,
                        span_start=match.start(),
                        span_end=match.end()
                    ))

        return extracted

    def _spacy_based_extraction(self, text: str) -> List[ExtractedSymptom]:
        """Extract symptoms using spaCy NLP"""
        extracted = []
        doc = self.nlp(text)

        # Process noun chunks as potential symptoms
        for chunk in doc.noun_chunks:
            # Check if the chunk contains any known symptom term
            for symptom in self.medical_terms["symptoms"]:
                if symptom in chunk.text:
                    # Get modifiers (adjectives) connected to this chunk
                    modifiers = []

                    # Check for negation
                    is_negated = False
                    for token in chunk.root.children:
                        # Check for negation
                        if token.dep_ == "neg":
                            is_negated = True
                            break

                        # Collect modifiers
                        if token.dep_ in ("amod", "advmod") and token.text in self.medical_terms["modifiers"]:
                            modifiers.append(token.text)

                    if not is_negated:
                        # Try to find associated location
                        location = None

                        # Look for prepositional phrases indicating location
                        for token in doc:
                            if token.text in self.medical_terms["locations"]:
                                # Check if this location is related to our symptom
                                if any(t.text == symptom for t in token.ancestors):
                                    location = token.text
                                    break

                        # Calculate confidence score
                        confidence = 0.75  # Base confidence for spaCy extraction
                        confidence += len(modifiers) * 0.05  # More modifiers increase confidence
                        if location:
                            confidence += 0.05  # Location increases confidence
                        confidence = min(confidence, 0.9)  # Cap at 0.9

                        extracted.append(ExtractedSymptom(
                            text=symptom,
                            confidence=confidence,
                            severity=None,
                            duration=None,
                            location=location,
                            modifiers=modifiers,
                            source_text=chunk.text,
                            span_start=chunk.start_char,
                            span_end=chunk.end_char
                        ))

        # Look for entity-based symptoms
        for ent in doc.ents:
            if ent.label_ in ("DISEASE", "PROBLEM", "SYMPTOM") or ent.text in self.medical_terms["symptoms"]:
                # Check for negation
                is_negated = False
                for token in doc:
                    if token.idx < ent.start_char and token.is_negation:
                        # Check if this negation is within reasonable distance
                        if ent.start_char - token.idx < 10:
                            is_negated = True
                            break

                if not is_negated:
                    extracted.append(ExtractedSymptom(
                        text=ent.text,
                        confidence=0.8,  # Higher confidence for named entities
                        severity=None,
                        duration=None,
                        location=None,
                        modifiers=[],
                        source_text=ent.sent.text,
                        span_start=ent.start_char,
                        span_end=ent.end_char
                    ))

        return extracted

    def _transformer_based_extraction(self, text: str) -> List[ExtractedSymptom]:
        """Extract symptoms using transformer models"""
        extracted = []

        if not self.use_transformers:
            return extracted

        try:
            # Split text into manageable chunks if needed
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

            # Process each chunk with zero-shot classification
            all_results = []
            for chunk in chunks:
                # Skip empty chunks
                if not chunk.strip():
                    continue

                # Use zero-shot classification to identify symptoms
                candidate_labels = self.medical_terms["symptoms"][:20]  # Limit to reasonable number
                result = self.zero_shot(
                    chunk,
                    candidate_labels=candidate_labels,
                    multi_label=True
                )

                # Filter to only symptoms with reasonable confidence
                for label, score in zip(result["labels"], result["scores"]):
                    if score > 0.4:  # Threshold for zero-shot model
                        all_results.append((label, score, chunk))

            # Use medical NER if available
            ner_results = []
            if self.medical_ner:
                for chunk in chunks:
                    if not chunk.strip():
                        continue

                    # Extract medical entities
                    entities = self.medical_ner(chunk)

                    for entity in entities:
                        if entity["entity_group"] in ("DISEASE", "SYMPTOM", "PROBLEM", "B-SYMPTOM", "I-SYMPTOM"):
                            ner_results.append((
                                entity["word"],
                                entity["score"],
                                chunk,
                                entity["start"],
                                entity["end"]
                            ))

            # Process zero-shot results
            for symptom_text, confidence, context in all_results:
                # Check if this symptom is negated using simple rule
                negation_terms = ["no", "not", "don't", "doesn't", "didn't", "haven't",
                                 "hasn't", "hadn't", "never", "none", "deny", "denies",
                                 "denied", "absent", "negative for"]

                # Check for nearby negation terms
                is_negated = False
                for term in negation_terms:
                    if re.search(r'\b' + re.escape(term) + r'.*?\b' + re.escape(symptom_text) + r'\b', context) or \
                       re.search(r'\b' + re.escape(symptom_text) + r'.*?\b' + re.escape(term) + r'\b', context):
                        is_negated = True
                        break

                if not is_negated:
                    # Estimate severity using sentiment analysis
                    severity = None
                    try:
                        # Get the context around the symptom
                        start_idx = context.find(symptom_text)
                        if start_idx >= 0:
                            context_start = max(0, start_idx - 50)
                            context_end = min(len(context), start_idx + len(symptom_text) + 50)
                            symptom_context = context[context_start:context_end]

                            # Analyze sentiment of the context
                            sentiment_result = self.sentiment_analyzer(symptom_context)

                            # Map sentiment to severity (negative sentiment = more severe)
                            if sentiment_result[0]["label"] == "NEGATIVE":
                                severity_score = sentiment_result[0]["score"]
                                # Scale from 0-1 to 1-10 for severity
                                severity = 1 + 9 * severity_score
                    except Exception as e:
                        logger.warning(f"Error estimating severity: {e}")

                    # Extract location using pattern matching
                    location = None
                    for loc in self.medical_terms["locations"]:
                        if loc in context:
                            location = loc
                            break

                    extracted.append(ExtractedSymptom(
                        text=symptom_text,
                        confidence=confidence,
                        severity=severity,
                        duration=None,
                        location=location,
                        modifiers=[],
                        source_text=context,
                        span_start=None,
                        span_end=None
                    ))

            # Process NER results
            for symptom_text, confidence, context, start, end in ner_results:
                # Check for negation around the entity
                context_start = max(0, start - 50)
                context_end = min(len(context), end + 50)
                entity_context = context[context_start:context_end]

                negation_terms = ["no", "not", "don't", "doesn't", "didn't", "haven't",
                                 "hasn't", "hadn't", "never", "none", "deny", "denies",
                                 "denied", "absent", "negative for"]

                is_negated = any(term in entity_context for term in negation_terms)

                if not is_negated:
                    extracted.append(ExtractedSymptom(
                        text=symptom_text,
                        confidence=confidence,
                        severity=None,
                        duration=None,
                        location=None,
                        modifiers=[],
                        source_text=entity_context,
                        span_start=start,
                        span_end=end
                    ))

        except Exception as e:
            logger.error(f"Error in transformer-based extraction: {e}")

        return extracted

    def _merge_symptoms(self, *symptom_lists: List[ExtractedSymptom]) -> List[ExtractedSymptom]:
        """Merge and deduplicate symptoms from multiple extraction methods"""
        all_symptoms = []
        for symptom_list in symptom_lists:
            all_symptoms.extend(symptom_list)

        # Group by symptom text
        grouped_symptoms = {}
        for symptom in all_symptoms:
            if symptom.text not in grouped_symptoms:
                grouped_symptoms[symptom.text] = []
            grouped_symptoms[symptom.text].append(symptom)

        # Merge symptoms with the same text
        merged_symptoms = []
        for symptom_text, symptom_group in grouped_symptoms.items():
            if len(symptom_group) == 1:
                merged_symptoms.append(symptom_group[0])
            else:
                # Take the highest confidence
                max_confidence = max(s.confidence for s in symptom_group)

                # Combine modifiers
                all_modifiers = []
                for s in symptom_group:
                    all_modifiers.extend(s.modifiers or [])
                all_modifiers = list(set(all_modifiers))  # Deduplicate

                # Take the most specific location (non-None)
                all_locations = [s.location for s in symptom_group if s.location]
                location = all_locations[0] if all_locations else None

                # Take the most specific duration (non-None)
                all_durations = [s.duration for s in symptom_group if s.duration]
                duration = all_durations[0] if all_durations else None

                # Take the highest severity if available
                all_severities = [s.severity for s in symptom_group if s.severity is not None]
                severity = max(all_severities) if all_severities else None

                # Take the most informative source text (longest)
                all_source_texts = [s.source_text for s in symptom_group if s.source_text]
                source_text = max(all_source_texts, key=len) if all_source_texts else None

                # Get span information if available
                span_starts = [s.span_start for s in symptom_group if s.span_start is not None]
                span_ends = [s.span_end for s in symptom_group if s.span_end is not None]

                span_start = min(span_starts) if span_starts else None
                span_end = max(span_ends) if span_ends else None

                merged_symptoms.append(ExtractedSymptom(
                    text=symptom_text,
                    confidence=max_confidence,
                    severity=severity,
                    duration=duration,
                    location=location,
                    modifiers=all_modifiers,
                    source_text=source_text,
                    span_start=span_start,
                    span_end=span_end
                ))

        # Sort by confidence (highest first)
        merged_symptoms.sort(key=lambda s: s.confidence, reverse=True)
        return merged_symptoms

    def _enrich_symptoms(self, symptoms: List[ExtractedSymptom], original_text: str) -> List[ExtractedSymptom]:
        """Enrich symptoms with additional context and information"""
        enriched_symptoms = []

        for symptom in symptoms:
            # If we don't have a severity, try to infer it from modifiers
            if symptom.severity is None:
                severity_modifiers = {
                    "severe": 9.0,
                    "extreme": 10.0,
                    "excruciating": 10.0,
                    "worst": 10.0,
                    "unbearable": 10.0,
                    "terrible": 8.0,
                    "intense": 8.0,
                    "bad": 7.0,
                    "significant": 7.0,
                    "moderate": 5.0,
                    "mild": 3.0,
                    "slight": 2.0,
                    "minimal": 1.0,
                    "minor": 1.0
                }

                for modifier in symptom.modifiers:
                    if modifier in severity_modifiers:
                        symptom.severity = severity_modifiers[modifier]
                        break

            # If no duration is specified, try to infer from text
            if symptom.duration is None and symptom.source_text:
                duration_patterns = [
                    (r'for\s+(\d+)\s+day', 'days'),
                    (r'for\s+(\d+)\s+week', 'weeks'),
                    (r'for\s+(\d+)\s+month', 'months'),
                    (r'for\s+(\d+)\s+year', 'years'),
                    (r'for\s+(\d+)\s+hour', 'hours'),
                    (r'for\s+(\d+)\s+minute', 'minutes'),
                    (r'since\s+(\w+)', 'since'),
                    (r'started\s+(\d+)\s+day', 'days'),
                    (r'started\s+(\d+)\s+week', 'weeks'),
                    (r'started\s+(\d+)\s+month', 'months')
                ]

                for pattern, unit in duration_patterns:
                    match = re.search(pattern, symptom.source_text)
                    if match:
                        if unit == 'since':
                            symptom.duration = f"since {match.group(1)}"
                        else:
                            symptom.duration = f"{match.group(1)} {unit}"
                        break

            # Check if this is an emergency symptom
            emergency_status = False
            for emergency_symptom in self.medical_terms["emergency_symptoms"]:
                if symptom.text in emergency_symptom or emergency_symptom in symptom.text:
                    emergency_status = True
                    # Increase confidence for emergency symptoms
                    symptom.confidence = max(symptom.confidence, 0.9)
                    break

            # Add to enriched list
            enriched_symptoms.append(symptom)

        return enriched_symptoms

    def calculate_symptom_similarity(self, user_symptoms: List[str], disease_symptoms: List[str]) -> float:
        """Calculate similarity between user symptoms and disease symptoms"""
        if not user_symptoms or not disease_symptoms:
            return 0.0

        # Create document representations
        docs = [' '.join(user_symptoms), ' '.join(disease_symptoms)]

        try:
            # Transform to TF-IDF features
            tfidf_matrix = self.vectorizer.fit_transform(docs)

            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating symptom similarity: {e}")
            # Fallback similarity calculation
            common_symptoms = set(user_symptoms).intersection(set(disease_symptoms))
            combined_symptoms = set(user_symptoms).union(set(disease_symptoms))

            if not combined_symptoms:
                return 0.0

            return len(common_symptoms) / len(combined_symptoms)

    def check_emergency_symptoms(self, symptoms: List[str]) -> Tuple[bool, List[str]]:
        """Check if any symptoms indicate a medical emergency"""
        emergency_symptoms = self.medical_terms["emergency_symptoms"]

        matched_emergency = []
        for symptom in symptoms:
            for emergency in emergency_symptoms:
                if emergency in symptom.lower() or symptom.lower() in emergency:
                    matched_emergency.append(symptom)
                    break

        # Specific pattern matching for critical conditions
        text = ' '.join(symptoms)

        # Pattern for stroke (BE FAST)
        stroke_patterns = [
            r'\b(face|facial)\s+(droop|drooping|weakness|numbness|paralysis)\b',
            r'\b(arm|leg|limb)\s+(weakness|numbness|paralysis|limp)\b',
            r'\b(speech|speaking)\s+(slurred|difficult|difficulty|confused|confusion)\b',
            r'\bsudden\s+(numbness|weakness|confusion|trouble\s+speaking|trouble\s+seeing|trouble\s+walking|dizziness|loss\s+of\s+balance|severe\s+headache)\b'
        ]

        for pattern in stroke_patterns:
            if re.search(pattern, text, re.IGNORECASE) and "stroke symptoms" not in matched_emergency:
                matched_emergency.append("stroke symptoms")
                break

        # Pattern for heart attack
        heart_attack_patterns = [
            r'\b(chest\s+pain|chest\s+pressure|chest\s+tightness|chest\s+heaviness)\b',
            r'\b(pain|discomfort)\s+(radiating|spreading)\s+to\s+(arm|jaw|neck|back)\b',
            r'\b(shortness\s+of\s+breath|difficulty\s+breathing)\s+with\s+(chest\s+pain|chest\s+pressure|chest\s+discomfort)\b'
        ]

        for pattern in heart_attack_patterns:
            if re.search(pattern, text, re.IGNORECASE) and "heart attack symptoms" not in matched_emergency:
                matched_emergency.append("heart attack symptoms")
                break

        # Pattern for severe allergic reaction
        allergy_patterns = [
            r'\b(throat\s+closing|throat\s+swelling|difficulty\s+swallowing|can\'t\s+swallow)\b',
            r'\b(hives|rash)\s+with\s+(shortness\s+of\s+breath|difficulty\s+breathing)\b',
            r'\b(face\s+swelling|lip\s+swelling|tongue\s+swelling)\s+after\b'
        ]

        for pattern in allergy_patterns:
            if re.search(pattern, text, re.IGNORECASE) and "severe allergic reaction" not in matched_emergency:
                matched_emergency.append("severe allergic reaction")
                break

        return bool(matched_emergency), matched_emergency

    def generate_prompt_for_medical_literature(self, symptoms: List[ExtractedSymptom], top_matches: Dict[str, Any] = None) -> str:
        """Generate a search prompt for medical literature based on extracted symptoms"""
        if not symptoms:
            return ""

        # Use top symptoms by confidence
        top_symptoms = sorted(symptoms, key=lambda s: s.confidence, reverse=True)[:5]

        # Generate prompts for different scenarios
        if top_matches and len(top_matches) > 0:
            # We have condition matches, create a targeted prompt
            top_condition = list(top_matches.keys())[0]
            prompt = f"Recent medical research on {top_condition} with symptoms including "
            symptom_texts = [s.text for s in top_symptoms]
            prompt += ", ".join(symptom_texts[:-1]) + " and " + symptom_texts[-1] if len(symptom_texts) > 1 else symptom_texts[0]
        else:
            # No condition matches, create a symptom-based prompt
            prompt = "Recent medical research on patients presenting with "

            # Add symptom details
            symptom_descriptions = []
            for symptom in top_symptoms:
                desc = symptom.text
                if symptom.modifiers:
                    desc = f"{' '.join(symptom.modifiers)} {desc}"
                if symptom.location:
                    desc = f"{desc} in the {symptom.location}"
                if symptom.duration:
                    desc = f"{desc} for {symptom.duration}"
                symptom_descriptions.append(desc)

            prompt += ", ".join(symptom_descriptions[:-1]) + " and " + symptom_descriptions[-1] if len(symptom_descriptions) > 1 else symptom_descriptions[0]

        return prompt

    def extract_keywords_for_search(self, symptoms: List[ExtractedSymptom], demographic_info: Dict[str, Any] = None) -> List[str]:
        """Extract keywords for medical search from symptoms and demographics"""
        if not symptoms:
            return []

        keywords = []

        # Add top symptom terms
        top_symptoms = sorted(symptoms, key=lambda s: s.confidence, reverse=True)[:5]
        for symptom in top_symptoms:
            # Add the primary symptom
            keywords.append(symptom.text)

            # Add with modifiers for more specific searches
            if symptom.modifiers:
                for modifier in symptom.modifiers[:2]:  # Limit to top 2 modifiers
                    keywords.append(f"{modifier} {symptom.text}")

            # Add with location for more specific searches
            if symptom.location:
                keywords.append(f"{symptom.text} {symptom.location}")

        # Add demographic-specific terms if available
        if demographic_info:
            if "age" in demographic_info:
                age = demographic_info["age"]
                if age <= 12:
                    keywords.append("pediatric")
                elif age >= 65:
                    keywords.append("geriatric")
                    keywords.append("elderly")

            if "gender" in demographic_info:
                gender = demographic_info["gender"]
                if gender.lower() in ["female", "f", "woman"]:
                    keywords.append("female")
                elif gender.lower() in ["male", "m", "man"]:
                    keywords.append("male")

        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)

        return unique_keywords

    def generate_doctor_questions(self, symptoms: List[ExtractedSymptom], condition: str = None) -> List[str]:
        """Generate questions to ask a doctor based on symptoms and potential condition"""
        if not symptoms:
            return ["What might be causing my symptoms?",
                   "What tests should I consider?",
                   "When should I be concerned about my symptoms?"]

        questions = []

        # Basic questions based on symptoms
        top_symptoms = sorted(symptoms, key=lambda s: s.confidence, reverse=True)[:3]
        symptom_texts = [s.text for s in top_symptoms]

        if condition:
            # Condition-specific questions
            questions.append(f"Could my symptoms of {', '.join(symptom_texts)} be consistent with {condition}?")
            questions.append(f"What tests would confirm a diagnosis of {condition}?")
            questions.append(f"What is the standard treatment for {condition}?")
            questions.append(f"What lifestyle changes would help manage {condition}?")
            questions.append(f"What's the typical prognosis for {condition}?")
        else:
            # General symptom-based questions
            symptoms_text = ", ".join(symptom_texts[:-1]) + " and " + symptom_texts[-1] if len(symptom_texts) > 1 else symptom_texts[0]
            questions.append(f"What could be causing my {symptoms_text}?")
            questions.append(f"How concerned should I be about these symptoms?")
            questions.append(f"What tests might you recommend for these symptoms?")

        # Add symptom-specific questions
        symptom_specific_questions = {
            "pain": "What pain management options would you recommend?",
            "fever": "At what temperature should I be concerned about my fever?",
            "rash": "Is this rash contagious?",
            "dizziness": "Is it safe for me to drive with this dizziness?",
            "fatigue": "Could my fatigue be related to a vitamin deficiency?",
            "headache": "Could my headaches be related to stress or something more serious?",
            "cough": "How long should I expect this cough to last?",
            "shortness of breath": "Should I be monitoring my oxygen levels at home?",
            "nausea": "Are there any dietary changes that could help with my nausea?",
            "chest pain": "What tests can determine if my chest pain is heart-related?",
            "abdominal pain": "Could my abdominal pain be related to something I'm eating?",
            "anxiety": "What techniques do you recommend to manage anxiety?",
            "depression": "Would you recommend therapy, medication, or both for my symptoms?",
            "insomnia": "What sleep hygiene practices might help my insomnia?"
        }

        for symptom in symptoms:
            for key, question in symptom_specific_questions.items():
                if key in symptom.text.lower() and question not in questions:
                    questions.append(question)
                    break

        # Add follow-up questions based on duration
        has_chronic_symptoms = any(s.duration and ("chronic" in s.duration or
                                                  "month" in s.duration or
                                                  "year" in s.duration)
                                  for s in symptoms)
        if has_chronic_symptoms:
            questions.append("Should I consider seeing a specialist for these long-term symptoms?")
            questions.append("Are there any support groups for people with chronic conditions like mine?")

        # Limit to a reasonable number of questions
        if len(questions) > 8:
            questions = questions[:8]

        return questions

# Singleton instance for reuse
_nlp_engine_instance = None

def get_nlp_engine(use_transformers=False, model_cache_dir=None):
    """Get or create the NLP engine singleton instance"""
    global _nlp_engine_instance

    if _nlp_engine_instance is None:
        _nlp_engine_instance = MedicalNLPEngine(
            use_transformers=use_transformers,
            model_cache_dir=model_cache_dir
        )

    return _nlp_engine_instance
