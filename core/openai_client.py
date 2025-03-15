import os
import json
import logging
import random
import time
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Mock OpenAI client that generates responses locally without using the OpenAI API.
    Provides methods for generating responses, analyzing symptoms, and processing medical text.

    This mock client doesn't require an API key and returns predefined responses.
    """

    def __init__(self):
        """Initialize the mock OpenAI client."""
        self.api_key = None
        self.health_tips = [
            "Stay hydrated throughout the day. Proper hydration supports all bodily functions, improves energy levels, and helps maintain concentration.",
            "Aim for 7-9 hours of quality sleep each night. Good sleep hygiene improves immune function, mood, and cognitive performance.",
            "Include a variety of colorful fruits and vegetables in your diet to ensure you get a wide range of nutrients and antioxidants.",
            "Regular physical activity helps maintain a healthy weight, reduces disease risk, and improves mental health. Aim for at least 150 minutes of moderate exercise per week.",
            "Practice mindfulness or meditation for stress reduction. Even 5-10 minutes daily can help reduce anxiety and improve focus.",
            "Take regular breaks when working at a computer to reduce eye strain and prevent repetitive stress injuries.",
            "Limit processed foods and added sugars, which can contribute to inflammation and chronic health issues.",
            "Maintain social connections, as strong relationships are linked to better health outcomes and longevity.",
            "Regular health check-ups can help catch potential problems early. Don't skip your preventive screenings.",
            "Practice good hand hygiene to prevent the spread of infections. Wash hands thoroughly for at least 20 seconds."
        ]
        self.symptom_descriptions = {
            "fever": "An elevated body temperature above the normal range of 97°F to 99°F (36.1°C to 37.2°C), often indicating infection or inflammation.",
            "cough": "A sudden expulsion of air from the lungs that helps clear the airways of irritants, mucus, or foreign particles.",
            "headache": "Pain or discomfort in the head, scalp, or neck, which can range from mild to severe and may be associated with various conditions.",
            "fatigue": "Extreme tiredness resulting from mental or physical exertion, illness, or stress that doesn't improve with rest.",
            "nausea": "An unpleasant sensation in the stomach with an urge to vomit, often accompanied by dizziness and general discomfort.",
            "chest pain": "Discomfort or pain in the chest area, which can feel like pressure, burning, tightness, or sharpness.",
            "shortness of breath": "Difficulty breathing or a feeling of not getting enough air, which can result from various cardiac and respiratory conditions.",
            "dizziness": "A sensation of lightheadedness, unsteadiness, or feeling faint, which can be caused by various factors including dehydration or inner ear issues.",
            "abdominal pain": "Pain felt between the chest and pelvic regions, ranging from mild to severe, and can be caused by numerous conditions.",
            "sore throat": "Pain, irritation, or scratchiness in the throat, often worsened by swallowing, commonly due to infections or irritation."
        }
        self.condition_information = {
            "common cold": "A viral infectious disease of the upper respiratory tract affecting the nose, throat, sinuses, and larynx. Symptoms include runny nose, sneezing, sore throat, cough, and mild fever. Treatment focuses on symptom relief as it typically resolves within 7-10 days.",
            "influenza": "A contagious respiratory illness caused by influenza viruses. Symptoms include fever, cough, sore throat, body aches, fatigue, and sometimes vomiting and diarrhea. More severe than the common cold, it can lead to complications, particularly in vulnerable populations.",
            "migraine": "A neurological condition characterized by recurrent, severe headaches often accompanied by nausea, vomiting, and sensitivity to light and sound. Triggers can include stress, certain foods, hormonal changes, and environmental factors.",
            "gastroenteritis": "Inflammation of the lining of the stomach and intestines, typically resulting from infection. Symptoms include diarrhea, vomiting, abdominal pain, and sometimes fever. Most cases resolve within a few days with proper hydration and rest.",
            "hypertension": "Persistently elevated blood pressure in the arteries, often without noticeable symptoms. Long-term hypertension increases the risk of heart disease, stroke, and kidney failure. Lifestyle modifications and medications can help manage this condition.",
            "bronchitis": "Inflammation of the bronchial tubes that carry air to and from the lungs. Symptoms include coughing with mucus, shortness of breath, mild fever, and chest discomfort. Acute bronchitis is often caused by viruses and resolves within two weeks.",
            "pneumonia": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. Severity can range from mild to life-threatening, depending on factors like age and health status.",
            "anxiety disorder": "A mental health disorder characterized by persistent feelings of worry, fear, or stress that interfere with daily activities. Physical symptoms may include rapid heartbeat, shortness of breath, and increased sweating. Treatment options include therapy and medication."
        }

        logger.info("Initialized mock OpenAI client")

    def set_api_key(self, api_key: str):
        """
        Set the API key (not actually used in this mock version).

        Args:
            api_key: OpenAI API key
        """
        # Store API key for compatibility with real client
        self.api_key = api_key
        logger.info("API key set (Note: not actually used in mock version)")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a text response based on the prompt without using the OpenAI API.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        # Log the request
        logger.info(f"Generating response for prompt: {prompt[:50]}...")

        # Add a small delay to simulate API call
        time.sleep(0.5)

        # Check for common prompt types and provide appropriate responses
        prompt_lower = prompt.lower()

        # Health tip request
        if "health tip" in prompt_lower or "wellness tip" in prompt_lower:
            return random.choice(self.health_tips)

        # Symptom information request
        for symptom, description in self.symptom_descriptions.items():
            if symptom in prompt_lower and ("what is" in prompt_lower or "explain" in prompt_lower):
                return f"{symptom.capitalize()}: {description}"

        # Condition information request
        for condition, information in self.condition_information.items():
            if condition in prompt_lower and ("what is" in prompt_lower or "explain" in prompt_lower):
                return f"{condition.capitalize()}: {information}"

        # Default responses for common medical inquiries
        if "should i see a doctor" in prompt_lower:
            return "While I can provide general information, it's important to consult with a healthcare professional for personalized medical advice. If you're experiencing severe symptoms, persistent symptoms, or are concerned about your health, it's always best to speak with a doctor."

        if "is it serious" in prompt_lower:
            return "I cannot determine the severity of medical conditions without a proper medical evaluation. Many symptoms can be associated with both minor and serious conditions. It's always best to consult with a healthcare professional who can perform a proper assessment."

        if "treatment" in prompt_lower or "cure" in prompt_lower or "remedy" in prompt_lower:
            return "Treatment depends on the specific condition and individual factors. While some conditions may benefit from rest, hydration, and over-the-counter medications, others may require prescription medications or medical procedures. A healthcare professional can provide personalized treatment recommendations based on a proper diagnosis."

        # Generic response for other queries
        return "I'm a health assistant designed to provide general health information. While I can offer educational content about symptoms and conditions, I cannot diagnose medical conditions or provide personalized medical advice. For specific health concerns, please consult with a qualified healthcare professional."

    def analyze_symptoms(self, symptoms: List[str], user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze symptoms to identify potential conditions without using the OpenAI API.

        Args:
            symptoms: List of symptom names or IDs
            user_profile: Optional user profile information

        Returns:
            Analysis results including potential conditions
        """
        # Log the request
        logger.info(f"Analyzing symptoms: {symptoms}")

        # Add a small delay to simulate API call
        time.sleep(1.0)

        # Default response structure
        response = {
            "possible_conditions": [],
            "severity_level": "unknown",
            "recommendation": "",
            "confidence_score": 0.0
        }

        # Convert symptoms to lowercase for matching
        symptoms_lower = [s.lower() for s in symptoms]

        # Check for common symptom combinations and provide appropriate responses

        # Cold/Flu symptoms
        cold_flu_symptoms = ["fever", "cough", "headache", "fatigue", "sore throat"]
        cold_flu_count = sum(1 for s in cold_flu_symptoms if any(s in symptom for symptom in symptoms_lower))

        # Migraine symptoms
        migraine_symptoms = ["headache", "nausea", "dizziness"]
        migraine_count = sum(1 for s in migraine_symptoms if any(s in symptom for symptom in symptoms_lower))

        # Gastroenteritis symptoms
        gastro_symptoms = ["nausea", "abdominal pain"]
        gastro_count = sum(1 for s in gastro_symptoms if any(s in symptom for symptom in symptoms_lower))

        # Respiratory issues
        respiratory_symptoms = ["cough", "shortness of breath", "chest pain"]
        respiratory_count = sum(1 for s in respiratory_symptoms if any(s in symptom for symptom in symptoms_lower))

        # Determine possible conditions based on symptom combinations
        possible_conditions = []

        if cold_flu_count >= 3:
            if any("fever" in s for s in symptoms_lower):
                possible_conditions.append({"name": "Influenza", "probability": 0.7})
            else:
                possible_conditions.append({"name": "Common Cold", "probability": 0.8})

        if migraine_count >= 2:
            possible_conditions.append({"name": "Migraine", "probability": 0.75})

        if gastro_count >= 2:
            possible_conditions.append({"name": "Gastroenteritis", "probability": 0.65})

        if respiratory_count >= 2:
            if respiratory_count == 3:
                possible_conditions.append({"name": "Pneumonia", "probability": 0.5})
            possible_conditions.append({"name": "Bronchitis", "probability": 0.6})

        # Add to response
        response["possible_conditions"] = possible_conditions

        # Determine severity level
        if len(possible_conditions) == 0:
            response["severity_level"] = "low"
            response["recommendation"] = "Monitor your symptoms. If they persist or worsen, consider consulting a healthcare professional."
            response["confidence_score"] = 0.4
        elif any(c["name"] in ["Pneumonia"] for c in possible_conditions):
            response["severity_level"] = "high"
            response["recommendation"] = "These symptoms could indicate a serious condition. Please consult a healthcare professional promptly."
            response["confidence_score"] = 0.7
        elif any(c["name"] in ["Influenza", "Bronchitis"] for c in possible_conditions):
            response["severity_level"] = "medium"
            response["recommendation"] = "Your symptoms suggest a condition that may benefit from medical attention. Consider scheduling an appointment with a healthcare provider."
            response["confidence_score"] = 0.6
        else:
            response["severity_level"] = "low"
            response["recommendation"] = "Your symptoms are commonly associated with minor conditions. Rest, stay hydrated, and monitor for changes. If symptoms persist beyond a few days or worsen, consult a healthcare professional."
            response["confidence_score"] = 0.5

        return response

    def extract_symptoms_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract symptoms from natural language text without using the OpenAI API.

        Args:
            text: Natural language text describing symptoms

        Returns:
            List of extracted symptoms with confidence scores
        """
        # Log the request
        logger.info(f"Extracting symptoms from text: {text[:50]}...")

        # Add a small delay to simulate API call
        time.sleep(0.7)

        # Convert text to lowercase for matching
        text_lower = text.lower()

        # Check for common symptoms in the text
        extracted_symptoms = []

        for symptom, description in self.symptom_descriptions.items():
            if symptom in text_lower:
                # Extract symptoms with confidence scores
                extracted_symptoms.append({
                    "symptom": symptom,
                    "confidence": 0.9,
                    "matches": [symptom]
                })
            # Check for related terms
            elif symptom == "fever" and any(term in text_lower for term in ["hot", "temperature", "burning up"]):
                extracted_symptoms.append({
                    "symptom": "fever",
                    "confidence": 0.7,
                    "matches": ["temperature"]
                })
            elif symptom == "headache" and any(term in text_lower for term in ["head hurts", "head pain", "migraine"]):
                extracted_symptoms.append({
                    "symptom": "headache",
                    "confidence": 0.8,
                    "matches": ["head pain"]
                })
            elif symptom == "fatigue" and any(term in text_lower for term in ["tired", "exhausted", "no energy"]):
                extracted_symptoms.append({
                    "symptom": "fatigue",
                    "confidence": 0.8,
                    "matches": ["tired"]
                })
            elif symptom == "nausea" and any(term in text_lower for term in ["sick to my stomach", "feel like vomiting", "queasy"]):
                extracted_symptoms.append({
                    "symptom": "nausea",
                    "confidence": 0.8,
                    "matches": ["queasy"]
                })

        return extracted_symptoms

    def summarize_medical_literature(self, query: str, max_length: int = 500) -> Dict[str, Any]:
        """
        Summarize medical literature on a topic without using the OpenAI API.

        Args:
            query: Search query
            max_length: Maximum length of the summary

        Returns:
            Summary and reference information
        """
        # Log the request
        logger.info(f"Summarizing medical literature for query: {query}")

        # Add a small delay to simulate API call
        time.sleep(1.5)

        # Default response structure
        response = {
            "summary": "",
            "references": [],
            "query": query
        }

        # Convert query to lowercase for matching
        query_lower = query.lower()

        # Check for common topics and provide appropriate responses

        # Common cold
        if "cold" in query_lower or "common cold" in query_lower:
            response["summary"] = (
                "The common cold is a viral infection of the upper respiratory tract, primarily caused by rhinoviruses. "
                "Symptoms typically include runny nose, sore throat, cough, sneezing, and mild fever. The condition usually "
                "resolves within 7-10 days without specific treatment. Management focuses on symptom relief through rest, "
                "hydration, and over-the-counter medications. While antibiotics are ineffective against viral infections, "
                "some evidence suggests zinc lozenges and vitamin C may slightly reduce symptom duration. Preventive measures "
                "include frequent handwashing and avoiding close contact with infected individuals."
            )
            response["references"] = [
                {"title": "Common Cold: Evaluation and Management", "authors": "Fashner J, et al.", "journal": "American Family Physician", "year": 2021},
                {"title": "Update on the epidemiology and management of the common cold", "authors": "Eccles R", "journal": "Therapeutic Advances in Respiratory Disease", "year": 2019}
            ]

        # Influenza
        elif "flu" in query_lower or "influenza" in query_lower:
            response["summary"] = (
                "Influenza (flu) is a contagious respiratory illness caused by influenza viruses that infect the nose, throat, "
                "and lungs. It can cause mild to severe illness, and at times can lead to death. The flu is different from a "
                "cold and usually comes on suddenly. Symptoms include fever, cough, sore throat, runny nose, body aches, "
                "headaches, and fatigue. Most people recover in less than two weeks, but some develop complications like "
                "pneumonia. Annual vaccination is the most effective method for preventing influenza. Antiviral medications "
                "can be prescribed to treat influenza, especially for high-risk individuals, and are most effective when started "
                "within 48 hours of symptom onset."
            )
            response["references"] = [
                {"title": "Influenza: Diagnosis and Treatment", "authors": "Kalil AC, et al.", "journal": "Critical Care Clinics", "year": 2023},
                {"title": "Influenza Vaccination: Current and Future Practices", "authors": "Grohskopf LA, et al.", "journal": "Pharmacotherapy", "year": 2022}
            ]

        # Hypertension
        elif "hypertension" in query_lower or "high blood pressure" in query_lower:
            response["summary"] = (
                "Hypertension (high blood pressure) is a common condition in which the long-term force of the blood against "
                "artery walls is high enough that it may eventually cause health problems. Guidelines define hypertension as "
                "blood pressure above 130/80 mm Hg. Risk factors include age, family history, obesity, sedentary lifestyle, "
                "excess sodium intake, and certain chronic conditions. Long-term hypertension is a major risk factor for "
                "coronary artery disease, stroke, heart failure, and kidney disease. Management typically includes lifestyle "
                "modifications (diet, exercise, weight management, sodium reduction) and medication therapy. Regular monitoring "
                "is essential for effective management."
            )
            response["references"] = [
                {"title": "Diagnosis and Management of Hypertension", "authors": "Whelton PK, et al.", "journal": "Journal of the American College of Cardiology", "year": 2023},
                {"title": "Guidelines for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults", "authors": "American College of Cardiology", "journal": "Hypertension", "year": 2022}
            ]

        # Generic response for other queries
        else:
            response["summary"] = (
                f"Research on {query} indicates this is an area of ongoing medical investigation. Studies have examined various "
                f"aspects including potential causes, risk factors, diagnostic approaches, and treatment options. Current evidence "
                f"suggests multiple factors may contribute to this condition, though consensus on optimal management continues to evolve. "
                f"Healthcare providers typically recommend an individualized approach based on specific patient characteristics "
                f"and medical history. Recent advances have improved understanding, but additional research is needed to fully "
                f"clarify mechanisms and develop more targeted interventions."
            )
            response["references"] = [
                {"title": f"Current Perspectives on {query.capitalize()}", "authors": "Various Authors", "journal": "Journal of Medical Research", "year": 2023},
                {"title": f"Clinical Management of {query.capitalize()}: A Systematic Review", "authors": "Research Team", "journal": "Medical Reviews Journal", "year": 2022}
            ]

        # Ensure summary doesn't exceed max length
        if len(response["summary"]) > max_length:
            response["summary"] = response["summary"][:max_length].rsplit(".", 1)[0] + "."

        return response

    def generate_health_report(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive health report without using the OpenAI API.

        Args:
            user_data: User health data including symptoms, profile, and history

        Returns:
            Generated health report
        """
        # Log the request
        logger.info("Generating health report")

        # Add a small delay to simulate API call
        time.sleep(2.0)

        # Default report structure
        report = {
            "summary": "",
            "recommendations": [],
            "risk_factors": [],
            "health_trends": [],
            "suggested_actions": []
        }

        # Extract user information
        profile = user_data.get("profile", {})
        symptoms = user_data.get("symptoms", [])
        risk_level = user_data.get("risk_level", "unknown")

        # Generate summary based on risk level
        if risk_level == "high":
            report["summary"] = (
                "Based on your reported symptoms and health data, your current health status indicates some concerning issues that may require prompt medical attention. "
                "The combination of symptoms you're experiencing could be associated with conditions that benefit from professional evaluation. "
                "While this assessment is not a diagnosis, it's recommended that you consult with a healthcare provider to address these concerns."
            )
            report["recommendations"] = [
                "Schedule an appointment with your healthcare provider within the next 1-2 days",
                "Monitor your symptoms closely and seek immediate care if they worsen",
                "Keep a detailed log of your symptoms, including timing and severity",
                "Prepare a list of questions for your healthcare provider",
                "Consider having someone accompany you to your medical appointment"
            ]
        elif risk_level == "medium":
            report["summary"] = (
                "Your health data suggests a moderate level of concern. The symptoms you've reported may be associated with several conditions that typically "
                "benefit from medical evaluation, though they may not require urgent care. Monitoring these symptoms and discussing them with a healthcare "
                "provider is recommended to determine appropriate next steps."
            )
            report["recommendations"] = [
                "Schedule an appointment with your healthcare provider within the next week",
                "Monitor your symptoms and note any changes",
                "Maintain adequate hydration and rest",
                "Avoid activities that might exacerbate your symptoms",
                "Consider over-the-counter remedies for symptom relief, following package instructions"
            ]
        else:  # low or unknown
            report["summary"] = (
                "Based on your reported information, your current health concerns appear to be mild. Your symptoms may be associated with common, self-limiting "
                "conditions that often resolve with basic self-care. While medical attention doesn't seem urgent at this time, continued monitoring is recommended, "
                "especially if your symptoms persist or worsen."
            )
            report["recommendations"] = [
                "Monitor your symptoms over the next few days",
                "Maintain good hydration and adequate rest",
                "Consider over-the-counter remedies for symptom relief, following package instructions",
                "Schedule a routine check-up if symptoms persist beyond 7 days",
                "Practice preventive measures such as handwashing and healthy diet"
            ]

        # Generate risk factors based on profile
        age = profile.get("age", 0)
        gender = profile.get("gender", "unknown")

        if age > 60:
            report["risk_factors"].append("Age over 60 may increase risk for certain conditions")

        if "smoking" in profile.get("history", []):
            report["risk_factors"].append("Smoking history may exacerbate respiratory symptoms")

        if "hypertension" in profile.get("conditions", []):
            report["risk_factors"].append("History of hypertension may impact cardiovascular health")

        if "diabetes" in profile.get("conditions", []):
            report["risk_factors"].append("Diabetes may affect immune response and symptom presentation")

        # Generate health trends (mock data)
        report["health_trends"] = [
            "Your symptom frequency has been stable over the past month",
            "Sleep quality shows improvement compared to previous reports",
            "Stress levels appear to correlate with symptom intensity"
        ]

        # Generate suggested actions
        report["suggested_actions"] = [
            "Complete your health profile with any missing information",
            "Set up regular symptom tracking reminders",
            "Review educational materials about your symptoms",
            "Consider lifestyle modifications to support overall health"
        ]

        # Add specific actions based on symptoms
        if "fever" in symptoms or "temperature" in symptoms:
            report["suggested_actions"].append("Monitor temperature regularly and record readings")

        if "fatigue" in symptoms or "tired" in symptoms:
            report["suggested_actions"].append("Track energy levels throughout the day and note patterns")

        if "headache" in symptoms:
            report["suggested_actions"].append("Note headache triggers, location, and intensity")

        return report

    def chat_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Simulate a chat conversation without using the OpenAI API.

        Args:
            messages: List of message dictionaries with roles and content

        Returns:
            Response message
        """
        # Log the request
        logger.info(f"Processing chat conversation with {len(messages)} messages")

        # Add a small delay to simulate API call
        time.sleep(0.8)

        # Get the last user message
        last_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                last_message = message.get("content", "")
                break

        if not last_message:
            return "I don't see a message to respond to. How can I help you with your health questions today?"

        # Convert to lowercase for matching
        message_lower = last_message.lower()

        # Check for common message types and provide appropriate responses

        # Greetings
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm your health assistant. I can provide information about symptoms, conditions, and general health topics. How can I help you today?"

        # Symptom inquiries
        if "symptom" in message_lower:
            return "I can provide information about common symptoms. Could you tell me which specific symptom you'd like to learn about? Or if you're experiencing symptoms, you could describe them to me for general information."

        # Condition inquiries
        if "condition" in message_lower or "disease" in message_lower:
            return "I can provide educational information about various health conditions. Which specific condition would you like to learn more about?"

        # Thank you
        if "thank" in message_lower or "thanks" in message_lower:
            return "You're welcome! I'm here to help with your health questions. Is there anything else you'd like to know?"

        # Questions about capabilities
        if "what can you do" in message_lower or "how can you help" in message_lower:
            return "I can help with several health-related tasks:\n\n- Provide information about symptoms and conditions\n- Analyze reported symptoms for potential causes\n- Offer general health tips and recommendations\n- Summarize medical information on various topics\n- Track your health data over time\n\nWhat would you like help with today?"

        # Default response
        return "I'm here to provide general health information and support. While I can offer educational content about symptoms and conditions, I cannot provide personalized medical advice or diagnoses. For specific health concerns, it's always best to consult with a qualified healthcare professional."

    def generate_symptom_analysis(self, symptoms: List[str], user_age: int = 35, user_gender: str = "unknown") -> Dict[str, Any]:
        """
        Generate a detailed analysis of symptoms without using the OpenAI API.

        Args:
            symptoms: List of symptom names
            user_age: User's age
            user_gender: User's gender

        Returns:
            Detailed symptom analysis
        """
        # Log the request
        logger.info(f"Generating symptom analysis for {len(symptoms)} symptoms")

        # Add a small delay to simulate API call
        time.sleep(1.5)

        # Default analysis structure
        analysis = {
            "potential_causes": [],
            "severity_assessment": "",
            "recommended_actions": [],
            "time_sensitivity": "",
            "lifestyle_recommendations": []
        }

        # Convert symptoms to lowercase for matching
        symptoms_lower = [s.lower() for s in symptoms]

        # Map symptoms to potential causes
        potential_causes = {}

        for symptom in symptoms_lower:
            if "fever" in symptom:
                potential_causes.setdefault("Infection", 0)
                potential_causes["Infection"] += 1
                potential_causes.setdefault("Inflammatory conditions", 0)
                potential_causes["Inflammatory conditions"] += 0.5

            if "cough" in symptom:
                potential_causes.setdefault("Respiratory infection", 0)
                potential_causes["Respiratory infection"] += 1
                potential_causes.setdefault("Allergies", 0)
                potential_causes["Allergies"] += 0.5
                potential_causes.setdefault("Asthma", 0)
                potential_causes["Asthma"] += 0.3

            if "headache" in symptom:
                potential_causes.setdefault("Tension", 0)
                potential_causes["Tension"] += 1
                potential_causes.setdefault("Migraine", 0)
                potential_causes["Migraine"] += 0.7
                potential_causes.setdefault("Dehydration", 0)
                potential_causes["Dehydration"] += 0.5

            if "fatigue" in symptom:
                potential_causes.setdefault("Sleep disturbance", 0)
                potential_causes["Sleep disturbance"] += 1
                potential_causes.setdefault("Viral infection", 0)
                potential_causes["Viral infection"] += 0.7
                potential_causes.setdefault("Anemia", 0)
                potential_causes["Anemia"] += 0.4

            if "nausea" in symptom:
                potential_causes.setdefault("Gastrointestinal infection", 0)
                potential_causes["Gastrointestinal infection"] += 1
                potential_causes.setdefault("Food poisoning", 0)
                potential_causes["Food poisoning"] += 0.8
                potential_causes.setdefault("Migraine", 0)
                potential_causes["Migraine"] += 0.4

            if "chest pain" in symptom:
                potential_causes.setdefault("Musculoskeletal strain", 0)
                potential_causes["Musculoskeletal strain"] += 0.7
                potential_causes.setdefault("Acid reflux", 0)
                potential_causes["Acid reflux"] += 0.6
                potential_causes.setdefault("Anxiety", 0)
                potential_causes["Anxiety"] += 0.5
                potential_causes.setdefault("Cardiovascular concerns", 0)
                potential_causes["Cardiovascular concerns"] += 1

            if "shortness of breath" in symptom:
                potential_causes.setdefault("Anxiety", 0)
                potential_causes["Anxiety"] += 0.6
                potential_causes.setdefault("Respiratory infection", 0)
                potential_causes["Respiratory infection"] += 0.8
                potential_causes.setdefault("Asthma", 0)
                potential_causes["Asthma"] += 0.7

            if "dizziness" in symptom:
                potential_causes.setdefault("Dehydration", 0)
                potential_causes["Dehydration"] += 0.8
                potential_causes.setdefault("Low blood pressure", 0)
                potential_causes["Low blood pressure"] += 0.7
                potential_causes.setdefault("Inner ear issues", 0)
                potential_causes["Inner ear issues"] += 0.6

        # Sort causes by score and format for output
        sorted_causes = sorted(potential_causes.items(), key=lambda x: x[1], reverse=True)
        analysis["potential_causes"] = [{"cause": cause, "likelihood": min(score * 20, 100)} for cause, score in sorted_causes[:4]]

        # Determine severity assessment
        high_concern_symptoms = ["chest pain", "shortness of breath", "severe headache", "high fever"]
        moderate_concern_symptoms = ["persistent cough", "dizziness", "fatigue", "nausea"]

        high_concern_count = sum(1 for symptom in symptoms_lower if any(concern in symptom for concern in high_concern_symptoms))
        moderate_concern_count = sum(1 for symptom in symptoms_lower if any(concern in symptom for concern in moderate_concern_symptoms))

        if high_concern_count >= 1:
            analysis["severity_assessment"] = "These symptoms include some that may require prompt medical attention. While many causes are benign, some may need evaluation."
            analysis["time_sensitivity"] = "Consider seeking medical attention soon, especially if symptoms are severe or worsening."
        elif moderate_concern_count >= 2:
            analysis["severity_assessment"] = "These symptoms suggest a moderate level of concern. While they could be due to self-limiting conditions, they might benefit from evaluation."
            analysis["time_sensitivity"] = "Monitor symptoms for 24-48 hours; seek care if they persist or worsen."
        else:
            analysis["severity_assessment"] = "Based on the reported symptoms, the concern level appears relatively mild. Many such symptoms resolve with self-care."
            analysis["time_sensitivity"] = "These symptoms can typically be monitored at home initially, with medical attention if they persist beyond 5-7 days."

        # Recommended actions
        if high_concern_count >= 1:
            analysis["recommended_actions"] = [
                "Contact a healthcare provider within 24 hours",
                "Document symptom timing, severity, and aggravating factors",
                "Avoid strenuous activity until evaluated",
                "Have someone stay with you if symptoms are severe",
                "Prepare a list of all medications you're taking"
            ]
        elif moderate_concern_count >= 2:
            analysis["recommended_actions"] = [
                "Rest and monitor symptoms for 1-2 days",
                "Stay well-hydrated",
                "Consider over-the-counter remedies appropriate for your symptoms",
                "Schedule an appointment if not improving in 48 hours",
                "Keep a symptom diary noting timing and triggers"
            ]
        else:
            analysis["recommended_actions"] = [
                "Rest and adequate hydration",
                "Over-the-counter remedies appropriate for specific symptoms",
                "Monitor for changes in symptom intensity or new symptoms",
                "Basic self-care appropriate to specific symptoms",
                "Schedule routine appointment if symptoms persist beyond a week"
            ]

        # Lifestyle recommendations
        analysis["lifestyle_recommendations"] = [
            "Ensure adequate sleep (7-9 hours nightly)",
            "Stay well-hydrated throughout the day",
            "Maintain a balanced diet rich in fruits and vegetables",
            "Practice stress-reduction techniques",
            "Moderate physical activity as tolerated"
        ]

        # Age-specific recommendations
        if user_age > 65:
            analysis["recommended_actions"].append("Given your age, consider a lower threshold for seeking medical care")

        return analysis
