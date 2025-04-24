"""
Health Data Analyzer for MedExplain AI Pro.

This module provides advanced analytics for health data, identifying
patterns, trends, and insights from user symptom history.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import calendar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logger
logger = logging.getLogger(__name__)

class HealthDataAnalyzer:
    """
    Advanced analytics for health data, including trend analysis,
    pattern recognition, and predictive insights.

    This class is responsible for:
    - Analyzing symptom patterns over time
    - Identifying correlations between symptoms
    - Generating health risk assessments
    - Providing predictive insights
    """

    def __init__(self, health_history: Optional[List[Dict[str, Any]]] = None,
                user_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the health data analyzer.

        Args:
            health_history: List of health history entries
            user_profile: User profile data
        """
        self.health_history = health_history or []
        self.user_profile = user_profile or {}

        logger.info("HealthDataAnalyzer initialized")

    def prepare_symptom_dataframe(self) -> pd.DataFrame:
        """
        Prepare a DataFrame from health history for analysis.

        Returns:
            DataFrame with dates and symptom indicators
        """
        if not self.health_history:
            logger.warning("No health history data available for analysis")
            return pd.DataFrame()

        # Get all unique symptoms
        all_symptoms = set()
        for entry in self.health_history:
            all_symptoms.update(entry.get("symptoms", []))

        logger.debug("Found %d unique symptoms in health history", len(all_symptoms))

        # Create a list of dictionaries for the DataFrame
        data = []

        for entry in self.health_history:
            date_str = entry.get("date", "")
            if not date_str:
                continue

            try:
                # Convert string date to datetime
                if isinstance(date_str, str):
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:
                    date_obj = date_str

                # Create a dictionary for this entry
                entry_dict = {"date": date_obj}

                # Add symptom indicators (1 if present, 0 if not)
                for symptom in all_symptoms:
                    entry_dict[symptom] = 1 if symptom in entry.get("symptoms", []) else 0

                data.append(entry_dict)
            except Exception as e:
                logger.error("Error processing entry date: %s", str(e))
                continue

        # Create DataFrame
        df = pd.DataFrame(data)

        # Sort by date
        if not df.empty and "date" in df.columns:
            df = df.sort_values("date")

        logger.debug("Created DataFrame with %d rows and %d columns",
                   df.shape[0] if not df.empty else 0,
                   df.shape[1] if not df.empty else 0)

        return df

    def analyze_symptom_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in symptoms over time.

        Returns:
            Dictionary of pattern analysis results
        """
        df = self.prepare_symptom_dataframe()

        if df.empty or len(df.columns) <= 1:
            logger.warning("Insufficient data for pattern analysis")
            return {"error": "Insufficient data for pattern analysis"}

        # Extract features (symptoms) excluding date
        symptom_cols = [col for col in df.columns if col != "date"]

        if not symptom_cols:
            logger.warning("No symptom data available for analysis")
            return {"error": "No symptom data available for analysis"}

        # Get frequency of each symptom
        symptom_frequency = {}
        for symptom in symptom_cols:
            symptom_frequency[symptom] = df[symptom].sum()

        # Sort by frequency (descending)
        symptom_frequency = dict(sorted(
            symptom_frequency.items(),
            key=lambda item: item[1],
            reverse=True
        ))

        logger.debug("Calculated frequency for %d symptoms", len(symptom_frequency))

        # Calculate correlations between symptoms
        symptom_correlations = {}

        # Only calculate if we have at least 2 symptoms with data
        if len(symptom_cols) >= 2 and df.shape[0] >= 3:
            # Calculate correlation matrix
            corr_matrix = df[symptom_cols].corr()

            # Extract top correlations
            for i, symptom1 in enumerate(symptom_cols):
                for j, symptom2 in enumerate(symptom_cols):
                    if i < j:  # Only take each pair once
                        correlation = corr_matrix.loc[symptom1, symptom2]
                        if not np.isnan(correlation) and abs(correlation) > 0.3:
                            symptom_correlations[(symptom1, symptom2)] = correlation

            # Sort correlations by strength (absolute value, descending)
            symptom_correlations = dict(sorted(
                symptom_correlations.items(),
                key=lambda item: abs(item[1]),
                reverse=True
            ))

            logger.debug("Found %d significant symptom correlations", len(symptom_correlations))

        # Identify temporal patterns if enough data
        temporal_patterns = {}
        if df.shape[0] >= 5:  # Need at least a few data points
            # Add month and weekday columns
            df["month"] = df["date"].dt.month
            df["weekday"] = df["date"].dt.weekday

            # Check for monthly patterns
            monthly_data = {}
            for symptom in symptom_cols:
                # Group by month and calculate mean occurrence
                monthly_means = df.groupby("month")[symptom].mean()

                # Check if there's significant variation by month
                if monthly_means.std() > 0.15:
                    peak_month = monthly_means.idxmax()
                    peak_value = monthly_means.max()

                    monthly_data[symptom] = {
                        "peak_month": calendar.month_name[peak_month],
                        "peak_value": peak_value,
                        "monthly_values": monthly_means.to_dict()
                    }

            # Check for weekly patterns (by day of week)
            weekly_data = {}
            for symptom in symptom_cols:
                # Group by weekday and calculate mean occurrence
                weekday_means = df.groupby("weekday")[symptom].mean()

                # Check if there's significant variation by weekday
                if weekday_means.std() > 0.15:
                    peak_day = weekday_means.idxmax()
                    peak_value = weekday_means.max()

                    weekly_data[symptom] = {
                        "peak_day": calendar.day_name[peak_day],
                        "peak_value": peak_value,
                        "weekday_values": weekday_means.to_dict()
                    }

            temporal_patterns = {
                "monthly": monthly_data,
                "weekly": weekly_data
            }

            logger.debug("Identified temporal patterns: %d monthly, %d weekly",
                       len(monthly_data), len(weekly_data))

        # Try to identify clusters of symptoms if enough data
        clusters = {}
        if len(symptom_cols) >= 3 and df.shape[0] >= 5:
            try:
                # Scale the data
                scaler = StandardScaler()
                symptom_data = df[symptom_cols]
                scaled_data = scaler.fit_transform(symptom_data)

                # Apply PCA for dimensionality reduction
                pca = PCA(n_components=min(2, len(symptom_cols)))
                pca_result = pca.fit_transform(scaled_data)

                # Prepare for clustering
                X = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

                # Determine optimal number of clusters (simplified)
                optimal_k = min(3, len(symptom_cols))

                # Apply KMeans clustering
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                X["cluster"] = kmeans.fit_predict(X)

                # Get symptom importance for each cluster
                for cluster_id in range(optimal_k):
                    # Get original data for this cluster
                    cluster_mask = X["cluster"] == cluster_id
                    cluster_data = symptom_data.loc[cluster_mask]

                    # Calculate mean values for each symptom in this cluster
                    symptom_means = cluster_data.mean()

                    # Sort by mean (descending)
                    symptom_means = symptom_means.sort_values(ascending=False)

                    # Store top symptoms for this cluster
                    clusters[f"cluster_{cluster_id}"] = {
                        "top_symptoms": symptom_means[:3].index.tolist(),
                        "symptom_weights": symptom_means[:3].to_dict()
                    }

                logger.debug("Identified %d symptom clusters", optimal_k)
            except Exception as e:
                logger.error("Error in clustering: %s", str(e))
                pass

        return {
            "symptom_frequency": symptom_frequency,
            "symptom_correlations": {f"{s1}_{s2}": corr for (s1, s2), corr in symptom_correlations.items()},
            "temporal_patterns": temporal_patterns,
            "symptom_clusters": clusters
        }

    def generate_risk_assessment(self, health_data_manager=None) -> Dict[str, Any]:
        """
        Generate a risk assessment based on user health history and profile.

        Args:
            health_data_manager: Optional HealthDataManager for additional reference data

        Returns:
            Risk assessment results
        """
        # Base risk assessment
        risk_assessment = {
            "overall_risk": "low",
            "risk_factors": [],
            "protective_factors": [],
            "recommendations": []
        }

        # Check if we have enough data
        if not self.health_history or not self.user_profile:
            risk_assessment["overall_risk"] = "unknown"
            risk_assessment["risk_factors"].append("Insufficient health data for accurate assessment")
            risk_assessment["recommendations"].append("Complete your health profile and track symptoms regularly for more personalized insights")
            logger.warning("Insufficient data for risk assessment")
            return risk_assessment

        # Calculate risk score
        risk_score = 0

        # Extract age and apply age-related risk factors
        age = self.user_profile.get("age", 0)
        if age > 65:
            risk_assessment["risk_factors"].append("Age over 65")
            risk_score += 15
        elif age > 50:
            risk_assessment["risk_factors"].append("Age over 50")
            risk_score += 10
        elif age < 18:
            risk_assessment["risk_factors"].append("Pediatric age group")
            risk_score += 5

        # Check for chronic conditions
        chronic_conditions = self.user_profile.get("chronic_conditions", [])
        high_risk_conditions = ["heart_disease", "diabetes", "hypertension", "copd", "asthma", "immunocompromised"]

        for condition in chronic_conditions:
            condition_lower = condition.lower()
            for high_risk in high_risk_conditions:
                if high_risk in condition_lower:
                    risk_assessment["risk_factors"].append(f"Pre-existing condition: {condition}")
                    risk_score += 15
                    break

        # Analyze symptoms from health history
        high_risk_symptoms = ["chest_pain", "shortness_of_breath", "severe_headache", "high_fever"]
        symptom_frequency = {}

        for entry in self.health_history:
            for symptom in entry.get("symptoms", []):
                if symptom not in symptom_frequency:
                    symptom_frequency[symptom] = 0
                symptom_frequency[symptom] += 1

                # Check for high-risk symptoms
                if symptom in high_risk_symptoms:
                    risk_assessment["risk_factors"].append(f"History of high-risk symptom: {symptom}")
                    risk_score += 10

        # Check for recurrent symptoms (appearing in >30% of checks)
        recurrent_threshold = max(3, len(self.health_history) * 0.3)
        recurrent_symptoms = [
            symptom for symptom, count in symptom_frequency.items()
            if count >= recurrent_threshold
        ]

        if recurrent_symptoms:
            risk_assessment["risk_factors"].append(f"Recurrent symptoms: {', '.join(recurrent_symptoms)}")
            risk_score += 5 * len(recurrent_symptoms)

        # Analyze protective factors
        # Regular exercise
        if self.user_profile.get("exercise_frequency", "").lower() in ["daily", "regular", "active"]:
            risk_assessment["protective_factors"].append("Regular physical activity")
            risk_score -= 10

        # Non-smoker
        if self.user_profile.get("smoking_status", "").lower() in ["never", "former", "non-smoker"]:
            risk_assessment["protective_factors"].append("Non-smoker")
            risk_score -= 10

        # Calculate overall risk based on risk score
        if risk_score <= 0:
            risk_assessment["overall_risk"] = "low"
        elif risk_score <= 25:
            risk_assessment["overall_risk"] = "moderate"
        else:
            risk_assessment["overall_risk"] = "high"

        # Add risk score to assessment
        risk_assessment["risk_score"] = risk_score

        # Generate recommendations
        if risk_assessment["overall_risk"] == "high":
            risk_assessment["recommendations"].append("Consider scheduling a comprehensive health check-up")

            if "Age over 65" in risk_assessment["risk_factors"]:
                risk_assessment["recommendations"].append("Regular preventive screenings are particularly important for your age group")

            if any("Pre-existing condition" in factor for factor in risk_assessment["risk_factors"]):
                risk_assessment["recommendations"].append("Regular monitoring of pre-existing conditions is recommended")

        elif risk_assessment["overall_risk"] == "moderate":
            risk_assessment["recommendations"].append("Consider discussing your symptoms with a healthcare provider at your next visit")
            risk_assessment["recommendations"].append("Monitor your symptoms and note any changes in frequency or severity")

        else:  # low risk
            risk_assessment["recommendations"].append("Continue healthy lifestyle habits")
            risk_assessment["recommendations"].append("Keep tracking your symptoms to maintain awareness of your health patterns")

        # Always add general recommendations
        risk_assessment["recommendations"].append("Stay hydrated and maintain a balanced diet")
        risk_assessment["recommendations"].append("Ensure you're getting adequate sleep regularly")

        logger.info("Generated risk assessment with overall risk: %s", risk_assessment["overall_risk"])

        return risk_assessment

    def generate_insights_report(self, health_data_manager=None) -> Dict[str, Any]:
        """
        Generate a comprehensive insights report based on user health data.

        Args:
            health_data_manager: Optional HealthDataManager for additional reference data

        Returns:
            Dictionary with sections of the insights report
        """
        # Initialize report
        report = {
            "summary": {},
            "symptom_patterns": {},
            "risk_assessment": {},
            "recommendations": [],
            "visualizations": {}
        }

        # Check if we have enough data
        if not self.health_history:
            report["summary"] = {
                "message": "Not enough health data to generate insights.",
                "recommendation": "Use the Symptom Analyzer to track your symptoms over time."
            }
            logger.warning("Insufficient data for insights report")
            return report

        # Generate symptom patterns analysis
        patterns = self.analyze_symptom_patterns()
        if "error" not in patterns:
            report["symptom_patterns"] = patterns

        # Generate risk assessment
        risk_assessment = self.generate_risk_assessment(health_data_manager)
        report["risk_assessment"] = risk_assessment

        # Generate recommendations based on patterns and risk assessment
        # Add recommendations from risk assessment
        report["recommendations"].extend(risk_assessment.get("recommendations", []))

        # Add pattern-based recommendations
        if "temporal_patterns" in patterns and patterns["temporal_patterns"]:
            monthly_patterns = patterns["temporal_patterns"].get("monthly", {})
            if monthly_patterns:
                for symptom, data in monthly_patterns.items():
                    peak_month = data.get("peak_month", "")
                    if peak_month:
                        report["recommendations"].append(
                            f"Consider taking preventive measures for {symptom} around {peak_month}."
                        )

        # Generate summary
        symptom_frequency = patterns.get("symptom_frequency", {})
        top_symptoms = list(symptom_frequency.keys())[:3] if symptom_frequency else []

        summary = {
            "total_checks": len(self.health_history),
            "unique_symptoms": len(symptom_frequency) if symptom_frequency else 0,
            "top_symptoms": top_symptoms,
            "overall_risk": risk_assessment.get("overall_risk", "unknown")
        }

        report["summary"] = summary

        logger.info("Generated insights report with %d recommendations", len(report["recommendations"]))

        return report

    def predict_future_symptoms(self) -> Dict[str, Any]:
        """
        Predict potential future symptoms based on patterns.

        Returns:
            Dictionary with prediction results
        """
        # This is a simplified predictive model that could be expanded
        # with more sophisticated ML techniques in a production system

        if not self.health_history or len(self.health_history) < 5:
            logger.warning("Insufficient data for symptom prediction")
            return {
                "error": "Insufficient data for prediction",
                "message": "Need more health history data to make predictions"
            }

        # Get patterns
        patterns = self.analyze_symptom_patterns()

        # Initialize results
        predictions = {
            "upcoming_risks": [],
            "seasonal_patterns": {},
            "confidence": "low"  # Default confidence
        }

        # Check for temporal patterns
        if "temporal_patterns" in patterns and patterns["temporal_patterns"]:
            temporal = patterns["temporal_patterns"]
            monthly = temporal.get("monthly", {})

            # Get current month
            current_month = datetime.now().month
            next_month = (current_month % 12) + 1  # Wrap around to January after December

            # Check for symptoms that peak in the upcoming month
            upcoming_risks = []

            for symptom, data in monthly.items():
                monthly_values = data.get("monthly_values", {})
                if next_month in monthly_values and monthly_values[next_month] > 0.5:
                    upcoming_risks.append({
                        "symptom": symptom,
                        "likelihood": monthly_values[next_month] * 100,
                        "reason": f"Historical pattern shows increased occurrence in {calendar.month_name[next_month]}"
                    })

            # Sort by likelihood
            upcoming_risks = sorted(upcoming_risks, key=lambda x: x["likelihood"], reverse=True)
            predictions["upcoming_risks"] = upcoming_risks

            # Add full seasonal patterns
            predictions["seasonal_patterns"] = {
                symptom: {
                    "peak_month": data.get("peak_month", ""),
                    "pattern": [data.get("monthly_values", {}).get(m, 0) * 100 for m in range(1, 13)]
                }
                for symptom, data in monthly.items()
            }

            # Determine confidence based on data quality
            if len(self.health_history) > 10 and len(upcoming_risks) > 0:
                predictions["confidence"] = "moderate"

            if len(self.health_history) > 20 and len(upcoming_risks) > 2:
                predictions["confidence"] = "high"

        logger.info("Generated symptom predictions with %d upcoming risks",
                   len(predictions["upcoming_risks"]))

        return predictions
