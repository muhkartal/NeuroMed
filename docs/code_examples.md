# Code Examples

This document provides practical code examples for integrating with and utilizing the MedExplain AI Pro API across various programming languages and scenarios.

## Table of Contents

1. [Authentication](#authentication)
2. [Symptom Analysis](#symptom-analysis)
3. [Medical Literature Search](#medical-literature-search)
4. [Health Records Management](#health-records-management)
5. [User Profile Management](#user-profile-management)
6. [Webhook Integration](#webhook-integration)
7. [Error Handling](#error-handling)
8. [Complete Applications](#complete-applications)

## Authentication

### Python

```python
import requests

API_KEY = "your_api_key"
BASE_URL = "https://api.medexplain.ai/v1"

def get_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def make_request(endpoint, method="GET", data=None, params=None):
    url = f"{BASE_URL}/{endpoint}"
    headers = get_headers()

    if method == "GET":
        response = requests.get(url, headers=headers, params=params)
    elif method == "POST":
        response = requests.post(url, headers=headers, json=data)
    elif method == "PATCH":
        response = requests.patch(url, headers=headers, json=data)
    elif method == "DELETE":
        response = requests.delete(url, headers=headers)

    response.raise_for_status()
    return response.json()
```

### JavaScript

```javascript
const API_KEY = "your_api_key";
const BASE_URL = "https://api.medexplain.ai/v1";

async function makeRequest(endpoint, method = "GET", data = null, params = null) {
   const url = new URL(`${BASE_URL}/${endpoint}`);

   // Add query parameters if provided
   if (params) {
      Object.keys(params).forEach((key) => {
         if (params[key] !== null && params[key] !== undefined) {
            url.searchParams.append(key, params[key]);
         }
      });
   }

   const options = {
      method: method,
      headers: {
         Authorization: `Bearer ${API_KEY}`,
         "Content-Type": "application/json",
         Accept: "application/json",
      },
   };

   if (data && ["POST", "PATCH", "PUT"].includes(method)) {
      options.body = JSON.stringify(data);
   }

   const response = await fetch(url.toString(), options);

   if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error.message || "API request failed");
   }

   return response.json();
}
```

### Java

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Map;
import java.util.stream.Collectors;
import com.fasterxml.jackson.databind.ObjectMapper;

public class MedExplainClient {
    private static final String API_KEY = "your_api_key";
    private static final String BASE_URL = "https://api.medexplain.ai/v1";
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    public MedExplainClient() {
        this.httpClient = HttpClient.newHttpClient();
        this.objectMapper = new ObjectMapper();
    }

    public <T> T makeRequest(String endpoint, String method, Object data,
                             Map<String, String> params, Class<T> responseType)
                             throws Exception {
        // Build URL with query parameters
        String url = BASE_URL + "/" + endpoint;
        if (params != null && !params.isEmpty()) {
            String queryParams = params.entrySet().stream()
                .filter(e -> e.getValue() != null)
                .map(e -> e.getKey() + "=" + e.getValue())
                .collect(Collectors.joining("&"));
            url += "?" + queryParams;
        }

        // Build request
        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .header("Authorization", "Bearer " + API_KEY)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        // Set method and body if needed
        switch (method) {
            case "GET":
                requestBuilder.GET();
                break;
            case "POST":
                requestBuilder.POST(HttpRequest.BodyPublishers.ofString(
                    objectMapper.writeValueAsString(data)));
                break;
            case "PATCH":
                requestBuilder.method("PATCH", HttpRequest.BodyPublishers.ofString(
                    objectMapper.writeValueAsString(data)));
                break;
            case "DELETE":
                requestBuilder.DELETE();
                break;
        }

        // Send request and process response
        HttpResponse<String> response = httpClient.send(
            requestBuilder.build(),
            HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() >= 200 && response.statusCode() < 300) {
            return objectMapper.readValue(response.body(), responseType);
        } else {
            throw new RuntimeException("API request failed with status: " +
                response.statusCode() + ", body: " + response.body());
        }
    }
}
```

### PHP

```php
<?php

class MedExplainClient {
    private $apiKey;
    private $baseUrl;

    public function __construct($apiKey) {
        $this->apiKey = $apiKey;
        $this->baseUrl = "https://api.medexplain.ai/v1";
    }

    public function makeRequest($endpoint, $method = "GET", $data = null, $params = null) {
        // Build URL with query parameters
        $url = $this->baseUrl . "/" . $endpoint;
        if ($params) {
            $queryString = http_build_query($params);
            $url .= "?" . $queryString;
        }

        // Initialize curl session
        $curl = curl_init();

        // Set common curl options
        curl_setopt_array($curl, [
            CURLOPT_URL => $url,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                "Authorization: Bearer " . $this->apiKey,
                "Content-Type: application/json",
                "Accept: application/json"
            ]
        ]);

        // Set method-specific options
        switch ($method) {
            case "POST":
                curl_setopt($curl, CURLOPT_POST, true);
                if ($data) {
                    curl_setopt($curl, CURLOPT_POSTFIELDS, json_encode($data));
                }
                break;
            case "PATCH":
                curl_setopt($curl, CURLOPT_CUSTOMREQUEST, "PATCH");
                if ($data) {
                    curl_setopt($curl, CURLOPT_POSTFIELDS, json_encode($data));
                }
                break;
            case "DELETE":
                curl_setopt($curl, CURLOPT_CUSTOMREQUEST, "DELETE");
                break;
        }

        // Execute request and get response
        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);

        // Handle errors
        if (curl_errno($curl)) {
            throw new Exception(curl_error($curl));
        }

        curl_close($curl);

        // Parse and return JSON response
        $responseData = json_decode($response, true);

        if ($httpCode >= 200 && $httpCode < 300) {
            return $responseData;
        } else {
            throw new Exception(
                "API request failed with status: " . $httpCode .
                ", message: " . ($responseData['error']['message'] ?? "Unknown error")
            );
        }
    }
}
```

## Symptom Analysis

### Python - Basic Symptom Analysis

```python
def analyze_symptoms(symptoms, demographics):
    """
    Analyze symptoms using the MedExplain API.

    Args:
        symptoms (list): List of symptom dictionaries
        demographics (dict): User demographic information

    Returns:
        dict: Analysis results
    """
    endpoint = "symptom-analysis"
    data = {
        "symptoms": symptoms,
        "demographics": demographics,
        "options": {
            "max_results": 5,
            "min_confidence": 0.3,
            "include_literature": True
        }
    }

    return make_request(endpoint, method="POST", data=data)

# Example usage
if __name__ == "__main__":
    symptoms = [
        {
            "name": "headache",
            "severity": "moderate",
            "duration": "3 days",
            "description": "Throbbing pain on right side of head"
        },
        {
            "name": "nausea",
            "severity": "mild",
            "duration": "1 day",
            "description": "Occasional nausea especially after eating"
        }
    ]

    demographics = {
        "age": 35,
        "sex": "female",
        "existing_conditions": ["allergies"]
    }

    try:
        results = analyze_symptoms(symptoms, demographics)

        # Print top condition
        top_condition = results["potential_conditions"][0]
        print(f"Top condition: {top_condition['name']} (confidence: {top_condition['confidence']})")

        # Print recommended actions
        print("\nRecommended actions:")
        for action in results["recommendations"]["suggested_actions"]:
            print(f"- {action['description']}")

    except Exception as e:
        print(f"Error: {e}")
```

### JavaScript - Natural Language Symptom Input

```javascript
async function analyzeNaturalLanguageSymptoms(description, userContext) {
   try {
      const endpoint = "symptom-analysis/natural-language";

      const data = {
         description: description,
         user_context: userContext,
         options: {
            extract_symptoms: true,
            max_results: 5,
            include_literature: true,
         },
      };

      const results = await makeRequest(endpoint, "POST", data);

      // Process and display results
      const extractedSymptoms = results.extracted_symptoms;
      console.log("Extracted symptoms:");
      extractedSymptoms.forEach((symptom) => {
         console.log(`- ${symptom.name} (${symptom.severity || "unknown severity"})`);
      });

      console.log("\nPotential conditions:");
      results.potential_conditions.forEach((condition) => {
         console.log(`- ${condition.name} (${(condition.confidence * 100).toFixed(1)}%)`);
      });

      return results;
   } catch (error) {
      console.error("Error analyzing symptoms:", error);
      throw error;
   }
}

// Example usage
const description =
   "I've had a throbbing headache on the right side of my head for the past 3 days. It gets worse with bright light, and I've also felt slightly nauseated, especially in the morning.";

const userContext = {
   age: 35,
   sex: "female",
   existing_conditions: ["seasonal allergies"],
   current_medications: ["loratadine 10mg daily"],
};

analyzeNaturalLanguageSymptoms(description, userContext)
   .then((results) => {
      // Further processing can be done here
      console.log("\nSeverity assessment:", results.recommendations.severity_assessment);
   })
   .catch((error) => {
      console.error("Analysis failed:", error.message);
   });
```

### Java - Bulk Symptom Analysis

```java
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;

public class BulkSymptomAnalyzer {
    private final MedExplainClient client;

    public BulkSymptomAnalyzer(MedExplainClient client) {
        this.client = client;
    }

    public List<Map<String, Object>> analyzeBulkSymptoms(
            List<Map<String, Object>> symptomEntries,
            Map<String, Object> demographics) throws Exception {

        List<Map<String, Object>> results = new ArrayList<>();

        for (Map<String, Object> entry : symptomEntries) {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> symptoms = (List<Map<String, Object>>) entry.get("symptoms");

            Map<String, Object> requestData = new HashMap<>();
            requestData.put("symptoms", symptoms);
            requestData.put("demographics", demographics);
            requestData.put("options", Map.of(
                "max_results", 3,
                "min_confidence", 0.3,
                "include_literature", false
            ));

            Map<String, Object> result = client.makeRequest(
                "symptom-analysis",
                "POST",
                requestData,
                null,
                Map.class
            );

            // Add date information from the original entry
            result.put("date", entry.get("date"));
            results.add(result);
        }

        return results;
    }

    // Example usage
    public static void main(String[] args) {
        try {
            MedExplainClient client = new MedExplainClient();
            BulkSymptomAnalyzer analyzer = new BulkSymptomAnalyzer(client);

            // Prepare symptom entries
            List<Map<String, Object>> symptomEntries = new ArrayList<>();

            // Entry 1
            Map<String, Object> entry1 = new HashMap<>();
            entry1.put("date", "2023-06-15");

            List<Map<String, Object>> symptoms1 = new ArrayList<>();
            symptoms1.add(Map.of(
                "name", "headache",
                "severity", "moderate",
                "duration", "3 days"
            ));
            symptoms1.add(Map.of(
                "name", "nausea",
                "severity", "mild",
                "duration", "1 day"
            ));
            entry1.put("symptoms", symptoms1);

            symptomEntries.add(entry1);

            // Entry 2
            Map<String, Object> entry2 = new HashMap<>();
            entry2.put("date", "2023-06-20");

            List<Map<String, Object>> symptoms2 = new ArrayList<>();
            symptoms2.add(Map.of(
                "name", "cough",
                "severity", "moderate",
                "duration", "5 days"
            ));
            symptoms2.add(Map.of(
                "name", "fever",
                "severity", "mild",
                "duration", "2 days"
            ));
            entry2.put("symptoms", symptoms2);

            symptomEntries.add(entry2);

            // Demographics
            Map<String, Object> demographics = Map.of(
                "age", 35,
                "sex", "female",
                "existing_conditions", List.of("allergies")
            );

            // Analyze all entries
            List<Map<String, Object>> results = analyzer.analyzeBulkSymptoms(
                symptomEntries, demographics);

            // Process results
            for (Map<String, Object> result : results) {
                System.out.println("Date: " + result.get("date"));

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> conditions =
                    (List<Map<String, Object>>) result.get("potential_conditions");

                System.out.println("Top conditions:");
                for (Map<String, Object> condition : conditions) {
                    System.out.println("- " + condition.get("name") +
                        " (confidence: " + condition.get("confidence") + ")");
                }
                System.out.println();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## Medical Literature Search

### Python - Literature Search with Filters

```python
def search_medical_literature(query, condition=None, year_range=None, limit=10):
    """
    Search medical literature with filters.

    Args:
        query (str): Search query
        condition (str, optional): Filter by condition
        year_range (tuple, optional): Min and max year range (e.g., (2020, 2023))
        limit (int, optional): Maximum results to return

    Returns:
        dict: Search results
    """
    endpoint = "literature/search"
    params = {
        "query": query,
        "limit": limit
    }

    if condition:
        params["condition"] = condition

    if year_range:
        params["year_min"] = year_range[0]
        params["year_max"] = year_range[1]

    return make_request(endpoint, params=params)

def format_literature_results(results):
    """Format and print literature search results."""
    print(f"Found {results['total_results']} results. Showing {len(results['results'])}:")

    for i, item in enumerate(results['results'], 1):
        print(f"\n{i}. {item['title']}")
        print(f"   {item['journal']} ({item['year']})")
        print(f"   Authors: {item['authors']}")
        print(f"   Conditions: {', '.join(item['conditions'])}")
        print(f"   Summary: {item['summary'][:150]}...")

# Example usage
if __name__ == "__main__":
    try:
        # Search for migraine treatments published in the last 3 years
        results = search_medical_literature(
            query="treatment options",
            condition="migraine",
            year_range=(2020, 2023),
            limit=5
        )

        format_literature_results(results)

        # Show pagination info
        pagination = results['pagination']
        if pagination['offset'] + pagination['limit'] < results['total_results']:
            next_offset = pagination['next_offset']
            print(f"\nMore results available. Use offset={next_offset} to see the next page.")

    except Exception as e:
        print(f"Error searching literature: {e}")
```

### JavaScript - Get Literature Details and Related Articles

```javascript
async function getLiteratureDetails(literatureId) {
   try {
      const endpoint = `literature/${literatureId}`;
      return await makeRequest(endpoint);
   } catch (error) {
      console.error(`Error fetching literature details: ${error.message}`);
      throw error;
   }
}

async function getRelatedLiterature(literatureId, limit = 5) {
   try {
      const endpoint = `literature/${literatureId}/related`;
      const params = { limit };
      return await makeRequest(endpoint, "GET", null, params);
   } catch (error) {
      console.error(`Error fetching related literature: ${error.message}`);
      throw error;
   }
}

async function displayLiteratureWithRelated(literatureId) {
   try {
      // Get main article details
      const article = await getLiteratureDetails(literatureId);

      // Display main article
      console.log(`Title: ${article.title}`);
      console.log(`Journal: ${article.journal} (${article.year})`);
      console.log(`Authors: ${article.authors.map((a) => a.name).join(", ")}`);
      console.log(`\nAbstract: ${article.abstract}\n`);

      // Get and display related articles
      const related = await getRelatedLiterature(literatureId);

      console.log("Related Articles:");
      related.results.forEach((item, index) => {
         console.log(`${index + 1}. ${item.title} (${item.year})`);
         console.log(`   ${item.journal}`);
         console.log(`   Relevance: ${(item.relevance_score * 100).toFixed(1)}%\n`);
      });

      return { main: article, related: related.results };
   } catch (error) {
      console.error(`Error in literature display: ${error.message}`);
      throw error;
   }
}

// Example usage
const literatureId = "lit_1a2b3c4d5e";

displayLiteratureWithRelated(literatureId)
   .then((result) => {
      console.log(`Successfully retrieved article and ${result.related.length} related articles`);
   })
   .catch((error) => {
      console.error("Failed to display literature:", error);
   });
```

## Health Records Management

### Python - Creating and Retrieving Health Records

```python
def create_health_record(record_type, date, details):
    """
    Create a new health record.

    Args:
        record_type (str): Type of record (symptom, medication, etc.)
        date (str): ISO format date string
        details (dict): Record-specific details

    Returns:
        dict: Created record data
    """
    endpoint = "health-records"
    data = {
        "record_type": record_type,
        "date": date,
        "details": details
    }

    return make_request(endpoint, method="POST", data=data)

def get_health_records(record_type=None, date_range=None, limit=20):
    """
    Retrieve health records with optional filters.

    Args:
        record_type (str, optional): Filter by record type
        date_range (tuple, optional): Start and end dates (ISO format)
        limit (int, optional): Maximum records to return

    Returns:
        dict: Health records data
    """
    endpoint = "health-records"
    params = {"limit": limit}

    if record_type:
        params["record_type"] = record_type

    if date_range:
        params["start_date"] = date_range[0]
        params["end_date"] = date_range[1]

    return make_request(endpoint, params=params)

# Example usage
if __name__ == "__main__":
    from datetime import datetime, timedelta

    # Create a symptom record
    try:
        today = datetime.now().isoformat()

        symptom_record = create_health_record(
            record_type="symptom",
            date=today,
            details={
                "symptoms": [
                    {
                        "name": "headache",
                        "severity": "moderate",
                        "duration": "3 hours"
                    },
                    {
                        "name": "fatigue",
                        "severity": "mild",
                        "duration": "all day"
                    }
                ],
                "triggers": ["stress", "lack of sleep"],
                "notes": "Started after lunch meeting"
            }
        )

        print(f"Created symptom record with ID: {symptom_record['record_id']}")

        # Create a medication record
        medication_record = create_health_record(
            record_type="medication",
            date=today,
            details={
                "medication": "Ibuprofen",
                "dosage": "400mg",
                "time_taken": datetime.now().strftime("%H:%M"),
                "notes": "Taken for headache"
            }
        )

        print(f"Created medication record with ID: {medication_record['record_id']}")

        # Retrieve records from the last week
        one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()

        records = get_health_records(
            date_range=(one_week_ago, datetime.now().isoformat())
        )

        print(f"\nRetrieved {len(records['records'])} records from the past week:")

        for record in records["records"]:
            print(f"- {record['record_type']} on {record['date']}")

    except Exception as e:
        print(f"Error managing health records: {e}")
```

### JavaScript - Advanced Health Record Filtering and Analysis

```javascript
async function getFilteredHealthRecords(options = {}) {
   try {
      const endpoint = "health-records";
      const params = { ...options };

      return await makeRequest(endpoint, "GET", null, params);
   } catch (error) {
      console.error(`Error fetching health records: ${error.message}`);
      throw error;
   }
}

// Analyze symptom frequency over time
async function analyzeSymptomFrequency(symptomName, timeframe = 90) {
   try {
      // Calculate date range
      const endDate = new Date().toISOString();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - timeframe);

      // Get all symptom records in the timeframe
      const records = await getFilteredHealthRecords({
         record_type: "symptom",
         start_date: startDate.toISOString(),
         end_date: endDate,
         limit: 100,
      });

      // Process records to find the target symptom
      const symptomFrequency = {};

      records.records.forEach((record) => {
         // Extract date (just the day part)
         const date = record.date.split("T")[0];

         // Extract symptoms from the record
         const recordSymptoms = record.details.symptoms || [];

         // Check if the target symptom is in this record
         const hasTargetSymptom = recordSymptoms.some(
            (s) => s.name.toLowerCase() === symptomName.toLowerCase()
         );

         // Count occurrences by date
         if (hasTargetSymptom) {
            if (symptomFrequency[date]) {
               symptomFrequency[date]++;
            } else {
               symptomFrequency[date] = 1;
            }
         }
      });

      // Convert to array for easier visualization
      const frequencyArray = Object.entries(symptomFrequency).map(([date, count]) => ({
         date,
         count,
      }));

      // Sort by date
      frequencyArray.sort((a, b) => new Date(a.date) - new Date(b.date));

      return {
         symptom: symptomName,
         timeframe,
         totalOccurrences: frequencyArray.reduce((sum, item) => sum + item.count, 0),
         frequencyData: frequencyArray,
      };
   } catch (error) {
      console.error(`Error analyzing symptom frequency: ${error.message}`);
      throw error;
   }
}

// Example usage
async function generateSymptomReport() {
   try {
      // Analyze headache frequency over the last 30 days
      const headacheAnalysis = await analyzeSymptomFrequency("headache", 30);

      console.log(`Headache Frequency Analysis (Last ${headacheAnalysis.timeframe} days)`);
      console.log(`Total occurrences: ${headacheAnalysis.totalOccurrences}`);

      console.log("\nFrequency by date:");
      headacheAnalysis.frequencyData.forEach((item) => {
         console.log(`${item.date}: ${item.count} occurrence(s)`);
      });

      // Calculate average occurrences per week
      const avgPerWeek = (
         headacheAnalysis.totalOccurrences /
         (headacheAnalysis.timeframe / 7)
      ).toFixed(1);
      console.log(`\nAverage occurrences per week: ${avgPerWeek}`);

      return headacheAnalysis;
   } catch (error) {
      console.error("Failed to generate symptom report:", error);
      throw error;
   }
}

// Run the analysis
generateSymptomReport()
   .then((report) => {
      console.log("Report generated successfully");
   })
   .catch((error) => {
      console.error("Report generation failed:", error);
   });
```

## User Profile Management

### Python - Updating User Profile

```python
def get_user_profile():
    """
    Retrieve the current user profile.

    Returns:
        dict: User profile data
    """
    endpoint = "user/profile"
    return make_request(endpoint)

def update_user_profile(update_data):
    """
    Update the user profile.

    Args:
        update_data (dict): Data to update in the profile

    Returns:
        dict: Updated profile data
    """
    endpoint = "user/profile"
    return make_request(endpoint, method="PATCH", data=update_data)

# Example usage
if __name__ == "__main__":
    try:
        # Get current profile
        profile = get_user_profile()

        print("Current Profile:")
        print(f"Email: {profile['email']}")
        print(f"Age: {profile.get('demographics', {}).get('age', 'Not set')}")
        print(f"Existing conditions: {', '.join(profile.get('medical_history', {}).get('existing_conditions', ['None']))}")

        # Update profile
        update_data = {
            "demographics": {
                "weight": 68,  # in kg
                "height": 175  # in cm
            },
            "medical_history": {
                "medications": ["Vitamin D", "Fish Oil"],
            }
        }

        updated_profile = update_user_profile(update_data)

        print("\nProfile updated successfully!")
        print(f"Weight updated to: {updated_profile['demographics']['weight']} kg")
        print(f"Height updated to: {updated_profile['demographics']['height']} cm")
        print(f"Medications updated to: {', '.join(updated_profile['medical_history']['medications'])}")

    except Exception as e:
        print(f"Error managing user profile: {e}")
```

### Java - Complete Profile Management

```java
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

public class UserProfileManager {
    private final MedExplainClient client;

    public UserProfileManager(MedExplainClient client) {
        this.client = client;
    }

    public Map<String, Object> getUserProfile() throws Exception {
        return client.makeRequest("user/profile", "GET", null, null, Map.class);
    }

    public Map<String, Object> updateUserProfile(Map<String, Object> updateData) throws Exception {
        return client.makeRequest("user/profile", "PATCH", updateData, null, Map.class);
    }

    public Map<String, Object> addMedicalCondition(String condition) throws Exception {
        // Get current profile
        Map<String, Object> profile = getUserProfile();

        // Get current medical history
        @SuppressWarnings("unchecked")
        Map<String, Object> medicalHistory =
            (Map<String, Object>) profile.getOrDefault("medical_history", new HashMap<>());

        // Get current conditions
        @SuppressWarnings("unchecked")
        List<String> conditions =
            (List<String>) medicalHistory.getOrDefault("existing_conditions", new ArrayList<>());

        // Add new condition if not already present
        if (!conditions.contains(condition)) {
            conditions.add(condition);
        }

        // Create update data
        Map<String, Object> updateData = new HashMap<>();
        Map<String, Object> newMedicalHistory = new HashMap<>();
        newMedicalHistory.put("existing_conditions", conditions);
        updateData.put("medical_history", newMedicalHistory);

        // Update profile
        return updateUserProfile(updateData);
    }

    public Map<String, Object> updateMedications(List<String> medications) throws Exception {
        Map<String, Object> updateData = new HashMap<>();
        Map<String, Object> newMedicalHistory = new HashMap<>();
        newMedicalHistory.put("medications", medications);
        updateData.put("medical_history", newMedicalHistory);

        return updateUserProfile(updateData);
    }

    public Map<String, Object> updateDemographics(
            Integer age, String sex, Integer height, Integer weight) throws Exception {

        Map<String, Object> demographics = new HashMap<>();

        if (age != null) demographics.put("age", age);
        if (sex != null) demographics.put("sex", sex);
        if (height != null) demographics.put("height", height);
        if (weight != null) demographics.put("weight", weight);

        Map<String, Object> updateData = new HashMap<>();
        updateData.put("demographics", demographics);

        return updateUserProfile(updateData);
    }

    public Map<String, Object> updatePreferences(
            Boolean notificationEmail, Boolean dataSharingEnabled) throws Exception {

        Map<String, Object> preferences = new HashMap<>();

        if (notificationEmail != null) preferences.put("notification_email", notificationEmail);
        if (dataSharingEnabled != null) preferences.put("data_sharing", dataSharingEnabled);

        Map<String, Object> updateData = new HashMap<>();
        updateData.put("preferences", preferences);

        return updateUserProfile(updateData);
    }

    // Example usage
    public static void main(String[] args) {
        try {
            MedExplainClient client = new MedExplainClient();
            UserProfileManager profileManager = new UserProfileManager(client);

            // Get current profile
            Map<String, Object> profile = profileManager.getUserProfile();
            System.out.println("Current profile: " + profile);

            // Update demographics
            profileManager.updateDemographics(35, "female", 165, 63);
            System.out.println("Demographics updated.");

            // Add a medical condition
            profileManager.addMedicalCondition("Seasonal allergies");
            System.out.println("Medical condition added.");

            // Update medications
            List<String> medications = new ArrayList<>();
            medications.add("Cetirizine 10mg");
            medications.add("Vitamin D 2000IU");
            profileManager.updateMedications(medications);
            System.out.println("Medications updated.");

            // Update preferences
            profileManager.updatePreferences(true, false);
            System.out.println("Preferences updated.");

            // Get updated profile
            Map<String, Object> updatedProfile = profileManager.getUserProfile();
            System.out.println("Updated profile: " + updatedProfile);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## Webhook Integration

### JavaScript - Webhook Handler (Node.js)

```javascript
const express = require("express");
const bodyParser = require("body-parser");
const crypto = require("crypto");

const app = express();
app.use(bodyParser.json());

const WEBHOOK_SECRET = "your_webhook_secret";

// Verify webhook signature
function verifySignature(req) {
   const signature = req.headers["x-medexplain-signature"];
   if (!signature) {
      return false;
   }

   const payload = JSON.stringify(req.body);
   const computed = crypto.createHmac("sha256", WEBHOOK_SECRET).update(payload).digest("hex");

   return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(computed));
}

// Handle analysis.completed webhook
app.post("/webhooks/analysis-completed", (req, res) => {
   // Verify webhook signature
   if (!verifySignature(req)) {
      console.error("Invalid webhook signature");
      return res.status(401).json({ error: "Invalid signature" });
   }

   // Process the webhook data
   const { event_type, timestamp, data } = req.body;

   if (event_type !== "analysis.completed") {
      return res.status(400).json({ error: "Unexpected event type" });
   }

   console.log(`Received analysis completion at ${timestamp}`);
   console.log(`Analysis ID: ${data.analysis_id}`);
   console.log(`Top condition: ${data.top_condition} (${data.confidence})`);

   // Store analysis result in database (implementation depends on your database)
   storeAnalysisResult(data)
      .then(() => {
         console.log("Analysis result stored successfully");
      })
      .catch((error) => {
         console.error("Error storing analysis result:", error);
      });

   // Acknowledge receipt of the webhook
   res.status(200).json({ received: true });
});

// Handle record.created webhook
app.post("/webhooks/record-created", (req, res) => {
   if (!verifySignature(req)) {
      return res.status(401).json({ error: "Invalid signature" });
   }

   const { event_type, timestamp, data } = req.body;

   if (event_type !== "record.created") {
      return res.status(400).json({ error: "Unexpected event type" });
   }

   console.log(`Received record creation at ${timestamp}`);
   console.log(`Record ID: ${data.record_id}`);
   console.log(`Record type: ${data.record_type}`);

   // Process record creation (implementation specific)
   processNewRecord(data)
      .then(() => {
         console.log("New record processed successfully");
      })
      .catch((error) => {
         console.error("Error processing new record:", error);
      });

   res.status(200).json({ received: true });
});

// Placeholder functions (implement based on your specific needs)
async function storeAnalysisResult(data) {
   // Implementation depends on your database and data model
   // This would typically store the analysis result in your database
   // and perform any necessary updates or notifications
}

async function processNewRecord(data) {
   // Implementation specific to how you handle new health records
   // This might include updating statistics, triggering notifications,
   // or syncing with other systems
}

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
   console.log(`Webhook handler running on port ${PORT}`);
});
```

### Python - Webhook Handler (Flask)

```python
from flask import Flask, request, jsonify
import hmac
import hashlib
import json
import time
from datetime import datetime

app = Flask(__name__)

# Your webhook secret from the MedExplain Developer Portal
WEBHOOK_SECRET = "your_webhook_secret"

def verify_signature(payload, signature):
    """Verify the webhook signature."""
    computed = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(computed, signature)

@app.route('/webhooks/medexplain', methods=['POST'])
def handle_webhook():
    # Get the signature from headers
    signature = request.headers.get('X-MedExplain-Signature')
    if not signature:
        return jsonify({"error": "Missing signature"}), 401

    # Get the raw payload
    payload = request.data.decode('utf-8')

    # Verify signature
    if not verify_signature(payload, signature):
        return jsonify({"error": "Invalid signature"}), 401

    # Parse the JSON data
    data = json.loads(payload)

    # Extract common fields
    event_type = data.get('event_type')
    timestamp = data.get('timestamp')
    event_data = data.get('data', {})

    print(f"Received webhook: {event_type} at {timestamp}")

    # Process based on event type
    if event_type == 'analysis.completed':
        process_analysis_completed(event_data, timestamp)
    elif event_type == 'record.created':
        process_record_created(event_data, timestamp)
    elif event_type == 'record.updated':
        process_record_updated(event_data, timestamp)
    elif event_type == 'user.updated':
        process_user_updated(event_data, timestamp)
    else:
        print(f"Unknown event type: {event_type}")

    # Always acknowledge receipt
    return jsonify({"received": True, "timestamp": int(time.time())}), 200

def process_analysis_completed(data, timestamp):
    """Process an analysis.completed webhook event."""
    analysis_id = data.get('analysis_id')
    user_id = data.get('user_id')
    top_condition = data.get('top_condition')
    confidence = data.get('confidence')

    print(f"Analysis completed for user {user_id}")
    print(f"Top condition: {top_condition} (confidence: {confidence})")

    # Example: Store in database, send notification, etc.
    # db.save_analysis_result(analysis_id, user_id, top_condition, confidence, timestamp)

    # Example: Check if high confidence result requires notification
    if confidence > 0.85:
        print(f"High confidence result detected: {top_condition}")
        # notify_healthcare_provider(user_id, analysis_id, top_condition)

def process_record_created(data, timestamp):
    """Process a record.created webhook event."""
    record_id = data.get('record_id')
    user_id = data.get('user_id')
    record_type = data.get('record_type')

    print(f"New {record_type} record created for user {user_id}: {record_id}")

    # Example: Update analytics, sync with other systems, etc.
    # update_user_health_stats(user_id)

def process_record_updated(data, timestamp):
    """Process a record.updated webhook event."""
    record_id = data.get('record_id')
    user_id = data.get('user_id')

    print(f"Record updated for user {user_id}: {record_id}")

    # Example: Handle record updates
    # refresh_cached_health_data(user_id)

def process_user_updated(data, timestamp):
    """Process a user.updated webhook event."""
    user_id = data.get('user_id')
    updated_fields = data.get('updated_fields', [])

    print(f"User profile updated for {user_id}")
    print(f"Updated fields: {', '.join(updated_fields)}")

    # Example: Handle profile updates
    # if 'demographics' in updated_fields:
    #     recalculate_risk_factors(user_id)

if __name__ == '__main__':
    # Run the webhook handler
    app.run(port=5000, debug=True)
```

## Error Handling

### Python - Comprehensive Error Handling

```python
import requests
import time
import logging
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("medexplain_client")

class MedExplainError(Exception):
    """Base exception for MedExplain API errors."""
    def __init__(self, message, status_code=None, error_code=None, error_details=None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.error_details = error_details
        super().__init__(self.message)

class RateLimitError(MedExplainError):
    """Exception raised when rate limit is exceeded."""
    pass

class AuthenticationError(MedExplainError):
    """Exception raised for authentication failures."""
    pass

class ResourceNotFoundError(MedExplainError):
    """Exception raised when a requested resource doesn't exist."""
    pass

class ValidationError(MedExplainError):
    """Exception raised for invalid request parameters."""
    pass

class ServerError(MedExplainError):
    """Exception raised for server-side errors."""
    pass

class MedExplainClient:
    """Client for interacting with the MedExplain API with robust error handling."""

    def __init__(self, api_key: str, base_url: str = "https://api.medexplain.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def _handle_error_response(self, response: requests.Response) -> None:
        """Process error responses and raise appropriate exceptions."""
        status_code = response.status_code

        try:
            error_data = response.json().get("error", {})
            error_code = error_data.get("code")
            error_message = error_data.get("message", "Unknown error")
            error_details = error_data.get("details")
        except (ValueError, KeyError):
            error_code = None
            error_message = response.text or f"HTTP error {status_code}"
            error_details = None

        if status_code == 401:
            raise AuthenticationError(
                f"Authentication failed: {error_message}",
                status_code, error_code, error_details
            )
        elif status_code == 404:
            raise ResourceNotFoundError(
                f"Resource not found: {error_message}",
                status_code, error_code, error_details
            )
        elif status_code == 400:
            raise ValidationError(
                f"Invalid request: {error_message}",
                status_code, error_code, error_details
            )
        elif status_code == 429:
            # Extract rate limit headers if available
            reset_time = response.headers.get("X-RateLimit-Reset")
            if reset_time:
                try:
                    reset_time = int(reset_time)
                    reset_seconds = max(0, reset_time - int(time.time()))
                    error_message += f" (Rate limit resets in {reset_seconds} seconds)"
                except (ValueError, TypeError):
                    pass

            raise RateLimitError(
                f"Rate limit exceeded: {error_message}",
                status_code, error_code, error_details
            )
        elif 500 <= status_code < 600:
            raise ServerError(
                f"Server error: {error_message}",
                status_code, error_code, error_details
            )
        else:
            raise MedExplainError(
                f"API error ({status_code}): {error_message}",
                status_code, error_code, error_details
            )

    def make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_on_server_error: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to the MedExplain API with retry logic and error handling.

        Args:
            endpoint: API endpoint (without base URL)
            method: HTTP method (GET, POST, PATCH, DELETE)
            data: Request body data for POST/PATCH
            params: Query parameters
            retry_on_server_error: Whether to retry on server errors

        Returns:
            Response data as dictionary

        Raises:
            AuthenticationError: Authentication failed
            ResourceNotFoundError: Resource not found
            ValidationError: Invalid request parameters
            RateLimitError: Rate limit exceeded
            ServerError: Server-side error
            MedExplainError: Other API errors
            requests.RequestException: Network or connection errors
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        retry_count = 0
        last_exception = None

        while retry_count < self.max_retries:
            try:
                logger.debug(f"Making {method} request to {url}")

                if method == "GET":
                    response = self.session.get(url, params=params)
                elif method == "POST":
                    response = self.session.post(url, json=data, params=params)
                elif method == "PATCH":
                    response = self.session.patch(url, json=data, params=params)
                elif method == "DELETE":
                    response = self.session.delete(url, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Log rate limit info
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = response.headers['X-RateLimit-Remaining']
                    logger.debug(f"Rate limit remaining: {remaining}")

                # Check if request was successful
                if response.status_code >= 200 and response.status_code < 300:
                    return response.json()

                # For rate limit errors, check if we should wait and retry
                if response.status_code == 429:
                    reset_time = response.headers.get("X-RateLimit-Reset")
                    if reset_time:
                        try:
                            reset_time = int(reset_time)
                            current_time = int(time.time())
                            wait_time = max(0, reset_time - current_time) + 1

                            if wait_time <= 30:  # Only wait if reset is within 30 seconds
                                logger.info(f"Rate limit hit. Waiting {wait_time} seconds before retry.")
                                time.sleep(wait_time)
                                retry_count += 1
                                continue
                        except (ValueError, TypeError):
                            pass

                # For server errors, retry with exponential backoff
                if 500 <= response.status_code < 600 and retry_on_server_error:
                    wait_time = self.retry_delay * (2 ** retry_count)
                    logger.warning(f"Server error {response.status_code}. Retrying in {wait_time} seconds.")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue

                # Handle error response
                self._handle_error_response(response)

            except (requests.RequestException, ConnectionError) as e:
                # Network or connection errors
                last_exception = e
                wait_time = self.retry_delay * (2 ** retry_count)
                logger.warning(f"Network error: {e}. Retrying in {wait_time} seconds.")
                time.sleep(wait_time)
                retry_count += 1

        # If we get here, all retries failed
        if last_exception:
            logger.error(f"Request failed after {self.max_retries} retries: {last_exception}")
            raise last_exception

        raise MedExplainError(f"Request failed after {self.max_retries} retries")

    # Symptom Analysis Methods
    def analyze_symptoms(self, symptoms, demographics=None, options=None):
        """Analyze symptoms with robust error handling."""
        try:
            data = {
                "symptoms": symptoms,
                "demographics": demographics or {},
                "options": options or {}
            }
            return self.make_request("symptom-analysis", method="POST", data=data)
        except ValidationError as e:
            # Handle validation errors specifically
            if e.error_code == "invalid_symptom_format":
                # Attempt to fix symptom format and retry
                logger.info("Attempting to fix symptom format and retry")
                fixed_symptoms = self._fix_symptom_format(symptoms)
                data["symptoms"] = fixed_symptoms
                return self.make_request("symptom-analysis", method="POST", data=data)
            else:
                raise

    def _fix_symptom_format(self, symptoms):
        """Attempt to fix common symptom format issues."""
        fixed_symptoms = []

        for symptom in symptoms:
            if isinstance(symptom, str):
                # Convert string to proper format
                fixed_symptoms.append({"name": symptom, "severity": "moderate"})
            elif isinstance(symptom, dict):
                # Ensure required fields are present
                fixed_symptom = symptom.copy()
                if "name" not in fixed_symptom:
                    continue  # Skip invalid symptoms
                if "severity" not in fixed_symptom:
                    fixed_symptom["severity"] = "moderate"
                fixed_symptoms.append(fixed_symptom)

        return fixed_symptoms

# Example usage
if __name__ == "__main__":
    client = MedExplainClient(api_key="your_api_key")

    try:
        # Example with correct format
        symptoms = [
            {"name": "headache", "severity": "moderate", "duration": "3 days"},
            {"name": "nausea", "severity": "mild"}
        ]

        results = client.analyze_symptoms(symptoms, {"age": 35, "sex": "female"})
        print("Analysis successful!")

        # Example with incorrect format that will be fixed
        malformatted_symptoms = ["headache", "nausea"]

        results = client.analyze_symptoms(malformatted_symptoms, {"age": 35, "sex": "female"})
        print("Analysis with fixed format successful!")

    except AuthenticationError as e:
        print(f"Authentication error: {e}")
        print("Please check your API key.")

    except ResourceNotFoundError as e:
        print(f"Resource not found: {e}")

    except ValidationError as e:
        print(f"Validation error: {e}")
        if e.error_details:
            print(f"Details: {e.error_details}")

    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        print("Please try again later or contact support to increase your rate limit.")

    except ServerError as e:
        print(f"Server error: {e}")
        print("This is likely a temporary issue. Please try again later.")

    except MedExplainError as e:
        print(f"API error: {e}")

    except requests.RequestException as e:
        print(f"Network error: {e}")
        print("Please check your internet connection.")

    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Complete Applications

### React - Symptom Analyzer Component

```javascript
import React, { useState, useEffect } from "react";
import "./SymptomAnalyzer.css";

// API client utility
const API_KEY = "your_api_key";
const BASE_URL = "https://api.medexplain.ai/v1";

async function makeApiRequest(endpoint, method = "GET", data = null, params = null) {
   // API request implementation (see Authentication section)
   // ...
}

const SymptomAnalyzer = ({ userProfile }) => {
   // State management
   const [symptoms, setSymptoms] = useState([]);
   const [inputMode, setInputMode] = useState("structured"); // 'structured', 'text', or 'voice'
   const [textInput, setTextInput] = useState("");
   const [isRecording, setIsRecording] = useState(false);
   const [isLoading, setIsLoading] = useState(false);
   const [results, setResults] = useState(null);
   const [error, setError] = useState(null);
   const [availableSymptoms, setAvailableSymptoms] = useState([]);
   const [newSymptom, setNewSymptom] = useState({ name: "", severity: "moderate", duration: "" });

   // Fetch available symptoms on component mount
   useEffect(() => {
      const fetchSymptoms = async () => {
         try {
            const response = await makeApiRequest("reference/symptoms");
            setAvailableSymptoms(response.symptoms || []);
         } catch (err) {
            console.error("Error fetching available symptoms:", err);
            setError("Unable to load symptom reference data");
         }
      };

      fetchSymptoms();
   }, []);

   // Handle structured symptom input
   const handleAddSymptom = () => {
      if (!newSymptom.name) return;

      setSymptoms([...symptoms, { ...newSymptom }]);
      setNewSymptom({ name: "", severity: "moderate", duration: "" });
   };

   const handleRemoveSymptom = (index) => {
      const updatedSymptoms = [...symptoms];
      updatedSymptoms.splice(index, 1);
      setSymptoms(updatedSymptoms);
   };

   const handleSymptomChange = (e) => {
      const { name, value } = e.target;
      setNewSymptom({ ...newSymptom, [name]: value });
   };

   // Handle text input
   const handleTextInputChange = (e) => {
      setTextInput(e.target.value);
   };

   // Handle voice input (simplified - would need a real speech recognition implementation)
   const toggleRecording = () => {
      if (isRecording) {
         // Stop recording and process result
         setIsRecording(false);
         // In a real implementation, this would come from speech recognition
         const transcribedText = "I've had a headache for 3 days and some mild nausea";
         setTextInput(transcribedText);
      } else {
         // Start recording
         setIsRecording(true);
         setTextInput("");
         // In a real implementation, this would initialize speech recognition
      }
   };

   // Process symptoms and submit analysis request
   const handleAnalyze = async () => {
      try {
         setIsLoading(true);
         setError(null);

         let analysisData;

         if (inputMode === "structured") {
            // Use structured symptoms input
            if (symptoms.length === 0) {
               setError("Please add at least one symptom");
               setIsLoading(false);
               return;
            }

            analysisData = {
               symptoms: symptoms,
               demographics: {
                  age: userProfile?.demographics?.age || null,
                  sex: userProfile?.demographics?.sex || null,
                  existing_conditions: userProfile?.medical_history?.existing_conditions || [],
               },
               options: {
                  max_results: 5,
                  min_confidence: 0.3,
                  include_literature: true,
               },
            };

            const results = await makeApiRequest("symptom-analysis", "POST", analysisData);
            setResults(results);
         } else {
            // Use text or voice input (they both end up as text)
            if (!textInput.trim()) {
               setError("Please enter or speak your symptoms");
               setIsLoading(false);
               return;
            }

            analysisData = {
               description: textInput,
               user_context: {
                  age: userProfile?.demographics?.age || null,
                  sex: userProfile?.demographics?.sex || null,
                  existing_conditions: userProfile?.medical_history?.existing_conditions || [],
               },
               options: {
                  extract_symptoms: true,
                  max_results: 5,
                  include_literature: true,
               },
            };

            const results = await makeApiRequest(
               "symptom-analysis/natural-language",
               "POST",
               analysisData
            );
            setResults(results);
         }
      } catch (err) {
         console.error("Error analyzing symptoms:", err);
         setError(err.message || "An error occurred during analysis");
      } finally {
         setIsLoading(false);
      }
   };

   // Handle saving analysis to health records
   const handleSaveAnalysis = async () => {
      if (!results) return;

      try {
         setIsLoading(true);

         const recordData = {
            record_type: "symptom_analysis",
            date: new Date().toISOString(),
            details: {
               analysis_id: results.analysis_id,
               symptoms: inputMode === "structured" ? symptoms : results.extracted_symptoms,
               potential_conditions: results.potential_conditions.map((c) => ({
                  name: c.name,
                  confidence: c.confidence,
               })),
               notes: textInput || "Structured symptom entry",
            },
         };

         await makeApiRequest("health-records", "POST", recordData);

         alert("Analysis saved to your health records");
      } catch (err) {
         console.error("Error saving analysis:", err);
         setError(err.message || "Failed to save analysis to health records");
      } finally {
         setIsLoading(false);
      }
   };

   // Render the component
   return (
      <div className="symptom-analyzer-container">
         <h2>Symptom Analyzer</h2>

         {/* Input mode selector */}
         <div className="input-mode-selector">
            <button
               className={inputMode === "structured" ? "active" : ""}
               onClick={() => setInputMode("structured")}
            >
               Structured Input
            </button>
            <button
               className={inputMode === "text" ? "active" : ""}
               onClick={() => setInputMode("text")}
            >
               Text Description
            </button>
            <button
               className={inputMode === "voice" ? "active" : ""}
               onClick={() => setInputMode("voice")}
            >
               Voice Input
            </button>
         </div>

         {/* Structured input */}
         {inputMode === "structured" && (
            <div className="structured-input">
               <h3>Add Your Symptoms</h3>

               <div className="symptom-form">
                  <div className="form-row">
                     <label>
                        Symptom:
                        <select name="name" value={newSymptom.name} onChange={handleSymptomChange}>
                           <option value="">Select a symptom</option>
                           {availableSymptoms.map((s) => (
                              <option key={s.id} value={s.id}>
                                 {s.name}
                              </option>
                           ))}
                        </select>
                     </label>

                     <label>
                        Severity:
                        <select
                           name="severity"
                           value={newSymptom.severity}
                           onChange={handleSymptomChange}
                        >
                           <option value="mild">Mild</option>
                           <option value="moderate">Moderate</option>
                           <option value="severe">Severe</option>
                        </select>
                     </label>

                     <label>
                        Duration:
                        <input
                           type="text"
                           name="duration"
                           placeholder="e.g., 3 days"
                           value={newSymptom.duration}
                           onChange={handleSymptomChange}
                        />
                     </label>
                  </div>

                  <button
                     onClick={handleAddSymptom}
                     disabled={!newSymptom.name}
                     className="add-button"
                  >
                     Add Symptom
                  </button>
               </div>

               {/* Display added symptoms */}
               {symptoms.length > 0 && (
                  <div className="added-symptoms">
                     <h4>Added Symptoms:</h4>
                     <ul>
                        {symptoms.map((s, index) => (
                           <li key={index}>
                              <span className="symptom-name">
                                 {availableSymptoms.find((as) => as.id === s.name)?.name || s.name}
                              </span>
                              <span className="symptom-details">
                                 {s.severity}, {s.duration || "unknown duration"}
                              </span>
                              <button
                                 onClick={() => handleRemoveSymptom(index)}
                                 className="remove-button"
                              >
                                 ✕
                              </button>
                           </li>
                        ))}
                     </ul>
                  </div>
               )}
            </div>
         )}

         {/* Text input */}
         {inputMode === "text" && (
            <div className="text-input">
               <h3>Describe Your Symptoms</h3>
               <textarea
                  placeholder="Describe your symptoms in detail, including severity, duration, and any factors that make them better or worse..."
                  value={textInput}
                  onChange={handleTextInputChange}
                  rows={5}
               />
            </div>
         )}

         {/* Voice input */}
         {inputMode === "voice" && (
            <div className="voice-input">
               <h3>Speak Your Symptoms</h3>
               <div className="voice-controls">
                  <button
                     onClick={toggleRecording}
                     className={`record-button ${isRecording ? "recording" : ""}`}
                  >
                     {isRecording ? "Stop Recording" : "Start Recording"}
                  </button>

                  {isRecording && (
                     <div className="recording-indicator">
                        Recording... Speak clearly and describe your symptoms
                     </div>
                  )}
               </div>

               {textInput && (
                  <div className="transcription">
                     <h4>Transcription:</h4>
                     <p>{textInput}</p>
                     <button onClick={() => setTextInput("")}>Clear</button>
                  </div>
               )}
            </div>
         )}

         {/* Analysis button */}
         <div className="action-buttons">
            <button
               onClick={handleAnalyze}
               disabled={
                  isLoading ||
                  (inputMode === "structured" && symptoms.length === 0) ||
                  ((inputMode === "text" || inputMode === "voice") && !textInput.trim())
               }
               className="analyze-button"
            >
               {isLoading ? "Analyzing..." : "Analyze Symptoms"}
            </button>
         </div>

         {/* Error display */}
         {error && <div className="error-message">{error}</div>}

         {/* Results display */}
         {results && (
            <div className="analysis-results">
               <h3>Analysis Results</h3>

               {/* Extracted symptoms (for text/voice input) */}
               {results.extracted_symptoms && (
                  <div className="extracted-symptoms">
                     <h4>Identified Symptoms:</h4>
                     <ul>
                        {results.extracted_symptoms.map((s, i) => (
                           <li key={i}>
                              <span className="symptom-name">{s.name}</span>
                              {s.severity && <span className="symptom-severity">{s.severity}</span>}
                              {s.duration && <span className="symptom-duration">{s.duration}</span>}
                           </li>
                        ))}
                     </ul>
                  </div>
               )}

               {/* Potential conditions */}
               <div className="potential-conditions">
                  <h4>Potential Conditions:</h4>
                  <div className="conditions-list">
                     {results.potential_conditions.map((condition, i) => (
                        <div key={i} className="condition-card">
                           <div className="condition-header">
                              <h5>{condition.name}</h5>
                              <div className="confidence-score">
                                 <span className="confidence-label">Confidence:</span>
                                 <div className="confidence-bar">
                                    <div
                                       className="confidence-fill"
                                       style={{ width: `${condition.confidence * 100}%` }}
                                    />
                                 </div>
                                 <span className="confidence-percent">
                                    {Math.round(condition.confidence * 100)}%
                                 </span>
                              </div>
                           </div>

                           <p className="condition-description">{condition.description}</p>

                           <div className="matching-symptoms">
                              <h6>Matching Symptoms:</h6>
                              <ul>
                                 {condition.matching_symptoms.map((s, j) => (
                                    <li key={j}>
                                       {s.name}
                                       <span className="relevance-score">
                                          (Relevance: {Math.round(s.relevance * 100)}%)
                                       </span>
                                    </li>
                                 ))}
                              </ul>
                           </div>

                           {condition.literature_references &&
                              condition.literature_references.length > 0 && (
                                 <div className="literature-references">
                                    <h6>Related Literature:</h6>
                                    <ul>
                                       {condition.literature_references.map((ref, j) => (
                                          <li key={j}>
                                             <a
                                                href={ref.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                             >
                                                {ref.title} ({ref.year})
                                             </a>
                                          </li>
                                       ))}
                                    </ul>
                                 </div>
                              )}
                        </div>
                     ))}
                  </div>
               </div>

               {/* Recommendations */}
               {results.recommendations && (
                  <div className="recommendations">
                     <h4>Recommendations:</h4>
                     <p className="severity-assessment">
                        <strong>Severity Assessment:</strong>{" "}
                        {results.recommendations.severity_assessment}
                     </p>

                     <ul className="suggested-actions">
                        {results.recommendations.suggested_actions.map((action, i) => (
                           <li key={i}>
                              <strong>{action.action.replace("_", " ")}:</strong>{" "}
                              {action.description}
                           </li>
                        ))}
                     </ul>
                  </div>
               )}

               {/* Save button */}
               <div className="result-actions">
                  <button onClick={handleSaveAnalysis} disabled={isLoading} className="save-button">
                     {isLoading ? "Saving..." : "Save to Health Records"}
                  </button>
               </div>
            </div>
         )}
      </div>
   );
};

export default SymptomAnalyzer;
```

---

Developed by Muhammad Ibrahim Kartal | [kartal.dev](https://kartal.dev)
