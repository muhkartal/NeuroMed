# MedExplain AI Pro - API Documentation

This document provides comprehensive information about the MedExplain AI Pro API, which allows developers to integrate medical analysis capabilities into their applications.

## API Overview

The MedExplain AI Pro API enables programmatic access to symptom analysis, medical literature retrieval, health data management, and other core functionality of the MedExplain platform.

## Base URL

All API requests should be made to the following base URL:

```
https://api.medexplain.ai/v1/
```

## Authentication

### API Keys

Authentication is performed using API keys. To obtain an API key:

1. Create an account on the MedExplain Developer Portal
2. Navigate to API Keys in your account settings
3. Generate a new API key with appropriate permissions

Include your API key in all requests using the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

### Rate Limiting

The API implements the following rate limits:

| Plan         | Requests per minute | Requests per day |
| ------------ | ------------------- | ---------------- |
| Basic        | 60                  | 5,000            |
| Professional | 300                 | 30,000           |
| Enterprise   | 1,000               | 100,000          |

Rate limit headers are included in all API responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1620000000
```

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of requests:

| Code | Description                             |
| ---- | --------------------------------------- |
| 200  | OK - Request succeeded                  |
| 400  | Bad Request - Invalid parameters        |
| 401  | Unauthorized - Authentication failed    |
| 403  | Forbidden - Insufficient permissions    |
| 404  | Not Found - Resource does not exist     |
| 429  | Too Many Requests - Rate limit exceeded |
| 500  | Internal Server Error - Server error    |

Error responses include a JSON object with details:

```json
{
   "error": {
      "code": "invalid_parameter",
      "message": "The parameter 'symptoms' is required",
      "details": {
         "field": "symptoms",
         "issue": "missing_required_field"
      }
   }
}
```

## Endpoints

### Symptom Analysis

#### Analyze Symptoms

```
POST /symptom-analysis
```

Analyzes symptoms and returns potential conditions with confidence scores.

**Request Body:**

```json
{
   "symptoms": [
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
   ],
   "demographics": {
      "age": 35,
      "sex": "female",
      "existing_conditions": ["allergies"]
   },
   "options": {
      "max_results": 5,
      "min_confidence": 0.3,
      "include_literature": true
   }
}
```

**Response:**

```json
{
   "analysis_id": "anl_8f7d6e5c4b3a2",
   "processed_symptoms": [
      {
         "name": "headache",
         "normalized_name": "cephalgia",
         "body_system": "neurological",
         "severity_normalized": 0.6
      },
      {
         "name": "nausea",
         "normalized_name": "nausea",
         "body_system": "digestive",
         "severity_normalized": 0.3
      }
   ],
   "potential_conditions": [
      {
         "name": "Migraine",
         "confidence": 0.87,
         "description": "A neurological condition characterized by recurrent headaches...",
         "matching_symptoms": [
            { "name": "headache", "relevance": 0.9 },
            { "name": "nausea", "relevance": 0.7 }
         ],
         "literature_references": [
            {
               "title": "Novel CGRP Receptor Antagonists for Migraine Treatment",
               "journal": "Neurology Today",
               "year": 2023,
               "url": "https://api.medexplain.ai/literature/9876543"
            }
         ]
      },
      {
         "name": "Tension Headache",
         "confidence": 0.65,
         "description": "A common form of primary headache characterized by...",
         "matching_symptoms": [
            { "name": "headache", "relevance": 0.8 },
            { "name": "nausea", "relevance": 0.2 }
         ],
         "literature_references": [
            {
               "title": "Clinical Management of Tension-type Headache",
               "journal": "Pain Management",
               "year": 2022,
               "url": "https://api.medexplain.ai/literature/5432109"
            }
         ]
      }
   ],
   "recommendations": {
      "severity_assessment": "moderate",
      "suggested_actions": [
         {
            "action": "monitor_symptoms",
            "description": "Keep track of headache patterns including triggers"
         },
         {
            "action": "consider_consultation",
            "description": "If symptoms persist beyond 7 days, consult healthcare provider"
         }
      ]
   }
}
```

#### Retrieve Analysis

```
GET /symptom-analysis/{analysis_id}
```

Retrieves a previously created symptom analysis.

**Parameters:**

| Name        | Type   | Description                        |
| ----------- | ------ | ---------------------------------- |
| analysis_id | string | The ID of the analysis to retrieve |

**Response:**

Returns the same structure as the POST /symptom-analysis endpoint.

### Medical Literature

#### Search Literature

```
GET /literature/search
```

Searches medical literature based on query parameters.

**Parameters:**

| Name      | Type    | Description                                           |
| --------- | ------- | ----------------------------------------------------- |
| query     | string  | Search terms                                          |
| condition | string  | (Optional) Filter by condition                        |
| year_min  | integer | (Optional) Minimum publication year                   |
| year_max  | integer | (Optional) Maximum publication year                   |
| limit     | integer | (Optional) Maximum results to return (default: 10)    |
| offset    | integer | (Optional) Results offset for pagination (default: 0) |

**Response:**

```json
{
   "total_results": 243,
   "results": [
      {
         "id": "lit_1a2b3c4d5e",
         "title": "Novel CGRP Receptor Antagonists for Migraine Treatment",
         "journal": "Neurology Today",
         "year": 2023,
         "authors": "Johnson K, Smith P, et al.",
         "summary": "This review examines recent developments in CGRP antagonists for migraine...",
         "conditions": ["Migraine"],
         "url": "https://api.medexplain.ai/literature/1a2b3c4d5e"
      },
      {
         "id": "lit_6f7g8h9i0j",
         "title": "Migraine and Sleep: Understanding the Relationship",
         "journal": "Journal of Sleep Medicine",
         "year": 2022,
         "authors": "Martinez A, Williams T, et al.",
         "summary": "This study investigates the bidirectional relationship between migraines and sleep disorders...",
         "conditions": ["Migraine", "Insomnia"],
         "url": "https://api.medexplain.ai/literature/6f7g8h9i0j"
      }
   ],
   "pagination": {
      "limit": 10,
      "offset": 0,
      "next_offset": 10
   }
}
```

#### Get Literature Details

```
GET /literature/{literature_id}
```

Retrieves detailed information about a specific literature item.

**Parameters:**

| Name          | Type   | Description                   |
| ------------- | ------ | ----------------------------- |
| literature_id | string | The ID of the literature item |

**Response:**

```json
{
   "id": "lit_1a2b3c4d5e",
   "title": "Novel CGRP Receptor Antagonists for Migraine Treatment",
   "journal": "Neurology Today",
   "volume": 45,
   "issue": 3,
   "pages": "123-145",
   "year": 2023,
   "doi": "10.1234/neurol.2023.1234",
   "authors": [
      {
         "name": "Johnson, K.",
         "affiliation": "University of Medical Research"
      },
      {
         "name": "Smith, P.",
         "affiliation": "Institute of Neurological Studies"
      }
   ],
   "abstract": "Recent advances in calcitonin gene-related peptide (CGRP) antagonists have shown promising results in migraine treatment...",
   "summary": "This review examines recent developments in CGRP antagonists for migraine, showing significant reduction in frequency and intensity of migraine attacks with fewer side effects than older treatments.",
   "conditions": ["Migraine"],
   "keywords": ["CGRP", "migraine", "headache", "treatment", "antagonists"],
   "references": [
      {
         "id": "lit_2b3c4d5e6f",
         "title": "The role of CGRP in migraine pathophysiology",
         "year": 2021
      }
   ],
   "full_text_url": "https://api.medexplain.ai/literature/1a2b3c4d5e/full-text"
}
```

### Health Records

#### Create Health Record

```
POST /health-records
```

Creates a new health record entry for a user.

**Request Body:**

```json
{
   "record_type": "symptom",
   "date": "2023-06-15T14:30:00Z",
   "details": {
      "symptoms": [
         {
            "name": "headache",
            "severity": "moderate",
            "duration": "3 hours"
         }
      ],
      "triggers": ["stress", "lack of sleep"],
      "notes": "Occurred after working late"
   }
}
```

**Response:**

```json
{
   "record_id": "rec_1a2b3c4d5e",
   "record_type": "symptom",
   "date": "2023-06-15T14:30:00Z",
   "created_at": "2023-06-15T15:00:00Z",
   "details": {
      "symptoms": [
         {
            "name": "headache",
            "severity": "moderate",
            "duration": "3 hours"
         }
      ],
      "triggers": ["stress", "lack of sleep"],
      "notes": "Occurred after working late"
   }
}
```

#### List Health Records

```
GET /health-records
```

Retrieves health records for the authenticated user.

**Parameters:**

| Name        | Type    | Description                                           |
| ----------- | ------- | ----------------------------------------------------- |
| record_type | string  | (Optional) Filter by record type                      |
| start_date  | string  | (Optional) Start date filter (ISO format)             |
| end_date    | string  | (Optional) End date filter (ISO format)               |
| limit       | integer | (Optional) Maximum results to return (default: 20)    |
| offset      | integer | (Optional) Results offset for pagination (default: 0) |

**Response:**

```json
{
   "total_records": 45,
   "records": [
      {
         "record_id": "rec_1a2b3c4d5e",
         "record_type": "symptom",
         "date": "2023-06-15T14:30:00Z",
         "created_at": "2023-06-15T15:00:00Z",
         "details": {
            "symptoms": [
               {
                  "name": "headache",
                  "severity": "moderate",
                  "duration": "3 hours"
               }
            ]
         }
      },
      {
         "record_id": "rec_2b3c4d5e6f",
         "record_type": "medication",
         "date": "2023-06-14T09:00:00Z",
         "created_at": "2023-06-14T09:15:00Z",
         "details": {
            "medication": "Ibuprofen",
            "dosage": "400mg",
            "notes": "Taken for headache"
         }
      }
   ],
   "pagination": {
      "limit": 20,
      "offset": 0,
      "next_offset": 20
   }
}
```

### User Management

#### Get User Profile

```
GET /user/profile
```

Retrieves the profile information for the authenticated user.

**Response:**

```json
{
   "user_id": "usr_1a2b3c4d5e",
   "email": "user@example.com",
   "created_at": "2023-01-15T10:30:00Z",
   "demographics": {
      "age": 35,
      "sex": "female",
      "height": 165,
      "weight": 65
   },
   "medical_history": {
      "existing_conditions": ["allergies", "asthma"],
      "medications": ["Albuterol", "Cetirizine"],
      "allergies": ["Penicillin"]
   },
   "preferences": {
      "notification_email": true,
      "data_sharing": false
   }
}
```

#### Update User Profile

```
PATCH /user/profile
```

Updates the profile information for the authenticated user.

**Request Body:**

```json
{
   "demographics": {
      "weight": 63
   },
   "medical_history": {
      "medications": ["Albuterol", "Cetirizine", "Vitamin D"]
   },
   "preferences": {
      "notification_email": false
   }
}
```

**Response:**

Returns the updated user profile using the same structure as GET /user/profile.

## Webhooks

MedExplain AI Pro supports webhooks for real-time notifications of events.

### Configuring Webhooks

To set up a webhook:

1. Navigate to the Developer Portal
2. Go to Webhooks in your account settings
3. Add a new webhook URL
4. Select the events you want to subscribe to

### Event Types

| Event Type         | Description                               |
| ------------------ | ----------------------------------------- |
| analysis.completed | A symptom analysis has completed          |
| record.created     | A new health record has been created      |
| record.updated     | A health record has been updated          |
| user.updated       | User profile information has been updated |

### Webhook Payload

```json
{
   "event_type": "analysis.completed",
   "timestamp": "2023-06-15T15:05:23Z",
   "data": {
      "analysis_id": "anl_8f7d6e5c4b3a2",
      "user_id": "usr_1a2b3c4d5e",
      "top_condition": "Migraine",
      "confidence": 0.87
   }
}
```

### Webhook Security

Webhook requests include a signature in the X-MedExplain-Signature header for verification:

1. Retrieve your webhook secret from the Developer Portal
2. Compute an HMAC-SHA256 hash of the request body using your secret
3. Compare this with the value in the X-MedExplain-Signature header

## SDK Support

We provide official SDKs for the following languages:

-  Python: [GitHub](https://github.com/medexplain/medexplain-python)
-  JavaScript/Node.js: [GitHub](https://github.com/medexplain/medexplain-node)
-  Java: [GitHub](https://github.com/medexplain/medexplain-java)
-  Ruby: [GitHub](https://github.com/medexplain/medexplain-ruby)

## Best Practices

1. **Rate Limiting**: Implement backoff strategies to handle rate limit errors
2. **Caching**: Cache responses where appropriate to reduce API calls
3. **Error Handling**: Process error responses and retry transient failures
4. **Webhooks**: Use webhooks for real-time updates rather than polling
5. **Security**: Keep your API keys secure and rotate them periodically

## Support

For API support:

-  Developer Documentation: [docs.medexplain.ai/api](https://docs.medexplain.ai/api)
-  API Status: [status.medexplain.ai](https://status.medexplain.ai)
-  Developer Forum: [community.medexplain.ai/developers](https://community.medexplain.ai/developers)
-  Email Support: api-support@medexplain.ai

---

Developed by Muhammad Ibrahim Kartal | [kartal.dev](https://kartal.dev)

Copyright © 2025 MedExplain AI Pro. All rights reserved.
