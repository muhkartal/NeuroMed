# Vital AI Pro - Developer Guide

This comprehensive guide is designed for developers who wish to contribute to or extend the Vital AI Pro platform. It covers the technical architecture, development environment setup, coding standards, and best practices.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Organization](#code-organization)
4. [Core Components](#core-components)
5. [Machine Learning Subsystem](#machine-learning-subsystem)
6. [UI Development](#ui-development)
7. [Testing Framework](#testing-framework)
8. [Contributing Guidelines](#contributing-guidelines)
9. [Performance Optimization](#performance-optimization)
10.   [Security Considerations](#security-considerations)

## Architecture Overview

Vital AI Pro is built on a modular architecture that separates concerns and enables scalable development. The system follows a layered approach:

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                          │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐   ┌───────────────────┐   │
│  │  Dashboard UI  │  │  Symptom UI   │   │  Medical Lit. UI  │   │
│  └───────────────┘  └───────────────┘   └───────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────┐
│                      Application Layer                           │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐   ┌───────────────────┐   │
│  │ Health Data   │  │ ML Pipeline   │   │ Analytics Engine  │   │
│  │    Manager    │  │               │   │                   │   │
│  └───────────────┘  └───────────────┘   └───────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────┐
│                       Service Layer                              │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐   ┌───────────────────┐   │
│  │ OpenAI Client │  │ Medical Data  │   │ User Profile      │   │
│  │               │  │ Service       │   │ Service           │   │
│  └───────────────┘  └───────────────┘   └───────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────┐
│                       Data Layer                                 │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐   ┌───────────────────┐   │
│  │ Local Storage │  │ Database      │   │ External APIs     │   │
│  │               │  │               │   │                   │   │
│  └───────────────┘  └───────────────┘   └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Components receive dependencies rather than creating them
3. **Interface-Based Design**: Components interact through well-defined interfaces
4. **Data Flow**: Unidirectional data flow for predictable state management
5. **Configurability**: External configuration for environment-specific settings

## Development Environment Setup

### Prerequisites

-  Python 3.8 or higher
-  Git
-  Virtual environment tool (venv or conda)
-  IDE with Python support (VS Code, PyCharm, etc.)
-  Docker (optional, for containerized development)

### Initial Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Vital-ai-pro.git
   cd Vital-ai-pro
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

4. Set up pre-commit hooks:

   ```bash
   pre-commit install
   ```

5. Configure environment variables:

   ```bash
   cp .env.example .env.dev
   ```

   Edit the `.env.dev` file with appropriate development values.

### Development Tools

-  **Code Linting**: flake8, pylint
-  **Code Formatting**: black, isort
-  **Type Checking**: mypy
-  **Testing**: pytest
-  **Documentation**: Sphinx
-  **Dependency Management**: pip-tools

### IDE Configuration

#### VS Code

Create a `.vscode/settings.json` file:

```json
{
   "python.linting.enabled": true,
   "python.linting.flake8Enabled": true,
   "python.linting.pylintEnabled": true,
   "python.formatting.provider": "black",
   "python.formatting.blackArgs": ["--line-length", "88"],
   "editor.formatOnSave": true,
   "python.testing.pytestEnabled": true,
   "python.testing.unittestEnabled": false,
   "python.testing.nosetestsEnabled": false,
   "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm

-  Enable Black as the formatter in Settings → Tools → Black
-  Configure pytest as the default test runner
-  Enable Pylint and Flake8 in Settings → Editor → Inspections → Python

## Code Organization

The Vital AI Pro codebase follows a modular structure:

```
Vital/                        # Main package
├── core/                          # Core functionality
│   ├── health_data_manager.py     # Health data management
│   ├── user_profile_manager.py    # User profile management
│   └── openai_client.py           # OpenAI API integration
│
├── ml/                            # Machine learning components
│   ├── symptom_predictor.py       # Symptom prediction model
│   ├── symptom_extractor.py       # NLP symptom extraction
│   └── risk_assessor.py           # Risk assessment model
│
├── analytics/                     # Analytics components
│   ├── health_analyzer.py         # Health data analyzer
│   └── visualization.py           # Data visualization utilities
│
├── ui/                            # User interface components
│   ├── dashboard.py               # Dashboard UI
│   ├── chat.py                    # Chat interface
│   ├── symptom_analyzer.py        # Symptom analyzer UI
│   ├── medical_literature.py      # Medical literature UI
│   ├── health_history.py          # Health history UI
│   └── settings.py                # Settings UI
│
├── utils/                         # Utility functions
│   ├── logging.py                 # Logging configuration
│   ├── validation.py              # Input validation
│   └── formatting.py              # Data formatting utilities
│
├── config.py                      # Configuration settings
└── app.py                         # Main application
```

### Module Responsibilities

-  **core**: Essential system functionality and services
-  **ml**: Machine learning models and data processing
-  **analytics**: Data analysis and visualization
-  **ui**: User interface components and pages
-  **utils**: Helper functions and utilities

## Core Components

### HealthDataManager

The `HealthDataManager` is responsible for storing, retrieving, and managing health-related data.

```python
# Example usage of HealthDataManager
from Vital.core.health_data_manager import HealthDataManager

health_manager = HealthDataManager()

# Adding a new symptom record
health_manager.add_symptom_record(
    user_id="user123",
    symptoms=["headache", "nausea"],
    severity="moderate",
    timestamp="2023-06-15T14:30:00Z",
    notes="After working late"
)

# Retrieving symptom history
symptom_history = health_manager.get_symptom_history(
    user_id="user123",
    start_date="2023-06-01T00:00:00Z",
    end_date="2023-06-30T23:59:59Z"
)
```

### UserProfileManager

The `UserProfileManager` handles user-specific data and preferences.

```python
# Example usage of UserProfileManager
from Vital.core.user_profile_manager import UserProfileManager

user_manager = UserProfileManager()

# Updating user profile
user_manager.update_profile(
    user_id="user123",
    demographics={
        "age": 35,
        "sex": "female",
        "weight": 65,
        "height": 165
    },
    medical_conditions=["allergies", "asthma"],
    medications=["Albuterol", "Cetirizine"]
)

# Getting user profile
user_profile = user_manager.get_profile(user_id="user123")
```

### OpenAIClient

The `OpenAIClient` provides an interface to OpenAI API services.

```python
# Example usage of OpenAIClient
from Vital.core.openai_client import OpenAIClient

openai_client = OpenAIClient(api_key="your_api_key")

# Analyzing symptoms using NLP
analysis_result = openai_client.analyze_symptoms(
    symptoms="I've had a throbbing headache on the right side of my head for 3 days, along with mild nausea.",
    user_context={
        "age": 35,
        "sex": "female",
        "medical_history": ["allergies"]
    }
)
```

## Machine Learning Subsystem

### ML Pipeline Architecture

The ML subsystem consists of several interconnected components:

1. **Data Preprocessing**: Cleans and normalizes input data
2. **Feature Extraction**: Extracts relevant features from symptoms and user data
3. **Model Inference**: Applies trained models to predict potential conditions
4. **Post-processing**: Formats results and calculates confidence scores

### Symptom Extractor

The `SymptomExtractor` identifies medical entities from natural language descriptions.

```python
# Example usage of SymptomExtractor
from Vital.ml.symptom_extractor import SymptomExtractor

extractor = SymptomExtractor()

# Extract symptoms from text
extracted_symptoms = extractor.extract_from_text(
    "I've had a throbbing headache on the right side of my head for 3 days, along with mild nausea."
)

# Result:
# [
#   {"symptom": "headache", "attributes": {"location": "right side", "duration": "3 days", "quality": "throbbing"}},
#   {"symptom": "nausea", "attributes": {"severity": "mild"}}
# ]
```

### Risk Assessor

The `RiskAssessor` evaluates the severity of symptoms and recommends actions.

```python
# Example usage of RiskAssessor
from Vital.ml.risk_assessor import RiskAssessor

risk_assessor = RiskAssessor()

# Assess risk level
risk_assessment = risk_assessor.assess(
    symptoms=[
        {"symptom": "headache", "severity": "moderate", "duration": "3 days"},
        {"symptom": "nausea", "severity": "mild", "duration": "1 day"}
    ],
    user_context={
        "age": 35,
        "sex": "female"
    }
)

# Result:
# {
#   "risk_level": "moderate",
#   "recommended_actions": ["monitor_symptoms", "consider_consultation_if_persistent"],
#   "urgency": "non-urgent"
# }
```

### Model Training Workflow

For contributors working on ML model improvements:

1. Data preparation scripts are in `scripts/data_preparation/`
2. Training scripts are in `scripts/model_training/`
3. Evaluation scripts are in `scripts/model_evaluation/`
4. Model artifacts should be saved to `data/ml_models/`

## UI Development

The UI is built using Streamlit, with a focus on modularity and reusability.

### UI Component Structure

Each UI component follows a consistent pattern:

```python
def render_component(data, options=None):
    """
    Render a UI component with the provided data.

    Args:
        data: The data to display
        options: Optional display options

    Returns:
        None (renders directly to Streamlit)
    """
    # Component implementation
    st.header("Component Title")

    # Display logic
    for item in data:
        st.write(item)

    # Interactive elements
    if st.button("Action Button"):
        # Handle action
        pass
```

### State Management

State is managed using Streamlit's session state:

```python
# Setting state
st.session_state.current_view = "dashboard"
st.session_state.selected_condition = "migraine"

# Reading state
if "current_view" in st.session_state:
    current_view = st.session_state.current_view
else:
    current_view = "dashboard"  # Default value
```

### Creating Custom Visualizations

For custom visualizations:

```python
def create_symptom_timeline(symptom_history):
    """
    Create a timeline visualization of symptoms.

    Args:
        symptom_history: List of symptom records

    Returns:
        fig: Plotly figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=[record["date"] for record in symptom_history],
            y=[record["severity"] for record in symptom_history],
            name="Symptom Severity"
        )
    )

    # Customize layout
    fig.update_layout(
        title="Symptom Severity Timeline",
        xaxis_title="Date",
        yaxis_title="Severity",
        legend_title="Legend"
    )

    return fig
```

## Testing Framework

Vital AI Pro uses pytest for testing. Tests are organized to mirror the module structure:

```
tests/
├── unit/                      # Unit tests
│   ├── core/                  # Tests for core components
│   ├── ml/                    # Tests for ML components
│   └── utils/                 # Tests for utilities
│
├── integration/               # Integration tests
│   ├── api/                   # API integration tests
│   └── ui/                    # UI integration tests
│
├── performance/               # Performance tests
└── conftest.py                # Test configuration
```

### Writing Tests

Each test file should follow this pattern:

```python
import pytest
from Vital.core.health_data_manager import HealthDataManager

class TestHealthDataManager:

    @pytest.fixture
    def health_manager(self):
        """Create a test instance of HealthDataManager."""
        return HealthDataManager()

    def test_add_symptom_record(self, health_manager):
        """Test adding a symptom record."""
        # Arrange
        user_id = "test_user"
        symptoms = ["headache"]

        # Act
        result = health_manager.add_symptom_record(
            user_id=user_id,
            symptoms=symptoms,
            severity="mild"
        )

        # Assert
        assert result["success"] is True
        assert result["record_id"] is not None

    def test_get_symptom_history(self, health_manager, mocker):
        """Test retrieving symptom history."""
        # Arrange
        user_id = "test_user"
        mock_data = [{"symptom": "headache", "severity": "mild"}]
        mocker.patch.object(
            health_manager,
            "_retrieve_data",
            return_value=mock_data
        )

        # Act
        history = health_manager.get_symptom_history(user_id=user_id)

        # Assert
        assert len(history) == 1
        assert history[0]["symptom"] == "headache"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=Vital

# Run specific test file
pytest tests/unit/core/test_health_data_manager.py

# Run tests matching a pattern
pytest -k "health_data"
```

## Contributing Guidelines

### Development Workflow

1. **Issue Creation**: Start by creating or claiming an issue in the tracker
2. **Branch Creation**: Create a feature branch from `develop`
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```
3. **Development**: Implement your changes with tests
4. **Code Quality**: Ensure code passes linting and formatting
   ```bash
   flake8 Vital
   black Vital
   isort Vital
   mypy Vital
   ```
5. **Testing**: Run tests to verify your changes
   ```bash
   pytest tests/
   ```
6. **Documentation**: Update documentation as needed
7. **Pull Request**: Submit a PR to the `develop` branch
8. **Code Review**: Address review feedback
9. **Merge**: After approval, your changes will be merged

### Commit Message Format

Follow the conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:

-  **feat**: A new feature
-  **fix**: A bug fix
-  **docs**: Documentation changes
-  **style**: Code style changes (formatting, etc.)
-  **refactor**: Code refactoring
-  **test**: Adding or updating tests
-  **chore**: Maintenance tasks

Example:

```
feat(symptom-analyzer): add severity assessment feature

Implement algorithm to evaluate symptom severity based on duration,
intensity, and impact on daily activities.

Closes #123
```

### Code Standards

-  **PEP 8**: Follow Python style guidelines
-  **Docstrings**: Use Google-style docstrings
-  **Type Annotations**: Include type hints for all functions
-  **Test Coverage**: Maintain >90% test coverage for new code
-  **Comments**: Comment complex logic but prefer self-explanatory code

Example of a properly formatted function:

```python
def analyze_symptoms(
    symptoms: List[Dict[str, Any]],
    user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze symptoms to identify potential conditions.

    Args:
        symptoms: List of symptom dictionaries with name, severity, and duration
        user_context: User demographic and medical history information

    Returns:
        Dictionary containing analysis results with potential conditions
        and confidence scores

    Raises:
        ValueError: If symptoms list is empty
    """
    if not symptoms:
        raise ValueError("Symptoms list cannot be empty")

    # Implementation details
    # ...

    return {
        "potential_conditions": conditions,
        "confidence_scores": scores
    }
```

## Performance Optimization

### Profiling

Use the built-in profiling tools to identify bottlenecks:

```python
from Vital.utils.profiling import profile_function

@profile_function
def your_function():
    # Function implementation
    pass
```

### Optimization Guidelines

1. **Caching**: Use `@st.cache_data` for Streamlit components that process large datasets
2. **Lazy Loading**: Defer loading resources until needed
3. **Batch Processing**: Process data in batches rather than individually
4. **Asynchronous Operations**: Use async/await for I/O-bound operations
5. **Optimized Data Structures**: Choose appropriate data structures for operations

### Memory Management

For handling large datasets:

```python
import pandas as pd

def process_large_dataset(file_path):
    # Use chunks for large files
    chunk_size = 10000
    chunks = pd.read_csv(file_path, chunksize=chunk_size)

    results = []
    for chunk in chunks:
        # Process each chunk
        processed = process_chunk(chunk)
        results.append(processed)

    # Combine results
    return pd.concat(results)
```

## Security Considerations

### Data Protection

-  **Encryption**: Sensitive data should be encrypted at rest and in transit
-  **Anonymization**: Use anonymization techniques for medical data
-  **Access Control**: Implement proper access controls and authentication
-  **Input Validation**: Validate all user inputs to prevent injection attacks
-  **Dependency Security**: Regularly update dependencies to patch vulnerabilities

### Secure Coding Practices

1. **Sanitize Inputs**: All user inputs must be validated and sanitized
2. **Parameterized Queries**: Use parameterized queries for database operations
3. **Secrets Management**: Use environment variables for sensitive information
4. **Error Handling**: Implement proper error handling without leaking sensitive information
5. **Audit Logging**: Log security-relevant events for audit purposes

Example of secure input handling:

```python
def process_user_input(input_text: str) -> str:
    """Process user input securely."""
    # Validate input
    if not input_text or len(input_text) > 1000:
        raise ValueError("Invalid input length")

    # Sanitize input
    sanitized = sanitize_text(input_text)

    # Process input
    result = perform_processing(sanitized)

    return result
```

---

Developed by Muhammad Ibrahim Kartal | [kartal.dev](https://kartal.dev)

Copyright © 2025 Vital AI Pro. All rights reserved.
