# Code Standards

This document outlines the coding standards and best practices for the MedExplain AI Pro Medical Literature module. These standards ensure code consistency, readability, and maintainability across the project.

## Python Standards

### General Guidelines

-  Follow PEP 8 for code style and formatting
-  Use 4 spaces for indentation (no tabs)
-  Maximum line length of 88 characters
-  Use UTF-8 encoding for all Python files
-  Include module-level docstrings explaining the purpose of each file
-  Maintain consistent naming conventions

### Naming Conventions

| Type              | Convention                 | Example                    |
| ----------------- | -------------------------- | -------------------------- |
| Modules           | Snake case                 | `medical_literature.py`    |
| Classes           | Pascal case                | `MedicalLiteratureUI`      |
| Functions/Methods | Snake case                 | `get_medical_literature()` |
| Variables         | Snake case                 | `condition_data`           |
| Constants         | Uppercase with underscores | `MAX_RESULTS`              |
| Private members   | Prefix with underscore     | `_conditions`              |

### Docstrings

All modules, classes, and methods should include Google-style docstrings:

```python
def search_medical_database(query: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search for medical information related to the query.

    Args:
        query (str): The search query string

    Returns:
        Dict: Dictionary with categories of search results
            - conditions: List of condition dictionaries
            - symptoms: List of symptom dictionaries
            - literature: List of literature dictionaries

    Raises:
        ValueError: If query is empty
    """
```

### Type Annotations

-  Use type hints for all function parameters and return values
-  Import types from the `typing` module (`Dict`, `List`, `Optional`, etc.)
-  Use `Any` only when necessary, prefer more specific types
-  For optional parameters, use `Optional[Type]` rather than `Type = None`

```python
from typing import Dict, List, Any, Optional, Union

def get_condition_info(condition_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific health condition."""
```

### Error Handling

-  Use explicit exception handling with try/except blocks
-  Catch specific exceptions rather than generic `Exception` when possible
-  Log exceptions with appropriate context
-  Provide meaningful error messages
-  Return sensible default values when appropriate

```python
def get_symptom_info(symptom_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific symptom."""
    try:
        return self._symptoms.get(symptom_id)
    except KeyError:
        logger.error(f"Symptom ID not found: {symptom_id}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving symptom info for {symptom_id}: {e}")
        return None
```

## Streamlit UI Standards

### Component Organization

-  Organize UI code into logical functions with clear purposes
-  Group related UI elements into logical sections
-  Maintain a consistent visual hierarchy
-  Use meaningful variable names for UI elements
-  Apply enterprise styling consistently

### UI Component Hierarchy

```
render()
├── _render_common_conditions()
│   └── _display_condition_details()
├── _render_recent_research()
├── _render_search_interface()
│   └── _process_literature_search()
│       └── _display_search_results()
└── _render_personalized_recommendations()
```

### Styling Standards

-  Use the defined enterprise styling in `_add_enterprise_styling()`
-  Apply consistent color schemes matching the design system
-  Maintain responsive layouts for different screen sizes
-  Use consistent spacing between UI elements
-  Follow the established card and container design patterns

### HTML and CSS

When using HTML and CSS via `st.markdown()`:

-  Use meaningful class names following the established pattern
-  Keep HTML structure semantic and accessible
-  Avoid inline styles when possible, prefer CSS classes
-  Include appropriate ARIA attributes for accessibility
-  Test rendering across different browsers

```python
st.markdown(f"""
<div class="symptom-item">
    <strong>{symptom_info['name']}</strong><br>
    <span>{symptom_info['description']}</span>
</div>
""", unsafe_allow_html=True)
```

## State Management

-  Use `st.session_state` for managing application state
-  Set clear, descriptive keys for state variables
-  Check if keys exist before accessing them
-  Initialize state variables with default values when appropriate
-  Use consistent patterns for state updates

```python
# Setting state
st.session_state.viewing_condition = condition_id

# Checking and using state
if 'viewing_condition' in st.session_state:
    condition_id = st.session_state.viewing_condition
    condition_data = self.get_condition_info(condition_id)
    self._display_condition_details(condition_id, condition_data)
```

## Data Structure Standards

### Medical Data Schemas

Maintain consistent data structures for medical information:

**Condition Schema:**

```python
{
    "id": str,                       # Unique identifier
    "name": str,                     # Display name
    "description": str,              # Detailed description
    "typical_duration": str,         # Duration information
    "treatment": str,                # Treatment approaches
    "when_to_see_doctor": str,       # Medical advice
    "symptoms": List[str],           # List of symptom IDs
    "category": str                  # Medical category
}
```

**Symptom Schema:**

```python
{
    "id": str,                       # Unique identifier
    "name": str,                     # Display name
    "description": str,              # Detailed description
    "category": str                  # Symptom category
}
```

**Literature Schema:**

```python
{
    "title": str,                    # Article title
    "journal": str,                  # Journal name
    "year": int,                     # Publication year
    "summary": str,                  # Plain-language summary
    "authors": str                   # Author information
}
```

### Return Values

-  Use consistent return types for similar functions
-  Document return structures clearly in docstrings
-  Include error information in returns when appropriate
-  Follow established patterns for success/failure responses

## Logging Standards

-  Configure logging using the `logging` module
-  Use appropriate log levels:
   -  DEBUG: Detailed information for debugging
   -  INFO: Confirmation of expected functionality
   -  WARNING: Unexpected behavior that doesn't cause errors
   -  ERROR: Errors that prevent function execution
   -  CRITICAL: Critical errors affecting the application
-  Include context in log messages
-  Don't expose sensitive information in logs

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Initializing medical literature data")
logger.info("Medical Literature UI initialized with embedded data")
logger.warning("No condition information found for ID: %s", condition_id)
logger.error("Error processing literature search: %s", str(e))
```

## Performance Considerations

-  Use efficient data structures appropriate for the task
-  Cache results when appropriate using `@st.cache_data`
-  Optimize database queries and API calls
-  Be mindful of memory usage with large datasets
-  Profile and optimize slow operations

```python
@st.cache_data
def get_all_conditions() -> List[Dict[str, Any]]:
    """
    Returns a list of all available health conditions.
    Results are cached for improved performance.
    """
```

## Security Guidelines

-  Sanitize user inputs to prevent injection attacks
-  Validate input parameters before using them
-  Don't expose sensitive information in the UI or logs
-  Follow the principle of least privilege
-  Implement appropriate access controls

## Testing Requirements

-  Write unit tests for all non-UI functionality
-  Test edge cases and error conditions
-  Mock external dependencies for reliable testing
-  Maintain high test coverage for core functionality
-  Document testing patterns for UI components

## Documentation Standards

-  Include a module-level docstring explaining the file's purpose
-  Document classes with their responsibilities and usage
-  Explain complex algorithms or business logic
-  Comment non-obvious code sections
-  Keep comments up-to-date with code changes

## Code Review Checklist

When reviewing code, verify:

-  [ ] Code follows the style guidelines
-  [ ] Proper error handling is implemented
-  [ ] Functions have appropriate docstrings
-  [ ] Type annotations are present and correct
-  [ ] UI elements follow the design system
-  [ ] No security vulnerabilities exist
-  [ ] No performance issues are evident
-  [ ] Tests cover the new functionality
-  [ ] Documentation is complete and accurate
-  [ ] Medical information is correct and properly sourced

---

Developed by Muhammad Ibrahim Kartal | [kartal.dev](https://kartal.dev)
