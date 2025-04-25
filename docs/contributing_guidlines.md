# Contributing Guidelines

Thank you for your interest in contributing to the MedExplain AI Pro Medical Literature module. This document provides guidelines for contributing to the project and outlines the development workflow.

## Getting Started

### Prerequisites

Before contributing to the Medical Literature module, ensure you have:

-  Python 3.8 or higher
-  Streamlit 1.24.0 or higher
-  Git for version control
-  Basic understanding of Streamlit UI components
-  Familiarity with medical data structures

### Setting Up the Development Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medexplain-ai-pro.git
   cd medexplain-ai-pro
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application in development mode:
   ```bash
   streamlit run medexplain/app.py
   ```

## Code Standards

### Python Style Guidelines

-  Follow PEP 8 style guidelines
-  Use Google-style docstrings for all classes and methods
-  Include type hints for function parameters and return values
-  Maximum line length of 88 characters
-  Use meaningful variable and function names

Example of properly formatted function:

```python
def get_medical_literature(condition_id: str) -> List[Dict[str, Any]]:
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
```

### UI Component Standards

When creating or modifying UI components:

1. Follow the enterprise styling defined in `_add_enterprise_styling()`
2. Maintain responsive design principles
3. Ensure accessibility compliance
4. Use consistent naming conventions for CSS classes
5. Apply appropriate error handling with user-friendly messages

### Commits and Pull Requests

-  Use descriptive commit messages that explain the purpose of the change
-  Keep commits focused on a single logical change
-  Reference issue numbers in commit messages when applicable
-  Create pull requests with clear descriptions of changes and their purpose
-  Link pull requests to relevant issues

## Development Workflow

### Feature Development

1. **Issue Creation**: Start by creating or claiming an issue in the tracker
2. **Branch Creation**: Create a feature branch from `develop`
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/medical-literature-improvement
   ```
3. **Development**:
   -  Implement your changes with appropriate tests
   -  Follow the code standards outlined above
   -  Test your changes locally
4. **Code Review**:
   -  Submit a pull request to the `develop` branch
   -  Address review feedback
5. **Merge**: After approval, changes will be merged

### Bug Fixes

1. **Issue Identification**: Reproduce and document the bug in an issue
2. **Branch Creation**: Create a bugfix branch
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b fix/medical-literature-bug
   ```
3. **Fix Implementation**:
   -  Identify root cause
   -  Implement the fix
   -  Add tests to prevent regression
4. **Review and Merge**: Follow the same process as feature development

## Testing Guide

### Writing Tests

Write tests for any new functionality or bug fixes:

```python
def test_get_medical_literature():
    """Test retrieving medical literature for a condition."""
    manager = MedicalLiteratureUI()
    literature = manager.get_medical_literature("migraine")

    # Verify that literature is returned
    assert isinstance(literature, list)
    assert len(literature) > 0

    # Verify structure of literature items
    for item in literature:
        assert "title" in item
        assert "journal" in item
        assert "year" in item
        assert "summary" in item
```

### Running Tests

Run tests using pytest:

```bash
pytest tests/test_medical_literature.py
```

### UI Testing

For UI components:

1. Test with various screen sizes to ensure responsiveness
2. Verify that interactions work as expected
3. Ensure appropriate error states are handled
4. Test with different datasets to ensure robustness

## Documentation

When adding or modifying features, update the corresponding documentation:

1. Add docstrings to all new methods and classes
2. Update user guide if the feature affects user interaction
3. Add code examples for API changes
4. Update architecture documentation for structural changes

## Medical Data Guidelines

When working with medical data, ensure:

1. **Accuracy**: All medical information must be accurate and verified
2. **Citation**: Include references for medical claims and information
3. **Terminology**: Use standard medical terminology with plain language explanations
4. **Categorization**: Follow established medical categorization systems
5. **Disclaimers**: Maintain appropriate medical disclaimers in the UI

## Submission Checklist

Before submitting your pull request, verify:

-  [ ] Code follows the style guidelines
-  [ ] Tests have been added or updated
-  [ ] Documentation has been updated
-  [ ] The code runs without errors in the Streamlit interface
-  [ ] All existing tests pass
-  [ ] Medical information is accurate and properly sourced
-  [ ] The UI maintains enterprise-level styling
-  [ ] Error handling is robust

---

Developed by Muhammad Ibrahim Kartal | [kartal.dev](https://kartal.dev)
