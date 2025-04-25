# Testing Guide

This guide outlines the testing strategy and procedures for the Vital AI Pro Medical Literature module. It provides guidance on writing, running, and maintaining tests to ensure the quality and reliability of the application.

## Testing Philosophy

Our testing approach is designed to:

1. **Catch bugs early**: Identify issues before they reach production
2. **Verify functionality**: Ensure features work as intended
3. **Prevent regressions**: Avoid reintroducing fixed bugs
4. **Document behavior**: Tests serve as executable documentation
5. **Enable refactoring**: Allow code improvements with confidence

## Test Types

### Unit Tests

Unit tests verify individual components in isolation:

-  **Scope**: Individual functions, methods, and classes
-  **Dependencies**: Mocked or stubbed
-  **Speed**: Fast execution (milliseconds)
-  **Coverage**: High coverage (85%+)
-  **Location**: `tests/unit/`

### Integration Tests

Integration tests verify interactions between components:

-  **Scope**: Multiple components working together
-  **Dependencies**: Real dependencies for integration points
-  **Speed**: Moderate execution time
-  **Coverage**: Medium coverage
-  **Location**: `tests/integration/`

### UI Tests

UI tests verify the Streamlit interface functionality:

-  **Scope**: User interface components and interactions
-  **Dependencies**: May use real or mocked backend components
-  **Speed**: Slower execution
-  **Coverage**: Key user flows
-  **Location**: `tests/ui/`

### Manual Tests

Manual tests verify aspects difficult to automate:

-  **Scope**: Visual appearance, usability, complex interactions
-  **Documentation**: Test scripts in `tests/manual/`
-  **Frequency**: Before major releases

## Test Structure

### Unit Test Structure

We follow the Arrange-Act-Assert pattern:

```python
def test_get_medical_literature():
    """Test retrieving medical literature for a condition."""
    # Arrange
    ui = MedicalLiteratureUI()
    condition_id = "migraine"

    # Act
    literature = ui.get_medical_literature(condition_id)

    # Assert
    assert isinstance(literature, list)
    assert len(literature) > 0
    for item in literature:
        assert "title" in item
        assert "journal" in item
        assert "year" in item
        assert "summary" in item
```

### Test Class Structure

Group related tests in test classes:

```python
class TestMedicalLiteratureUI:
    """Tests for the MedicalLiteratureUI class."""

    @pytest.fixture
    def literature_ui(self):
        """Create a test instance of MedicalLiteratureUI."""
        return MedicalLiteratureUI()

    def test_get_all_conditions(self, literature_ui):
        """Test retrieving all conditions."""
        conditions = literature_ui.get_all_conditions()
        assert isinstance(conditions, list)
        assert len(conditions) > 0

    def test_get_condition_info(self, literature_ui):
        """Test retrieving information for a specific condition."""
        condition = literature_ui.get_condition_info("migraine")
        assert condition is not None
        assert condition["name"] == "Migraine"
        assert "description" in condition
        assert "symptoms" in condition
```

## Writing Effective Tests

### Test Naming

Name tests clearly to indicate what they verify:

-  Use the pattern `test_[function]_[scenario]_[expectation]`
-  Example: `test_get_condition_info_nonexistent_returns_none`

### Test Independence

Each test should be independent:

-  Don't rely on state from other tests
-  Reset any shared state before/after tests
-  Use fixtures to set up preconditions

### Test Data

Use appropriate test data:

-  **Fixtures**: For common test setups
-  **Constants**: For shared test data
-  **Factories**: For generating test data

Example fixture:

```python
@pytest.fixture
def sample_literature_data():
    """Sample literature data for testing."""
    return [
        {
            "title": "Test Article 1",
            "journal": "Test Journal",
            "year": 2023,
            "summary": "This is a test summary.",
            "authors": "Test Author et al."
        },
        {
            "title": "Test Article 2",
            "journal": "Another Journal",
            "year": 2022,
            "summary": "Another test summary.",
            "authors": "Another Author et al."
        }
    ]
```

### Mocking

Use mocking to isolate components:

```python
def test_search_medical_database(mocker):
    """Test searching the medical database with mocked internal data."""
    # Arrange
    ui = MedicalLiteratureUI()

    # Mock the internal data access
    mocker.patch.object(
        ui,
        '_conditions',
        {
            "migraine": {
                "id": "migraine",
                "name": "Migraine",
                "description": "Test description",
                "category": "Neurological"
            }
        }
    )

    # Act
    results = ui.search_medical_database("migraine")

    # Assert
    assert "conditions" in results
    assert len(results["conditions"]) == 1
    assert results["conditions"][0]["name"] == "Migraine"
```

### Testing Error Cases

Always test error handling:

```python
def test_get_symptom_info_handles_exception(mocker):
    """Test that get_symptom_info handles exceptions gracefully."""
    # Arrange
    ui = MedicalLiteratureUI()

    # Mock _symptoms to raise an exception when accessed
    mocker.patch.object(ui, '_symptoms', side_effect=Exception("Test exception"))

    # Act
    result = ui.get_symptom_info("headache")

    # Assert
    assert result is None
```

## Testing Streamlit UI

Testing Streamlit UI components requires special approaches:

### Component Testing

For testing specific UI rendering:

```python
def test_render_common_conditions(monkeypatch):
    """Test rendering conditions list."""
    # Mock streamlit functions
    mock_subheader = MagicMock()
    monkeypatch.setattr(st, "subheader", mock_subheader)

    mock_write = MagicMock()
    monkeypatch.setattr(st, "write", mock_write)

    # Create UI instance with test data
    ui = MedicalLiteratureUI()

    # Call the render function
    ui._render_common_conditions()

    # Verify expected streamlit calls
    mock_subheader.assert_called_once_with("Common Health Conditions")
```

### Session State Testing

For testing session state interactions:

```python
def test_display_condition_details(monkeypatch):
    """Test displaying condition details sets session state."""
    # Mock session_state
    mock_session_state = {}
    monkeypatch.setattr(st, "session_state", mock_session_state)

    # Mock streamlit functions
    mock_markdown = MagicMock()
    monkeypatch.setattr(st, "markdown", mock_markdown)

    # Create UI instance
    ui = MedicalLiteratureUI()

    # Test condition
    condition_id = "migraine"
    condition_data = ui.get_condition_info(condition_id)

    # Call the function
    ui._display_condition_details(condition_id, condition_data)

    # Verify session state was updated
    assert "viewing_condition" in mock_session_state
    assert mock_session_state["viewing_condition"] == condition_id
```

## Running Tests

### Running Tests Locally

Execute tests using pytest:

```bash
# Run all tests
pytest

# Run tests in a specific file
pytest tests/unit/test_medical_literature.py

# Run a specific test
pytest tests/unit/test_medical_literature.py::test_get_medical_literature

# Run tests with a specific marker
pytest -m "ui"
```

### Test Coverage

Measure test coverage:

```bash
# Generate coverage report
pytest --cov=Vital

# Generate HTML coverage report
pytest --cov=Vital --cov-report=html

# Verify minimum coverage
pytest --cov=Vital --cov-fail-under=85
```

### Continuous Integration

Tests run automatically in CI:

1. On pull request creation
2. On updates to pull requests
3. On merges to develop or main

## Test Organization

### Test Directory Structure

```
tests/
├── conftest.py                # Shared fixtures
├── unit/                      # Unit tests
│   ├── core/                  # Core component tests
│   │   └── test_health_data_manager.py
│   ├── ui/                    # UI component tests
│   │   └── test_medical_literature.py
│   └── utils/                 # Utility tests
│       └── test_logging.py
├── integration/               # Integration tests
│   └── test_medical_literature_integration.py
├── ui/                        # UI tests
│   └── test_medical_literature_ui.py
└── manual/                    # Manual test scripts
    └── medical_literature_ui_test.md
```

### Fixtures

Define fixtures in `conftest.py`:

```python
# tests/conftest.py
import pytest
from Vital.ui.medical_literature import MedicalLiteratureUI

@pytest.fixture
def literature_ui():
    """Create a MedicalLiteratureUI instance for testing."""
    return MedicalLiteratureUI()

@pytest.fixture
def sample_conditions():
    """Sample condition data for testing."""
    return {
        "migraine": {
            "id": "migraine",
            "name": "Migraine",
            "description": "A neurological condition characterized by headaches.",
            "symptoms": ["headache", "sensitivity_to_light", "nausea"],
            "category": "Neurological"
        },
        "hypertension": {
            "id": "hypertension",
            "name": "Hypertension",
            "description": "High blood pressure.",
            "symptoms": ["headache", "shortness_of_breath"],
            "category": "Cardiovascular"
        }
    }
```

### Test Markers

Use markers to categorize tests:

```python
# In test files
import pytest

@pytest.mark.ui
def test_render_conditions():
    """Test UI rendering of conditions."""
    ...

@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance with large dataset."""
    ...
```

Define markers in `pytest.ini`:

```ini
[pytest]
markers =
    ui: UI component tests
    integration: Integration tests
    slow: Tests that take longer to run
```

## Test-Driven Development

Follow TDD principles when appropriate:

1. **Write a failing test** that defines the desired behavior
2. **Write minimal code** to make the test pass
3. **Refactor** the code while keeping tests passing

TDD is especially useful for:

-  Bug fixes (reproduce the bug in a test first)
-  New features with clear requirements
-  Complex algorithms

## Performance Testing

For performance-critical components:

```python
def test_search_performance_with_large_dataset(benchmark):
    """Test search performance with a large dataset."""
    # Arrange
    ui = MedicalLiteratureUI()
    # Prepare large dataset for testing

    # Act & Assert with benchmark
    result = benchmark(lambda: ui.search_medical_database("common query"))

    # Verify results are correct
    assert "conditions" in result
    assert len(result["conditions"]) > 0
```

## Regression Testing

When fixing bugs:

1. Write a test that reproduces the bug
2. Fix the code to make the test pass
3. Include the test case ID in the commit message

Example:

```python
def test_literature_pagination_bug_issue_123():
    """Test for pagination bug reported in issue #123."""
    # Arrange
    ui = MedicalLiteratureUI()

    # Act - search with parameters that trigger the bug
    results = ui.search_medical_database("test", {"limit": 10, "offset": 10})

    # Assert - verify the bug is fixed
    assert results["pagination"]["next_offset"] == 20
```

## Testing Checklist

Before submitting code for review:

-  [ ] Unit tests for new functions
-  [ ] Integration tests for component interactions
-  [ ] Edge case tests (empty inputs, error conditions)
-  [ ] Performance tests for critical paths
-  [ ] UI tests for new interface elements
-  [ ] All tests pass locally
-  [ ] Test coverage meets standards (85%+)

## Troubleshooting Tests

### Common Issues

1. **Tests interfering with each other**:

   -  Check for shared state
   -  Ensure proper setup/teardown

2. **Inconsistent failures**:

   -  Look for race conditions
   -  Check for external dependencies

3. **Slow tests**:
   -  Use profiling to identify bottlenecks
   -  Consider marking as slow and excluding from regular runs

### Debugging Failed Tests

1. **Run with increased verbosity**:

   ```bash
   pytest -v tests/unit/test_problem.py
   ```

2. **Use print debugging**:

   ```bash
   pytest -v tests/unit/test_problem.py -s
   ```

3. **Use pytest's interactive debugger**:
   ```bash
   pytest --pdb tests/unit/test_problem.py
   ```

## Maintaining Tests

### Refactoring Tests

When refactoring tests:

1. Extract common setup into fixtures
2. Keep tests focused on a single assertion
3. Use parameterized tests for similar test cases

### Parameterized Tests

Use parameterization for similar test cases:

```python
@pytest.mark.parametrize("condition_id,expected_name", [
    ("migraine", "Migraine"),
    ("hypertension", "Hypertension"),
    ("diabetes", "Diabetes"),
    ("asthma", "Asthma")
])
def test_get_condition_info_returns_correct_name(
    literature_ui, condition_id, expected_name
):
    """Test retrieving condition info returns the correct name."""
    condition = literature_ui.get_condition_info(condition_id)
    assert condition is not None
    assert condition["name"] == expected_name
```

### Test Reviews

During code reviews, verify:

1. Tests cover the functionality completely
2. Tests are maintainable and readable
3. Tests include edge cases and error conditions
4. Tests are efficient and don't slow down the test suite

---

Developed by Muhammad Ibrahim Kartal | [kartal.dev](https://kartal.dev)
