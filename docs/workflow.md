# Development Workflow

This document outlines the development workflow for the Vital AI Pro Medical Literature module. It provides a structured approach to development, testing, and deployment to ensure code quality and project stability.

## Development Environment Setup

### Initial Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Vital-ai-pro.git
   cd Vital-ai-pro
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set Up Pre-commit Hooks**

   ```bash
   pre-commit install
   ```

5. **Configure Editor**
   -  For VS Code, configure settings in `.vscode/settings.json`
   -  For PyCharm, configure PEP 8 compliance and linter integration

### Local Development Server

To run the application locally:

```bash
streamlit run Vital/app.py
```

The application will be available at `http://localhost:8501`.

## Git Workflow

We follow a modified Gitflow workflow:

### Branch Structure

-  `main`: Production-ready code, deployed to production
-  `develop`: Integration branch for features, deployed to staging
-  `feature/*`: Feature branches for new development
-  `bugfix/*`: Bug fix branches
-  `hotfix/*`: Emergency fixes for production issues
-  `release/*`: Release preparation branches

### Feature Development

1. **Create a Branch**

   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/medical-literature-enhancement
   ```

2. **Regular Commits**

   -  Make small, focused commits with descriptive messages
   -  Format commit messages with a subject line and detailed body

   ```
   feat(medical-literature): add filter by publication date

   Implemented date range filter for the medical literature search
   interface. Added UI controls and backend filtering logic.

   References #123
   ```

3. **Push Feature Branch**

   ```bash
   git push origin feature/medical-literature-enhancement
   ```

4. **Create Pull Request**
   -  Create a PR against the `develop` branch
   -  Fill in the PR template with details about changes
   -  Request reviews from appropriate team members
   -  Link to related issues

### Bug Fixes

1. **Create a Bug Fix Branch**

   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b bugfix/fix-literature-pagination
   ```

2. **Implement and Test Fix**

   -  Make minimal changes required to fix the issue
   -  Add tests to verify the fix
   -  Ensure existing tests pass

3. **Push and Create PR**
   ```bash
   git push origin bugfix/fix-literature-pagination
   ```
   -  Create a PR with details about the bug and fix
   -  Reference the issue number

### Hot Fixes

For urgent production issues:

1. **Create Hot Fix Branch from Main**

   ```bash
   git checkout main
   git pull origin main
   git checkout -b hotfix/critical-literature-bug
   ```

2. **Implement and Test Fix**

   -  Make minimal, focused changes to address the issue
   -  Add tests to verify the fix

3. **Push and Create PR to Main**
   ```bash
   git push origin hotfix/critical-literature-bug
   ```
   -  Create a PR targeting the `main` branch
   -  After merging to `main`, cherry-pick or merge to `develop`

## Code Review Process

### Preparing for Review

Before submitting a PR for review:

1. Ensure all tests pass
2. Run linters and formatters
   ```bash
   flake8 Vital
   black Vital
   mypy Vital
   ```
3. Check test coverage
   ```bash
   pytest --cov=Vital
   ```
4. Update documentation if necessary

### Review Guidelines

When reviewing code:

1. **Functionality**: Does the code work as intended?
2. **Design**: Is the code well-structured and maintainable?
3. **Standards**: Does it follow project code standards?
4. **Tests**: Are there appropriate tests for the changes?
5. **Documentation**: Is the code properly documented?
6. **Performance**: Are there any performance concerns?
7. **Security**: Are there any security issues?

### Review Workflow

1. **Reviewer Assignment**

   -  Assign at least one reviewer with domain expertise
   -  For UI changes, include a designer or UX specialist

2. **Review Process**

   -  Reviewers provide comments on specific lines
   -  Use constructive language and suggest improvements
   -  Categorize issues as "required" or "optional"

3. **Addressing Feedback**

   -  Address all required changes
   -  Respond to all comments with actions taken
   -  Push additional commits to the PR branch

4. **PR Approval**
   -  PR must have approval from all assigned reviewers
   -  All CI checks must pass
   -  PR can be merged after approvals

## Testing Strategy

### Types of Tests

-  **Unit Tests**: Test individual functions and methods
-  **Integration Tests**: Test interactions between components
-  **UI Tests**: Test user interface functionality
-  **Performance Tests**: Test system performance under load

### Writing Tests

Place tests in the `tests/` directory, mirroring the package structure:

```
tests/
├── unit/
│   └── ui/
│       └── test_medical_literature.py
└── integration/
    └── test_medical_literature_integration.py
```

Example unit test:

```python
def test_get_medical_literature():
    """Test medical literature retrieval for a condition."""
    # Arrange
    ui = MedicalLiteratureUI()

    # Act
    literature = ui.get_medical_literature("migraine")

    # Assert
    assert isinstance(literature, list)
    assert len(literature) > 0
    for item in literature:
        assert "title" in item
        assert "journal" in item
        assert "summary" in item
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/ui/test_medical_literature.py

# Run tests with coverage
pytest --cov=Vital --cov-report=html
```

## Continuous Integration

Our CI pipeline includes:

1. **Linting**: Check code style with flake8
2. **Formatting**: Verify code formatting with black
3. **Type Checking**: Validate type annotations with mypy
4. **Unit Tests**: Run unit tests with pytest
5. **Integration Tests**: Run integration tests
6. **Coverage**: Generate test coverage report

CI runs automatically on:

-  Pull request creation
-  Updates to pull requests
-  Merges to develop or main branches

## Deployment Process

### Staging Deployment

When a PR is merged to `develop`:

1. CI builds the application
2. Tests are run in the staging environment
3. The application is deployed to the staging server
4. Smoke tests verify basic functionality

### Production Deployment

When ready for production:

1. Create a release branch from `develop`

   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/v1.2.0
   ```

2. Perform final testing and fixes on the release branch

3. Create a PR to merge the release branch to `main`

4. After approval and merge to `main`:

   -  CI builds the production version
   -  Automated tests run in the production environment
   -  The application is deployed to production
   -  Post-deployment verification tests run

5. Tag the release in git

   ```bash
   git checkout main
   git pull origin main
   git tag -a v1.2.0 -m "Release v1.2.0"
   git push origin v1.2.0
   ```

6. Merge `main` back to `develop`
   ```bash
   git checkout develop
   git merge main
   git push origin develop
   ```

## Documentation Workflow

### Types of Documentation

1. **Code Documentation**

   -  Docstrings for modules, classes, and functions
   -  Comments for complex sections of code

2. **API Documentation**

   -  Generated from docstrings
   -  REST API endpoints, parameters, and responses

3. **User Documentation**

   -  User guides
   -  Feature documentation
   -  FAQs

4. **Developer Documentation**
   -  Setup instructions
   -  Architecture documentation
   -  Contribution guidelines

### Documentation Process

1. **Update Documentation with Code Changes**

   -  Update docstrings for modified code
   -  Update API documentation for API changes
   -  Update user guides for UI changes

2. **Review Documentation**

   -  Documentation is reviewed as part of the PR process
   -  Check for completeness, clarity, and accuracy

3. **Generate API Documentation**
   ```bash
   cd docs
   sphinx-build -b html source build
   ```

## Version Control Guidelines

### Commit Messages

Follow the Conventional Commits specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:

-  `feat`: A new feature
-  `fix`: A bug fix
-  `docs`: Documentation only changes
-  `style`: Changes that don't affect code behavior
-  `refactor`: Code change that neither fixes a bug nor adds a feature
-  `perf`: Code change that improves performance
-  `test`: Adding or correcting tests
-  `chore`: Changes to the build process or tools

Examples:

```
feat(search): add advanced filter options for literature search

fix(pagination): correct page count calculation in results

docs(readme): update installation instructions
```

### Pull Request Guidelines

PR titles should follow the same convention as commit messages.

PR descriptions should include:

-  What changes were made
-  Why the changes were made
-  How to test the changes
-  Screenshots for UI changes
-  Links to related issues

## Issue Management

### Issue Types

-  **Bug**: Something isn't working as expected
-  **Feature**: New functionality to be added
-  **Enhancement**: Improvement to existing functionality
-  **Documentation**: Documentation updates
-  **Technical Debt**: Code improvements without changing functionality

### Issue Workflow

1. **Creation**

   -  Use the appropriate issue template
   -  Provide clear descriptions and reproduction steps
   -  Add appropriate labels

2. **Triage**

   -  Product owner prioritizes issues
   -  Issues are assigned to developers or placed in the backlog

3. **Implementation**

   -  Developer assigns themselves to the issue
   -  Creates a branch and implements the solution
   -  Creates a PR referencing the issue

4. **Closure**
   -  Issue is closed automatically when the PR is merged
   -  Manually close if no code changes are needed

## Release Management

### Versioning

We follow Semantic Versioning (SemVer):

-  **Major** (x.0.0): Incompatible API changes
-  **Minor** (0.x.0): New functionality in a backward-compatible manner
-  **Patch** (0.0.x): Backward-compatible bug fixes

### Release Process

1. **Release Planning**

   -  Determine features and fixes for the release
   -  Assign version number based on changes

2. **Release Preparation**

   -  Create release branch
   -  Update version numbers
   -  Update changelog

3. **Testing**

   -  Perform final testing on release branch
   -  Fix any critical issues found

4. **Release Deployment**

   -  Merge to main
   -  Deploy to production
   -  Tag release in git

5. **Post-Release**
   -  Monitor for issues
   -  Prepare hotfixes if needed

### Changelog

For each release, update the CHANGELOG.md file:

```markdown
## [1.2.0] - 2023-06-15

### Added

-  Advanced filtering in medical literature search
-  Publication date range selector
-  Journal filtering options

### Fixed

-  Pagination in search results
-  Incorrect display of author names
-  Performance issue with large result sets

### Changed

-  Improved UI for condition cards
-  Enhanced enterprise styling for consistency
```

---

Developed by Muhammad Ibrahim Kartal | [kartal.dev](https://kartal.dev)
