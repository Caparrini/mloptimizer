Development Process and CI/CD Pipeline
======================================

This document outlines the branching strategy, development workflow, and GitHub Actions CI/CD pipeline used in the project to maintain code quality and streamline deployment.

Branching Strategy
------------------

The project follows a structured branching strategy to separate development, staging, and production environments:

- **feature branches** Branch**: Each feature or fix is developed in its own branch, prefixed with "feature" (e.g., `feature_123`). This branch is where individual developers work until the feature is ready for testing.
- **dev**: The main development branch where validated feature branches are merged. It serves as an integration branch for all features under development.
- **alpha**: A pre-release or staging branch that reflects a stable version of `dev`. Once `dev` reaches a stable point, it is merged into `alpha` for final testing before production.
- **master**: The production branch. Only stable and fully tested code is merged here, and this branch is automatically deployed to PyPI upon tagging.

Pushes and Merge Workflow
-------------------------

1. **Feature Branch Development**:
   - Each developer creates a `featureXXX` branch to work on their feature.
   - When the feature is complete, a pull request (PR) is made to merge `featureXXX` into `dev`.
   - GitHub Actions will automatically run tests and generate a coverage report on the PR.
   - If the tests pass, the feature can be merged into `dev`; if tests fail, the PR cannot be merged until issues are resolved.

2. **Development Integration on `dev`**:
   - Multiple `featureXXX` branches may be merged into `dev` concurrently.
   - This branch is intended for integrating and testing new features collectively before staging.
   - When `dev` reaches a stable state with passing tests, it can proceed to the `alpha` branch.

3. **Alpha Testing on `alpha`**:
   - The `dev` branch is merged into `alpha` for pre-release validation.
   - GitHub Actions re-runs tests and generates a coverage report to verify stability.
   - If all tests pass, the version is tagged on `alpha`, marking it as ready for release.
   - Build and installation tests are also performed to ensure the package can be installed correctly.

4. **Production Release on `master`**:
   - The fully tested and working code is merged into `master`.
   - The version is tagged on `master` to mark it as a stable release.
   - Once there is a tag, a release is created on GitHub, documenting the changes.
   - The released version is automatically published to PyPI, making the package available to users.


GitHub Actions CI/CD Pipelines
------------------------------

The project utilizes four primary GitHub Actions workflows:

1. **Feature Branch Validation**:

   - **Workflow File**: `code_quality.yml`
   - **Trigger**: Initiates on pull requests targeting the `dev` branch from feature branches (e.g., `featureXXX`).
   - **Functionality**:

     - Runs static code analysis using Qodana.
     - Ensures code quality and standards compliance before merging.
     - Executes within an isolated environment to verify quality without modifying the branch.
     - Validates that the code aligns with development standards before it can be integrated into `dev`.

2. **Alpha Branch Validation**:

   - **Workflow Files**: `CI.yml` and `build-test-package.yml`
   - **Trigger**: Runs on any push to the `alpha` branch.
   - **Functionality**:

     - Executes tests across different Python versions to ensure compatibility.
     - Uploads test coverage reports to Codecov, maintaining accountability for test quality.
     - Builds the package, installs it in a virtual environment, and runs tests to confirm the package's standalone functionality.
     - Acts as the final pre-production gate, validating that `alpha` is stable before being merged into `master`.

3. **Production Release to PyPI**:

   - **Workflow File**: `publish-to-pypi.yml`
   - **Trigger**: Executes upon tagged releases to `master`.
   - **Functionality**:

     - Builds the package into source and wheel distributions.
     - Publishes the package to PyPI, ensuring the release is available in production.
     - This workflow guarantees that only a stable, tested, and tagged version is deployed to PyPI.

4. **Build and Install Test for Package Validation**:

   - **Workflow File**: `build-test-package.yml`
   - **Trigger**: Activates on any push or pull request to `alpha`.
   - **Functionality**:

     - Builds the package and verifies that it can be installed in an isolated environment.
     - Ensures the package can run independently of the source, verifying import dependencies and package structure.
     - Provides confidence that the package can be distributed and used without issues related to build or installation.

By following this CI/CD strategy, the project maintains high code quality, consistent testing, and reliable releases with minimal manual intervention.