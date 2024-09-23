# Testing 

To ensure the quality and functionality of our code, we use automated tests.

## Installation
from PyPI: 
```bash
pip install pytest
```

## Running Tests
We use pytest for our tests. Below are the commands for running tests in different scopes:


### All Tests
Run all tests for both speech and text with:
```bash
pytest 
```

### Specific Test Files
Run tests for text processing only:
```bash
pytest tests/test_text.py
```

Run tests for speech processing only:
```bash
pytest tests/test_speech.py
```

### Specific Test Methods
Run a specific test method by specifying the test file and method name
(replacing the `test_text.py` with the desired test file and `test_method_name` by the desited test method):
```bash 
pytest tests/test_text.py::test_method_name
```

### Clear Cache
We use some caching in our tests. If you encounter issues that might be related to cached test results or configurations, you can clear the pytest cache with:
```bash
pytest --cache-clear
```
This command removes all items from the cache, ensuring that your next test run is completely clean.

## Troubleshooting Common Issues
If tests behave unexpectedly or fail after changes, consider clearing the pytest cache or re-running the tests to verify if the issue persists. Always ensure that your environment matches the required configurations as specified in our setup guidelines.