import pytest
import sys

if __name__ == "__main__":
    # Run the database tests
    sys.exit(pytest.main(["-v", "tests/test_database.py"]))
