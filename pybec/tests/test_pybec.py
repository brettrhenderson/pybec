"""
Unit and regression test for the pybec package.
"""

# Import package, test suite, and other packages as needed
import pybec
import pytest
import sys

def test_pybec_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pybec" in sys.modules
