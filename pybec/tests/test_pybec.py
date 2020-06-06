"""
Unit and regression test for the pybec package.
"""

# Import package, test suite, and other packages as needed
import pybec
import pytest
import sys

class TestImports:

    def test_pybec_imported(self):
        """Sample test, will always pass so long as import statement worked"""
        assert "pybec" in sys.modules

    def test_modules_imported(self):
        """Sample test, will always pass so long as import statement worked"""
        assert all(elem in dir(pybec) for elem in "parsers analysis output utils plotters".split())
