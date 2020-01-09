"""
Unit and regression test for the hdphmm package.
"""

# Import package, test suite, and other packages as needed
import hdphmm
import pytest
import sys

def test_hdphmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "hdphmm" in sys.modules
