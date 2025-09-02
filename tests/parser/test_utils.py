import pytest
import sys
import os

# Add project root for consistent imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from parser.process_pdfs import sanitize_filename, clean_text

def test_sanitize_filename():
    assert sanitize_filename("test*file?.pdf") == "test_file_.pdf"
    assert sanitize_filename("  leading_trailing_spaces  .pdf") == "leading_trailing_spaces.pdf"
    assert sanitize_filename("no_bad_chars.pdf") == "no_bad_chars.pdf"
    assert sanitize_filename("") == "invalid_filename"
    assert sanitize_filename(".pdf") == "invalid_filename"
    assert sanitize_filename("...pdf") == "invalid_filename"

def test_clean_text_basic():
    assert clean_text("  Hello   World  ") == "Hello World"
    assert clean_text("Hello\nWorld") == "Hello World"
    assert clean_text("Hyphen-\\-nation") == "Hyphen-nation" # Test hyphen at line break
    assert clean_text("Multiple   spaces and\nnewlines") == "Multiple spaces and newlines"
    assert clean_text(None) == ""
    assert clean_text("") == ""

def test_clean_text_hyphen_at_line_break():
    assert clean_text("This is a test-\nfor hyphenation.") == "This is a testfor hyphenation."
    assert clean_text("Another test-\nwith new line.") == "Another testwith new line."

def test_clean_text_newlines():
    assert clean_text("Line one\nLine two\n\nLine three") == "Line one Line two Line three"
