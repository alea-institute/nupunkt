"""Pytest configuration for nupunkt tests."""

import os
import pytest
from pathlib import Path
from typing import Dict, List

from nupunkt import PunktParameters, PunktLanguageVars, PunktToken
from nupunkt import PunktTrainer, PunktSentenceTokenizer


@pytest.fixture
def sample_text() -> str:
    """Return a sample text for testing."""
    return """
    This is a sample text. It contains multiple sentences, including abbreviations like Dr. Johnson and Mr. Smith.
    The U.S.A. is a country in North America. This example has numbers like 3.14, which aren't abbreviations!
    Prof. Jones works at the university. She has a Ph.D. in Computer Science.
    """


@pytest.fixture
def legal_text() -> str:
    """Return a sample legal text for testing."""
    return """
    The Court finds as follows. Pursuant to 28 U.S.C. § 1332, diversity jurisdiction exists.
    The plaintiff, Mr. Smith, filed suit against Corp. Inc. on Jan. 15, 2023. Judge Davis presided.
    The case was dismissed with prejudice. See Smith v. Corp., 123 F.Supp.2d 456 (N.D. Cal. 2023).
    """


@pytest.fixture
def scientific_text() -> str:
    """Return a sample scientific text for testing."""
    return """
    Recent studies show promising results. Fig. 3 demonstrates the correlation between variables A and B.
    Dr. Williams et al. (2023) found that approx. 75% of samples exhibited the effect. The p-value was 0.01.
    The solution was diluted to 0.5 mg/ml. This conc. was found to be optimal for cell growth.
    """


@pytest.fixture
def common_abbreviations() -> List[str]:
    """Return a list of common abbreviations for testing."""
    return ["dr", "mr", "mrs", "ms", "prof", "etc", "e.g", "i.e", "u.s.a", "ph.d"]


@pytest.fixture
def punkt_params() -> PunktParameters:
    """Return a basic PunktParameters object for testing."""
    params = PunktParameters()
    params.abbrev_types.update(["dr", "mr", "prof", "etc", "e.g", "i.e"])
    return params


@pytest.fixture
def test_tokenizer(punkt_params) -> PunktSentenceTokenizer:
    """Return a pre-configured tokenizer for testing."""
    return PunktSentenceTokenizer(punkt_params)


@pytest.fixture
def data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def create_test_data(data_dir) -> None:
    """Create some test data files."""
    # Create a small JSONL file for testing
    os.makedirs(data_dir, exist_ok=True)
    jsonl_path = data_dir / "test_small.jsonl"
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write('{"text": "This is sentence one. This is sentence two."}\n')
        f.write('{"text": "Dr. Smith visited Mr. Jones. They discussed the U.S.A."}\n')
        f.write('{"text": "This contains a number 3.14 which is not an abbreviation."}\n')
    
    # Create a mixed test file
    text_path = data_dir / "mixed_text.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("""
        This is a paragraph with multiple sentences. It has abbreviations like Dr. and Mr. Smith.
        
        This is another paragraph. The U.S.A. is mentioned here. Also Prof. Jones with his Ph.D.
        
        Here are some numbers: 3.14, 2.71, and 1.62 which should not be treated as abbreviations.
        """)
    
    return None