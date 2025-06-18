import pytest
from src.naming import normalize_species_identifier


@pytest.mark.parametrize(
    "species_name, expected",
    [
        # Standard binomial names
        ("Arabidopsis thaliana", "Athaliana"),
        ("Homo sapiens", "Hsapiens"),
        ("Escherichia coli", "Ecoli"),
        # Names with capitalization variations
        ("arabidopsis thaliana", "Athaliana"),
        ("HOMO SAPIENS", "Hsapiens"),
        # Names with extra spaces
        ("  Arabidopsis   thaliana  ", "Athaliana"),
        # Names with extra parts (subspecies, strain, etc.)
        ("Homo sapiens sapiens", "Hsapiens"),
        ("Arabidopsis thaliana col-0", "Athaliana"),
        # Single-word inputs
        ("Arabidopsis", "Arabidopsis"),
        ("HUMAN", "HUMAN"),
        # Empty inputs
        ("", ""),
        (None, ""),
        ("   ", ""),
    ],
)
def test_normalize_species_identifier(species_name, expected):
    result = normalize_species_identifier(species_name)
    assert result == expected
