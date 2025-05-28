
from src.config import SpeciesConfig


def normalize_species_identifier(species_name: str) -> str:
    """Convert full species name to a normalized identifier format.
    
    Parameters
    ----------
    species_name : str
        A species name, preferably in binomial format (genus + species)
    
    Returns
    -------
    str
        Normalized species identifier
    
    Examples
    --------
    >>> normalize_species_identifier("Arabidopsis thaliana")
    "Athaliana"
    >>> normalize_species_identifier("Homo sapiens")
    "Hsapiens"
    """
    if not (species_name or "").strip():
        return ""
    parts = [p.strip() for p in species_name.split() if p.strip()]
    return parts[0] if len(parts) == 1 else f"{parts[0][0].upper()}{parts[1].lower()}"


def get_species_data_dir(config: SpeciesConfig) -> str:
    if config.split.use_in_training:
        return "training_data"
    elif config.split.use_in_evaluation:
        return "testing_data"
    else:
        raise ValueError(f"Invalid split configuration for species {config.id}")
    

def normalize_genomic_region_label(label: str, strict: bool = True) -> str | None:
    label = label.lower()
    if label in ["intergenic", "intron", "five_prime_utr", "cds", "three_prime_utr"]:
        return label
    if label == "mrna":
        return "transcript"
    if label == "coding_sequence":
        return "cds"
    if strict:
        raise ValueError(f"Invalid label: {label}")
    return None