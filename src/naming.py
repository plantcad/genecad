
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