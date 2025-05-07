from dataclasses import dataclass, field

WINDOW_SIZE = 8192

@dataclass
class SpeciesConfig:
    """Configuration for a specific species genome and annotation data.
    
    Parameters
    ----------
    chromosome_map : dict[str, str]
        Mapping from source chromosome names to standardized names
    gff_properties : dict[str, str]
        Properties specific to GFF file handling for this species
    fasta_properties : dict[str, str]
        Properties specific to FASTA file handling for this species
    """
    chromosome_map: dict[str, str]
    gff_properties: dict[str, str] = field(default_factory=dict)
    fasta_properties: dict[str, str] = field(default_factory=dict)

# Zmays configuration
zmays_config = SpeciesConfig(
    chromosome_map={
        "chr1": "chr1",
        "chr2": "chr2",
        "chr3": "chr3",
        "chr4": "chr4",
        "chr5": "chr5",
        "chr6": "chr6",
        "chr7": "chr7",
        "chr8": "chr8",
        "chr9": "chr9",
        "chr10": "chr10",
    }
)

# Athaliana configuration
athaliana_config = SpeciesConfig(
    chromosome_map={
        "Chr1": "chr1",
        "Chr2": "chr2",
        "Chr3": "chr3",
        "Chr4": "chr4",
        "Chr5": "chr5",
        "ChrC": "chrc",
        "ChrM": "chrm",
    }
)

# Osativa (Rice) configuration
osativa_config = SpeciesConfig(
    chromosome_map={
        "Chr1": "chr1",
        "Chr2": "chr2",
        "Chr3": "chr3",
        "Chr4": "chr4",
        "Chr5": "chr5",
        "Chr6": "chr6",
        "Chr7": "chr7",
        "Chr8": "chr8",
        "Chr9": "chr9",
        "Chr10": "chr10",
        "Chr11": "chr11",
        "Chr12": "chr12",
        "ChrSy": "chrsy",
        "ChrUn": "chrun",
    }
)

# Pvulgaris configuration
pvulgaris_config = SpeciesConfig(
    chromosome_map={
        "PvulFLAVERTChr01": "chr1",
    }
)

# Jregia (Walnut) configuration
jregia_config = SpeciesConfig(
    chromosome_map={
        "1": "chr1",
    }
)

# Carabica (Coffee) configuration
carabica_config = SpeciesConfig(
    chromosome_map={
        # Ignore chr1e; see: https://openathena.slack.com/archives/C086EMP18P4/p1746786888058559?thread_ts=1746641974.195829&cid=C086EMP18P4
        "chr1c": "chr1",
    }
)

# Species configuration registry
SPECIES_CONFIGS = {
    "Zmays": zmays_config,
    "Athaliana": athaliana_config,
    "Osativa": osativa_config,
    "Pvulgaris": pvulgaris_config,
    "Jregia": jregia_config,
    "Carabica": carabica_config,
}