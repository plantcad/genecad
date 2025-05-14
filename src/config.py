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

# Nicotiana sylvestris configuration
nsylvestris_config = SpeciesConfig(
    chromosome_map={
        "chr01": "chr1",
        "chr03": "chr3",
        "chr05": "chr5",
        "chr06": "chr6",
        "chr07": "chr7",
        "chr08": "chr8",
        "chr10": "chr10",
        "chr11": "chr11",
        "chr16": "chr16",
        "chr18": "chr18",
        "chr20": "chr20",
        "chr22": "chr22",
    },
    # There are two ids in Nicotiana_sylvestris_annotation.gff3: "ID" and "id":
    # - The "ID" field is present in 100% of features while "id" is only present in ~.2% of them
    # - Nicotiana_tabacum_annotation.gff3 contains only an "ID" attribute, not "id"
    # - The "ID" field also contains values like 'Nisyl01G0000100.1' and 'NisylMtG0005500.1:exon:1'
    #   which match the format of IDs in Nicotiana_tabacum_annotation.gff3
    # - For these reasons, we drop the "id" attribute to prevent normalized column name collisions
    gff_properties={"drop_attributes": ["id"]}
)

# Nicotiana tabacum configuration
ntabacum_config = SpeciesConfig(
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
        "Chr13": "chr13",
        "Chr14": "chr14",
        "Chr15": "chr15",
        "Chr16": "chr16",
        "Chr17": "chr17",
        "Chr18": "chr18",
        "Chr19": "chr19",
        "Chr20": "chr20",
        "Chr21": "chr21",
        "Chr22": "chr22",
        "Chr23": "chr23",
        "Chr24": "chr24",
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
    "Nsylvestris": nsylvestris_config,
    "Ntabacum": ntabacum_config,
}