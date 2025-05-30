from enum import IntEnum, StrEnum
from typing import Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo



# -------------------------------------------------------------------------------------------------
# Schema definitions
# -------------------------------------------------------------------------------------------------

class SentinelType(StrEnum):
    INTERGENIC = 'intergenic'
    MASK = 'mask'
    
    @classmethod
    def value_to_index(cls) -> dict["SentinelType", int]:
        return {SentinelType.INTERGENIC: 0, SentinelType.MASK: -1}
    
    @classmethod
    def index_to_value(cls) -> dict[int, "SentinelType"]:
        return {i: ft for ft, i in cls.value_to_index().items()}


class ModelingFeatureType(StrEnum):
    INTERGENIC = 'intergenic'
    INTRON = 'intron'
    FIVE_PRIME_UTR = 'five_prime_utr'
    CDS = 'cds'
    THREE_PRIME_UTR = 'three_prime_utr'

SEQUENCE_MODELING_FEATURES = [
    e.value for e in ModelingFeatureType
]

class RegionType(StrEnum):
    GENE = 'gene'
    TRANSCRIPT = 'transcript'
    EXON = 'exon'
    INTRON = 'intron'
    FIVE_PRIME_UTR = 'five_prime_utr'
    THREE_PRIME_UTR = 'three_prime_utr'
    CODING_SEQUENCE = 'coding_sequence'

    @classmethod
    def value_to_index(cls) -> dict["RegionType", int]:
        return {ft: i for i, ft in enumerate(RegionType)}
    
    @classmethod
    def intergenic(cls) -> str:
        return SentinelType.INTERGENIC.value
    
    @classmethod
    def value_to_feature_type(cls) -> dict["RegionType", "GffFeatureType"]:
        return {
            RegionType.GENE: GffFeatureType.GENE,
            RegionType.TRANSCRIPT: GffFeatureType.MRNA,
            RegionType.FIVE_PRIME_UTR: GffFeatureType.FIVE_PRIME_UTR,
            RegionType.CODING_SEQUENCE: GffFeatureType.CDS,
            RegionType.THREE_PRIME_UTR: GffFeatureType.THREE_PRIME_UTR,
        }
    
    def get_index(self) -> int:
        return self.value_to_index()[self]

class FeatureLevel(IntEnum):
    GENE = 0
    TRANSCRIPT = 1
    ANNOTATION = 2

class GffFeatureType(StrEnum):
    GENE = 'gene'
    MRNA = 'mRNA'
    FIVE_PRIME_UTR = 'five_prime_UTR'
    CDS = 'CDS'
    THREE_PRIME_UTR = 'three_prime_UTR'

    @classmethod
    def value_to_index(cls) -> dict["GffFeatureType", int]:
        return {ft: i for i, ft in enumerate(GffFeatureType)}
    
    @classmethod
    def index_to_value(cls) -> dict[int, "GffFeatureType"]:
        return {i: ft for ft, i in cls.value_to_index().items()}
    
    @classmethod
    def value_to_level(cls) -> dict["GffFeatureType", int]:
        mapping = {
            GffFeatureType.GENE: FeatureLevel.GENE.value,
            GffFeatureType.MRNA: FeatureLevel.TRANSCRIPT.value,
            GffFeatureType.FIVE_PRIME_UTR: FeatureLevel.ANNOTATION.value,
            GffFeatureType.CDS: FeatureLevel.ANNOTATION.value,
            GffFeatureType.THREE_PRIME_UTR: FeatureLevel.ANNOTATION.value,
        }
        assert set(GffFeatureType) == set(mapping.keys())
        return mapping
    
    @classmethod
    def value_to_slug(cls) -> dict["GffFeatureType", int]:
        mapping = {
            GffFeatureType.GENE: "gene",
            GffFeatureType.MRNA: "mrna",
            GffFeatureType.FIVE_PRIME_UTR: "five_prime_utr",
            GffFeatureType.CDS: "cds",
            GffFeatureType.THREE_PRIME_UTR: "three_prime_utr",
        }
        assert set(GffFeatureType) == set(mapping.keys())
        return mapping
    
    @classmethod
    def get_values(cls, level: int) -> list["GffFeatureType"]:
        mapping = cls.value_to_level()
        return [ft for ft in GffFeatureType if mapping[ft] == level]

    def get_index(self) -> int:
        return self.value_to_index()[self]
    
    def get_level(self) -> int:
        return self.value_to_level()[self]

def required_field(description: str = "", **kwargs: Any) -> Any:
    """Create a required field with description."""
    return Field(..., description=description, **kwargs)


def optional_field(description: str = "", default: Any = None, **kwargs: Any) -> Any:
    """Create an optional field with default value and description."""
    return Field(default, description=description, **kwargs)

# fmt: off

class PositionInfo(BaseModel):
    strand: int = required_field("Strand information: 1 for forward, -1 for reverse")
    start:  int = required_field("Start position (inclusive)")
    stop:   int = required_field("Stop position (exclusive)")
    
    @field_validator('strand', mode='after')
    def validate_strand(cls, v):
        if v not in [-1, 1]:
            raise ValueError(f"Strand must be either 1 (forward) or -1 (reverse), got {v}")
        return v
    
    @field_validator('stop', mode='after')
    def stop_greater_than_start(cls, v, info: ValidationInfo):
        if v <= (start := info.data['start']):
            raise ValueError(f"Stop position {v} must be greater than start position {start}")
        return v


class SequenceFeature(BaseModel):
    species_id:              str         = required_field("Species identifier (e.g. Athaliana)")
    species_name:            str         = required_field("Species name (e.g. Arabidopsis thaliana)")

    chromosome_id:           str         = required_field("Chromosome identifier (e.g. Chr1)")
    chromosome_name:         str         = required_field("Chromosome name")
    chromosome_length:       int         = required_field("Length of the chromosome")
    
    gene_id:                 str         = required_field("Gene identifier")
    gene_name:               str | None  = optional_field("Gene name")
    gene_strand:             int         = required_field("Gene strand: 1 for forward, -1 for reverse")
    gene_start:              int         = required_field("Gene start position (inclusive)")
    gene_stop:               int         = required_field("Gene stop position (exclusive)")
    
    transcript_id:           str         = required_field("Transcript identifier")
    transcript_name:         str | None  = optional_field("Transcript name")
    transcript_strand:       int         = required_field("Transcript strand: 1 for forward, -1 for reverse")
    transcript_is_canonical: bool        = required_field("Whether this is the canonical transcript")
    transcript_start:        int         = required_field("Transcript start position (inclusive)")
    transcript_stop:         int         = required_field("Transcript stop position (exclusive)")
    
    feature_id:              str         = required_field("Feature identifier")
    feature_name:            str | None  = optional_field("Feature name")
    feature_strand:          int         = required_field("Feature strand: 1 for forward, -1 for reverse")
    feature_type:            GffFeatureType = required_field("Feature type (e.g. CDS, UTR)")
    feature_start:           int         = required_field("Feature start position (inclusive)")
    feature_stop:            int         = required_field("Feature stop position (exclusive)")

    filename:                str         = required_field("Source GFF file name")

# fmt: on
