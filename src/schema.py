from enum import IntEnum, StrEnum
from typing import Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from src.sequence import BILUO_TAGS


# -------------------------------------------------------------------------------------------------
# Schema definitions
# -------------------------------------------------------------------------------------------------


class TokenBiluoClass(StrEnum):
    """Token classification labels for sequence modeling."""

    INTERGENIC = "intergenic"
    B_INTRON = "B-intron"
    I_INTRON = "I-intron"
    L_INTRON = "L-intron"
    U_INTRON = "U-intron"
    B_FIVE_PRIME_UTR = "B-five_prime_utr"
    I_FIVE_PRIME_UTR = "I-five_prime_utr"
    L_FIVE_PRIME_UTR = "L-five_prime_utr"
    U_FIVE_PRIME_UTR = "U-five_prime_utr"
    B_CDS = "B-cds"
    I_CDS = "I-cds"
    L_CDS = "L-cds"
    U_CDS = "U-cds"
    B_THREE_PRIME_UTR = "B-three_prime_utr"
    I_THREE_PRIME_UTR = "I-three_prime_utr"
    L_THREE_PRIME_UTR = "L-three_prime_utr"
    U_THREE_PRIME_UTR = "U-three_prime_utr"


class FilterReason(StrEnum):
    """Reasons for GFF feature filtering typically used to generate training masks"""

    INCOMPLETE_FEATURES = "incomplete_features"
    OVERLAPPING_GENE = "overlapping_gene"
    OVERLAPPING_FEATURES = "overlapping_features"
    NO_CANONICAL_TRANSCRIPT = "no_canonical_transcript"
    MULTIPLE_CANONICAL_TRANSCRIPTS = "multiple_canonical_transcripts"


class SentinelType(StrEnum):
    INTERGENIC = "intergenic"
    MASK = "mask"

    @classmethod
    def value_to_index(cls) -> dict["SentinelType", int]:
        return {SentinelType.INTERGENIC: 0, SentinelType.MASK: -1}

    @classmethod
    def index_to_value(cls) -> dict[int, "SentinelType"]:
        return {i: ft for ft, i in cls.value_to_index().items()}


SENTINEL_MASK = SentinelType.value_to_index()[SentinelType.MASK]


class ModelingFeatureType(StrEnum):
    """Modeling features represent the canonical feature set for sequence modeling.

    This are intended to represent a combination of GffFeatureType and RegionType
    spans.  See `analysis.py` for more details on how this is often used.
    """

    INTERGENIC = "intergenic"
    INTRON = "intron"
    FIVE_PRIME_UTR = "five_prime_utr"
    CDS = "cds"
    THREE_PRIME_UTR = "three_prime_utr"


# pyrefly: ignore[not-iterable]
SEQUENCE_MODELING_FEATURES = [e.value for e in ModelingFeatureType]

BILUO_TAG_CLASS_INFO = [
    {"name": SentinelType.MASK.value, "index": -1, "feature": None},
    {
        "name": SentinelType.INTERGENIC.value,
        "index": 0,
        "feature": ModelingFeatureType.INTERGENIC,
    },
] + [
    {"name": f"{tag}-{feature.value}", "index": i + 1, "feature": feature}
    for i, (tag, feature) in enumerate(
        (tag, feature)
        for feature in [
            ModelingFeatureType.INTRON,
            ModelingFeatureType.FIVE_PRIME_UTR,
            ModelingFeatureType.CDS,
            ModelingFeatureType.THREE_PRIME_UTR,
        ]
        for tag in BILUO_TAGS
    )
]

MODELING_FEATURE_CLASS_INFO = [
    {"name": SentinelType.MASK.value, "index": -1, "feature": None},
] + [
    {"name": feature.value, "index": i, "feature": feature}
    # pyrefly: ignore[bad-argument-type]
    for i, feature in enumerate(ModelingFeatureType)
]


class RegionType(StrEnum):
    """Regions generally comprise contiguous blocks of underlying features.

    For example, a CDS region spans all CDS exons (min start to max stop).
    There is only one CDS region per transcript, and the same is true for
    the FIVE_PRIME_UTR and THREE_PRIME_UTR regions.

    The INTRON and EXON regions are slighly different.  There can be multiple
    per transcript where EXON regions are the union of CDS, THREE_PRIME_UTR, or
    FIVE_PRIME_UTR regions.  INTRON regions are the regions between CDS, THREE_PRIME_UTR,
    and FIVE_PRIME_UTR regions.
    """

    GENE = "gene"
    TRANSCRIPT = "transcript"
    EXON = "exon"
    INTRON = "intron"
    FIVE_PRIME_UTR = "five_prime_utr"
    THREE_PRIME_UTR = "three_prime_utr"
    CDS = "cds"

    @classmethod
    def value_to_index(cls) -> dict["RegionType", int]:
        # pyrefly: ignore[bad-argument-type]
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
            RegionType.CDS: GffFeatureType.CDS,
            RegionType.THREE_PRIME_UTR: GffFeatureType.THREE_PRIME_UTR,
        }

    def get_index(self) -> int:
        return self.value_to_index()[self]


class FeatureLevel(IntEnum):
    GENE = 0
    TRANSCRIPT = 1
    ANNOTATION = 2


class GffFeatureType(StrEnum):
    GENE = "gene"
    MRNA = "mRNA"
    FIVE_PRIME_UTR = "five_prime_UTR"
    CDS = "CDS"
    THREE_PRIME_UTR = "three_prime_UTR"

    @classmethod
    def value_to_index(cls) -> dict["GffFeatureType", int]:
        # pyrefly: ignore[bad-argument-type]
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
        # pyrefly: ignore[no-matching-overload]
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
        # pyrefly: ignore[no-matching-overload]
        assert set(GffFeatureType) == set(mapping.keys())
        return mapping

    @classmethod
    def get_values(cls, level: int) -> list["GffFeatureType"]:
        mapping = cls.value_to_level()
        # pyrefly: ignore  # not-iterable
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
