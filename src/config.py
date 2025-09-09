from dataclasses import dataclass
import re
from typing import Sequence


WINDOW_SIZE = 8192


@dataclass
class SpeciesGffConfig:
    filename: str
    attributes_to_drop: list[str] | None = None


@dataclass
class SpeciesFastaConfig:
    filename: str


@dataclass
class DataSplitConfig:
    use_in_training: bool
    use_in_validation: bool
    use_in_evaluation: bool


@dataclass
class SpeciesConfig:
    """Configuration for a specific species genome and annotation data.

    Parameters
    ----------
    id : str
        Unique identifier for the species
    name : str
        Human-readable name for the species
    chromosome_map : dict[str, str]
        Mapping from source chromosome names to standardized names
    gff : SpeciesGffConfig
        Configuration for GFF file handling for this species
    fasta : SpeciesFastaConfig
        Configuration for FASTA file handling for this species
    split : DataSplitConfig
        Configuration for data splits (training/validation/evaluation)
    """

    id: str
    name: str
    chromosome_map: dict[str, str]
    gff: SpeciesGffConfig
    fasta: SpeciesFastaConfig
    split: DataSplitConfig

    @classmethod
    def parse_chromosome_number(cls, chrom_id: str) -> int | None:
        """Parse a chromosome number from a chromosome ID.

        Parameters
        ----------
        chrom_id : str

        Returns
        -------
        int | None
            The chromosome number if found, otherwise None
        """
        if re.match(r"^chr\d+$", chrom_id):
            return int(re.match(r"chr(\d+)", chrom_id).group(1))
        return None

    def get_chromosome_number(self, chrom_id: str) -> int | None:
        """Get the chromosome number from a chromosome ID.

        Parameters
        ----------
        chrom_id : str
            The chromosome ID to get the number from (e.g. "chr1", "Chr2", "Chr3_EL10_PGA_scaffold7", etc.)

        Returns
        -------
        int | None
            The chromosome number if found, otherwise None
        """
        if chrom_id in self.chromosome_map:
            chrom_id = self.chromosome_map[chrom_id]
        return self.parse_chromosome_number(chrom_id)

    def sort_chromosome_ids_by_number(self, chrom_ids: Sequence[str]) -> list[str]:
        """Sort chromosome IDs by their numerical value.

        Parameters
        ----------
        chrom_ids : Sequence[str]
            Sequence of chromosome IDs to sort

        Returns
        -------
        list[str]
            Sorted list of chromosome IDs

        Examples
        --------
        >>> config.sort_chromosome_ids_by_number(["chr1", "chr10", "chr2"])
        ['chr1', 'chr2', 'chr10']
        """

        def chrom_num(key):
            num = self.get_chromosome_number(key)
            if num is None:
                raise ValueError(
                    f"Unable to determine sort order for chromosome ID: {key}"
                )
            return num

        return sorted(chrom_ids, key=chrom_num)


def get_species_configs(
    species_ids: Sequence[str] | None = None,
) -> list[SpeciesConfig]:
    """Get configurations for the specified species.

    Parameters
    ----------
    species_ids : Sequence[str] | None, default=None
        Sequence of species IDs to get configurations for; if None, all species configs will be returned

    Returns
    -------
    list[SpeciesConfig]
        List of species configurations

    Raises
    ------
    ValueError
        If any of the specified species IDs cannot be found
    """
    if species_ids is None:
        return list(SPECIES_CONFIGS.values())
    missing_ids = [sid for sid in species_ids if sid not in SPECIES_CONFIGS]
    if missing_ids:
        available_ids = list(SPECIES_CONFIGS.keys())
        raise ValueError(
            f"Could not find configs for species IDs: {missing_ids}. Available IDs: {available_ids}"
        )
    return [SPECIES_CONFIGS[sid] for sid in species_ids]


def get_species_config(species_id: str) -> SpeciesConfig:
    return SPECIES_CONFIGS[species_id]


# -------------------------------------------------------------------------------------------------
# Evaluation species configurations
# -------------------------------------------------------------------------------------------------

# fmt: off
# Zmays configuration
zmays_config = SpeciesConfig(
    id="Zmays",
    name="Zea mays",
    chromosome_map={
        f"chr{i}": f"chr{i}" for i in range(1, 11)
    },
    gff=SpeciesGffConfig(filename="Zea_mays-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"),
    fasta=SpeciesFastaConfig(filename="Zea_mays-B73-REFERENCE-NAM-5.0.fa.gz"),
    split=DataSplitConfig(use_in_training=False, use_in_validation=False, use_in_evaluation=True)
)

# Pvulgaris configuration
pvulgaris_config = SpeciesConfig(
    id="Pvulgaris",
    name="Phaseolus vulgaris",
    chromosome_map={
        "PvulFLAVERTChr01": "chr1",
    },
    gff=SpeciesGffConfig(filename="Phaseolus_vulgaris-2.0.2_chr1.gff"),
    fasta=SpeciesFastaConfig(filename="Phaseolus_vulgaris-2.0.2_chr1.fasta"),
    split=DataSplitConfig(use_in_training=False, use_in_validation=False, use_in_evaluation=True)
)

# Jregia (Walnut) configuration
jregia_config = SpeciesConfig(
    id="Jregia",
    name="Juglans regia",
    chromosome_map={
        "1": "chr1",
    },
    gff=SpeciesGffConfig(filename="Juglans_regia.Walnut_2.0.60_chr1.gff3"),
    fasta=SpeciesFastaConfig(filename="Juglans_regia.Walnut_2.0.dna.toplevel_chr1.fa"),
    split=DataSplitConfig(use_in_training=False, use_in_validation=False, use_in_evaluation=True)
)

# Carabica (Coffee) configuration
carabica_config = SpeciesConfig(
    id="Carabica",
    name="Coffea arabica",
    chromosome_map={
        # Ignore chr1e; see: https://openathena.slack.com/archives/C086EMP18P4/p1746786888058559?thread_ts=1746641974.195829&cid=C086EMP18P4
        "chr1c": "chr1",
    },
    gff=SpeciesGffConfig(filename="Coffea_arabica_geisha.annotation_chr1.gff3"),
    fasta=SpeciesFastaConfig(filename="Coffea_arabica_geisha.final_chr1.fasta"),
    split=DataSplitConfig(use_in_training=False, use_in_validation=False, use_in_evaluation=True)
)

# Nicotiana sylvestris configuration
nsylvestris_config = SpeciesConfig(
    id="Nsylvestris",
    name="Nicotiana sylvestris",
    chromosome_map={
        **{f"chr{i:02d}": f"chr{i}" for i in [1, 3, 5, 6, 7, 8, 10, 11, 16, 18, 20, 22]}
    },
    # There are two ids in Nicotiana_sylvestris_annotation.gff3: "ID" and "id":
    # - The "ID" field is present in 100% of features while "id" is only present in ~.2% of them
    # - Nicotiana_tabacum_annotation.gff3 contains only an "ID" attribute, not "id"
    # - The "ID" field also contains values like 'Nisyl01G0000100.1' and 'NisylMtG0005500.1:exon:1'
    #   which match the format of IDs in Nicotiana_tabacum_annotation.gff3
    # - For these reasons, we drop the "id" attribute to prevent normalized column name collisions
    gff=SpeciesGffConfig(
        filename="Nicotiana_sylvestris_annotation.gff3",
        attributes_to_drop=["id"]
    ),
    fasta=SpeciesFastaConfig(filename="Nicotiana_sylvestris_genome_chr1_clean.fa.gz"),
    split=DataSplitConfig(use_in_training=False, use_in_validation=False, use_in_evaluation=True)
)

# Nicotiana tabacum configuration
ntabacum_config = SpeciesConfig(
    id="Ntabacum",
    name="Nicotiana tabacum",
    chromosome_map={
        f"Chr{i}": f"chr{i}" for i in range(1, 25)
    },
    gff=SpeciesGffConfig(filename="Nicotiana_tabacum_annotation.gff3"),
    fasta=SpeciesFastaConfig(filename="Nicotiana_tabacum_genome_chr1_clean.fa.gz"),
    split=DataSplitConfig(use_in_training=False, use_in_validation=False, use_in_evaluation=True)
)

# -------------------------------------------------------------------------------------------------
# Training species configurations
# -------------------------------------------------------------------------------------------------

# Athaliana configuration
athaliana_config = SpeciesConfig(
    id="Athaliana",
    name="Arabidopsis thaliana",
    chromosome_map={
        **{f"Chr{i}": f"chr{i}" for i in range(1, 6)},
        "ChrC": "chr6",
        "ChrM": "chr7",
    },
    gff=SpeciesGffConfig(filename="Athaliana_447_Araport11.gene.gff3"),
    fasta=SpeciesFastaConfig(filename="Athaliana_447.fasta"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=True, use_in_evaluation=False)
)

# Osativa (Rice) configuration
osativa_config = SpeciesConfig(
    id="Osativa",
    name="Oryza sativa",
    chromosome_map={
        **{f"Chr{i}": f"chr{i}" for i in range(1, 13)},
        "ChrSy": "chr13",
        "ChrUn": "chr14",
    },
    gff=SpeciesGffConfig(filename="Osativa_323_v7.0.gene.gff3"),
    fasta=SpeciesFastaConfig(filename="Osativa_323.fasta"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=True, use_in_evaluation=False)
)

# Ananas comosus (Pineapple) configuration
acomosus_config = SpeciesConfig(
    id="Acomosus",
    name="Ananas comosus",
    chromosome_map={
        f"LG{i:02d}": f"chr{i}" for i in range(1, 26)
    },
    gff=SpeciesGffConfig(filename="Acomosus_321_v3.gene.no_scaffold.gff3.gz"),
    fasta=SpeciesFastaConfig(filename="Acomosus_321_v3.no_scaffold.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# Aquilegia coerulea (Columbine) configuration
acoerulea_config = SpeciesConfig(
    id="Acoerulea",
    name="Aquilegia coerulea",
    chromosome_map={
        f"Chr_{i:02d}": f"chr{i}" for i in range(1, 8)
    },
    gff=SpeciesGffConfig(filename="Acoerulea_322_v3.1.gene.no_scaffold.gff3.gz"),
    fasta=SpeciesFastaConfig(filename="Acoerulea_322_v3.no_scaffold.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# Asparagus officinalis (Asparagus) configuration
aofficinalis_config = SpeciesConfig(
    id="Aofficinalis",
    name="Asparagus officinalis",
    chromosome_map={
        f"AsparagusV1_{i:02d}": f"chr{i}" for i in range(1, 11)
    },
    gff=SpeciesGffConfig(filename="Aofficinalis_498_V1.1.gene.no_scaffold.gff3.gz"),
    fasta=SpeciesFastaConfig(filename="Aofficinalis_498_Aspof.V1.no_scaffold.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# Betula platyphylla (Japanese white birch) configuration
bplatyphylla_config = SpeciesConfig(
    id="Bplatyphylla",
    name="Betula platyphylla",
    chromosome_map={
        f"Chr{i:02d}": f"chr{i}" for i in range(1, 15)
    },
    gff=SpeciesGffConfig(filename="Bplatyphylla_679_v1.1.gene.no_scaffold.gff3.gz"),
    fasta=SpeciesFastaConfig(filename="Bplatyphylla_679_v1.0.no_scaffold.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# Beta vulgaris (Beet) configuration
bvulgaris_config = SpeciesConfig(
    id="Bvulgaris",
    name="Beta vulgaris",
    chromosome_map={
        "Chr1_EL10_PGA_scaffold3": "chr1",
        "Chr2_EL10_PGA_scaffold6": "chr2",
        "Chr3_EL10_PGA_scaffold7": "chr3",
        "Chr4_EL10_PGA_scaffold1": "chr4",
        "Chr5_EL10_PGA_scaffold2": "chr5",
        "Chr6_EL10_PGA_scaffold0": "chr6",
        "Chr7_EL10_PGA_scaffold5": "chr7",
        "Chr8_EL10_PGA_scaffold4": "chr8",
        "Chr9_EL10_PGA_scaffold8": "chr9"
    },
    gff=SpeciesGffConfig(filename="Bvulgaris_548_EL10_1.0.gene.no_scaffold.gff3.gz"),
    fasta=SpeciesFastaConfig(filename="Bvulgaris_548_EL10_1.0.no_scaffold.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# Carya illinoinensis (Pecan) configuration
cillinoinensis_config = SpeciesConfig(
    id="Cillinoinensis",
    name="Carya illinoinensis",
    chromosome_map={
        f"Chr{i:02d}": f"chr{i}" for i in range(1, 17)
    },
    gff=SpeciesGffConfig(filename="Cillinoinensis_573_v1.1.gene.no_scaffold.gff3.gz"),
    fasta=SpeciesFastaConfig(filename="Cillinoinensis_573_v1.0.no_scaffold.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# Glycine max (Soybean) configuration
gmax_config = SpeciesConfig(
    id="Gmax",
    name="Glycine max",
    chromosome_map={
        f"Gm{i:02d}": f"chr{i}" for i in range(1, 21)
    },
    gff=SpeciesGffConfig(filename="Gmax_880_Wm82.a6.v1.gene.gff3"),
    fasta=SpeciesFastaConfig(filename="Gmax_880_v6.0.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# Hordeum vulgare (Barley) configuration
hvulgare_config = SpeciesConfig(
    id="Hvulgare",
    name="Hordeum vulgare",
    chromosome_map={
        **{f"chr{i}H": f"chr{i}" for i in range(1, 8)},
        "chrUn": "chr8",
    },
    gff=SpeciesGffConfig(filename="HvulgareMorex_702_V3.gene.gff3"),
    fasta=SpeciesFastaConfig(filename="HvulgareMorex_702_V3.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# Populus trichocarpa (Black cottonwood) configuration
ptrichocarpa_config = SpeciesConfig(
    id="Ptrichocarpa",
    name="Populus trichocarpa",
    chromosome_map={
        f"Chr{i:02d}": f"chr{i}" for i in range(1, 20)
    },
    gff=SpeciesGffConfig(filename="Ptrichocarpa_533_v4.1.gene.gff3"),
    fasta=SpeciesFastaConfig(filename="Ptrichocarpa_533_v4.0.fa.gz"),
    split=DataSplitConfig(use_in_training=True, use_in_validation=False, use_in_evaluation=False)
)

# fmt: on

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
    "Acomosus": acomosus_config,
    "Acoerulea": acoerulea_config,
    "Aofficinalis": aofficinalis_config,
    "Bplatyphylla": bplatyphylla_config,
    "Bvulgaris": bvulgaris_config,
    "Cillinoinensis": cillinoinensis_config,
    "Gmax": gmax_config,
    "Hvulgare": hvulgare_config,
    "Ptrichocarpa": ptrichocarpa_config,
}
