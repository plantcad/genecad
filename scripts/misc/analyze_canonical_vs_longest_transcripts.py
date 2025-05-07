import pandas as pd
from src.gff_reader import read_gff3

path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/predict/zmays/chr1/gff/predictions__strand_positive__minlen_03__valid_only.gff"
df = read_gff3(path)

def tap(df, fn):
    fn(df)
    return df

def assert_true(df, fn):
    assert fn(df)
    return df

(
    df 
    .pipe(lambda df: df[df["type"] == "mRNA"])
    .pipe(lambda df: (
        pd.concat([
            df.drop(columns=["attributes"]),
            # Parse and merge into attributes, e.g.:
            # ID=Zm00001eb000010_T001;Parent=Zm00001eb000010;biotype=protein_coding;transcript_id=Zm00001eb000010_T001;canonical_transcript=1
            pd.DataFrame([
                {
                    e.split("=")[0].lower(): e.split("=")[1]
                    for e in v.split(";")
                }
                for v in df.attributes.values
            ], index=df.index)
            .add_prefix("attr_"),
        ], axis=1)
    ))
    .pipe(tap, lambda df: print(df))
    .pipe(tap, lambda df: print("Biotype counts\n", df["attr_biotype"].fillna("").value_counts()))
    .pipe(tap, lambda df: print("Canonical transcript counts\n", df["attr_canonical_transcript"].fillna("").value_counts()))
    .assign(length=lambda df: df.end - df.start)
    .pipe(tap, lambda df: print("Transcript length summary\n", df["length"].describe()))
    .assign(is_canonical_transcript=lambda df: df["attr_canonical_transcript"].str.strip().str.len() > 0)
    # Group by `attr_parent` to get the longest transcript for each parent, and then
    # assign `is_longest` to True if the transcript is the longest for its parent.
    .assign(is_longest_transcript=lambda df: df.groupby("attr_parent")["length"].transform(lambda x: x == x.max()))
    .pipe(assert_true, lambda df: (
        df.groupby("attr_parent")[["is_longest_transcript", "length"]]
        .apply(lambda g: g[g["is_longest_transcript"]]["length"].max() == g["length"].max())
        .all()
    ))
    .pipe(tap, lambda df: print(
        "Canonical vs longest status for transcripts\n",
        df[["is_canonical_transcript", "is_longest_transcript"]].value_counts()
        .unstack()
    ))
    .pipe(tap, lambda df: print(
        "Number of genes with explicit canonical transcripts\n",
        df.groupby("attr_parent")["is_canonical_transcript"].max().value_counts()
    ))
)


# path = "/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/training_data/gff/Athaliana_447_Araport11.gene.gff3"
# ** Does not have explicit canonical transcripts
# path = "/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/training_data/gff/Osativa_323_v7.0.gene.gff3"
# ** Also does not have explicit canonical transcripts


# path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/predict/zmays/chr1/Zea_mays-B73-REFERENCE-NAM-5.0_Zm00001eb.1.chr1.gff3"
# Biotype counts
#  attr_biotype
# protein_coding    10967
# Name: count, dtype: int64
# Canonical transcript counts
#  attr_canonical_transcript
# 1    5892
#      5075
# Name: count, dtype: int64
# Transcript length summary
#  count     10967.000000
# mean       5511.813623
# std        7747.849303
# min         218.000000
# 25%        1960.000000
# 50%        3632.000000
# 75%        6114.000000
# max      398212.000000
# Name: length, dtype: float64
# Canonical vs longest status for transcripts
#  is_longest_transcript    False  True
# is_canonical_transcript
# False                     2917   2158
# True                       819   5073
# Number of genes with explicit canonical transcripts
#  is_canonical_transcript
# True    5892
# Name: count, dtype: int64


# path = "/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/testing_data/gff/Zea_mays-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
# Biotype counts
#  attr_biotype
# protein_coding    72539
# Name: count, dtype: int64
# Canonical transcript counts
#  attr_canonical_transcript
# 1    39756
#      32783
# Name: count, dtype: int64
# Transcript length summary
#  count     72539.000000
# mean       5375.993962
# std        8317.202586
# min         212.000000
# 25%        1879.500000
# 50%        3569.000000
# 75%        5978.000000
# max      745091.000000
# Name: length, dtype: float64
# Canonical vs longest status for transcripts
#  is_longest_transcript    False  True
# is_canonical_transcript
# False                    18769  14014
# True                      5120  34636
# Number of genes with explicit canonical transcripts
#  is_canonical_transcript
# True    39756
# Name: count, dtype: int64
# Out[31]:
#            seq_id source  type  start    end score strand phase               attr_id      attr_parent    attr_biotype    attr_transcript_id attr_canonical_transcript  length  is_canonical_transcript  is_longest_transcript
# 2            chr1    NAM  mRNA  34617  40204     .      +     .  Zm00001eb000010_T001  Zm00001eb000010  protein_coding  Zm00001eb000010_T001                         1    5587                     True                   True
# 24           chr1    NAM  mRNA  41263  46050     .      -     .  Zm00001eb000020_T002  Zm00001eb000020  protein_coding  Zm00001eb000020_T002                       NaN    4787                    False                  False
# 43           chr1    NAM  mRNA  41214  43902     .      -     .  Zm00001eb000020_T004  Zm00001eb000020  protein_coding  Zm00001eb000020_T004                       NaN    2688                    False                  False
# 62           chr1    NAM  mRNA  41314  46039     .      -     .  Zm00001eb000020_T003  Zm00001eb000020  protein_coding  Zm00001eb000020_T003                       NaN    4725                    False                  False
# 85           chr1    NAM  mRNA  41214  46762     .      -     .  Zm00001eb000020_T001  Zm00001eb000020  protein_coding  Zm00001eb000020_T001                         1    5548                     True                   True
# ...           ...    ...   ...    ...    ...   ...    ...   ...                   ...              ...             ...                   ...                       ...     ...                      ...                    ...
# 1143773  scaf_675    NAM  mRNA  23981  25052     .      +     .  Zm00001eb442990_T001  Zm00001eb442990  protein_coding  Zm00001eb442990_T001                         1    1071                     True                   True
# 1143778  scaf_675    NAM  mRNA  25430  26259     .      -     .  Zm00001eb443000_T001  Zm00001eb443000  protein_coding  Zm00001eb443000_T001                         1     829                     True                   True
# 1143785  scaf_692    NAM  mRNA  17668  21429     .      -     .  Zm00001eb443010_T001  Zm00001eb443010  protein_coding  Zm00001eb443010_T001                         1    3761                     True                   True
# 1143791  scaf_692    NAM  mRNA  26462  30223     .      -     .  Zm00001eb443020_T001  Zm00001eb443020  protein_coding  Zm00001eb443020_T001                         1    3761                     True                   True
# 1143798  scaf_695    NAM  mRNA   2336   4226     .      -     .  Zm00001eb443030_T001  Zm00001eb443030  protein_coding  Zm00001eb443030_T001                         1    1890                     True                   True

# [72539 rows x 16 columns]