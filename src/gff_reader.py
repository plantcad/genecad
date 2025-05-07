# Modified from https://github.com/foerstner-lab/gffpandas/blob/1266b4c430ab558e8675b74f8884d32f3cfdd9fc/gffpandas/gffpandas.py
import itertools
from pathlib import Path
import pandas as pd

PathLike = str | Path

def read_gff3_object(input_file: PathLike) -> "Gff3DataFrame":
    return Gff3DataFrame(input_file)

def read_gff3(input_file: PathLike, parse_attributes: bool = True, attributes_prefix: str | None = None) -> pd.DataFrame:
    gdf = read_gff3_object(input_file)
    if parse_attributes:
        df = gdf.attributes_to_columns(prefix=attributes_prefix)
    else:
        df = gdf.df
    df.attrs = {"input_file": str(input_file), "header": gdf.header}
    return df

def write_gff3(df: pd.DataFrame, output_file: PathLike) -> None:
    if "header" not in df.attrs:
        raise ValueError(f"DataFrame does not contain required 'header' attribute; {df.attrs=}")
    gdf = Gff3DataFrame(input_df=df, input_header=df.attrs["header"])
    gdf.to_gff3(output_file)

class Gff3DataFrame(object):

    def __init__(self, input_gff_file=None, input_df=None, input_header=None) -> None:
        if input_gff_file is not None:
            self._gff_file = input_gff_file
            self._read_gff3_to_df()
            self._read_gff_header()
        else:
            self.df = input_df
            self.header = input_header

    def _read_gff3_to_df(self) -> pd.DataFrame:
        self.df = pd.read_table(
            self._gff_file,
            comment="#",
            names=[
                "seq_id",
                "source",
                "type",
                "start",
                "end",
                "score",
                "strand",
                "phase",
                "attributes",
            ],
            keep_default_na=False,
        )
        return self.df

    def _read_gff_header(self) -> str:
        """Create a header.

        The header of the gff file is read, means all lines,
        which start with '#'."""
        self.header = ""
        try:
            with open(self._gff_file) as file_content:
                self.header = ''.join([line for line in file_content.readlines() if line.startswith("#")])
        except:
            pass
        return self.header

    def _to_xsv(self, output_file=None, sep=None) -> None:
        self.df.to_csv(
            output_file,
            sep=sep,
            index=False,
            header=[
                "seq_id",
                "source",
                "type",
                "start",
                "end",
                "score",
                "strand",
                "phase",
                "attributes",
            ],
        )

    def to_csv(self, output_file=None) -> None:
        self._to_xsv(output_file=output_file, sep=",")

    def to_tsv(self, output_file=None) -> None:
        self._to_xsv(output_file=output_file, sep="\t")

    def to_gff3(self, gff_file) -> None:
        df_nine_col = self.df[
            [
                "seq_id",
                "source",
                "type",
                "start",
                "end",
                "score",
                "strand",
                "phase",
                "attributes",
            ]
        ]
        gff_feature = df_nine_col.to_csv(sep="\t", index=False, header=None)
        with open(gff_file, "w") as fh:
            fh.write(self.header)
            fh.write(gff_feature)

    def attributes_to_columns(self, prefix: str | None = None) -> pd.DataFrame:
        attribute_df = self.df.copy()
        df_attributes = attribute_df.loc[:, "seq_id":"attributes"]
        attribute_df["at_dic"] = attribute_df.attributes.apply(
            lambda attributes: dict(
                [
                    key_value_pair.split(sep="=", maxsplit=1)
                    for key_value_pair in attributes.split(";")
                ]
            )
        )
        attribute_df["at_dic_keys"] = attribute_df["at_dic"].apply(
            lambda at_dic: list(at_dic.keys())
        )
        merged_attribute_list = list(
            itertools.chain.from_iterable(attribute_df["at_dic_keys"])
        )
        nonredundant_list = sorted(list(set(merged_attribute_list)))
        for atr in nonredundant_list:
            col_name = atr
            if prefix:
                col_name = f"{prefix}{atr}"
            df_attributes[col_name] = attribute_df["at_dic"].apply(
                lambda at_dic: at_dic.get(atr)
            )
        return df_attributes
