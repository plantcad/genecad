"""Utilities for reading, writing, and manipulating GFF3 files using pandas.

This module provides functions to work with GFF3 (Generic Feature Format version 3) files,
converting them to and from pandas DataFrames. It handles the standard GFF3 format
including headers, attributes, and validation of required fields.

See Also
--------
https://github.com/the-sequence-ontology/specifications/blob/master/gff3.md
    The GFF3 specification
"""

import pandas as pd
import gzip
from pathlib import Path

PathLike = str | Path

GFF3_SPEC = [
    ("seq_id", str),
    ("source", str),
    ("type", str),
    ("start", int),
    ("end", int),
    ("score", float),
    ("strand", str),
    ("phase", pd.Int8Dtype()),
    ("attributes", str),
]
"""GFF3 column specification; see: https://github.com/the-sequence-ontology/specifications/blob/master/gff3.md"""

GFF3_COLUMNS = [col for col, _ in GFF3_SPEC]
GFF3_DTYPES = dict(GFF3_SPEC)

def read_gff3(path: PathLike, parse_attributes: bool = True, attributes_prefix: str | None = None) -> pd.DataFrame:
    """Read a GFF3 file into a DataFrame with optional attribute parsing.
    
    Parameters
    ----------
    path : PathLike
        Path to the GFF3 file
    parse_attributes : bool, default=True
        Whether to parse the attributes column into separate columns
    attributes_prefix : str | None, default=None
        Optional prefix to add to attribute column names
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing GFF3 data with parsed attributes if requested
    """
    features = read_gff3_data(path)
    if parse_attributes:
        attributes = parse_gff3_attributes(features)
        if attributes_prefix:
            attributes = attributes.add_prefix(attributes_prefix)
        for col in attributes:
            if col in features:
                raise ValueError(f"Attributes in path {path} contain column {col!r} which conflicts with reserved GFF3 column by the same name.")
            features[col] = attributes[col]
    features.attrs = {"path": str(path), "header": read_gff3_header(path)}
    return features

def read_gff3_data(path: PathLike) -> pd.DataFrame:
    """Read raw GFF3 data into a DataFrame without parsing attributes.
    
    Parameters
    ----------
    path : PathLike
        Path to the GFF3 file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing raw GFF3 data
    """
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        names=GFF3_COLUMNS,
        na_values=".",
        dtype=GFF3_DTYPES,
    )

def read_gff3_header(path: PathLike) -> str:
    """Read comment lines from the start of a GFF3 file.
    
    Parameters
    ----------
    path : PathLike
        Path to the GFF3 file (can be gzipped with .gz extension)
        
    Returns
    -------
    str
        All comment lines from the file concatenated together
        
    Examples
    --------
    >>> header = read_gff3_header("example.gff3")
    >>> print(header)
    ##gff-version 3
    ##source-version example 1.0
    # This is a test file
    """
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else open(path)) as f:
        return ''.join(line for line in f if line.startswith("#"))
    
def parse_gff3_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse GFF3 attribute strings into a DataFrame of key-value pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column named 'attributes' with GFF3 attribute strings
        
    Returns
    -------
    pd.DataFrame
        DataFrame with attribute key-value pairs as columns
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "attributes": [
    ...         "ID=gene1;Name=TestGene",
    ...         "ID=exon1;Parent=gene1",
    ...         "Note=Complex value with spaces"
    ...     ]
    ... })
    >>> attrs = parse_gff3_attributes(df)
    >>> for i, row in attrs.iterrows():
    ...     print(f"Row {i}:", {k: v for k, v in row.items() if pd.notna(v)})
    Row 0: {'ID': 'gene1', 'Name': 'TestGene'}
    Row 1: {'ID': 'exon1', 'Parent': 'gene1'}
    Row 2: {'Note': 'Complex value with spaces'}
    """
    def parse_attribute_string(attr_str: str) -> dict:
        if not attr_str:
            return {}
        result = {}
        for pair in attr_str.split(";"):
            if not pair:
                continue
            try:
                key, value = pair.split("=", maxsplit=1)
            except ValueError as e:
                raise ValueError(f"Invalid key-value pair in GFF3 attributes: {pair!r}") from e
            result[key] = value
        return result
    
    return pd.DataFrame(
        [parse_attribute_string(attr_str) for attr_str in df["attributes"]],
        index=df.index,
    )

def validate_gff3(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a DataFrame contains no missing values in GFF3 columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns
    -------
    pd.DataFrame
        The input DataFrame if validation passes
        
    Raises
    ------
    ValueError
        If any missing values are found in GFF3 columns
    """
    missing = df.isna()
    if missing.any().any():
        missing_cols = missing.sum()
        cols_with_missing = missing_cols[missing_cols > 0]
        error_msg = "Missing values found in columns: " + ", ".join(
            f"{col}({n})" for col, n in cols_with_missing.items()
        )
        raise ValueError(error_msg)
    return df

def write_gff3(df: pd.DataFrame, path: PathLike, require_header: bool = True, fill_missing: bool = True) -> None:
    """Write a DataFrame to a GFF3 file with optional header and missing value handling.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write
    path : PathLike
        Path where the GFF3 file will be written
    require_header : bool, default=True
        Whether to require a header in df.attrs
    fill_missing : bool, default=True
        Whether to fill missing values with '.'
        
    Raises
    ------
    ValueError
        If require_header=True and df.attrs['header'] is missing
    """
    if require_header and "header" not in df.attrs:
        raise ValueError(f"DataFrame does not contain required 'header' attribute; {df.attrs=}")
    header = df.attrs.get("header", "")
    df = df[GFF3_COLUMNS].copy()
    if fill_missing:
        for col in df.columns:
            if isinstance(df[col].values, pd.core.arrays.integer.IntegerArray):
                df[col] = df[col].astype(str).where(df[col].notna(), ".")
            else:
                df[col] = df[col].fillna(".")
    validate_gff3(df)
    data = df.to_csv(sep="\t", index=False, header=None)
    with open(path, "w") as fh:
        fh.write(header)
        fh.write(data)
