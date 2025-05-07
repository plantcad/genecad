import pytest
import pandas as pd
from src.gff_pandas import (
    read_gff3, read_gff3_data, read_gff3_header, 
    parse_gff3_attributes, GFF3_COLUMNS, write_gff3
)

# Sample GFF3 data for testing
SAMPLE_GFF3 = """##gff-version 3
##source-version example 1.0
# This is a test GFF3 file
chr1\tTestSource\tgene\t1000\t2000\t.\t+\t.\tID=gene1;Name=TestGene
chr1\tTestSource\texon\t1000\t1500\t.\t+\t.\tID=exon1;Parent=gene1
chr1\tTestSource\texon\t1600\t2000\t.\t+\t.\tID=exon2;Parent=gene1
"""

@pytest.fixture
def sample_gff3_file(tmp_path):
    """Create a temporary GFF3 file for testing."""
    gff3_path = tmp_path / "test.gff3"
    gff3_path.write_text(SAMPLE_GFF3)
    return gff3_path

@pytest.fixture
def sample_attributes_df():
    """Create a DataFrame with various attribute formats for testing."""
    return pd.DataFrame({
        "attributes": [
            "ID=gene1;Name=TestGene",
            "ID=exon1;Parent=gene1",
            "ID=gene2;Note=Complex value with spaces",
            "SingleAttribute=value1",
            "",  # Empty attributes
            "Key1=Value1;Key2=Value2;Key3=Value3",
            "Dbxref=GeneID:1234,HGNC:5678;Alias=gene-1,gene-X",  # Multiple values
        ]
    })

def test_read_gff3_header(sample_gff3_file):
    """Test reading GFF3 header."""
    header = read_gff3_header(sample_gff3_file)
    assert header.startswith("##gff-version 3")
    assert "##source-version example 1.0" in header
    assert "# This is a test GFF3 file" in header

def test_read_gff3_data(sample_gff3_file):
    """Test reading GFF3 data into DataFrame."""
    df = read_gff3_data(sample_gff3_file)
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Three features in our sample data
    assert list(df.columns) == GFF3_COLUMNS
    
    # Check specific values
    assert df.iloc[0]["seq_id"] == "chr1"
    assert df.iloc[0]["type"] == "gene"
    assert df.iloc[0]["start"] == 1000
    assert df.iloc[0]["end"] == 2000

def test_parse_gff3_attributes(sample_attributes_df):
    """Test parsing GFF3 attributes with various formats."""
    result = parse_gff3_attributes(sample_attributes_df)
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_attributes_df)
    
    # Check basic key-value pairs
    assert result.iloc[0]["ID"] == "gene1"
    assert result.iloc[0]["Name"] == "TestGene"
    assert result.iloc[1]["ID"] == "exon1"
    assert result.iloc[1]["Parent"] == "gene1"
    
    # Check complex values
    assert result.iloc[2]["Note"] == "Complex value with spaces"
    assert result.iloc[3]["SingleAttribute"] == "value1"
    assert result.iloc[5]["Key1"] == "Value1"
    assert result.iloc[5]["Key2"] == "Value2"
    assert result.iloc[6]["Dbxref"] == "GeneID:1234,HGNC:5678"
    
    # Check empty attributes
    assert pd.isna(result.iloc[4]["ID"])
    assert pd.isna(result.iloc[4]["Name"])

def test_parse_gff3_attributes_empty():
    """Test parsing empty attributes."""
    df = pd.DataFrame({"attributes": ["", "ID=gene1", ""]})
    result = parse_gff3_attributes(df)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    # Empty strings should result in all keys present but with empty values
    assert "ID" in result.columns
    assert pd.isna(result.iloc[0]["ID"])
    assert result.iloc[1]["ID"] == "gene1"
    assert pd.isna(result.iloc[2]["ID"])

def test_parse_gff3_attributes_malformed():
    """Test that malformed key-value pairs raise appropriate errors."""
    df = pd.DataFrame({
        "attributes": [
            "ID=gene1",  # valid
            "InvalidPair",  # missing =
            "Key=Value=Extra",  # multiple =
            "=NoKey",  # missing key
            "NoValue=",  # missing value
        ]
    })
    
    with pytest.raises(ValueError, match="Invalid key-value pair in GFF3 attributes: 'InvalidPair'"):
        parse_gff3_attributes(df)

def test_parse_gff3_attributes_from_file(sample_gff3_file):
    """Test parsing attributes from actual GFF3 file data."""
    df = read_gff3_data(sample_gff3_file)
    result = parse_gff3_attributes(df)
    
    # Test first feature (gene)
    assert result.iloc[0]["ID"] == "gene1"
    assert result.iloc[0]["Name"] == "TestGene"
    
    # Test exon features
    assert result.iloc[1]["ID"] == "exon1"
    assert result.iloc[1]["Parent"] == "gene1"
    assert result.iloc[2]["ID"] == "exon2"
    assert result.iloc[2]["Parent"] == "gene1"

def test_read_gff3_full(sample_gff3_file):
    """Test the complete read_gff3 function."""
    df = read_gff3(sample_gff3_file)
    
    # Check DataFrame structure and attributes
    assert isinstance(df, pd.DataFrame)
    assert "path" in df.attrs
    assert "header" in df.attrs
    assert df.attrs["path"] == str(sample_gff3_file)
    
    # Check that attributes were parsed and added as columns
    assert "ID" in df.columns
    assert "Name" in df.columns
    assert "Parent" in df.columns
    
    # Check specific values
    assert df.iloc[0]["ID"] == "gene1"
    assert df.iloc[0]["Name"] == "TestGene"
    assert df.iloc[1]["Parent"] == "gene1"

def test_read_gff3_with_attributes_prefix(sample_gff3_file):
    """Test read_gff3 with attributes prefix."""
    df = read_gff3(sample_gff3_file, attributes_prefix="attr_")
    
    # Check that attributes have the prefix
    assert "attr_ID" in df.columns
    assert "attr_Name" in df.columns
    assert "attr_Parent" in df.columns
    
    # Check specific values
    assert df.iloc[0]["attr_ID"] == "gene1"
    assert df.iloc[0]["attr_Name"] == "TestGene"

def test_write_gff3_roundtrip(sample_gff3_file, tmp_path):
    """Test that writing and reading a GFF3 file preserves all data."""
    # Read original file
    df_original = read_gff3(sample_gff3_file)
    
    # Write and read back
    output_path = tmp_path / "roundtrip.gff3"
    write_gff3(df_original, output_path)
    df_roundtrip = read_gff3(output_path)
    
    # Verify data is identical
    pd.testing.assert_frame_equal(df_original, df_roundtrip)
