import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from src.analysis import get_sequence_modeling_labels
from src.schema import SEQUENCE_MODELING_FEATURES
import xarray as xr
from numba import njit
from src.sequence import find_intervals
from src.dataset import open_datatree
import logging
import argparse

# Constants
MAX_LENGTH = 16_384

logger = logging.getLogger(__name__)

@njit
def _transition_counts(labels: np.ndarray, classes: int) -> np.ndarray:
    """
    Calculates the transition counts matrix for labels in sequences.

    Parameters
    ----------
    labels : np.ndarray
        A 2D array of shape (N, S) where N is the number of sequences
        and S is the sequence length. Masked labels should be -1.
    classes : int
        The total number of unique classes (including the mask value if relevant,
        though transitions involving -1 are ignored).

    Returns
    -------
    np.ndarray
        A square matrix of shape (classes, classes) where element (i, j)
        is the count of transitions from class i to class j.
    """
    counts = np.zeros((classes, classes), dtype=np.float64)
    N, S = labels.shape

    for i in range(N):
        for j in range(S - 1):
            label_from = labels[i, j]
            label_to = labels[i, j + 1]
            if label_from >= 0 and label_to >= 0:
                counts[label_from, label_to] += 1

    return counts


def format_transition_probs(df: pd.DataFrame, precision: int = 16) -> str:
    rows, cols = df.index.tolist(), df.columns.tolist()
    format_spec = f"{{:.{precision}e}}"

    # Calculate column widths based on formatted numbers and column headers
    col_widths = []
    for j, col_label in enumerate(cols):
        max_len = len(col_label)  # Start with header length
        for i in range(len(rows)):
            formatted_num = format_spec.format(df.iloc[i, j])
            max_len = max(max_len, len(formatted_num))
        col_widths.append(max_len)

    lines = ["np.array(["]

    # Add column header comment with padding
    padded_cols = [col.ljust(width) for col, width in zip(cols, col_widths)]
    header_comment = "# " + "  ".join(padded_cols)
    lines.append(f"    {header_comment}")

    # Add data rows with padding
    for i, row_label in enumerate(rows):
        row_values = []
        for j in range(len(cols)):
            formatted_num = format_spec.format(df.iloc[i, j])
            padded_num = formatted_num.ljust(col_widths[j])
            row_values.append(padded_num)
        
        row_str = ", ".join(row_values)
        # Add trailing comma for the row, followed by the row label comment
        lines.append(f"    [{row_str}],  # {row_label}")

    lines.append("], dtype=np.float64)") # Specify dtype for clarity
    return "\n".join(lines)

def compute_transition_stats(datasets: list[xr.Dataset], classes: list[str], limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    num_classes = len(classes)

    for dataset in tqdm.tqdm(datasets):
        if limit is not None:
            dataset = dataset.isel(sequence=slice(limit))
        species_id = dataset.attrs["species_id"]
        chromosome_id = dataset.attrs["chromosome_id"]
        assert set(dataset.strand.values) == {"positive", "negative"}
        for strand in dataset.strand.values:
            strand_dataset = dataset.sel(strand=strand)
            sequence_length = strand_dataset.sizes["sequence"]

            # Extract labels
            labels = get_sequence_modeling_labels(strand_dataset)
            assert labels.feature.values.tolist() == classes, f"{labels.feature.values.tolist()} != {classes}"
            assert labels.dims == ("sequence", "feature")

            # Extract masks
            masks = strand_dataset.label_masks.all(dim="reason")
            assert masks.dims == ("sequence",)
            assert masks.shape == (sequence_length,)

            # Select feature index
            assert labels.shape == (sequence_length, num_classes)
            assert (labels.sum(dim="feature") == 1).all().item()
            labels = labels.argmax(dim="feature")
            assert labels.dims == ("sequence",)
            assert labels.shape == (sequence_length,)
            assert labels.isin(list(range(num_classes))).all().item()

            # Apply masks
            labels = xr.where(masks, labels, -1).expand_dims("batch")
            assert labels.dims == ("batch", "sequence")
            assert labels.shape == (1, sequence_length)

            # Reverse negative strand labels
            if strand == "negative":
                labels = labels.isel(sequence=slice(None, None, -1))

            # Count transitions
            transition_counts = _transition_counts(labels.values, num_classes)
            transition_counts = (
                pd.DataFrame(transition_counts, index=classes, columns=classes)
                .rename_axis("from_class", axis=0)
                .rename_axis("to_class", axis=1)
                .stack()
                .rename("count")
                .reset_index()
                .assign(strand=strand, species_id=species_id, chromosome_id=chromosome_id)
            )

            # Calculate interval lengths
            feature_lengths = []
            for class_index, class_name in enumerate(classes):
                # Create binary array for current class
                class_labels = (labels.squeeze(dim="batch").values == class_index).astype(int)
                if class_labels.sum() == 0:
                    continue
                
                # Find intervals for this class
                intervals = find_intervals(class_labels, masks.values)
                
                # Calculate lengths (stop - start + 1 for inclusive intervals)
                lengths = intervals[1] - intervals[0] + 1
                # Clip lengths to MAX_LENGTH
                lengths = np.clip(lengths, 0, MAX_LENGTH)
                # Count frequency of each length
                unique_lengths, counts = np.unique(lengths, return_counts=True)
                
                for length, count in zip(unique_lengths, counts):
                    feature_lengths.append({
                        "class_name": class_name,
                        "length": int(length),
                        "count": int(count),
                        "strand": strand,
                        "species_id": species_id,
                        "chromosome_id": chromosome_id
                    })
            
            feature_lengths = pd.DataFrame(feature_lengths) if feature_lengths else pd.DataFrame(columns=["class_name", "length", "count", "strand", "species_id", "chromosome_id"])

            results.append((transition_counts, feature_lengths))
    transition_counts = pd.concat([r[0] for r in results], ignore_index=True)
    feature_lengths = pd.concat([r[1] for r in results], ignore_index=True)

    return transition_counts, feature_lengths



def get_transition_stats(counts: pd.DataFrame, groupby: list[str] = ["strand", "species_id", "chromosome_id"]) -> pd.DataFrame:
    key = groupby + ["from_class", "to_class"]
    grouped = counts.groupby(key, as_index=False)["count"].sum()
    
    # Calculate probabilities by normalizing within each group and from_class
    group_cols = groupby + ["from_class"]
    grouped["probability"] = (
        grouped["count"] / grouped.groupby(group_cols)["count"].transform("sum")
    )
    
    return grouped

def visualize_transition_stats(counts: pd.DataFrame, feature_names: list[str]):
    """Visualize transition statistics with detailed context and summaries."""
    
    # Print overall dataset summary
    total_transitions = counts["count"].sum()
    unique_species = counts["species_id"].nunique()
    unique_chromosomes = counts["chromosome_id"].nunique()
    
    print(f"=== TRANSITION STATISTICS SUMMARY ===")
    print(f"Total transitions: {total_transitions:,}")
    print(f"Unique species: {unique_species}")
    print(f"Unique chromosomes: {unique_chromosomes}")
    print(f"Feature classes: {', '.join(feature_names)}")
    print()
    
    # Overall transition probabilities
    print("=== OVERALL TRANSITION PROBABILITIES ===")
    overall_stats = get_transition_stats(counts, groupby=[])
    overall_probs = overall_stats.pivot(index="from_class", columns="to_class", values="probability")
    overall_probs = overall_probs.reindex(index=feature_names, columns=feature_names, fill_value=0.0)
    print(overall_probs.map(lambda x: f"{x:.6f}" if pd.notna(x) else "0.000000"))
    print()
    
    # Self-transition probabilities (diagonal elements)
    print("=== SELF-TRANSITION PROBABILITIES ===")
    self_transitions = pd.Series({cls: overall_probs.loc[cls, cls] for cls in feature_names})
    for cls in feature_names:
        prob = self_transitions[cls]
        print(f"{cls}: {prob:.6f} ({prob:.2%})")
    print()
    
    # Strand-specific analysis
    print("=== STRAND-SPECIFIC TRANSITION PROBABILITIES ===")
    strand_stats = get_transition_stats(counts, groupby=["strand"])
    strand_probs = strand_stats.pivot(index=["strand", "from_class"], columns="to_class", values="probability")
    
    for strand in ["positive", "negative"]:
        print(f"\n{strand.upper()} STRAND:")
        strand_data = strand_probs.loc[strand].reindex(index=feature_names, columns=feature_names, fill_value=0.0)
        print(strand_data.map(lambda x: f"{x:.6f}" if pd.notna(x) else "0.000000"))
    
    print()
    
    # Species-specific summary (if multiple species)
    if unique_species > 1:
        print("=== SPECIES-SPECIFIC SELF-TRANSITION RATES ===")
        species_stats = get_transition_stats(counts, groupby=["species_id"])
        species_probs = species_stats.pivot(index=["species_id", "from_class"], columns="to_class", values="probability")
        
        for species in counts["species_id"].unique():
            print(f"\n{species}:")
            species_data = species_probs.loc[species]
            self_trans = pd.Series({cls: species_data.loc[cls, cls] for cls in feature_names})
            for cls in feature_names:
                prob = self_trans[cls]
                print(f"  {cls}: {prob:.4f}")
        print()
    
    # Transition count summary by feature
    print("=== TRANSITION COUNT SUMMARY BY FEATURE ===")
    from_counts = counts.groupby("from_class")["count"].sum().reindex(feature_names, fill_value=0)
    to_counts = counts.groupby("to_class")["count"].sum().reindex(feature_names, fill_value=0)
    
    print("Outgoing transitions (from each feature):")
    for feature in feature_names:
        count = from_counts[feature]
        pct = count / total_transitions * 100
        print(f"  {feature}: {count:,} ({pct:.2f}%)")
    
    print("\nIncoming transitions (to each feature):")
    for feature in feature_names:
        count = to_counts[feature]
        pct = count / total_transitions * 100
        print(f"  {feature}: {count:,} ({pct:.2f}%)")

def visualize_feature_lengths(lengths: pd.DataFrame, feature_names: list[str]):
    """Visualize feature length statistics with summary information."""
    
    # Print overall summary
    total_features = lengths["count"].sum()
    unique_species = lengths["species_id"].nunique()
    unique_chromosomes = lengths["chromosome_id"].nunique()
    
    print(f"=== FEATURE LENGTH STATISTICS SUMMARY ===")
    print(f"Total feature instances: {total_features:,}")
    print(f"Unique species: {unique_species}")
    print(f"Unique chromosomes: {unique_chromosomes}")
    print(f"Feature classes: {', '.join(feature_names)}")
    print()
    
    # Calculate total counts for each feature class (for percentage calculations)
    feature_totals = lengths.groupby("class_name")["count"].sum()
    
    # Overall length statistics by feature
    print("=== LENGTH STATISTICS BY FEATURE ===")
    for feature in feature_names:
        feature_data = lengths[lengths["class_name"] == feature]
        if feature_data.empty:
            print(f"{feature}: No instances found")
            continue
            
        # Calculate weighted statistics
        total_count = feature_data["count"].sum()
        weighted_lengths = np.repeat(feature_data["length"].values, feature_data["count"].values)
        
        mean_len = np.mean(weighted_lengths)
        median_len = np.median(weighted_lengths)
        min_len = weighted_lengths.min()
        max_len = weighted_lengths.max()
        std_len = np.std(weighted_lengths)
        
        print(f"{feature}:")
        print(f"  Count: {total_count:,}")
        print(f"  Mean length: {mean_len:.1f}")
        print(f"  Median length: {median_len:.1f}")
        print(f"  Min/Max length: {min_len}/{max_len}")
        print(f"  Std deviation: {std_len:.1f}")
        print()
    
    # Length distribution summary
    print("=== LENGTH DISTRIBUTION SUMMARY ===")
    all_weighted_lengths = np.concatenate([
        np.repeat(lengths["length"].values, lengths["count"].values)
    ])
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("Overall length percentiles:")
    for p in percentiles:
        val = np.percentile(all_weighted_lengths, p)
        print(f"  {p}th percentile: {val:.1f}")
    print()
    
    # Detailed tail analysis
    print("=== DETAILED TAIL ANALYSIS ===")
    
    # Low-length tail analysis (lengths 1-20)
    print("LOW-LENGTH TAIL (1-20 bp):")
    low_tail_data = lengths[lengths["length"] <= 20]
    if not low_tail_data.empty:
        low_tail_total = low_tail_data["count"].sum()
        overall_total = lengths["count"].sum()
        low_tail_pct = low_tail_total / overall_total * 100
        print(f"  Total features ≤20 bp: {low_tail_total:,} ({low_tail_pct:.2f}% of all features)")
        
        # Per-feature breakdown for low lengths
        for feature in feature_names:
            feature_low = low_tail_data[low_tail_data["class_name"] == feature]
            if not feature_low.empty:
                feature_low_count = feature_low["count"].sum()
                feature_total = lengths[lengths["class_name"] == feature]["count"].sum()
                if feature_total > 0:
                    feature_low_pct = feature_low_count / feature_total * 100
                    print(f"    {feature}: {feature_low_count:,} ({feature_low_pct:.2f}% of {feature} features)")
        
        # Detailed breakdown by feature class (1-32 bp) in table format
        print("\n  Feature class distribution by length (1-32 bp):")
        
        # Collect data for table
        table_data = []
        for length in range(1, 33):
            length_data = low_tail_data[low_tail_data["length"] == length]
            if not length_data.empty:
                length_count = length_data["count"].sum()
                row = {"length": length, "total": length_count}
                
                # Get counts for each feature class
                for feature in feature_names:
                    feature_length_data = length_data[length_data["class_name"] == feature]
                    if not feature_length_data.empty:
                        feature_count = feature_length_data["count"].sum()
                        row[feature] = feature_count
                    else:
                        row[feature] = 0
                
                table_data.append(row)
        
        # Print table header
        print(f"    {'Len':>3} {'Total':>6} {'Inter':>8} {'Intron':>8} {'5UTR':>8} {'CDS':>8} {'3UTR':>8}")
        print(f"    {'-'*3:>3} {'-'*6:>6} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
        
        # Print table rows
        for row in table_data:
            intergenic = row.get("intergenic", 0)
            intron = row.get("intron", 0)
            five_utr = row.get("five_prime_utr", 0)
            cds = row.get("cds", 0)
            three_utr = row.get("three_prime_utr", 0)
            
            print(f"    {row['length']:>3} {row['total']:>6} {intergenic:>8} {intron:>8} {five_utr:>8} {cds:>8} {three_utr:>8}")
        
        print()
        
        # Print percentage table - FIXED: percentages across ALL lengths for each feature
        print("  Feature class percentages by length (1-32 bp):")
        print("  (Percentages show what fraction of each feature class has this specific length)")
        print(f"    {'Len':>3} {'Inter':>8} {'Intron':>8} {'5UTR':>8} {'CDS':>8} {'3UTR':>8}")
        print(f"    {'-'*3:>3} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
        
        # Print percentage rows
        for row in table_data:
            # Calculate percentages relative to total count for each feature class
            intergenic_pct = row.get("intergenic", 0) / feature_totals.get("intergenic", 1) * 100 if feature_totals.get("intergenic", 0) > 0 else 0
            intron_pct = row.get("intron", 0) / feature_totals.get("intron", 1) * 100 if feature_totals.get("intron", 0) > 0 else 0
            five_utr_pct = row.get("five_prime_utr", 0) / feature_totals.get("five_prime_utr", 1) * 100 if feature_totals.get("five_prime_utr", 0) > 0 else 0
            cds_pct = row.get("cds", 0) / feature_totals.get("cds", 1) * 100 if feature_totals.get("cds", 0) > 0 else 0
            three_utr_pct = row.get("three_prime_utr", 0) / feature_totals.get("three_prime_utr", 1) * 100 if feature_totals.get("three_prime_utr", 0) > 0 else 0
            
            print(f"    {row['length']:>3} {intergenic_pct:>8.3f} {intron_pct:>8.3f} {five_utr_pct:>8.3f} {cds_pct:>8.3f} {three_utr_pct:>8.3f}")
        
        print()
    else:
        print("  No features found with length ≤20 bp")
    
    print()
    
    # High-length tail analysis (lengths >1000)
    print("HIGH-LENGTH TAIL (>1000 bp):")
    high_tail_data = lengths[lengths["length"] > 1000]
    if not high_tail_data.empty:
        high_tail_total = high_tail_data["count"].sum()
        overall_total = lengths["count"].sum()
        high_tail_pct = high_tail_total / overall_total * 100
        print(f"  Total features >1000 bp: {high_tail_total:,} ({high_tail_pct:.2f}% of all features)")
        
        # Per-feature breakdown for high lengths
        for feature in feature_names:
            feature_high = high_tail_data[high_tail_data["class_name"] == feature]
            if not feature_high.empty:
                feature_high_count = feature_high["count"].sum()
                feature_total = lengths[lengths["class_name"] == feature]["count"].sum()
                if feature_total > 0:
                    feature_high_pct = feature_high_count / feature_total * 100
                    print(f"    {feature}: {feature_high_count:,} ({feature_high_pct:.2f}% of {feature} features)")
        
        # Binned breakdown for very long features in table format
        high_bins = [1000, 2000, 4000, 8000, MAX_LENGTH]
        bin_labels = []
        bin_table_data = []
        
        print(f"\n  Binned breakdown for high-length features:")
        for i in range(len(high_bins) - 1):
            start_bin = high_bins[i]
            end_bin = high_bins[i + 1]
            if end_bin == MAX_LENGTH:
                bin_data = high_tail_data[high_tail_data["length"] >= start_bin]
                bin_label = f"≥{start_bin}"
            else:
                bin_data = high_tail_data[
                    (high_tail_data["length"] >= start_bin) & 
                    (high_tail_data["length"] < end_bin)
                ]
                bin_label = f"{start_bin}-{end_bin-1}"
            
            if not bin_data.empty:
                bin_count = bin_data["count"].sum()
                bin_pct = bin_count / overall_total * 100
                print(f"    {bin_label} bp: {bin_count:,} features ({bin_pct:.3f}%)")
                
                # Collect data for table
                row = {"bin_label": bin_label, "total": bin_count}
                for feature in feature_names:
                    feature_bin_data = bin_data[bin_data["class_name"] == feature]
                    if not feature_bin_data.empty:
                        feature_count = feature_bin_data["count"].sum()
                        row[feature] = feature_count
                    else:
                        row[feature] = 0
                
                bin_table_data.append(row)
                bin_labels.append(bin_label)
        
        # Print counts table
        print(f"\n  Feature class counts by length bin:")
        print(f"    {'Bin':>12} {'Total':>6} {'Inter':>8} {'Intron':>8} {'5UTR':>8} {'CDS':>8} {'3UTR':>8}")
        print(f"    {'-'*12:>12} {'-'*6:>6} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
        
        for row in bin_table_data:
            intergenic = row.get("intergenic", 0)
            intron = row.get("intron", 0)
            five_utr = row.get("five_prime_utr", 0)
            cds = row.get("cds", 0)
            three_utr = row.get("three_prime_utr", 0)
            
            print(f"    {row['bin_label']:>12} {row['total']:>6} {intergenic:>8} {intron:>8} {five_utr:>8} {cds:>8} {three_utr:>8}")
        
        # Print percentages table - FIXED: percentages across ALL lengths for each feature
        print(f"\n  Feature class percentages by length bin:")
        print("  (Percentages show what fraction of each feature class falls in this length range)")
        print(f"    {'Bin':>12} {'Inter':>8} {'Intron':>8} {'5UTR':>8} {'CDS':>8} {'3UTR':>8}")
        print(f"    {'-'*12:>12} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
        
        for row in bin_table_data:
            # Calculate percentages relative to total count for each feature class
            intergenic_pct = row.get("intergenic", 0) / feature_totals.get("intergenic", 1) * 100 if feature_totals.get("intergenic", 0) > 0 else 0
            intron_pct = row.get("intron", 0) / feature_totals.get("intron", 1) * 100 if feature_totals.get("intron", 0) > 0 else 0
            five_utr_pct = row.get("five_prime_utr", 0) / feature_totals.get("five_prime_utr", 1) * 100 if feature_totals.get("five_prime_utr", 0) > 0 else 0
            cds_pct = row.get("cds", 0) / feature_totals.get("cds", 1) * 100 if feature_totals.get("cds", 0) > 0 else 0
            three_utr_pct = row.get("three_prime_utr", 0) / feature_totals.get("three_prime_utr", 1) * 100 if feature_totals.get("three_prime_utr", 0) > 0 else 0
            
            print(f"    {row['bin_label']:>12} {intergenic_pct:>8.3f} {intron_pct:>8.3f} {five_utr_pct:>8.3f} {cds_pct:>8.3f} {three_utr_pct:>8.3f}")
        
        print()
        
        # Summary of features at maximum length
        max_length_data = high_tail_data[high_tail_data["length"] == high_tail_data["length"].max()]
        if not max_length_data.empty:
            max_len = max_length_data["length"].iloc[0]
            max_count = max_length_data["count"].sum()
            print(f"  Features at maximum length ({max_len} bp): {max_count:,} total")
            max_by_class = max_length_data.groupby("class_name")["count"].sum()
            for feature in feature_names:
                if feature in max_by_class:
                    count = max_by_class[feature]
                    pct = count / max_count * 100
                    print(f"    {feature}: {count:,} ({pct:.1f}%)")
    else:
        print("  No features found with length >1000 bp")
    
    print()
    
    # Extreme percentile analysis
    print("EXTREME PERCENTILE ANALYSIS:")
    extreme_percentiles = [0.1, 0.5, 1, 2, 5, 95, 98, 99, 99.5, 99.9]
    print("Extreme length percentiles:")
    for p in extreme_percentiles:
        val = np.percentile(all_weighted_lengths, p)
        print(f"  {p:>5.1f}th percentile: {val:>8.1f} bp")
    print()

    # Strand comparison (if applicable)
    if "strand" in lengths.columns:
        print("=== STRAND COMPARISON ===")
        for strand in ["positive", "negative"]:
            strand_data = lengths[lengths["strand"] == strand]
            if strand_data.empty:
                continue
            strand_count = strand_data["count"].sum()
            strand_pct = strand_count / total_features * 100
            print(f"{strand.upper()} strand: {strand_count:,} features ({strand_pct:.1f}%)")
        print()

    # ASCII histograms for length distributions
    print("=== LENGTH DISTRIBUTION HISTOGRAMS ===")
    for feature in feature_names:
        feature_data = lengths[lengths["class_name"] == feature]
        if feature_data.empty:
            continue
            
        # Create weighted length array
        weighted_lengths = np.repeat(feature_data["length"].values, feature_data["count"].values)
        
        # Create histogram bins (more granular to show all data up to MAX_LENGTH)
        bins = [1, 5, 10, 25, 50, 100, 200, 300, 500, 700, 1000, MAX_LENGTH]
        hist, _ = np.histogram(weighted_lengths, bins=bins)
        
        print(f"\n{feature.upper()} LENGTH DISTRIBUTION:")
        max_count = hist.max()
        max_width = 50  # Maximum bar width
        
        for i, count in enumerate(hist):
            if count == 0:
                continue
            start_bin = bins[i]
            end_bin = bins[i + 1]
            
            # Calculate bar width proportional to count
            bar_width = int((count / max_count) * max_width) if max_count > 0 else 0
            bar = "█" * bar_width
            
            # Format bin range
            if end_bin == MAX_LENGTH:
                bin_label = f"{start_bin}-{end_bin}"
            else:
                bin_label = f"{start_bin}-{end_bin-1}"
            
            pct = count / len(weighted_lengths) * 100
            print(f"  {bin_label:>8} bp │{bar:<50} │ {count:>8,} ({pct:5.1f}%)")
    
    # Create and save probability matrix plot
    prob_data = lengths.groupby(["class_name", "length"], as_index=False)["count"].sum()
    class_totals = prob_data.groupby("class_name")["count"].transform("sum")
    prob_data["probability"] = prob_data["count"] / class_totals
    prob_matrix = prob_data.pivot(index="length", columns="class_name", values="probability").fillna(0)
    prob_matrix = prob_matrix.reindex(columns=feature_names, fill_value=0)
    
    # Define appropriate x-axis limits for each feature
    x_limits = {
        "intergenic": 8000,
        "intron": 2000,
        "five_prime_utr": 1500,
        "cds": 2000,
        "three_prime_utr": 1500
    }
    
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features), sharex=False, sharey=False)
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(feature_names):
        if feature in prob_matrix.columns:
            # Get the data for this feature
            feature_data = prob_matrix[feature].copy()
            x_max = x_limits.get(feature, 2000)
            
            # Accumulate probability mass beyond x_max at x_max
            beyond_max = feature_data.index > x_max
            accumulated_prob = 0
            if beyond_max.any():
                accumulated_prob = feature_data[beyond_max].sum()
                feature_data = feature_data[~beyond_max]
                if x_max in feature_data.index:
                    feature_data[x_max] += accumulated_prob
                else:
                    feature_data[x_max] = accumulated_prob
            
            # Plot the main distribution
            feature_data.plot(ax=axes[i], alpha=0.7, color=f"C{i}", linewidth=1)
            
            # Highlight the accumulated probability at x_max if significant
            if accumulated_prob > 0:
                axes[i].scatter([x_max], [feature_data[x_max]], 
                              color=f"C{i}", s=50, alpha=0.8, zorder=5)
            
            # Expand x-range slightly beyond the data
            x_padding = x_max * 0.05  # 5% padding
            axes[i].set_xlim(0, x_max + x_padding)
        
        axes[i].set_ylabel("Probability")
        axes[i].set_title(f"{feature} Length Distribution")
        axes[i].grid(True, alpha=0.3)
        # Add log scaling only for intergenic
        if feature == "intergenic":
            axes[i].set_yscale("log")
    
    axes[-1].set_xlabel("Length (bp)")
    plt.tight_layout()
    plt.savefig("local/scratch/feature_length_probabilities.png", dpi=300, bbox_inches="tight")
    logger.info("Saved feature length probability plot to local/scratch/feature_length_probabilities.png")
    plt.close()
    
    print()

def save_transition_stats(datasets: list[xr.Dataset]) -> None:
    logger.info(f"Computing transition counts for {len(datasets)} datasets")
    transition_counts, feature_lengths = compute_transition_stats(datasets, SEQUENCE_MODELING_FEATURES, limit=None)

    path = TRANSITION_COUNTS_PATH
    logger.info(f"Saving transition counts to {path}")
    transition_counts.to_parquet(path)

    path = FEATURE_LENGTHS_PATH
    logger.info(f"Saving feature lengths to {path}")
    feature_lengths.to_parquet(path)


def load_datasets() -> list[xr.Dataset]:
    """Load datasets from the hard-coded zarr path."""
    path = LABELS_ZARR_PATH
    dt = open_datatree(path)
    return [dt.ds for dt in dt.subtree if dt.is_leaf]


def load_and_visualize_stats() -> None:
    """Load saved transition stats and visualize them."""
    logger.info(f"Loading transition counts from {TRANSITION_COUNTS_PATH}")
    counts = pd.read_parquet(TRANSITION_COUNTS_PATH)
    visualize_transition_stats(counts, SEQUENCE_MODELING_FEATURES)
    
    logger.info(f"Loading feature lengths from {FEATURE_LENGTHS_PATH}")
    lengths = pd.read_parquet(FEATURE_LENGTHS_PATH)
    visualize_feature_lengths(lengths, SEQUENCE_MODELING_FEATURES)


def load_and_format_stats() -> None:
    """Load saved transition stats and format them as numpy arrays."""
    logger.info(f"Loading transition counts from {TRANSITION_COUNTS_PATH}")
    counts = pd.read_parquet(TRANSITION_COUNTS_PATH)
    
    overall_stats = get_transition_stats(counts, groupby=[])
    probs = overall_stats.pivot(index="from_class", columns="to_class", values="probability")
    probs = probs.reindex(index=SEQUENCE_MODELING_FEATURES, columns=SEQUENCE_MODELING_FEATURES, fill_value=0.0)
    
    print("=== TRANSITION PROBABILITIES ===")
    print("Transition probabilities (16-bit precision):")
    print(format_transition_probs(probs, precision=16))
    print("\nTransition probabilities (8-bit precision):")
    print(format_transition_probs(probs, precision=8))
    print("\nTransition probabilities as percentages:")
    print(probs.map(lambda x: f"{x:.6%}" if x > 0 else ""))
    print()
    
    # Feature length distributions
    logger.info(f"Loading feature lengths from {FEATURE_LENGTHS_PATH}")
    lengths = pd.read_parquet(FEATURE_LENGTHS_PATH)
    
    # Aggregate feature lengths across all strands, species, and chromosomes
    logger.info("Aggregating feature length distributions")
    aggregated_lengths = (
        lengths.groupby(["class_name", "length"], as_index=False)["count"]
        .sum()
    )

    # Calculate probabilities for each class
    class_totals = aggregated_lengths.groupby("class_name")["count"].transform("sum")
    aggregated_lengths["probability"] = aggregated_lengths["count"] / class_totals
    
    # Verify that probabilities sum to approximately 1 for each class
    prob_sums = aggregated_lengths.groupby("class_name")["probability"].sum()
    for class_name, prob_sum in prob_sums.items():
        assert abs(prob_sum - 1.0) < 1e-10, f"Probabilities for {class_name} sum to {prob_sum}, not 1.0"
    logger.info(f"Verified that probabilities sum to 1.0 for all {len(prob_sums)} feature classes")

    logger.info(f"Feature length distributions:\n{aggregated_lengths}")
    
    # Save aggregated feature length distributions
    output_path = "local/data/feature_length_distributions.parquet"
    logger.info(f"Saving aggregated feature length distributions to {output_path}")
    aggregated_lengths.to_parquet(output_path, index=False)



# Path constants
LABELS_ZARR_PATH = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/transform/v0.4/labels.zarr"
TRANSITION_COUNTS_PATH = "local/scratch/transition_counts.parquet"
FEATURE_LENGTHS_PATH = "local/scratch/feature_lengths.parquet"


def main():
    parser = argparse.ArgumentParser(description="Compute and analyze feature transition statistics")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Save command
    save_parser = subparsers.add_parser("save", help="Compute and save transition statistics")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Load and visualize saved transition statistics")
    
    # Format command
    format_parser = subparsers.add_parser("format", help="Load and format transition statistics as numpy arrays")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    if args.command == "save":
        datasets = load_datasets()
        save_transition_stats(datasets)
    elif args.command == "visualize":
        load_and_visualize_stats()
    elif args.command == "format":
        load_and_format_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()