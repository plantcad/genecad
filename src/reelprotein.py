#!/usr/bin/env python3
import os
import glob
import gc
import logging
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import torch
import xgboost as xgb
from transformers import T5EncoderModel, T5Tokenizer
from huggingface_hub import snapshot_download

# Initialize module logger
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMBEDDING_WIDTH = 1024

# =============================================================================
# PART 1: GENE PARSING & ORF EXTRACTION
# =============================================================================


def _iter_parent_ids(parent_attr: str) -> list[str]:
    return [parent_id for parent_id in parent_attr.split(",") if parent_id]


def _parse_cds_phase(
    phase: str,
    gene_id: str | None,
    start: int,
    end: int,
) -> int:
    try:
        phase_int = int(phase)
    except ValueError as exc:
        raise ValueError(
            f"Invalid CDS phase {phase!r} for gene {gene_id!r} at {start}-{end}"
        ) from exc
    if phase_int not in {0, 1, 2}:
        raise ValueError(
            f"Invalid CDS phase {phase!r} for gene {gene_id!r} at {start}-{end}"
        )
    return phase_int


@dataclass(slots=True, frozen=True)
class FeatureRowRecord:
    seqid: str
    source: str
    feature_type: str
    start: int
    end: int
    score: str
    strand: str
    phase: str
    attributes: dict[str, str]
    row_index: int

    @property
    def feature_id(self) -> str | None:
        return self.attributes.get("ID")

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return tuple(_iter_parent_ids(self.attributes.get("Parent", "")))

    @classmethod
    def from_fields(cls, fields: list[str], row_index: int) -> "FeatureRowRecord":
        if len(fields) != 9:
            raise ValueError(f"Expected 9 GFF fields, got {len(fields)}")
        return cls(
            seqid=fields[0],
            source=fields[1],
            feature_type=fields[2],
            start=int(fields[3]),
            end=int(fields[4]),
            score=fields[5],
            strand=fields[6],
            phase=fields[7],
            attributes=parse_attributes(fields[8]),
            row_index=row_index,
        )

    def to_fields(self, attributes: dict[str, str] | None = None) -> list[str]:
        attrs = self.attributes if attributes is None else attributes
        return [
            self.seqid,
            self.source,
            self.feature_type,
            str(self.start),
            str(self.end),
            self.score,
            self.strand,
            self.phase,
            format_attributes(attrs),
        ]

    def copy(self, attributes: dict[str, str] | None = None) -> "FeatureRowRecord":
        attrs = self.attributes.copy() if attributes is None else attributes.copy()
        return FeatureRowRecord(
            seqid=self.seqid,
            source=self.source,
            feature_type=self.feature_type,
            start=self.start,
            end=self.end,
            score=self.score,
            strand=self.strand,
            phase=self.phase,
            attributes=attrs,
            row_index=self.row_index,
        )


@dataclass(slots=True, frozen=True)
class ParsedGffGraph:
    header: list[str]
    features: list[FeatureRowRecord]
    features_by_id: dict[str, FeatureRowRecord]
    children_by_parent_id: dict[str, list[FeatureRowRecord]]


def _build_feature_row(fields: list[str], row_index: int) -> FeatureRowRecord:
    return FeatureRowRecord.from_fields(fields, row_index)


def _describe_feature(feature: FeatureRowRecord) -> str:
    return (
        f"{feature.feature_type} at {feature.seqid}:{feature.start}-{feature.end} "
        f"(row_index={feature.row_index}, id={feature.feature_id!r}, "
        f"parents={feature.parent_ids!r})"
    )


def _read_gff_feature_graph(gff_path: os.PathLike[str] | str) -> ParsedGffGraph:
    """Parse all GFF feature rows into an order-independent parent/child graph.

    This reads the full file before resolving relationships so child features,
    transcripts, and genes can appear in any order. It raises on duplicate IDs
    and missing parents instead of silently dropping malformed rows.
    """
    with open(gff_path) as fh:
        lines = fh.readlines()

    header: list[str] = []
    feature_lines: list[str] = []
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            header = lines[:i]
            feature_lines = lines[i:]
            break

    features: list[FeatureRowRecord] = []
    for row_index, raw in enumerate(feature_lines):
        fields = raw.rstrip("\n").split("\t")
        if len(fields) != 9:
            logger.warning(
                "Skipping malformed GFF row in %s at feature index %d: expected 9 "
                "tab-delimited fields, got %d: %r",
                gff_path,
                row_index,
                len(fields),
                raw.rstrip("\n"),
            )
            continue

        features.append(_build_feature_row(fields, row_index))

    features_by_id: dict[str, FeatureRowRecord] = {}
    children_by_parent_id: defaultdict[str, list[FeatureRowRecord]] = defaultdict(list)

    for feature in features:
        feature_id = feature.feature_id
        if feature_id:
            if feature_id in features_by_id:
                raise ValueError(
                    f"Duplicate GFF feature ID {feature_id!r} found in {gff_path}"
                )
            features_by_id[feature_id] = feature

    for feature in features:
        for parent_id in feature.parent_ids:
            if parent_id not in features_by_id:
                raise ValueError(
                    f"{_describe_feature(feature)} references missing parent "
                    f"{parent_id!r} in {gff_path}"
                )
            children_by_parent_id[parent_id].append(feature)

    return ParsedGffGraph(
        header=header,
        features=features,
        features_by_id=features_by_id,
        children_by_parent_id=dict(children_by_parent_id),
    )


def _resolve_ancestor_gene_id(
    feature: FeatureRowRecord,
    graph: ParsedGffGraph,
    cache: dict[int, str | None],
    active_row_indexes: set[int] | None = None,
) -> str | None:
    """Resolve the unique ancestor gene ID for a feature by walking parent links.

    Results are memoized by `row_index` because many sibling features share the
    same lineage. Cyclic parent chains and features that resolve to multiple
    genes raise immediately.
    """
    if feature.row_index in cache:
        return cache[feature.row_index]

    if active_row_indexes is None:
        active_row_indexes = set()
    if feature.row_index in active_row_indexes:
        raise ValueError(
            f"Detected cyclic parent chain for {_describe_feature(feature)}"
        )

    feature_id = feature.feature_id
    if feature.feature_type == "gene":
        if feature_id is None:
            raise ValueError(
                f"Gene feature is missing ID: {_describe_feature(feature)}"
            )
        cache[feature.row_index] = feature_id
        return feature_id

    active_row_indexes.add(feature.row_index)
    ancestor_gene_ids: set[str] = set()
    for parent_id in feature.parent_ids:
        parent = graph.features_by_id[parent_id]
        gene_id = _resolve_ancestor_gene_id(
            parent,
            graph,
            cache,
            active_row_indexes,
        )
        if gene_id is not None:
            ancestor_gene_ids.add(gene_id)
    active_row_indexes.remove(feature.row_index)

    if len(ancestor_gene_ids) > 1:
        raise ValueError(
            f"{_describe_feature(feature)} resolves to multiple ancestor genes: "
            f"{sorted(ancestor_gene_ids)!r}"
        )

    gene_id = next(iter(ancestor_gene_ids), None)
    cache[feature.row_index] = gene_id
    return gene_id


def _group_features_by_ancestor_gene(
    graph: ParsedGffGraph,
) -> tuple[dict[str, FeatureRowRecord], dict[str, list[FeatureRowRecord]]]:
    """Group non-gene features by their resolved ancestor gene.

    The returned child lists preserve input row order so downstream writers can
    reproduce the original feature ordering for each gene.
    """
    gene_features_by_id: dict[str, FeatureRowRecord] = {}
    features_by_gene_id: defaultdict[str, list[FeatureRowRecord]] = defaultdict(list)
    gene_cache: dict[int, str | None] = {}

    for feature in graph.features:
        if feature.feature_type == "gene":
            feature_id = feature.feature_id
            if feature_id is None:
                raise ValueError(
                    f"Gene feature is missing ID: {_describe_feature(feature)}"
                )
            gene_features_by_id[feature_id] = feature
            continue

        gene_id = _resolve_ancestor_gene_id(feature, graph, gene_cache)
        if gene_id is None:
            raise ValueError(
                f"Could not resolve ancestor gene for {_describe_feature(feature)}"
            )
        features_by_gene_id[gene_id].append(feature)

    for gene_id in features_by_gene_id:
        features_by_gene_id[gene_id].sort(key=lambda feature: feature.row_index)

    return gene_features_by_id, dict(features_by_gene_id)


@dataclass(slots=True, frozen=True)
class CdsSpanRecord:
    start: int
    end: int
    phase: int


@dataclass(slots=True, frozen=True)
class UtrSpanRecord:
    start: int
    end: int


@dataclass(slots=True, frozen=True)
class OrfGeneRecord:
    gene_id: str | None
    start: int
    end: int
    strand: str
    cds: list[CdsSpanRecord] = field(default_factory=list)
    utr5: list[UtrSpanRecord] = field(default_factory=list)
    utr3: list[UtrSpanRecord] = field(default_factory=list)
    feature_types: set[str] = field(default_factory=set)


def build_orf_gene_index(
    gff_file: os.PathLike[str] | str,
) -> dict[str, list[OrfGeneRecord]]:
    """Build the compact gene view used for ORF extraction.

    This projects the generic GFF graph down to gene spans plus CDS and UTR
    segments, keyed by chromosome and sorted by genomic start position.
    """
    graph = _read_gff_feature_graph(gff_file)
    gene_features_by_id, features_by_gene_id = _group_features_by_ancestor_gene(graph)
    genes: defaultdict[str, list[OrfGeneRecord]] = defaultdict(list)

    for gene_id, gene_feature in gene_features_by_id.items():
        cds: list[CdsSpanRecord] = []
        utr5: list[UtrSpanRecord] = []
        utr3: list[UtrSpanRecord] = []
        feature_types: set[str] = set()

        for feature in features_by_gene_id.get(gene_id, []):
            if feature.feature_type == "CDS":
                cds.append(
                    CdsSpanRecord(
                        start=feature.start,
                        end=feature.end,
                        phase=_parse_cds_phase(
                            feature.phase,
                            gene_id,
                            feature.start,
                            feature.end,
                        ),
                    )
                )
                feature_types.add(feature.feature_type)
            elif feature.feature_type == "five_prime_UTR":
                utr5.append(UtrSpanRecord(start=feature.start, end=feature.end))
                feature_types.add(feature.feature_type)
            elif feature.feature_type == "three_prime_UTR":
                utr3.append(UtrSpanRecord(start=feature.start, end=feature.end))
                feature_types.add(feature.feature_type)

        cds.sort(key=lambda feature: feature.start)
        utr5.sort(key=lambda feature: feature.start)
        utr3.sort(key=lambda feature: feature.start)
        genes[gene_feature.seqid].append(
            OrfGeneRecord(
                gene_id=gene_id,
                start=gene_feature.start,
                end=gene_feature.end,
                strand=gene_feature.strand,
                cds=cds,
                utr5=utr5,
                utr3=utr3,
                feature_types=feature_types,
            )
        )

    for chrom in genes:
        genes[chrom].sort(key=lambda gene: gene.start)

    return genes


def build_cds_string_for_gene(gene: OrfGeneRecord, chrom_seq: Seq) -> str:
    strand = gene.strand
    cds_list = gene.cds
    if not cds_list:
        return ""

    if strand == "+":
        sorted_cds = sorted(cds_list, key=lambda feature: feature.start)
    else:
        sorted_cds = sorted(cds_list, key=lambda feature: feature.start, reverse=True)

    # Explicit type annotation to satisfy pyrefly check
    parts: list[str] = []

    for cds in sorted_cds:
        a, b = cds.start, cds.end
        # BioPython seq slicing is 0-based, GFF is 1-based
        raw_seq = chrom_seq[a - 1 : b]
        seq_str = str(raw_seq).upper()
        if strand == "+":
            trimmed = seq_str[cds.phase :]
        else:
            rc = str(Seq(seq_str).reverse_complement())
            trimmed = rc[cds.phase :]
        parts.append(trimmed)

    return "".join(parts)


def extract_candidate_proteins(genes, genome_fasta):
    """
    Returns a dictionary {merged_id: protein_sequence_string}
    """
    STOPS = ("TAA", "TAG", "TGA")

    def is_complete_orf(nuc: str) -> bool:
        return len(nuc) >= 3 and nuc.startswith("ATG") and nuc[-3:] in STOPS

    logger.info(f"[Step 1] Loading Genome: {genome_fasta}")
    seqs = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))

    protein_candidates = {}

    for chrom, gene_list in genes.items():
        if chrom not in seqs:
            logger.warning(f"[WARN] Chromosome {chrom} not found in FASTA; skipping.")
            continue
        chrom_seq = seqs[chrom].seq
        id_to_index = {gene.gene_id: idx for idx, gene in enumerate(gene_list)}

        for strand in ("+", "-"):
            st_genes = sorted(
                (gene for gene in gene_list if gene.strand == strand),
                key=lambda gene: gene.start,
                reverse=(strand == "-"),
            )

            cds_strings = [build_cds_string_for_gene(g, chrom_seq) for g in st_genes]
            n = len(st_genes)

            for i in range(n):
                if not cds_strings[i]:
                    continue

                running = ""
                chain = [st_genes[i]]

                for j in range(i, n):
                    if j > i:
                        prev = st_genes[j - 1]
                        curr = st_genes[j]

                        # Stop if previous gene is a complete annotated model
                        required = {"five_prime_UTR", "CDS", "three_prime_UTR"}
                        if required.issubset(prev.feature_types):
                            break

                        # Stop if gap > 1 gene index
                        prev_idx = id_to_index.get(prev.gene_id)
                        curr_idx = id_to_index.get(curr.gene_id)
                        if (
                            prev_idx is None
                            or curr_idx is None
                            or abs(curr_idx - prev_idx) != 1
                        ):
                            break
                        chain.append(curr)

                    if not cds_strings[j]:
                        break
                    running += cds_strings[j]

                    if (
                        len(running) >= 3
                        and running.startswith("ATG")
                        and running[-3:] in STOPS
                    ):
                        # If merged and trailing gene is already valid, skip
                        if j > i and any(
                            is_complete_orf(cds_strings[k]) for k in range(i + 1, j + 1)
                        ):
                            break

                        # Generate ID and Protein
                        prot = str(Seq(running).translate(to_stop=False))
                        # Replace special characters in ID for file safety logic
                        header_ids = "|".join(gene.gene_id for gene in chain)
                        merged_id = f"{chrom}_{header_ids}_{strand}"

                        # Sanitize sequence for embedding
                        prot_clean = (
                            prot.replace("U", "X")
                            .replace("Z", "X")
                            .replace("O", "X")
                            .replace("-", "")
                        )
                        protein_candidates[merged_id] = prot_clean
                        break

    logger.info(
        f"[Step 1] Found {len(protein_candidates)} candidate sequences (single + merged)."
    )
    return protein_candidates


# =============================================================================
# PART 2: PROTTRANS EMBEDDING
# =============================================================================


def get_prot_t5_model():
    logger.info("[Step 2] Loading ProtT5 Model...")
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    return model, tokenizer


def generate_embeddings(seqs_dict, max_residues=4000, max_seq_len=1000, max_batch=1):
    """
    Returns a pandas DataFrame where index is ProteinID and columns are embedding features.
    """
    seq_items = sorted(seqs_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

    model, tokenizer = get_prot_t5_model()
    # TODO: Derive this from the loaded model config instead of keeping it hard-coded.
    cols = ["ProteinID"] + list(range(EMBEDDING_WIDTH))

    results_list = []
    batch = []
    logger.info(f"[Step 2] Generating embeddings for {len(seq_items)} sequences...")

    for seq_idx, (pdb_id, seq) in enumerate(seq_items, 1):
        seq_len = len(seq)
        seq_spaced = " ".join(list(seq))
        batch.append((pdb_id, seq_spaced, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

        # Process batch
        if (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or seq_idx == len(seq_items)
            or seq_len > max_seq_len
        ):
            pdb_ids, batch_seqs, batch_lens = zip(*batch)
            batch = []  # reset

            token_encoding = tokenizer.batch_encode_plus(
                batch_seqs, add_special_tokens=True, padding="longest"
            )
            input_ids = torch.tensor(token_encoding["input_ids"]).to(device)
            attention_mask = torch.tensor(token_encoding["attention_mask"]).to(device)

            with torch.no_grad():
                embedding_repr = model(input_ids, attention_mask=attention_mask)

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = batch_lens[batch_idx]
                # Slice off padding -> avg pool
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                protein_emb = emb.mean(dim=0).detach().cpu().numpy().squeeze()

                # Append to results
                results_list.append([identifier] + protein_emb.tolist())

    logger.info("[Step 2] Embedding generation complete.")
    df = pd.DataFrame(results_list, columns=cols)

    # Cleanup GPU memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return df


# =============================================================================
# PART 3: XGBOOST SCORING (With HF Support)
# =============================================================================


def score_proteins(features_df, model_source):
    """
    Scores proteins using XGBoost models downloaded from Hugging Face.
    model_source: Hugging Face Repo ID (e.g., 'plantcad/reelprotein').
    """
    logger.info(
        f"[Step 3] Downloading/Loading models from Hugging Face: {model_source}"
    )

    try:
        # Always use snapshot_download. It handles caching automatically.
        model_dir = snapshot_download(
            repo_id=model_source, allow_patterns=["*.json"], repo_type="model"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download models from Hugging Face ({model_source}): {e}"
        ) from e

    model_files = sorted(glob.glob(os.path.join(model_dir, "*.json")))

    if not model_files:
        raise FileNotFoundError(
            f"No .json models found in downloaded cache: {model_dir}"
        )

    # Separate IDs
    transcript_ids = features_df["ProteinID"].values
    # Drop ID column, keep only features.
    X = features_df.drop("ProteinID", axis=1)
    # Ensure columns are integers for XGBoost
    X.columns = range(len(X.columns))

    results_df = pd.DataFrame({"ProteinID": transcript_ids})

    for model_file in model_files:
        model_name = os.path.basename(model_file).replace(".json", "")
        model = xgb.XGBClassifier()
        model.load_model(model_file)
        # Predict
        pred_score = model.predict_proba(X)[:, 1]
        results_df[model_name] = pred_score

    # Calculate Mean
    score_columns = results_df.drop("ProteinID", axis=1)
    results_df["Mean_Score"] = score_columns.mean(axis=1)
    results_df["Predicted_Label"] = (results_df["Mean_Score"] >= 0.5).astype(int)

    logger.info("[Step 3] Scoring complete.")
    return results_df


# =============================================================================
# PART 4: GFF FILTRATION & MERGING
# =============================================================================


def parse_attributes(attr_str: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in attr_str.strip().split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k] = v
    return attrs


def format_attributes(attr_dict: dict[str, str]) -> str:
    return ";".join(f"{k}={v}" for k, v in attr_dict.items())


@dataclass(slots=True, frozen=True)
class GeneEntryRecord:
    gene_line: FeatureRowRecord
    subfeatures: list[FeatureRowRecord]
    attributes: dict[str, str]


def read_gff_gene_entries(
    gff_path: os.PathLike[str] | str,
) -> tuple[list[str], dict[str, GeneEntryRecord]]:
    """Build the gene-entry view used to rewrite the final output GFF.

    This preserves the original gene row and its descendant features after
    order-independent parent resolution, while copying records so output edits
    do not mutate the parsed graph.
    """
    graph = _read_gff_feature_graph(gff_path)
    gene_features_by_id, features_by_gene_id = _group_features_by_ancestor_gene(graph)
    gene_entries: dict[str, GeneEntryRecord] = {}

    for gene_id, gene in gene_features_by_id.items():
        gene_entries[gene_id] = GeneEntryRecord(
            gene_line=gene.copy(),
            subfeatures=[
                feature.copy() for feature in features_by_gene_id.get(gene_id, [])
            ],
            attributes=gene.attributes.copy(),
        )

    return graph.header, gene_entries


def merge_group_transcripts(
    grp: tuple[str, ...],
    gene_entries: dict[str, GeneEntryRecord],
    synthetic_mrna_id: str,
) -> tuple[
    FeatureRowRecord | None,
    list[tuple[FeatureRowRecord, dict[str, str]]],
]:
    """Create one synthetic mRNA row for a merged gene group.

    The synthetic transcript spans the min/max coordinates of the original
    transcripts. Child features are reused with copied attributes so only their
    `Parent` links change in the emitted output.
    """
    original_mrnas: list[FeatureRowRecord] = []
    other_children: list[FeatureRowRecord] = []
    for g in grp:
        if g not in gene_entries:
            continue
        for feature in gene_entries[g].subfeatures:
            if feature.feature_type == "mRNA":
                original_mrnas.append(feature)
            else:
                other_children.append(feature)

    if not original_mrnas:
        return None, []

    mrna_starts = [feature.start for feature in original_mrnas]
    mrna_ends = [feature.end for feature in original_mrnas]
    synthetic_start = min(mrna_starts)
    synthetic_end = max(mrna_ends)

    template = original_mrnas[0]
    synthetic_mrna = FeatureRowRecord(
        seqid=template.seqid,
        source=template.source,
        feature_type="mRNA",
        start=synthetic_start,
        end=synthetic_end,
        score=template.score,
        strand=template.strand,
        phase=template.phase,
        attributes={"ID": synthetic_mrna_id},
        row_index=template.row_index,
    )

    merged_children: list[tuple[FeatureRowRecord, dict[str, str]]] = []
    for feature in other_children:
        attrs = feature.attributes.copy()
        attrs["Parent"] = synthetic_mrna_id
        merged_children.append((feature, attrs))

    merged_children.sort(
        key=lambda item: item[0].start,
        reverse=(template.strand == "-"),
    )
    return synthetic_mrna, merged_children


def generate_final_gff(predictions_df, input_gff_path, output_gff_path):
    logger.info("[Step 4] Filtering and merging GFF based on predictions...")

    # Filter for positive predictions
    df1 = predictions_df[predictions_df["Predicted_Label"] == 1].copy()
    if df1.empty:
        logger.warning("[WARN] No positive predictions found. Writing empty GFF.")
        open(output_gff_path, "w").close()
        return

    # Extract Gene groupings from ProteinIDs
    single_genes = set()
    group_map = {}
    gene_to_group = {}
    group_strand = {}

    for seq_id in df1["ProteinID"]:
        if "_" not in seq_id:
            continue
        # ID Format: Chrom_GeneA|GeneB_Strand
        base, strand = seq_id.rsplit("_", 1)
        if "_" in base:
            _, gene_part = base.split("_", 1)
        else:
            gene_part = base

        genes = gene_part.split("|")
        if len(genes) == 1:
            single_genes.add(genes[0])
        else:
            key = tuple(genes)
            group_strand[key] = strand
            new_id = "concat_" + "_".join(genes)
            group_map[key] = new_id
            for g in genes:
                gene_to_group[g] = key

    single_genes.difference_update(gene_to_group)

    # Read original GFF
    header, gene_entries = read_gff_gene_entries(input_gff_path)

    # Build Output Items
    items = []
    for g in sorted(single_genes):
        if g in gene_entries:
            start = gene_entries[g].gene_line.start
            items.append({"type": "single", "id": g, "start": start})

    for group, new_id in group_map.items():
        member_entries = [gene_entries[g] for g in group if g in gene_entries]
        if not member_entries:
            continue
        starts = [entry.gene_line.start for entry in member_entries]
        ends = [entry.gene_line.end for entry in member_entries]
        items.append(
            {
                "type": "group",
                "genes": group,
                "new_id": new_id,
                "start": min(starts),
                "end": max(ends),
                "strand": group_strand.get(group, member_entries[0].gene_line.strand),
            }
        )

    items.sort(key=lambda x: x["start"])

    # Write Output
    with open(output_gff_path, "w") as out:
        out.writelines(header)
        for itm in items:
            if itm["type"] == "single":
                gene_entry = gene_entries[itm["id"]]
                out.write("\t".join(gene_entry.gene_line.to_fields()) + "\n")
                for feature in gene_entry.subfeatures:
                    out.write("\t".join(feature.to_fields()) + "\n")
            else:
                # Group logic
                grp = itm["genes"]
                gid = itm["new_id"]
                strand = itm.get("strand")
                first_valid = next((g for g in grp if g in gene_entries), None)
                if not first_valid:
                    continue

                first = gene_entries[first_valid].gene_line

                gene_attrs = {"ID": gid, "Note": "concatenated"}
                gene_fields = [
                    first.seqid,
                    first.source,
                    "gene",
                    str(itm["start"]),
                    str(itm["end"]),
                    first.score,
                    strand if strand else first.strand,
                    first.phase,
                    format_attributes(gene_attrs),
                ]
                out.write("\t".join(gene_fields) + "\n")

                synthetic_mrna_id = f"{gid}.t1"
                mrna_feature, merged_children = merge_group_transcripts(
                    grp, gene_entries, synthetic_mrna_id
                )

                if mrna_feature:
                    mrna_attrs = mrna_feature.attributes.copy()
                    mrna_attrs["Parent"] = gid
                    out.write("\t".join(mrna_feature.to_fields(mrna_attrs)) + "\n")
                    for feature, attrs in merged_children:
                        out.write("\t".join(feature.to_fields(attrs)) + "\n")
                else:
                    # Fallback
                    for g in grp:
                        if g not in gene_entries:
                            continue
                        for feature in gene_entries[g].subfeatures:
                            attributes = feature.attributes.copy()
                            if feature.feature_type == "mRNA":
                                attributes["Parent"] = gid
                            out.write("\t".join(feature.to_fields(attributes)) + "\n")

    logger.info(f"[Step 4] Final GFF written to {output_gff_path}")
