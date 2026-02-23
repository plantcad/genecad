#!/usr/bin/env python3
import os
import glob
import gc
import logging
from collections import defaultdict
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

# =============================================================================
# PART 1: GENE PARSING & ORF EXTRACTION
# =============================================================================


def parse_gff3(gff_file):
    genes = defaultdict(list)
    with open(gff_file) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 9:
                continue
            chrom, src, feature, start, end, score, strand, phase, attr = parts
            start, end = int(start), int(end)
            attrs = dict(kv.split("=", 1) for kv in attr.split(";") if "=" in kv)

            if feature == "gene":
                genes[chrom].append(
                    {
                        "id": attrs.get("ID"),
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "cds": [],
                        "utr5": [],
                        "utr3": [],
                        "feature_types": set(),
                    }
                )
            elif feature in ("CDS", "five_prime_UTR", "three_prime_UTR"):
                parent = attrs.get("Parent", "")
                for g in genes[chrom]:
                    if g["id"] and parent.startswith(g["id"] + "."):
                        g["feature_types"].add(feature)
                        if feature == "CDS":
                            try:
                                phase_int = int(phase)
                            except ValueError:
                                phase_int = 0
                            g["cds"].append(
                                {"start": start, "end": end, "phase": phase_int}
                            )
                        elif feature == "five_prime_UTR":
                            g["utr5"].append({"start": start, "end": end})
                        elif feature == "three_prime_UTR":
                            g["utr3"].append({"start": start, "end": end})
                        break

    # Deterministic ordering
    for chrom in genes:
        genes[chrom].sort(key=lambda g: g["start"])
        for g in genes[chrom]:
            g["cds"].sort(key=lambda x: x["start"])
            g["utr5"].sort(key=lambda x: x["start"])
            g["utr3"].sort(key=lambda x: x["start"])
    return genes


def build_cds_string_for_gene(gene, chrom_seq):
    strand = gene["strand"]
    cds_list = gene["cds"]
    if not cds_list:
        return ""

    if strand == "+":
        sorted_cds = sorted(cds_list, key=lambda x: x["start"])
    else:
        sorted_cds = sorted(cds_list, key=lambda x: x["start"], reverse=True)

    # Explicit type annotation to satisfy pyrefly check
    parts: list[str] = []

    for cds in sorted_cds:
        a, b = cds["start"], cds["end"]
        # BioPython seq slicing is 0-based, GFF is 1-based
        raw_seq = chrom_seq[a - 1 : b]
        seq_str = str(raw_seq).upper()
        if strand == "+":
            trimmed = seq_str[cds["phase"] :]
        else:
            rc = str(Seq(seq_str).reverse_complement())
            trimmed = rc[cds["phase"] :]
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
        for strand in ("+", "-"):
            st_genes = sorted(
                (g for g in gene_list if g["strand"] == strand),
                key=lambda g: g["start"],
                reverse=(strand == "-"),
            )
            
            # Use strand-specific indexing to avoid opposite-strand genes breaking contiguous chains
            id_to_index = {g["id"]: idx for idx, g in enumerate(st_genes)}

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

                        # Removed the strict required.issubset check here so that fragments
                        # with erroneously predicted UTRs can still be merged into a larger gene
                        # if they form a proper continuous orf.

                        # Stop if gap > 1 gene index
                        prev_idx = id_to_index.get(prev["id"])
                        curr_idx = id_to_index.get(curr["id"])
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
                        header_ids = "|".join(g["id"] for g in chain)
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
    Returns a pandas DataFrame where index is ProteinID and columns are 0-1023 (features).
    """
    model, tokenizer = get_prot_t5_model()

    # Sort sequences by length (descending)
    seq_items = sorted(seqs_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

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

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError as e:
                logger.error(f"RuntimeError during embedding: {e}")
                continue

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = batch_lens[batch_idx]
                # Slice off padding -> avg pool
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                protein_emb = emb.mean(dim=0).detach().cpu().numpy().squeeze()

                # Append to results
                results_list.append([identifier] + protein_emb.tolist())

    logger.info("[Step 2] Embedding generation complete.")
    cols = ["ProteinID"] + list(range(1024))
    df = pd.DataFrame(results_list, columns=cols)

    # Cleanup GPU memory
    del model
    del tokenizer
    del embedding_repr
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


def parse_attributes(attr_str):
    attrs = {}
    for part in attr_str.strip().split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k] = v
    return attrs


def format_attributes(attr_dict):
    return ";".join(f"{k}={v}" for k, v in attr_dict.items())


def read_gff_raw(gff_path):
    with open(gff_path) as f:
        lines = f.readlines()
    header = []
    feats = []
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            header = lines[:i]
            feats = lines[i:]
            break

    gene_entries = {}
    for line in feats:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 9:
            continue
        feature_type = fields[2]
        attrs = parse_attributes(fields[8])

        if feature_type == "gene":
            gid = attrs.get("ID")
            if gid:
                gene_entries[gid] = {
                    "gene_line": fields,
                    "subfeatures": [],
                    "attributes": attrs,
                }
        else:
            parent = attrs.get("Parent", "")
            if not parent:
                continue
            if feature_type == "mRNA":
                parent_gene = parent
            else:
                parent_gene = parent.split(".")[0]

            if parent_gene in gene_entries:
                gene_entries[parent_gene]["subfeatures"].append((fields, attrs))
    return header, gene_entries


def merge_group_transcripts(grp, gene_entries, synthetic_mrna_id):
    original_mrnas = []
    other_children = []
    for g in grp:
        if g not in gene_entries:
            continue
        for fields, attrs in gene_entries[g]["subfeatures"]:
            if fields[2] == "mRNA":
                original_mrnas.append((fields.copy(), attrs.copy(), g))
            else:
                other_children.append((fields.copy(), attrs.copy()))

    if not original_mrnas:
        return None, []

    mrna_starts = [int(f[0][3]) for f in original_mrnas]
    mrna_ends = [int(f[0][4]) for f in original_mrnas]
    synthetic_start = min(mrna_starts)
    synthetic_end = max(mrna_ends)

    template_fields = original_mrnas[0][0]
    seqid, source, _, _, _, score, strand, phase, _ = template_fields

    synthetic_attrs = {"ID": synthetic_mrna_id}
    synthetic_fields = [
        seqid,
        source,
        "mRNA",
        str(synthetic_start),
        str(synthetic_end),
        score,
        strand,
        phase,
        format_attributes(synthetic_attrs),
    ]

    merged_children = []
    for fields, attrs in other_children:
        attrs["Parent"] = synthetic_mrna_id
        fields[8] = format_attributes(attrs)
        merged_children.append((fields, attrs))

    merged_children.sort(key=lambda item: int(item[0][3]), reverse=(strand == "-"))
    return synthetic_fields, merged_children


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

    # Read original GFF
    header, gene_entries = read_gff_raw(input_gff_path)

    # Build Output Items
    items = []
    for g in sorted(single_genes):
        if g in gene_entries:
            start = int(gene_entries[g]["gene_line"][3])
            items.append({"type": "single", "id": g, "start": start})

    for group, new_id in group_map.items():
        member_entries = [gene_entries[g] for g in group if g in gene_entries]
        if not member_entries:
            continue
        starts = [int(e["gene_line"][3]) for e in member_entries]
        ends = [int(e["gene_line"][4]) for e in member_entries]
        items.append(
            {
                "type": "group",
                "genes": group,
                "new_id": new_id,
                "start": min(starts),
                "end": max(ends),
                "strand": group_strand.get(group, member_entries[0]["gene_line"][6]),
            }
        )

    items.sort(key=lambda x: x["start"])

    # Write Output
    with open(output_gff_path, "w") as out:
        out.writelines(header)
        for itm in items:
            if itm["type"] == "single":
                ge = gene_entries[itm["id"]]
                out.write("\t".join(ge["gene_line"]) + "\n")
                for fields, attrs in ge["subfeatures"]:
                    fields[8] = format_attributes(attrs)
                    out.write("\t".join(fields) + "\n")
            else:
                # Group logic
                grp = itm["genes"]
                gid = itm["new_id"]
                strand = itm.get("strand")
                first_valid = next((g for g in grp if g in gene_entries), None)
                if not first_valid:
                    continue

                first = gene_entries[first_valid]["gene_line"]
                seqid, source, _, _, _, score, _, phase, _ = first

                gene_attrs = {"ID": gid, "Note": "concatenated"}
                gene_fields = [
                    seqid,
                    source,
                    "gene",
                    str(itm["start"]),
                    str(itm["end"]),
                    score,
                    strand if strand else first[6],
                    phase,
                    format_attributes(gene_attrs),
                ]
                out.write("\t".join(gene_fields) + "\n")

                synthetic_mrna_id = f"{gid}.t1"
                mrna_fields, merged_children = merge_group_transcripts(
                    grp, gene_entries, synthetic_mrna_id
                )

                if mrna_fields:
                    mrna_attrs = parse_attributes(mrna_fields[8])
                    mrna_attrs["Parent"] = gid
                    mrna_fields[8] = format_attributes(mrna_attrs)
                    out.write("\t".join(mrna_fields) + "\n")
                    for fields, _ in merged_children:
                        out.write("\t".join(fields) + "\n")
                else:
                    # Fallback
                    for g in grp:
                        if g not in gene_entries:
                            continue
                        for fields, attrs in gene_entries[g]["subfeatures"]:
                            if fields[2] == "mRNA":
                                attrs["Parent"] = gid
                            fields[8] = format_attributes(attrs)
                            out.write("\t".join(fields) + "\n")

    logger.info(f"[Step 4] Final GFF written to {output_gff_path}")
