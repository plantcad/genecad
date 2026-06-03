#!/usr/bin/env python3
import gzip
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

# Default device (kept for get_prot_t5_model backward-compat; new code uses gpus= parameter)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# PART 1: GENE PARSING & ORF EXTRACTION
# =============================================================================


def parse_gff3(gff_file):
    genes = defaultdict(list)
    transcript_to_gene = {}
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
            elif feature == "mRNA":
                tx_id = attrs.get("ID")
                parent_gene = attrs.get("Parent")
                if tx_id and parent_gene:
                    transcript_to_gene[(chrom, tx_id)] = parent_gene
            elif feature in ("CDS", "five_prime_UTR", "three_prime_UTR"):
                parent = attrs.get("Parent", "")
                parent_gene = transcript_to_gene.get((chrom, parent))
                if parent_gene is None:
                    # Fallback for IDs like gene1.t1 where gene ID can be inferred.
                    parent_gene = parent.rsplit(".", 1)[0] if "." in parent else parent
                for g in genes[chrom]:
                    if g["id"] and g["id"] == parent_gene:
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
        if len(nuc) < 6:
            return False
        if len(nuc) % 3 != 0:
            return False
        if not nuc.startswith("ATG") or nuc[-3:] not in STOPS:
            return False
        for idx in range(3, len(nuc) - 3, 3):
            if nuc[idx : idx + 3] in STOPS:
                return False
        return True

    def is_high_confidence_merge(
        merged_orf: str, individual_orfs: list, chain_len: int
    ) -> bool:
        """
        Apply additional validation for multi-gene merges.
        We want to avoid low-confidence merges that combine many small fragments.

        Returns True only if:
        1. The merged ORF is reasonably longer than any individual gene
        2. The protein sequence is plausible (not too many rare codons or anomalies)
        3. For boundaries (0, 1, 2 offsets), prefer offset=0 unless there's strong evidence
        """
        if chain_len == 1:
            return True  # Single genes always pass

        merged_len = len(merged_orf)
        max_individual = max((len(o) for o in individual_orfs if o), default=0)

        # Merged ORF should be notably longer than the longest individual
        if merged_len <= max_individual:
            logger.debug(
                f"Merge rejected: merged_len ({merged_len}) <= max_individual ({max_individual})"
            )
            return False

        # Heuristic: merged should be at least 1.3x the longest individual for multi-fragment merges
        if chain_len > 2 and merged_len < max_individual * 1.3:
            logger.debug(
                f"Merge rejected: chain_len={chain_len} but merged_len ({merged_len}) < 1.3 * max_individual ({max_individual})"
            )
            return False

        return True

    def is_near_boundary(prev_gene: dict, curr_gene: dict, strand: str) -> bool:
        if strand == "+":
            gap = curr_gene["start"] - prev_gene["end"] - 1
        else:
            gap = prev_gene["start"] - curr_gene["end"] - 1
        return -2 <= gap <= 2

    def is_canonical_junction(
        chrom_seq, prev_gene: dict, curr_gene: dict, strand: str
    ) -> tuple:
        """
        Validate that genes can be merged at a canonical splice boundary.
        Returns (is_valid, offset_used, reason)

        Canonical splice dinucleotides: GT-AG (major), GC-AG, AT-AC
        For proper merging, we check:
        1. The junction region contains canonical splice sites
        2. The reading frame remains consistent
        3. Stop codons are not introduced at the boundary
        """
        canonical_donors = ("GT", "GC", "AT")  # 5' splice site (donor)
        canonical_acceptors = ("AG",)  # 3' splice site (acceptor)

        if strand == "+":
            # For forward strand: check donor site at end of prev_gene, acceptor at start of curr_gene
            prev_end = prev_gene["end"]
            curr_start = curr_gene["start"]
            gap = curr_start - prev_end - 1

            # Extract junction regions (±2 bp around boundary)
            try:
                # Donor site (end of prev gene): should have GT/GC/AT
                donor_region = str(chrom_seq[prev_end - 1 : prev_end + 1]).upper()
                # Acceptor site (start of curr gene): should have AG
                acceptor_region = str(
                    chrom_seq[curr_start - 3 : curr_start - 1]
                ).upper()

                is_canonical = (
                    donor_region in canonical_donors
                    and acceptor_region in canonical_acceptors
                )
            except (IndexError, TypeError):
                is_canonical = False
        else:
            # For reverse strand: coordinates are flipped
            prev_start = prev_gene["start"]
            curr_end = curr_gene["end"]
            gap = prev_start - curr_end - 1

            try:
                # Reverse complement the regions to check canonical sites
                donor_region = str(chrom_seq[curr_end - 2 : curr_end]).upper()
                donor_rc = str(Seq(donor_region).reverse_complement())
                acceptor_region = str(chrom_seq[prev_start : prev_start + 2]).upper()
                acceptor_rc = str(Seq(acceptor_region).reverse_complement())

                is_canonical = (
                    donor_rc in canonical_donors and acceptor_rc in canonical_acceptors
                )
            except (IndexError, TypeError):
                is_canonical = False

        # Log boundary analysis for debugging
        if -2 <= gap <= 2:
            logger.debug(
                f"Junction gap={gap}, canonical={is_canonical}, donor={donor_region if strand == '+' else donor_rc}, acceptor={acceptor_region if strand == '+' else acceptor_rc}"
            )

        return is_canonical, gap

    logger.info(f"[Step 1] Loading Genome: {genome_fasta}")
    _open = gzip.open if genome_fasta.endswith(".gz") else open
    with _open(genome_fasta, "rt") as _fh:
        seqs = SeqIO.to_dict(SeqIO.parse(_fh, "fasta"))

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

                    running_before = running
                    running += cds_strings[j]

                    # For the chain anchor (j == i) only offset 0 is valid —
                    # we never trim the first fragment.  For every subsequent
                    # fragment try offsets 0, 1, 2: a slight CDS boundary
                    # mis-prediction shifts the reading frame at the junction
                    # by 1–2 bases, and the correct offset produces a clean
                    # in-frame ORF.  `running` (offset 0) is always kept as
                    # the accumulator so chain extension to j+1 stays consistent.
                    if j == i:
                        candidates = [running]
                    else:
                        prev = st_genes[j - 1]
                        curr = st_genes[j]

                        # Strict boundary testing with canonical splice junction validation
                        offsets = [0]
                        use_offsets_1_2 = False

                        if is_near_boundary(prev, curr, strand):
                            # Only try offsets 1, 2 if the junction is canonical
                            is_canonical, gap = is_canonical_junction(
                                chrom_seq, prev, curr, strand
                            )
                            if is_canonical and -2 <= gap <= 2:
                                use_offsets_1_2 = True
                                logger.debug(
                                    f"Canonical junction found between {prev['id']} and {curr['id']} (gap={gap})"
                                )
                            else:
                                logger.debug(
                                    f"Non-canonical junction between {prev['id']} and {curr['id']} (gap={gap}, canonical={is_canonical}) — merging conservatively"
                                )

                        if use_offsets_1_2:
                            offsets = [0, 1, 2]

                        candidates = [
                            running_before + cds_strings[j][offset:]
                            for offset in offsets
                            if len(cds_strings[j]) > offset
                        ]

                    found_candidate = False
                    stop_chain = False
                    for offset_idx, candidate in enumerate(candidates):
                        if is_complete_orf(candidate):
                            # If merged and trailing gene is already valid, skip
                            if j > i and any(
                                is_complete_orf(cds_strings[k])
                                for k in range(i + 1, j + 1)
                            ):
                                logger.debug(
                                    f"Merge skipped: found valid individual ORF in chain at position {j}"
                                )
                                stop_chain = True
                                break

                            # High-confidence merge validation
                            cds_subset = [cds_strings[k] for k in range(i, j + 1)]
                            if not is_high_confidence_merge(
                                candidate, cds_subset, len(chain)
                            ):
                                logger.debug(
                                    f"Merge rejected by high-confidence filter (chain_len={len(chain)})"
                                )
                                continue

                            # Generate ID and Protein
                            prot = str(Seq(candidate).translate(to_stop=False))
                            # Replace special characters in ID for file safety logic
                            header_ids = "|".join(g["id"] for g in chain)
                            merged_id = f"{chrom}~{header_ids}~{strand}"

                            # Log merge decision
                            offset_used = (
                                [0, 1, 2][offset_idx]
                                if offset_idx < len([0, 1, 2])
                                else 0
                            )
                            if len(chain) > 1:
                                logger.info(
                                    f"[MERGE] {len(chain)} genes: {header_ids} (offset={offset_used})"
                                )

                            # Sanitize sequence for embedding
                            prot_clean = (
                                prot.replace("U", "X")
                                .replace("Z", "X")
                                .replace("O", "X")
                                .replace("-", "")
                            )
                            protein_candidates[merged_id] = prot_clean
                            found_candidate = True
                            break

                    if found_candidate or stop_chain:
                        break

    logger.info(
        f"[Step 1] Found {len(protein_candidates)} candidate sequences (single + merged)."
    )
    return protein_candidates


# =============================================================================
# PART 2: PROTTRANS EMBEDDING
# =============================================================================


def _embed_worker(args: tuple) -> list:
    """Embed a chunk of protein sequences on a single GPU.

    This is a module-level function so it can be pickled by ProcessPoolExecutor
    when spawning worker processes for multi-GPU embedding.

    Parameters
    ----------
    args : tuple
        (gpu_id, position, seq_items, max_residues, max_seq_len, max_batch)
        position: tqdm bar row (0 = top bar, 1 = second bar, …)

    Returns
    -------
    list
        Each element is [protein_id, feat_0, ..., feat_1023].
    """
    import logging as _logging
    from tqdm import tqdm

    gpu_id, position, seq_items, max_residues, max_seq_len, max_batch = args
    _logging.basicConfig(
        level=_logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    _log = _logging.getLogger(__name__)

    _device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    _log.info(
        f"[GPU {gpu_id}] Loading ProtT5 on {_device} ({len(seq_items)} sequences)..."
    )

    _model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    _model = _model.to(_device).eval()
    _tok = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    _log.info(f"[GPU {gpu_id}] Model loaded — starting inference...")

    results: list = []
    batch: list = []
    n_total = len(seq_items)

    pbar = tqdm(
        total=n_total,
        desc=f"GPU {gpu_id}",
        unit="seq",
        position=position,
        leave=True,
        dynamic_ncols=True,
    )

    for seq_idx, (pdb_id, seq) in enumerate(seq_items, 1):
        seq_len = len(seq)
        batch.append((pdb_id, " ".join(list(seq)), seq_len))
        n_res_batch = sum(s for _, _, s in batch) + seq_len

        if (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or seq_idx == n_total
            or seq_len > max_seq_len
        ):
            pdb_ids, batch_seqs, batch_lens = zip(*batch)
            batch = []

            enc = _tok(list(batch_seqs), add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(enc["input_ids"]).to(_device)
            attention_mask = torch.tensor(enc["attention_mask"]).to(_device)

            try:
                with torch.no_grad():
                    out = _model(input_ids, attention_mask=attention_mask)
            except RuntimeError as exc:
                _log.error(f"[GPU {gpu_id}] RuntimeError during embedding: {exc}")
                pbar.update(len(pdb_ids))
                continue

            for bi, pid in enumerate(pdb_ids):
                slen = batch_lens[bi]
                emb = (
                    out.last_hidden_state[bi, :slen]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                results.append([pid] + emb.tolist())

            pbar.update(len(pdb_ids))

    pbar.close()
    del _model, _tok
    torch.cuda.empty_cache()
    gc.collect()
    _log.info(f"[GPU {gpu_id}] Done — {len(results)} proteins embedded.")
    return results


def get_prot_t5_model():
    logger.info("[Step 2] Loading ProtT5 Model...")
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    return model, tokenizer


def generate_embeddings(
    seqs_dict, gpus=None, max_residues=None, max_seq_len=1300, max_batch=None
):
    """
    Returns a pandas DataFrame where index is ProteinID and columns are 0-1023 (features).

    Parameters
    ----------
    seqs_dict : dict
        {protein_id: sequence_string}
    gpus : list[int] | None
        GPU IDs to use (e.g. [0, 1, 2, 3]).  Defaults to [0].
        When more than one GPU is given, protein sequences are split across
        GPUs and embedded in parallel, each GPU running its own ProtT5 copy.
    """
    if gpus is None:
        gpus = [0]

    # Dynamic batch size based on free memory on the first GPU in the list.
    # Linear formulas scale continuously with every available GB (≈ 40% more
    # than the old fixed tiers) rather than jumping between coarse steps.
    #   max_residues = free_gb × 800   (e.g. 60 GB → 48 000)
    #   max_batch    = free_gb × 4     (e.g. 60 GB → 240)
    if max_residues is None or max_batch is None:
        try:
            free_mem, _ = torch.cuda.mem_get_info(gpus[0])
            free_vram_gb = free_mem / (1024**3)
            _res = max(1000, int(free_vram_gb * 800))
            _batch = max(4, int(free_vram_gb * 4))
            logger.info(
                f"[Step 2] GPU {gpus[0]}: {free_vram_gb:.1f} GB free VRAM -> "
                f"max_residues={_res}, max_batch={_batch}"
            )
        except Exception as e:
            logger.warning(
                f"[Step 2] VRAM detection failed ({e}). Using safe fallback: max_residues=3000, max_batch=8"
            )
            _res, _batch = 3000, 8
        max_residues = max_residues or _res
        max_batch = max_batch or _batch

    # Sort sequences by length descending for efficient batching
    seq_items = sorted(seqs_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    logger.info(
        f"[Step 2] Generating embeddings for {len(seq_items)} sequences "
        f"across {len(gpus)} GPU(s): {gpus}"
    )

    if len(gpus) == 1:
        results_list = _embed_worker(
            (gpus[0], 0, seq_items, max_residues, max_seq_len, max_batch)
        )
    else:
        import concurrent.futures
        import multiprocessing as _mp

        # Interleave sequences across GPUs so each GPU gets a balanced mix of
        # long and short proteins (seq_items is sorted longest-first)
        chunks = [seq_items[i :: len(gpus)] for i in range(len(gpus))]
        worker_args = [
            (gpu_id, position, chunk, max_residues, max_seq_len, max_batch)
            for position, (gpu_id, chunk) in enumerate(zip(gpus, chunks))
        ]

        ctx = _mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(gpus), mp_context=ctx
        ) as executor:
            futures = [executor.submit(_embed_worker, a) for a in worker_args]
            results_parts = [f.result() for f in futures]

        results_list = [item for part in results_parts for item in part]

    cols = ["ProteinID"] + list(range(1024))
    df = pd.DataFrame(results_list, columns=cols)
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("[Step 2] Embedding generation complete.")
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

    model_files = sorted(
        [
            f
            for f in glob.glob(os.path.join(model_dir, "*.json"))
            if os.path.basename(f) != "config.json"
        ]
    )

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
    transcript_to_gene = {}
    for line in feats:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 9:
            continue
        seqid = fields[0]
        feature_type = fields[2]
        attrs = parse_attributes(fields[8])

        if feature_type == "gene":
            gid = attrs.get("ID")
            if gid:
                gene_entries[(seqid, gid)] = {
                    "gene_line": fields,
                    "subfeatures": [],
                    "attributes": attrs,
                }
        else:
            parent = attrs.get("Parent", "")
            if not parent:
                continue
            if feature_type == "mRNA":
                tx_id = attrs.get("ID", "")
                parent_gene = parent
                if tx_id:
                    transcript_to_gene[(seqid, tx_id)] = parent_gene
            else:
                parent_gene = transcript_to_gene.get((seqid, parent))
                if parent_gene is None:
                    parent_gene = parent.rsplit(".", 1)[0] if "." in parent else parent

            key = (seqid, parent_gene)
            if key in gene_entries:
                gene_entries[key]["subfeatures"].append((fields, attrs))
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

    # Recalculate phases for all CDS features in the merged mRNA in 5'->3' order
    cds_items = [item for item in merged_children if item[0][2] == "CDS"]
    if cds_items:
        if strand == "+":
            cds_items.sort(key=lambda item: int(item[0][3]))
        else:
            cds_items.sort(key=lambda item: int(item[0][3]), reverse=True)
        
        cumulative_cds_bases = 0
        for fields, _ in cds_items:
            fields[7] = str((-cumulative_cds_bases) % 3)
            cds_len = int(fields[4]) - int(fields[3]) + 1
            cumulative_cds_bases += cds_len

    merged_children.sort(key=lambda item: int(item[0][3]), reverse=(strand == "-"))
    return synthetic_fields, merged_children


def generate_final_gff(
    predictions_df, input_gff_path, output_gff_path, keep_unmerged=True
):
    logger.info("[Step 4] Filtering and merging GFF based on predictions...")

    # Filter for positive predictions
    df1 = predictions_df[predictions_df["Predicted_Label"] == 1].copy()
    if df1.empty and not keep_unmerged:
        logger.warning("[WARN] No positive predictions found. Writing empty GFF.")
        open(output_gff_path, "w").close()
        return

    # Extract Gene groupings from ProteinIDs
    # single_genes stores (chrom, gene_id) tuples to avoid cross-chromosome deduplication
    single_genes = []
    group_map = {}
    gene_to_group = {}
    group_strand = {}
    group_chrom = {}

    for seq_id in df1["ProteinID"]:
        # ID Format: Chrom~GeneA|GeneB~Strand (using ~ separator)
        parts = seq_id.split("~")
        if len(parts) != 3:
            continue
        chrom, gene_part, strand = parts

        genes = gene_part.split("|")
        if len(genes) == 1:
            single_genes.append((chrom, genes[0]))
        else:
            key = (chrom, tuple(genes))
            group_strand[key] = strand
            group_chrom[key] = chrom
            new_id = "concat_" + "_".join(genes)
            group_map[key] = new_id
            for g in genes:
                gene_to_group[(chrom, g)] = key

    # Read original GFF
    header, gene_entries = read_gff_raw(input_gff_path)

    # Track merged genes to avoid duplicate output
    merged_genes = set()
    for chrom, gene_tuple in group_map.keys():
        for g in gene_tuple:
            merged_genes.add((chrom, g))

    # Build Output Items
    items = []
    seen_singles = set()

    if keep_unmerged:
        # Keep ALL unmerged genes (those not successfully grouped into a new chunk)
        for key, ge in gene_entries.items():
            if key not in merged_genes:
                start = int(ge["gene_line"][3])
                items.append({"type": "single", "key": key, "start": start})
    else:
        # Only keep the specifically predicted single genes
        for chrom, g in single_genes:
            composite_key = (chrom, g)
            if composite_key in seen_singles:
                continue
            seen_singles.add(composite_key)
            if composite_key in gene_entries and composite_key not in merged_genes:
                start = int(gene_entries[composite_key]["gene_line"][3])
                items.append({"type": "single", "key": composite_key, "start": start})

    for (chrom, gene_tuple), new_id in group_map.items():
        member_entries = [
            gene_entries[(chrom, g)] for g in gene_tuple if (chrom, g) in gene_entries
        ]
        if not member_entries:
            continue
        starts = [int(e["gene_line"][3]) for e in member_entries]
        ends = [int(e["gene_line"][4]) for e in member_entries]
        items.append(
            {
                "type": "group",
                "genes": gene_tuple,
                "chrom": chrom,
                "new_id": new_id,
                "start": min(starts),
                "end": max(ends),
                "strand": group_strand.get(
                    (chrom, gene_tuple), member_entries[0]["gene_line"][6]
                ),
            }
        )

    def _item_sort_key(item):
        if "chrom" in item:
            chrom = item["chrom"]
        else:
            chrom = item["key"][0]
        return (chrom, item["start"])

    items.sort(key=_item_sort_key)

    # Write Output
    with open(output_gff_path, "w") as out:
        out.writelines(header)
        for itm in items:
            if itm["type"] == "single":
                ge = gene_entries[itm["key"]]
                out.write("\t".join(ge["gene_line"]) + "\n")
                for fields, attrs in ge["subfeatures"]:
                    fields[8] = format_attributes(attrs)
                    out.write("\t".join(fields) + "\n")
            else:
                # Group logic
                grp = itm["genes"]
                chrom = itm["chrom"]
                gid = itm["new_id"]
                strand = itm.get("strand")
                first_valid_key = next(
                    ((chrom, g) for g in grp if (chrom, g) in gene_entries), None
                )
                if not first_valid_key:
                    continue

                first = gene_entries[first_valid_key]["gene_line"]
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
                # Pass composite keys for group members
                grp_keys = [g for g in grp if (chrom, g) in gene_entries]
                mrna_fields, merged_children = merge_group_transcripts(
                    grp_keys,
                    {g: gene_entries[(chrom, g)] for g in grp_keys},
                    synthetic_mrna_id,
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
                        if (chrom, g) not in gene_entries:
                            continue
                        for fields, attrs in gene_entries[(chrom, g)]["subfeatures"]:
                            if fields[2] == "mRNA":
                                attrs["Parent"] = gid
                            fields[8] = format_attributes(attrs)
                            out.write("\t".join(fields) + "\n")

    logger.info(f"[Step 4] Final GFF written to {output_gff_path}")
