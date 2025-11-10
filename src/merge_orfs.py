# src/merge_orfs.py
from __future__ import annotations

import os
import re
import glob
import logging
from typing import Dict, List, Tuple

from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from huggingface_hub import snapshot_download

from Bio import SeqIO
from Bio.Seq import Seq

# For scoring
import xgboost as xgb

logger = logging.getLogger(__name__)

# ----------------------------
# Config knobs
# ----------------------------

HF_REPO_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"

# You may set GENECAD_PROTT5_PATH to force a specific path for the model snapshot
ENV_FORCE_MODEL_PATH = "GENECAD_PROTT5_PATH"


# ============================
# Part 1. FASTA + GFF utilities (your ORF merger)
# ============================

def parse_gff3(gff_file: str) -> Dict[str, List[dict]]:
    """
    Parse GFF3 into:
      genes[chrom] = [ { id, start, end, strand,
                         cds=[{start,end,phase}],
                         utr5=[{start,end}],
                         utr3=[{start,end}],
                         feature_types=set([...])
                       }, ...] (sorted by start)
    """
    genes: Dict[str, List[dict]] = defaultdict(list)
    with open(gff_file) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 9:
                logger.warning(f"Skipping malformed line: {line}")
                continue
            chrom, src, feature, start, end, score, strand, phase, attr = parts
            start_i, end_i = int(start), int(end)
            attrs = dict(kv.split('=', 1) for kv in attr.split(';') if '=' in kv)

            if feature == 'gene':
                genes[chrom].append({
                    'id': attrs.get('ID'),
                    'start': start_i,
                    'end': end_i,
                    'strand': strand,
                    'cds': [],
                    'utr5': [],
                    'utr3': [],
                    'feature_types': set()
                })
            elif feature in ('CDS', 'five_prime_UTR', 'three_prime_UTR'):
                parent = attrs.get('Parent', '')
                # naive parent link: assumes Parent like geneID.xxx
                for g in genes[chrom]:
                    if g['id'] and parent.startswith(g['id'] + '.'):
                        g['feature_types'].add(feature)
                        if feature == 'CDS':
                            try:
                                phase_int = int(phase)
                            except ValueError:
                                phase_int = 0
                            g['cds'].append({'start': start_i, 'end': end_i, 'phase': phase_int})
                        elif feature == 'five_prime_UTR':
                            g['utr5'].append({'start': start_i, 'end': end_i})
                        elif feature == 'three_prime_UTR':
                            g['utr3'].append({'start': start_i, 'end': end_i})
                        break

    # deterministic ordering
    for chrom in genes:
        genes[chrom].sort(key=lambda g: g['start'])
        for g in genes[chrom]:
            g['cds'].sort(key=lambda x: x['start'])
            g['utr5'].sort(key=lambda x: x['start'])
            g['utr3'].sort(key=lambda x: x['start'])
    return genes


def build_cds_string_for_gene(gene: dict, chrom_seq: Seq) -> str:
    strand = gene['strand']
    cds_list = gene['cds']
    if not cds_list:
        return ''

    sorted_cds = sorted(cds_list, key=lambda x: x['start'], reverse=(strand == '-'))

    parts: List[str] = []
    for cds in sorted_cds:
        a, b = cds['start'], cds['end']
        raw_seq = chrom_seq[a - 1:b]
        seq_str = str(raw_seq).upper()
        if strand == '+':
            trimmed = seq_str[cds['phase']:]
        else:
            rc = str(Seq(seq_str).reverse_complement())
            trimmed = rc[cds['phase']:]
        parts.append(trimmed)

    return ''.join(parts)


def flatten_cds_in_transcript_order(genes_in_chain: List[dict]) -> List[dict]:
    strand = genes_in_chain[0]['strand']
    cds_segments: List[dict] = []
    for g in genes_in_chain:
        cds_segments.extend(sorted(g['cds'], key=lambda x: x['start'], reverse=(strand == '-')))
    return cds_segments


def compute_phases_for_segments(cds_segments: List[dict]) -> List[dict]:
    phases: List[dict] = []
    frame = 0  # start codon at frame 0
    for seg in cds_segments:
        start, end = seg['start'], seg['end']
        phases.append({'start': start, 'end': end, 'phase': frame})
        seg_len = end - start + 1
        frame = (frame + seg_len) % 3
    return phases


def pick_boundary_utrs(genes_in_chain: List[dict]) -> Tuple[List[dict], List[dict]]:
    strand = genes_in_chain[0]['strand']
    first_gene = genes_in_chain[0]
    last_gene  = genes_in_chain[-1]

    if strand == '+':
        utr5 = sorted(first_gene['utr5'], key=lambda x: x['start'])
        utr3 = sorted(last_gene['utr3'],  key=lambda x: x['start'])
    else:
        utr5 = sorted(first_gene['utr5'], key=lambda x: x['start'], reverse=True)
        utr3 = sorted(last_gene['utr3'],  key=lambda x: x['start'], reverse=True)
    return utr5, utr3


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def build_merged_gff_block_entries(chrom: str, genes_in_chain: List[dict], merged_id: str, source: str, block_idx: int) -> List[dict]:
    strand = genes_in_chain[0]['strand']
    gene_start = min(g['start'] for g in genes_in_chain)
    gene_end   = max(g['end'] for g in genes_in_chain)
    mrna_id = merged_id + ".m1"

    entries: List[dict] = []
    inblock = 0

    def add(feature: str, s: int, e: int, phase='.', id_sfx=None, parent=None):
        nonlocal inblock
        inblock += 1
        attr_parts = []
        if id_sfx is not None:
            attr_parts.append(f"ID={id_sfx}")
        if parent is not None:
            attr_parts.append(f"Parent={parent}")
        attrs = ';'.join(attr_parts) if attr_parts else '.'
        line = f"{chrom}\t{source}\t{feature}\t{s}\t{e}\t.\t{strand}\t{phase}\t{attrs}"
        entries.append({
            'chrom': chrom,
            'start': s,
            'end': e,
            'feature': feature,
            'strand': strand,
            'line': line,
            'block_start': gene_start,
            'block_end': gene_end,
            'block_idx': block_idx,
            'inblock_order': inblock
        })

    # gene & mRNA
    add('gene', gene_start, gene_end, '.', merged_id, None)
    add('mRNA', gene_start, gene_end, '.', mrna_id, merged_id)

    # UTRs
    utr5_list, utr3_list = pick_boundary_utrs(genes_in_chain)
    for i, u in enumerate(utr5_list, 1):
        add('five_prime_UTR', u['start'], u['end'], '.', f"{mrna_id}.5UTR{i}", mrna_id)

    # CDS (phases recomputed)
    cds_segments = flatten_cds_in_transcript_order(genes_in_chain)
    phased = compute_phases_for_segments(cds_segments)
    for i, seg in enumerate(phased, 1):
        add('CDS', seg['start'], seg['end'], str(seg['phase']), f"{mrna_id}.CDS{i}", mrna_id)

    for i, u in enumerate(utr3_list, 1):
        add('three_prime_UTR', u['start'], u['end'], '.', f"{mrna_id}.3UTR{i}", mrna_id)

    return entries


def sort_gff_entries_globally(entries: List[dict]) -> List[dict]:
    entries.sort(
        key=lambda x: (
            natural_key(x['chrom']),
            x['block_start'],
            x['block_end'],
            x['block_idx'],
            x['inblock_order']
        )
    )
    return entries


def find_orfs_translate_and_collect_gff(genes: Dict[str, List[dict]], genome_fasta: str, out_fasta: str) -> List[dict]:
    STOPS = ('TAA', 'TAG', 'TGA')

    def is_complete_orf(nuc: str) -> bool:
        return len(nuc) >= 3 and nuc.startswith('ATG') and nuc[-3:] in STOPS

    seqs = SeqIO.to_dict(SeqIO.parse(genome_fasta, 'fasta'))
    collected_entries: List[dict] = []
    block_counter = 0

    with open(out_fasta, 'w') as fasta_out:
        for chrom, gene_list in genes.items():
            if chrom not in seqs:
                logger.warning(f"Chromosome {chrom} not found in FASTA; skipping.")
                continue
            chrom_seq = seqs[chrom].seq

            id_to_index = {g['id']: idx for idx, g in enumerate(gene_list)}

            for strand in ('+', '-'):
                st_genes = sorted(
                    (g for g in gene_list if g['strand'] == strand),
                    key=lambda g: g['start'],
                    reverse=(strand == '-')
                )

                cds_strings = [build_cds_string_for_gene(g, chrom_seq) for g in st_genes]
                n = len(st_genes)

                for i in range(n):
                    if not cds_strings[i]:
                        continue

                    running = ''
                    chain = [st_genes[i]]

                    for j in range(i, n):
                        if j > i:
                            prev = st_genes[j - 1]
                            curr = st_genes[j]

                            required = {'five_prime_UTR', 'CDS', 'three_prime_UTR'}
                            if required.issubset(prev['feature_types']):
                                break

                            prev_idx = id_to_index.get(prev['id'])
                            curr_idx = id_to_index.get(curr['id'])
                            if prev_idx is None or curr_idx is None or abs(curr_idx - prev_idx) != 1:
                                break

                            chain.append(curr)

                        if not cds_strings[j]:
                            break

                        running += cds_strings[j]

                        # Found an ORF across the running chain?
                        if len(running) >= 3 and running.startswith('ATG') and running[-3:] in STOPS:
                            # If merged (j>i) and any trailing gene already forms its own ORF, do NOT emit merged.
                            if j > i and any(is_complete_orf(cds_strings[k]) for k in range(i+1, j+1)):
                                break

                            # Emit FASTA
                            prot = Seq(running).translate(to_stop=False)
                            header = '|'.join(g['id'] for g in chain)
                            fasta_out.write(f">{chrom}_{header}_{strand}\n{prot}\n")

                            # Collect GFF entries for this merged chain
                            block_counter += 1
                            merged_id = f"{chrom}_{header}_{strand}"
                            entries = build_merged_gff_block_entries(
                                chrom, chain, merged_id, source="merged_orf", block_idx=block_counter
                            )
                            collected_entries.extend(entries)
                            break

    return collected_entries


def write_sorted_gff(entries: List[dict], out_gff: str) -> None:
    sort_gff_entries_globally(entries)
    with open(out_gff, 'w') as out:
        out.write("##gff-version 3\n")
        for e in entries:
            out.write(e['line'] + "\n")


def merge_orfs(gff_in: str, genome_fa: str, proteins_fa: str, gff_out: str) -> None:
    logger.info("Parsing input GFF...")
    genes = parse_gff3(gff_in)
    logger.info("Generating merged ORFs and proteins...")
    entries = find_orfs_translate_and_collect_gff(genes, genome_fa, proteins_fa)
    logger.info("Writing sorted merged GFF...")
    write_sorted_gff(entries, gff_out)
    logger.info(f"Wrote proteins: {proteins_fa}")
    logger.info(f"Wrote sorted GFF: {gff_out}")


# ============================
# Part 2. ProtT5 loader (auto-cache & offline-friendly)
# ============================

def _is_offline() -> bool:
    """Return True if offline env flags are set."""
    def _truthy(v: str | None) -> bool:
        return str(v).lower() in {"1", "true", "yes", "on"}
    return _truthy(os.environ.get("TRANSFORMERS_OFFLINE")) or _truthy(os.environ.get("HF_HUB_OFFLINE"))


def _first_snapshot_under(base: str) -> str | None:
    pat = os.path.join(base, "models--Rostlab--prot_t5_xl_half_uniref50-enc", "snapshots", "*")
    snaps = sorted(glob.glob(pat))
    return snaps[-1] if snaps else None


def _candidate_cache_dirs() -> List[str]:
    """
    Where to look/put cache:
    - $HF_HOME
    - $TRANSFORMERS_CACHE
    - If neither set, default to ~/.cache/huggingface (used by huggingface_hub)
    """
    cands: List[str] = []
    hf_home = os.environ.get("HF_HOME")
    tf_cache = os.environ.get("TRANSFORMERS_CACHE")
    if hf_home:
        cands.append(hf_home)
    if tf_cache and tf_cache not in cands:
        cands.append(tf_cache)
    # fallback to huggingface default if nothing provided
    if not cands:
        default_base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        cands.append(default_base)
    return cands


def resolve_prott5_snapshot(download_if_needed: bool = True) -> str:
    """
    Return a local directory containing a valid ProtT5 snapshot.

    Order:
    1) GENECAD_PROTT5_PATH (if set) — used as-is.
    2) Search snapshots under HF_HOME / TRANSFORMERS_CACHE for the repo.
    3) If not found and not offline and download_if_needed=True, download via snapshot_download.

    Raises:
        RuntimeError with actionable message if nothing is available and offline/blocked.
    """
    # 1) forced path
    forced = os.environ.get(ENV_FORCE_MODEL_PATH)
    if forced:
        if os.path.isdir(forced):
            logger.info(f"Using ProtT5 from {ENV_FORCE_MODEL_PATH}={forced}")
            return forced
        raise RuntimeError(f"{ENV_FORCE_MODEL_PATH} is set but not a directory: {forced}")

    # 2) look in known caches
    for base in _candidate_cache_dirs():
        snap = _first_snapshot_under(base)
        if snap:
            logger.info(f"Found local ProtT5 snapshot under cache: {snap}")
            return snap

    # 3) not found — download if allowed
    if download_if_needed and not _is_offline():
        # pick first candidate as cache_dir to be explicit
        cache_dir = _candidate_cache_dirs()[0]
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Downloading {HF_REPO_ID} into cache_dir={cache_dir} ...")
        snap_dir = snapshot_download(repo_id=HF_REPO_ID, cache_dir=cache_dir)
        logger.info(f"Downloaded ProtT5 snapshot: {snap_dir}")
        return snap_dir

    # offline or download disabled -> error with instructions
    cache_hint = "Set HF_HOME or TRANSFORMERS_CACHE to a shared path (e.g., /workdir/zl843/hf_cache)."
    prefetch_hint = (
        "To prefetch on the host: \n"
        "  python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download(repo_id='Rostlab/prot_t5_xl_half_uniref50-enc', cache_dir='/path/to/cache')\"\n"
        "Then mount that directory into the container and export HF_HOME to it."
    )
    raise RuntimeError(
        "ProtT5 snapshot not found locally and downloads are disabled (offline). "
        f"{cache_hint}\n{prefetch_hint}"
    )


def _normalize_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_t5_model(device: torch.device):
    """
    Load the encoder-only ProtT5 strictly from a local snapshot directory.
    If not present and online, we fetch it automatically into cache.
    """
    snap_dir = resolve_prott5_snapshot(download_if_needed=True)

    # Local-only load to avoid any Hub resolution/requests inside the container
    model = T5EncoderModel.from_pretrained(snap_dir, local_files_only=True)
    tok = T5Tokenizer.from_pretrained(snap_dir, do_lower_case=False, local_files_only=True)

    model = model.to(device).eval()

    # Optional multi-GPU: wrap if >1 visible devices and device is CUDA
    if str(device).startswith("cuda") and torch.cuda.device_count() > 1:
        # DataParallel keeps it simple for inference
        model = torch.nn.DataParallel(model)

    return model, tok


# ============================
# Part 3. Embedding entrypoint
# ============================

def _read_fasta_simple(fasta_path: str, split_char: str = "!", id_field: int = 0) -> Dict[str, str]:
    """
    Very light FASTA reader modeled after your previous helper.
    Returns dict: {id: SEQUENCE(ACDE...)}
    """
    seqs: Dict[str, str] = {}
    cur_id = None
    with open(fasta_path, "r") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                cur_id = line[1:].strip().split(split_char)[id_field]
                cur_id = cur_id.replace("/", "_").replace(".", "_")
                seqs[cur_id] = ""
            else:
                s = "".join(line.split()).upper().replace("-", "")
                s = s.replace("U", "X").replace("Z", "X").replace("O", "X")
                if cur_id is not None:
                    seqs[cur_id] += s
    if not seqs:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")
    return seqs


@torch.inference_mode()
def embed_from_fasta(
    fasta_path: str,
    output_tsv: str,
    device: str | torch.device = "auto",
    per_residue: bool = False,
    max_residues: int = 4000,
    max_seq_len: int = 1000,
    max_batch:   int = 1,
) -> None:
    """
    Embed a protein FASTA with ProtT5 (encoder only).
    Writes a TSV: first column = ID, next 1024 columns = embedding dims.
    """
    dev = _normalize_device(device)
    logger.info(f"Loading ProtT5 encoder on {dev} ...")
    model, tokenizer = _get_t5_model(dev)

    # read sequences
    seqs = _read_fasta_simple(fasta_path)
    logger.info(f"Read {len(seqs)} sequences from {fasta_path}")

    results = {"protein_embs": {}}
    # sort by length (longest first) to reduce padding
    items = sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True)

    batch: List[tuple[str, str, int]] = []
    for idx, (sid, seq) in enumerate(items, 1):
        L = len(seq)
        spaced = " ".join(list(seq))
        batch.append((sid, spaced, L))

        # decide when to flush
        n_res_batch = sum(slen for _, _, slen in batch)
        flush = (
            len(batch) >= max_batch
            or n_res_batch + L >= max_residues
            or idx == len(items)
            or L > max_seq_len
        )
        if not flush:
            continue

        ids, spaced_seqs, lengths = zip(*batch)
        batch = []

        tok = tokenizer.batch_encode_plus(
            spaced_seqs, add_special_tokens=True, padding="longest", return_tensors="pt"
        )
        input_ids = tok["input_ids"].to(dev)
        attn_mask = tok["attention_mask"].to(dev)

        try:
            out = model(input_ids, attention_mask=attn_mask)
        except RuntimeError as e:
            logger.error(f"RuntimeError during embedding on batch with {len(ids)} seqs: {e}")
            continue

        for b_i, sid in enumerate(ids):
            slen = lengths[b_i]
            hidden = out.last_hidden_state[b_i, :slen]  # [L, 1024]
            if per_residue:
                # if needed later, add to results["residue_embs"][sid] = hidden.cpu().numpy()
                pass
            protein_emb = hidden.mean(dim=0)           # [1024]
            results["protein_embs"][sid] = protein_emb.detach().cpu().numpy().squeeze()

    # write TSV
    ids = list(results["protein_embs"].keys())
    mat = np.empty((len(ids), 1024 + 1), dtype=object)
    for i, sid in enumerate(ids):
        mat[i, 0] = sid
        mat[i, 1:] = results["protein_embs"][sid]
    np.savetxt(output_tsv, mat, fmt="%s", delimiter="\t")
    logger.info(f"Wrote embeddings TSV: {output_tsv}")


# ============================
# Part 4. XGBoost scoring on embeddings TSV
# ============================

def generate_protein_scores(input_tsv: str, model_dir: str, output_csv: str) -> None:
    """
    Load a TSV (ID + 1024 dims) produced by embed_from_fasta, ensemble several XGBoost
    classifiers in `model_dir` (*.json), and write a CSV with:
      ProteinID, <model1>, <model2>, ..., Mean_Score, Predicted_Label
    """
    # find models
    model_files = sorted(glob.glob(os.path.join(model_dir, "*.json")))
    if not model_files:
        raise FileNotFoundError(f"No .json models found under {model_dir}")

    logger.info(f"Found {len(model_files)} model files")

    # load features once
    logger.info(f"Loading features from {input_tsv}")
    try:
        feats = pd.read_csv(input_tsv, header=None, sep="\t")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input TSV not found: {input_tsv}") from e

    protein_ids = np.array(feats[0])
    X = feats.drop(columns=[0])
    # Match training script expectation: 0..n-1 numeric column names
    X.columns = range(0, X.shape[1])

    results = pd.DataFrame({"ProteinID": protein_ids})

    # each model → predict_proba
    for mpath in model_files:
        mname = os.path.basename(mpath).replace(".json", "")
        logger.info(f"Scoring with model: {mname}")
        model = xgb.XGBClassifier()
        model.load_model(mpath)
        prob = model.predict_proba(X)[:, 1]
        results[mname] = prob

    # aggregate + label
    score_cols = results.drop(columns=["ProteinID"])
    results["Mean_Score"] = score_cols.mean(axis=1)
    results["Predicted_Label"] = (results["Mean_Score"] >= 0.5).astype(int)

    results.to_csv(output_csv, index=False)
    logger.info(f"Wrote protein scores to: {output_csv}")


# ============================
# Part 5. CSV → GFF filtering/merging
# ============================

def _parse_attributes(attr_str: str) -> dict:
    """Parses a GFF attribute string into a dictionary."""
    attrs = {}
    for part in attr_str.strip().split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k] = v
    return attrs


def _format_attributes(attr_dict: dict) -> str:
    """Formats a dictionary into a GFF attribute string."""
    return ";".join(f"{k}={v}" for k, v in attr_dict.items())


def _extract_genes_and_strands(df: pd.DataFrame):
    """
    Extract single gene names and grouped gene information from the DataFrame.
    The identifier is read from the 'ProteinID' column.
    """
    single_genes = set()
    group_map = {}
    gene_to_group = {}
    group_strand = {}

    for seq in df["ProteinID"]:
        if "_" not in seq:
            continue
        base, strand = seq.rsplit("_", 1)
        if "_" in base:
            _, gene_part = base.split("_", 1)
        else:
            gene_part = base
        genes = gene_part.split("|")
        if len(genes) == 1:
            single_genes.add(genes[0])
        else:
            key = tuple(genes)
            if key in group_strand and group_strand[key] != strand:
                raise ValueError(f"Mixed strand for group {key}: {group_strand[key]} vs {strand}")
            group_strand[key] = strand
            new_id = "concat_" + "_".join(genes)
            group_map[key] = new_id
            for g in genes:
                gene_to_group[g] = key

    return single_genes, group_map, gene_to_group, group_strand


def _read_gff_as_gene_dict(gff_path: str):
    """Reads a GFF file and organizes features by parent gene ID."""
    with open(gff_path) as f:
        lines = f.readlines()

    header = []
    feats = []
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            header = lines[:i]
            feats = lines[i:]
            break
    else:
        raise RuntimeError("No feature lines in GFF")

    gene_entries = {}
    for line in feats:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 9:
            continue
        feature_type = fields[2]
        attrs = _parse_attributes(fields[8])

        if feature_type == "gene":
            gid = attrs.get("ID")
            if not gid:
                continue
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
            else:
                continue

    return header, gene_entries


def _build_items(single_genes, group_map, gene_to_group, group_strand, gene_entries):
    """Build a sorted list of items (single genes and groups) to be written to the GFF."""
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
        items.append({
            "type": "group",
            "genes": group,
            "new_id": new_id,
            "start": min(starts),
            "end": max(ends),
            "strand": group_strand.get(group, member_entries[0]["gene_line"][6]),
        })
    items.sort(key=lambda x: x["start"])
    return items


def _merge_group_transcripts(grp, gene_entries, synthetic_mrna_id):
    """
    For a grouped gene, collapse all original mRNA features into one synthetic mRNA.
    Returns:
      mrna_feature: list of fields for new mRNA
      child_features: list of (fields, attrs) for all original subfeatures updated to point to synthetic mRNA
    """
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

    # compute synthetic span
    mrna_starts = [int(f[0][3]) for f in original_mrnas]
    mrna_ends   = [int(f[0][4]) for f in original_mrnas]
    synthetic_start = min(mrna_starts)
    synthetic_end   = max(mrna_ends)

    # template from the first mRNA
    template_fields, _, _ = original_mrnas[0]
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
        _format_attributes(synthetic_attrs),
    ]

    merged_children = []
    for fields, attrs in other_children:
        attrs["Parent"] = synthetic_mrna_id
        fields[8] = _format_attributes(attrs)
        merged_children.append((fields, attrs))

    # sort children in transcript order; reverse on '-' strand
    merged_children.sort(key=lambda item: int(item[0][3]), reverse=(strand == "-"))
    return synthetic_fields, merged_children


def _write_filtered_gff(output_path, header, items, gene_entries):
    """Writes the final filtered and merged GFF file."""
    with open(output_path, "w") as out:
        out.writelines(header)
        for itm in items:
            if itm["type"] == "single":
                ge = gene_entries[itm["id"]]
                out.write("\t".join(ge["gene_line"]) + "\n")
                for fields, attrs in ge["subfeatures"]:
                    fields[8] = _format_attributes(attrs)
                    out.write("\t".join(fields) + "\n")
            else:  # group
                grp    = itm["genes"]
                gid    = itm["new_id"]
                strand = itm.get("strand")
                first_valid = next((g for g in grp if g in gene_entries), None)
                if not first_valid:
                    continue
                first = gene_entries[first_valid]["gene_line"]
                seqid, source, _, _, _, score, _, phase, _ = first

                gene_attrs  = {"ID": gid, "Note": "concatenated"}
                gene_fields = [
                    seqid,
                    source,
                    "gene",
                    str(itm["start"]),
                    str(itm["end"]),
                    score,
                    strand if strand else first[6],
                    phase,
                    _format_attributes(gene_attrs),
                ]
                out.write("\t".join(gene_fields) + "\n")

                synthetic_mrna_id = f"{gid}.t1"
                mrna_fields, merged_children = _merge_group_transcripts(grp, gene_entries, synthetic_mrna_id)

                if mrna_fields:
                    # attach Parent to mRNA
                    mrna_attrs = _parse_attributes(mrna_fields[8])
                    mrna_attrs["Parent"] = gid
                    mrna_fields[8] = _format_attributes(mrna_attrs)
                    out.write("\t".join(mrna_fields) + "\n")
                    for fields, _ in merged_children:
                        out.write("\t".join(fields) + "\n")
                else:
                    # fallback: write original subfeatures
                    for g in grp:
                        if g not in gene_entries:
                            continue
                        for fields, attrs in gene_entries[g]["subfeatures"]:
                            if fields[2] == "mRNA":
                                attrs["Parent"] = gid
                            fields[8] = _format_attributes(attrs)
                            out.write("\t".join(fields) + "\n")


def filter_merged_gff_by_predictions(pred_csv: str, merged_gff_in: str, output_gff: str) -> None:
    """
    Filter a *merged* GFF (from merge_orfs) based on a predictions CSV.
    Keeps any merged gene whose ID appears in ProteinID with Predicted_Label==1,
    plus its mRNA and subfeatures (CDS/UTRs).

    Assumes merged GFF structure:
      gene(ID=<merged_id>)
      mRNA(ID=<merged_id>.m1; Parent=<merged_id>)
      CDS/UTR(...; Parent=<merged_id>.m1)
    """
    import csv

    # 1) Collect positives from CSV
    keep_ids = set()
    with open(pred_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "ProteinID" not in reader.fieldnames or "Predicted_Label" not in reader.fieldnames:
            raise ValueError("CSV must have columns: ProteinID, Predicted_Label")
        for row in reader:
            try:
                lab = int(row["Predicted_Label"])
            except Exception:
                continue
            if lab == 1:
                keep_ids.add(row["ProteinID"])

    if not keep_ids:
        logger.warning("No Predicted_Label==1 rows; writing empty GFF.")
        with open(output_gff, "w") as out:
            out.write("##gff-version 3\n")
        return

    def _parse_attrs(attr: str) -> dict:
        d = {}
        for part in attr.strip().split(";"):
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                d[k] = v
        return d

    # 2) Stream merged GFF and keep matched blocks
    with open(merged_gff_in) as fin, open(output_gff, "w") as fout:
        header_written = False
        keep_current_block = False
        current_gene_id = None
        current_mrna_id = None

        for line in fin:
            if line.startswith("#"):
                if not header_written:
                    fout.write(line)
                continue

            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                continue

            feature = cols[2]
            attrs = _parse_attrs(cols[8])

            if feature == "gene":
                # New block starts. Decide whether to keep it.
                current_gene_id = attrs.get("ID")
                current_mrna_id = f"{current_gene_id}.m1" if current_gene_id else None
                keep_current_block = current_gene_id in keep_ids

                if keep_current_block:
                    if not header_written:
                        # ensure we have a header line at least
                        fout.write("##gff-version 3\n")
                        header_written = True
                    fout.write(line)
                continue

            if not keep_current_block:
                continue

            # Inside a kept block: keep mRNA (ID==<gene>.m1, Parent=<gene>)
            # and any children that declare Parent=<gene>.m1
            if feature == "mRNA":
                if attrs.get("ID") == current_mrna_id and attrs.get("Parent") == current_gene_id:
                    fout.write(line)
            else:
                if attrs.get("Parent") == current_mrna_id:
                    fout.write(line)

    logger.info(f"Filtered merged GFF written to {output_gff}")

# ============================
# Part 6. Convenience pipeline (optional)
# ============================

def score_and_filter_gff(embeddings_tsv: str, model_dir: str, scores_csv: str, input_gff: str, output_gff: str) -> None:
    """
    One-shot convenience:
      embeddings.tsv --(XGBoost models)--> scores.csv --(filter/merge)--> filtered.gff
    """
    generate_protein_scores(embeddings_tsv, model_dir, scores_csv)
    filter_merged_gff_by_predictions(scores_csv, input_gff, output_gff)
