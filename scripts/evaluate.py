#!/usr/bin/env python3
"""
Unified GFF evaluation script for GeneCAD predictions.

Five evaluation sections:

SECTION 1 – CDS-Based (ignores UTRs)
  Levels: Locus, Transcript, CDS-exon.

SECTION 2 – gffcompare-Style (full exon structure incl. UTRs)
  Levels: Base, Exon, Intron, Intron chain, Locus (intron-chain based).

SECTION 3 – Splice Site Analysis (requires --fasta)
  Checks GT-AG / GC-AG canonical intron boundaries in predicted transcripts.

SECTION 4 – BUSCO (requires --fasta, gffread and BUSCO in PATH / conda)
  Extracts transcripts via gffread and evaluates completeness with BUSCO.

SECTION 5 – Site-Level Error Breakdown
  Precision/Recall/F1 for TIS, TTS, Junction Donor, and Junction Acceptor
  sites independently, so you can see where errors are concentrated.

Usage:
    python evaluate_cds.py \\
        --ref   <ref.gff> \\
        --pred  <pred.gff> \\
        [--fasta  <genome.fna>] \\
        [--lineage embryophyta_odb10] \\
        [--cpu 32] \\
        [--busco-out busco_eval] \\
        [--output report.txt]
"""

import argparse
import os
import shlex
import subprocess
import sys
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────


def parse_attributes(attr_str):
    """Parse GFF3 attribute string into a key-value dict."""
    attrs = {}
    for field in attr_str.strip().split(";"):
        if "=" in field:
            k, _, v = field.strip().partition("=")
            attrs[k.strip()] = v.strip()
    return attrs


def parse_gff(path, ftypes):
    """Parse a GFF3 file, returning records whose feature type is in ftypes."""
    records = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in ftypes:
                continue
            attrs = parse_attributes(parts[8])
            records.append(
                {
                    "chrom": parts[0],
                    "ftype": parts[2],
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": parts[6],
                    "id": attrs.get("ID", ""),
                    "parent": attrs.get("Parent", ""),
                }
            )
    return records


def merge_intervals(ivs):
    """Merge a list of (start, end) tuples."""
    if not ivs:
        return []
    s = sorted(ivs)
    merged = [list(s[0])]
    for a, b in s[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(x) for x in merged]


def bases_in_merged(ivs):
    """Return the total number of bases covered by a list of (start, end) intervals."""
    return sum(e - s + 1 for s, e in merge_intervals(ivs))


def overlap_bases(ivs_a, ivs_b):
    """Return the number of bases shared between two lists of (start, end) intervals."""
    ma, mb = merge_intervals(ivs_a), merge_intervals(ivs_b)
    tp = i = j = 0
    while i < len(ma) and j < len(mb):
        os, oe = max(ma[i][0], mb[j][0]), min(ma[i][1], mb[j][1])
        if os <= oe:
            tp += oe - os + 1
        if ma[i][1] < mb[j][1]:
            i += 1
        else:
            j += 1
    return tp


def introns_from_exons(sorted_exons, chrom, strand):
    """Derive intron coordinates from a sorted list of (start, end) exon tuples."""
    return [
        (chrom, sorted_exons[i][1] + 1, sorted_exons[i + 1][0] - 1, strand)
        for i in range(len(sorted_exons) - 1)
        if sorted_exons[i][1] + 1 <= sorted_exons[i + 1][0] - 1
    ]


def prf(tp, fp, fn):
    """Compute precision, recall, and F1 from TP/FP/FN counts."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return dict(TP=tp, FP=fp, FN=fn, precision=p, recall=r, f1=f)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – CDS-Based Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _build_gene_tx(records, cds_ftype="CDS"):
    """Build gene→transcript→CDS hierarchy from parsed GFF records."""
    genes, tx2gene, tx_cds = {}, {}, defaultdict(list)
    for r in records:
        ft, rid = r["ftype"], r["id"]
        if ft == "gene":
            genes[rid] = {"chrom": r["chrom"], "strand": r["strand"], "transcripts": {}}
        elif ft == "mRNA":
            tx2gene[rid] = r["parent"]
            if r["parent"] in genes:
                genes[r["parent"]]["transcripts"][rid] = None
        elif ft == cds_ftype:
            tx_cds[r["parent"]].append((r["chrom"], r["start"], r["end"], r["strand"]))
    for tx_id, lst in tx_cds.items():
        g = tx2gene.get(tx_id)
        if g and g in genes:
            genes[g]["transcripts"][tx_id] = frozenset(lst)
    return genes


def build_ref_cds(records):
    """Build reference gene structures and a per-chromosome CDS frozenset index."""
    genes = _build_gene_tx(records)
    by_chrom = defaultdict(set)
    for gdata in genes.values():
        for fs in gdata["transcripts"].values():
            if fs:
                by_chrom[gdata["chrom"]].add(fs)
    return genes, by_chrom


def eval_cds(ref_genes, ref_by_chrom, pred_genes):
    """Score predictions at the locus, transcript, and CDS-exon level (CDS only, no UTRs)."""
    ref_cds_iv = defaultdict(set)
    for gdata in ref_genes.values():
        for fs in gdata["transcripts"].values():
            if fs:
                for ch, s, e, st in fs:
                    ref_cds_iv[ch].add((s, e, st))

    tp_loci, tp_tx = set(), set()
    pred_iv = defaultdict(set)

    for gid, gdata in pred_genes.items():
        ch, hit = gdata["chrom"], False
        for tx_id, fs in gdata["transcripts"].items():
            if not fs:
                continue
            for c, s, e, st in fs:
                pred_iv[ch].add((s, e, st))
            if fs in ref_by_chrom[ch]:
                tp_tx.add(tx_id)
                hit = True
        if hit:
            tp_loci.add(gid)

    n_pred_loci = len(pred_genes)
    n_pred_tx = sum(len(g["transcripts"]) for g in pred_genes.values())
    n_pred_cds = sum(len(v) for v in pred_iv.values())

    n_ref_loci = sum(
        1 for g in ref_genes.values() if any(fs for fs in g["transcripts"].values())
    )
    n_ref_tx = sum(
        1 for g in ref_genes.values() for fs in g["transcripts"].values() if fs
    )
    n_ref_cds = sum(len(v) for v in ref_cds_iv.values())

    ltp = len(tp_loci)
    ttp = len(tp_tx)
    ctp = sum(1 for ch, ivs in pred_iv.items() for iv in ivs if iv in ref_cds_iv[ch])

    return {
        "locus": prf(ltp, n_pred_loci - ltp, n_ref_loci - ltp),
        "tx": prf(ttp, n_pred_tx - ttp, n_ref_tx - ttp),
        "cds_ex": prf(ctp, n_pred_cds - ctp, n_ref_cds - ctp),
        "n_ref_loci": n_ref_loci,
        "n_ref_tx": n_ref_tx,
        "n_ref_cds": n_ref_cds,
        "n_pred_loci": n_pred_loci,
        "n_pred_tx": n_pred_tx,
        "n_pred_cds": n_pred_cds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – gffcompare-Style Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def build_ref_exon(records):
    """Build reference exon structure.

    Supports two reference GFF styles:
    - With explicit 'exon' features (e.g. Ensembl, NCBI)
    - Without 'exon' features (e.g. Phytozome): reconstructs exons from
      CDS + five_prime_UTR + three_prime_UTR sub-features, mirroring
      what build_pred_exon does for GeneCAD predictions.
    """
    EXON_FTYPES = {"CDS", "five_prime_UTR", "three_prime_UTR"}
    genes, tx2gene = {}, {}
    tx_exons_explicit = defaultdict(list)  # from 'exon' records
    tx_exons_fallback = defaultdict(list)  # from CDS/UTR records

    for r in records:
        ft, rid = r["ftype"], r["id"]
        if ft == "gene":
            genes[rid] = {"chrom": r["chrom"], "strand": r["strand"], "transcripts": {}}
        elif ft == "mRNA":
            tx2gene[rid] = r["parent"]
            if r["parent"] in genes:
                genes[r["parent"]]["transcripts"][rid] = None
        elif ft == "exon":
            if r["parent"] in tx2gene or r["parent"].startswith("rna"):
                tx_exons_explicit[r["parent"]].append((r["start"], r["end"]))
        elif ft in EXON_FTYPES:
            tx_exons_fallback[r["parent"]].append((r["start"], r["end"]))

    # Prefer explicit exon records; fall back to CDS+UTR when absent
    has_explicit_exons = bool(tx_exons_explicit)
    tx_exons = tx_exons_explicit if has_explicit_exons else tx_exons_fallback

    for tx_id, lst in tx_exons.items():
        g = tx2gene.get(tx_id)
        if g and g in genes:
            genes[g]["transcripts"][tx_id] = tuple(sorted(merge_intervals(lst)))
    return genes


def build_pred_exon(records):
    """Reconstruct exons for GeneCAD predictions by merging CDS + UTR sub-features."""
    genes, tx2gene = {}, {}
    tx_sub = defaultdict(list)
    EXON_FTYPES = {"CDS", "five_prime_UTR", "three_prime_UTR"}
    for r in records:
        ft, rid = r["ftype"], r["id"]
        if ft == "gene":
            genes[rid] = {"chrom": r["chrom"], "strand": r["strand"], "transcripts": {}}
        elif ft == "mRNA":
            tx2gene[rid] = r["parent"]
            if r["parent"] in genes:
                genes[r["parent"]]["transcripts"][rid] = None
        elif ft in EXON_FTYPES:
            tx_sub[r["parent"]].append((r["start"], r["end"]))
    for tx_id, lst in tx_sub.items():
        g = tx2gene.get(tx_id)
        if g and g in genes:
            genes[g]["transcripts"][tx_id] = tuple(sorted(merge_intervals(lst)))
    return genes


def eval_gffcompare(ref_genes, pred_genes):
    """Score predictions at base, exon, intron, intron-chain, and locus levels."""
    ref_exon_iv = defaultdict(set)
    ref_intron_iv = defaultdict(set)
    ref_ichain = defaultdict(set)
    ref_tx = defaultdict(set)
    ref_base_ivs = defaultdict(list)
    n_ref_loci = n_ref_tx = n_ref_loci_multi = 0

    for gdata in ref_genes.values():
        ch, st = gdata["chrom"], gdata["strand"]
        has = has_multi = False
        for exons in gdata["transcripts"].values():
            if not exons:
                continue
            n_ref_tx += 1
            has = True
            for s, e in exons:
                ref_exon_iv[ch].add((s, e, st))
                ref_base_ivs[ch].append((s, e))
            introns = introns_from_exons(exons, ch, st)
            for intr in introns:
                ref_intron_iv[ch].add((intr[1], intr[2], st))
            if introns:
                ref_ichain[ch].add(frozenset(introns))
                has_multi = True
            ref_tx[ch].add(exons)
        if has:
            n_ref_loci += 1
        if has_multi:
            n_ref_loci_multi += 1

    tp_loci_ichain = set()
    pred_exon_iv = defaultdict(set)
    pred_intron_iv = defaultdict(set)
    pred_ichain = defaultdict(set)
    pred_base_ivs = defaultdict(list)
    n_pred_loci = n_pred_tx = n_pred_loci_multi = 0

    for gid, gdata in pred_genes.items():
        ch, st = gdata["chrom"], gdata["strand"]
        has = locus_ichain_hit = has_multi = False
        for exons in gdata["transcripts"].values():
            if not exons:
                continue
            n_pred_tx += 1
            has = True
            for s, e in exons:
                pred_exon_iv[ch].add((s, e, st))
                pred_base_ivs[ch].append((s, e))
            introns = introns_from_exons(exons, ch, st)
            for intr in introns:
                pred_intron_iv[ch].add((intr[1], intr[2], st))
            if introns:
                ichain = frozenset(introns)
                pred_ichain[ch].add(ichain)
                has_multi = True
                if ichain in ref_ichain[ch]:
                    locus_ichain_hit = True
        if locus_ichain_hit:
            tp_loci_ichain.add(gid)
        if has:
            n_pred_loci += 1
        if has_multi:
            n_pred_loci_multi += 1

    def set_tp(pred_dict, ref_dict):
        return sum(
            len({x for x in vs if x in ref_dict[ch]}) for ch, vs in pred_dict.items()
        )

    exon_tp = set_tp(pred_exon_iv, ref_exon_iv)
    intron_tp = set_tp(pred_intron_iv, ref_intron_iv)
    ichain_tp = set_tp(pred_ichain, ref_ichain)
    locus_ic_tp = len(tp_loci_ichain)

    all_chroms = set(pred_base_ivs) | set(ref_base_ivs)
    base_tp = sum(
        overlap_bases(pred_base_ivs[ch], ref_base_ivs[ch]) for ch in all_chroms
    )
    pred_bases = sum(bases_in_merged(v) for v in pred_base_ivs.values())
    ref_bases = sum(bases_in_merged(v) for v in ref_base_ivs.values())

    def n(d):
        return sum(len(v) for v in d.values())

    return {
        "base": prf(base_tp, pred_bases - base_tp, ref_bases - base_tp),
        "exon": prf(exon_tp, n(pred_exon_iv) - exon_tp, n(ref_exon_iv) - exon_tp),
        "intron": prf(
            intron_tp, n(pred_intron_iv) - intron_tp, n(ref_intron_iv) - intron_tp
        ),
        "ichain": prf(ichain_tp, n(pred_ichain) - ichain_tp, n(ref_ichain) - ichain_tp),
        "locus_ic": prf(
            locus_ic_tp, n_pred_loci_multi - locus_ic_tp, n_ref_loci_multi - locus_ic_tp
        ),
        "n_ref_loci": n_ref_loci,
        "n_ref_loci_multi": n_ref_loci_multi,
        "n_ref_tx": n_ref_tx,
        "ref_bases": ref_bases,
        "n_pred_loci": n_pred_loci,
        "n_pred_loci_multi": n_pred_loci_multi,
        "n_pred_tx": n_pred_tx,
        "pred_bases": pred_bases,
        "n_ref_exons": n(ref_exon_iv),
        "n_pred_exons": n(pred_exon_iv),
        "n_ref_introns": n(ref_intron_iv),
        "n_pred_introns": n(pred_intron_iv),
        "n_ref_ichains": n(ref_ichain),
        "n_pred_ichains": n(pred_ichain),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Splice Site Analysis
# ─────────────────────────────────────────────────────────────────────────────


def load_fasta(fasta_path):
    """Load a FASTA file into a dict mapping sequence name to sequence string."""
    print(f"[INFO] Loading genome FASTA: {fasta_path}", file=sys.stderr)
    genome = {}
    chrom, seq = None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if chrom:
                    genome[chrom] = "".join(seq)
                chrom = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if chrom:
        genome[chrom] = "".join(seq)
    print(f"[INFO] Loaded {len(genome)} sequences.", file=sys.stderr)
    return genome


def _revcomp(seq):
    """Return the reverse complement of a DNA sequence."""
    comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    return seq.translate(comp)[::-1]


def eval_splice_sites(pred_exon_genes, genome):
    """
    For each predicted multi-exon transcript, check donor/acceptor dinucleotides.
    Uses the exon structure already built by build_pred_exon().
    """
    total = gt_ag = gc_ag = other = 0
    skipped_chroms = set()

    for gdata in pred_exon_genes.values():
        ch = gdata["chrom"]
        st = gdata["strand"]
        seq = genome.get(ch)
        if seq is None:
            skipped_chroms.add(ch)
            continue
        for exons in gdata["transcripts"].values():
            if not exons or len(exons) < 2:
                continue
            for i in range(len(exons) - 1):
                intr_s = exons[i][1] + 1  # 1-based inclusive intron start
                intr_e = exons[i + 1][0] - 1  # 1-based inclusive intron end
                if intr_e < intr_s:
                    continue
                total += 1
                # Donor: first 2 bases of intron (1-based intr_s, intr_s+1)
                #   → 0-based slice [intr_s-1 : intr_s+1]
                # Acceptor: last 2 bases of intron (1-based intr_e-1, intr_e)
                #   → 0-based slice [intr_e-2 : intr_e]
                #   (1-based inclusive end == 0-based exclusive end, so intr_e
                #    is used directly as the Python slice upper bound)
                # On '-' strand the donor/acceptor windows swap and are revcomped.
                if st == "+":
                    donor = seq[intr_s - 1 : intr_s + 1].upper()
                    acceptor = seq[intr_e - 2 : intr_e].upper()
                else:
                    donor = _revcomp(seq[intr_e - 2 : intr_e]).upper()
                    acceptor = _revcomp(seq[intr_s - 1 : intr_s + 1]).upper()
                if donor == "GT" and acceptor == "AG":
                    gt_ag += 1
                elif donor == "GC" and acceptor == "AG":
                    gc_ag += 1
                else:
                    other += 1

    return {
        "total": total,
        "gt_ag": gt_ag,
        "gc_ag": gc_ag,
        "other": other,
        "skipped_chroms": skipped_chroms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – BUSCO
# ─────────────────────────────────────────────────────────────────────────────


def extract_transcripts(pred_exon_genes, genome, output_path):
    """
    Extract spliced transcript sequences directly from the predicted exon
    structure — replaces the need for gffread.
    """
    count = skipped = 0
    with open(output_path, "w") as fh:
        for gdata in pred_exon_genes.values():
            ch, st = gdata["chrom"], gdata["strand"]
            seq = genome.get(ch)
            if seq is None:
                skipped += 1
                continue
            for tx_id, exons in gdata["transcripts"].items():
                if not exons:
                    continue
                # Concatenate exon slices (GFF is 1-based inclusive → 0-based)
                tx_seq = "".join(seq[s - 1 : e] for (s, e) in exons)
                if st == "-":
                    tx_seq = _revcomp(tx_seq)
                fh.write(f">{tx_id}\n{tx_seq}\n")
                count += 1
    print(f"[INFO] Extracted {count} transcript sequences.", file=sys.stderr)
    if skipped:
        print(f"[WARN] Skipped {skipped} genes (chrom not in FASTA).", file=sys.stderr)
    return count


def run_busco(
    pred_exon_genes,
    genome,
    lineage,
    cpu,
    busco_out_name,
    output_dir=None,
    conda_env="busco-5.5.0",
    busco_cmd=None,
    busco_activate_script=None,
    auto_install_busco=False,
):
    """
    1. Extract transcript FASTA via Python (no gffread needed).
    2. Run BUSCO in transcriptome mode.
    Returns (summary_path, busco_dir) or (None, None) on failure.
    """
    import shutil as _shutil

    def _check_busco_ready(cmd):
        r = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
        )
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        if r.returncode != 0:
            return False, combined.strip()
        if (
            "No module named 'busco'" in combined
            or 'No module named "busco"' in combined
        ):
            return False, combined.strip()
        return True, combined.strip()

    def _maybe_install_busco_env():
        for runner in ("conda", "mamba"):
            if not _shutil.which(runner):
                continue
            install_cmd = (
                f"{runner} create -n {shlex.quote(conda_env)} "
                f"-c bioconda -c conda-forge busco=5.5.0 -y"
            )
            print(
                f"[INFO] Attempting BUSCO install via: {runner} (env={conda_env})",
                file=sys.stderr,
            )
            r = subprocess.run(
                install_cmd,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                print("[INFO] BUSCO installation completed.", file=sys.stderr)
                return True
            print(
                f"[WARN] {runner} install failed (showing tail):\n{(r.stderr or r.stdout or '')[-2000:]}",
                file=sys.stderr,
            )
        return False

    base_busco_cmd = None

    # Priority 0: explicit command from user (most portable/reproducible)
    if busco_cmd:
        base_busco_cmd = busco_cmd.strip()
        print(f"[INFO] Using BUSCO from --busco-cmd: {base_busco_cmd}", file=sys.stderr)

    # Priority 1: busco already in PATH (user activated env themselves)
    if not base_busco_cmd and _shutil.which("busco"):
        ok, check_log = _check_busco_ready("busco --version")
        if ok:
            base_busco_cmd = "busco"
        else:
            # Under uv/venv, 'busco' can resolve but run with the wrong python.
            # Fall through to safer launchers instead of hard-failing early.
            print(
                "[INFO] PATH busco self-check failed; trying fallback launchers.",
                file=sys.stderr,
            )

    # Priority 2: conda/mamba/micromamba run (portable for local clones)
    if not base_busco_cmd:
        for runner in ("conda", "mamba", "micromamba"):
            if not _shutil.which(runner):
                continue
            candidate = f"{runner} run -n {shlex.quote(conda_env)} busco"
            ok, _ = _check_busco_ready(f"{candidate} --version")
            if ok:
                base_busco_cmd = candidate
                print(
                    f"[INFO] Using BUSCO via {runner} env '{conda_env}'",
                    file=sys.stderr,
                )
                break

    # Priority 3: source-activate fallback (HPC / site-specific setups)
    if not base_busco_cmd:
        activate_candidates = []
        if busco_activate_script:
            activate_candidates.append(busco_activate_script)
        env_activate = os.environ.get("BUSCO_ACTIVATE_SCRIPT")
        if env_activate:
            activate_candidates.append(env_activate)
        # Try common site path as a best-effort fallback.
        activate_candidates.append("/programs/miniconda3/bin/activate")

        seen = set()
        for act_script in activate_candidates:
            if not act_script:
                continue
            act_script = os.path.realpath(act_script)
            if act_script in seen or not os.path.isfile(act_script):
                continue
            seen.add(act_script)
            candidate = (
                f"source {shlex.quote(act_script)} {shlex.quote(conda_env)} "
                f'&& "$CONDA_PREFIX/bin/python" "$CONDA_PREFIX/bin/busco"'
            )
            ok, _ = _check_busco_ready(f"{candidate} --version")
            if ok:
                base_busco_cmd = candidate
                print(
                    f"[INFO] Using BUSCO via activate script: {act_script}",
                    file=sys.stderr,
                )
                break

    if not base_busco_cmd and auto_install_busco:
        if _maybe_install_busco_env():
            for runner in ("conda", "mamba", "micromamba"):
                if not _shutil.which(runner):
                    continue
                candidate = f"{runner} run -n {shlex.quote(conda_env)} busco"
                ok, _ = _check_busco_ready(f"{candidate} --version")
                if ok:
                    base_busco_cmd = candidate
                    print(
                        f"[INFO] Using newly installed BUSCO via {runner} env '{conda_env}'",
                        file=sys.stderr,
                    )
                    break

    if not base_busco_cmd:
        print("[WARN] BUSCO not found. To install:", file=sys.stderr)
        print(
            f"   conda create -n {conda_env} -c bioconda -c conda-forge busco=5.5.0 -y",
            file=sys.stderr,
        )
        print("   # Then either activate the env before running:", file=sys.stderr)
        print(
            f"   conda activate {conda_env} && python evaluate.py ...", file=sys.stderr
        )
        print("   # Or pass an explicit command:", file=sys.stderr)
        print(
            f"   python evaluate.py ... --busco-cmd 'conda run -n {conda_env} busco'",
            file=sys.stderr,
        )
        print("   # Or skip BUSCO for cross-platform runs:", file=sys.stderr)
        print("   python evaluate.py ... --skip-busco", file=sys.stderr)
        return None, None

    ready, busco_check_log = _check_busco_ready(f"{base_busco_cmd} --version")
    if not ready:
        print("[WARN] BUSCO executable is present but not runnable.", file=sys.stderr)
        if busco_check_log:
            print(
                f"[WARN] BUSCO self-check output:\n{busco_check_log[-2000:]}",
                file=sys.stderr,
            )
        print(
            "[WARN] This is usually a broken BUSCO environment (missing Python package/module).",
            file=sys.stderr,
        )
        print("[WARN] Recreate/fix the env, e.g.:", file=sys.stderr)
        print(
            f"   conda create -n {conda_env} -c bioconda -c conda-forge busco=5.5.0 -y",
            file=sys.stderr,
        )
        print(f"   conda activate {conda_env}", file=sys.stderr)
        print("   busco --version", file=sys.stderr)
        print(
            "[WARN] You can also bypass auto-detection with --busco-cmd.",
            file=sys.stderr,
        )
        return None, None

    # Determine output directory
    # Use realpath to resolve symlinks (e.g. /workdir -> /local/workdir),
    # so the transcript FASTA path and the BUSCO cwd are consistent.
    out_dir = os.path.realpath(output_dir if output_dir else os.getcwd())
    os.makedirs(out_dir, exist_ok=True)
    import uuid

    suffix = uuid.uuid4().hex[:8]
    trans_fa = os.path.join(out_dir, f"_busco_transcripts_{suffix}.fa")
    actual_busco_out_name = f"{busco_out_name}_{suffix}"
    busco_out = os.path.join(out_dir, actual_busco_out_name)

    # --- Step 1: extract transcripts with Python ---
    print("[INFO] Extracting transcript sequences ...", file=sys.stderr)
    n = extract_transcripts(pred_exon_genes, genome, trans_fa)
    if n == 0:
        print("[WARN] No transcripts extracted — skipping BUSCO.", file=sys.stderr)
        return None, None

    # --- Step 2: run BUSCO ---
    busco_cmd = (
        f"{base_busco_cmd} "
        f"--in {shlex.quote(trans_fa)} "
        f"--lineage_dataset {shlex.quote(lineage)} "
        f"--out {shlex.quote(actual_busco_out_name)} "
        f"--mode transcriptome "
        f"--cpu {cpu} "
        f"--force"
    )
    print(f"[INFO] Running BUSCO (lineage={lineage}) ...", file=sys.stderr)
    busco_failed = False
    r = subprocess.run(
        busco_cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        cwd=out_dir,
    )
    if r.returncode != 0:
        print(f"[WARN] BUSCO failed:\n{r.stderr[-2000:]}", file=sys.stderr)
        busco_failed = True

    # Clean up intermediate transcript FASTA
    try:
        os.remove(trans_fa)
        print(f"[INFO] Removed intermediate file: {trans_fa}", file=sys.stderr)
    except OSError:
        pass

    if busco_failed:
        return None, None

    # Locate summary file and return (summary_path, busco_dir)
    for root, dirs, files in os.walk(busco_out):
        for fn in files:
            if fn.startswith("short_summary") and fn.endswith(".txt"):
                return os.path.join(root, fn), busco_out
    return None, busco_out


def parse_busco_summary(summary_path, busco_dir=None):
    """Read BUSCO short_summary*.txt, then clean up the BUSCO directory."""
    result = {"path": summary_path, "lines": []}
    if summary_path and os.path.isfile(summary_path):
        with open(summary_path) as fh:
            result["lines"] = fh.readlines()
    if busco_dir:
        print(f"[INFO] BUSCO results kept in: {busco_dir}", file=sys.stderr)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Site-Level Error Breakdown
# ─────────────────────────────────────────────────────────────────────────────


def _collect_tis_tts(cds_genes):
    """
    Extract TIS and TTS from CDS-based genes (output of _build_gene_tx).

    Sites are (chrom, pos, strand) tuples (1-based GFF coordinates).

    TIS – Translation Initiation Site: first coding base on the coding strand.
          For '+': leftmost  start of all CDS intervals in the transcript.
          For '-': rightmost end   of all CDS intervals in the transcript.
    TTS – Translation Termination Site: last coding base on the coding strand.
          For '+': rightmost end   of all CDS intervals.
          For '-': leftmost  start of all CDS intervals.

    Because CDS-based gene dicts store transcripts as frozensets of
    (chrom, start, end, strand) tuples, we handle them directly here.
    """
    tis_set, tts_set = set(), set()
    for gdata in cds_genes.values():
        ch = gdata["chrom"]
        st = gdata["strand"]
        for tx_data in gdata["transcripts"].values():
            if not tx_data:
                continue
            if isinstance(tx_data, frozenset):
                intervals = sorted((s, e) for (_, s, e, _) in tx_data)
            else:
                intervals = sorted(tx_data)
            if not intervals:
                continue
            first_s = intervals[0][0]
            last_e = intervals[-1][1]
            if st == "+":
                tis_set.add((ch, first_s, st))
                tts_set.add((ch, last_e, st))
            else:
                tis_set.add((ch, last_e, st))
                tts_set.add((ch, first_s, st))
    return tis_set, tts_set


def _collect_junctions(exon_genes):
    """
    Extract splice-junction donor and acceptor sites from exon-based genes
    (output of build_ref_exon / build_pred_exon).

    Donor    – last  exon base immediately before each intron.
    Acceptor – first exon base immediately after  each intron.
    """
    donor_set, acceptor_set = set(), set()
    for gdata in exon_genes.values():
        ch = gdata["chrom"]
        st = gdata["strand"]
        for exons in gdata["transcripts"].values():
            if not exons or len(exons) < 2:
                continue
            for i in range(len(exons) - 1):
                exon_end = exons[i][1]
                exon_start = exons[i + 1][0]
                if st == "+":
                    donor_set.add((ch, exon_end, st))
                    acceptor_set.add((ch, exon_start, st))
                else:
                    donor_set.add((ch, exon_start, st))
                    acceptor_set.add((ch, exon_end, st))
    return donor_set, acceptor_set


def eval_sites(ref_cds_genes, pred_cds_genes, ref_exon_genes, pred_exon_genes):
    """
    Compare predicted vs reference genomic sites and return per-type
    precision / recall / F1.

    TIS/TTS  are derived from CDS-based genes (exact translation boundaries).
    Donor/Acceptor are derived from exon-based genes (splice junctions).
    """
    ref_tis, ref_tts = _collect_tis_tts(ref_cds_genes)
    pred_tis, pred_tts = _collect_tis_tts(pred_cds_genes)
    ref_don, ref_acc = _collect_junctions(ref_exon_genes)
    pred_don, pred_acc = _collect_junctions(pred_exon_genes)

    def _prf_sets(pred_s, ref_s):
        tp = len(pred_s & ref_s)
        fp = len(pred_s) - tp
        fn = len(ref_s) - tp
        return prf(tp, fp, fn)

    return {
        "TIS": _prf_sets(pred_tis, ref_tis),
        "TTS": _prf_sets(pred_tts, ref_tts),
        "donor": _prf_sets(pred_don, ref_don),
        "acceptor": _prf_sets(pred_acc, ref_acc),
        "n_ref_tis": len(ref_tis),
        "n_ref_tts": len(ref_tts),
        "n_ref_donor": len(ref_don),
        "n_ref_acceptor": len(ref_acc),
        "n_pred_tis": len(pred_tis),
        "n_pred_tts": len(pred_tts),
        "n_pred_donor": len(pred_don),
        "n_pred_acceptor": len(pred_acc),
    }


def format_site_report(r):
    """Format the site-level error breakdown (Section 5) as a string."""
    lines = [
        "=" * 62,
        "SECTION 5 – Site-Level Error Breakdown",
        "  Each site is a strand-aware (chrom, pos, strand) tuple.",
        "  TIS  = Translation Initiation Site (CDS start on coding strand)",
        "  TTS  = Translation Termination Site (CDS end on coding strand)",
        "  Donor    = last exon base before each intron",
        "  Acceptor = first exon base after each intron",
        "=" * 62,
        "",
        f"  {'Site':<12}  {'Ref':>7}  {'Pred':>7}  {'TP':>7}  {'FP':>7}  {'FN':>7}  "
        f"{'Prec':>7}  {'Rec':>7}  {'F1':>7}",
        "  " + "-" * 76,
    ]
    for site_key, label in [
        ("TIS", "TIS"),
        ("TTS", "TTS"),
        ("donor", "Junc.Donor"),
        ("acceptor", "Junc.Acc."),
    ]:
        m = r[site_key]
        ref = r[f"n_ref_{site_key.lower()}"]
        pred = r[f"n_pred_{site_key.lower()}"]
        lines.append(
            f"  {label:<12}  {ref:>7}  {pred:>7}  {m['TP']:>7}  {m['FP']:>7}  {m['FN']:>7}  "
            f"{m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}"
        )
    lines.append("")
    return "\n".join(lines)


def _row(title, m):
    """Format a single metric block (TP/FP/FN + P/R/F1) as a list of lines."""
    return [
        f"--- {title} ---",
        f"  TP: {m['TP']}   FP: {m['FP']}   FN: {m['FN']}",
        f"  Precision : {m['precision']:.4f}",
        f"  Recall    : {m['recall']:.4f}",
        f"  F1        : {m['f1']:.4f}",
        "",
    ]


def format_cds_report(r):
    """Format the CDS-based evaluation results (Section 1) as a string."""
    lines = [
        "=" * 62,
        "SECTION 1 – CDS-Based Evaluation (UTRs ignored)",
        "  Gene is correct if its CDS chain matches ANY ref isoform.",
        "=" * 62,
        "",
        f"  Reference : {r['n_ref_loci']} loci | {r['n_ref_tx']} transcripts | {r['n_ref_cds']} unique CDS exons",
        f"  Predicted : {r['n_pred_loci']} loci | {r['n_pred_tx']} transcripts | {r['n_pred_cds']} unique CDS exons",
        "",
    ]
    for title, key in [
        ("CDS-exon-level", "cds_ex"),
        ("Locus-level", "locus"),
        ("Transcript-level", "tx"),
    ]:
        lines += _row(title, r[key])
    return "\n".join(lines)


def format_gffcompare_report(r):
    """Format the gffcompare-style evaluation results (Section 2) as a string."""
    lines = [
        "=" * 62,
        "SECTION 2 – gffcompare-Style Evaluation (full exon incl. UTRs)",
        "  Counts are unique intervals per chromosome.",
        "  Intron chain / Locus[IC]: multi-exon transcripts only.",
        "=" * 62,
        "",
        f"  Reference : {r['n_ref_loci']} loci ({r['n_ref_loci_multi']} multi-exon) | "
        f"{r['n_ref_tx']} transcripts | {r['ref_bases']:,} bases",
        f"              {r['n_ref_exons']} exons | {r['n_ref_introns']} introns | "
        f"{r['n_ref_ichains']} intron chains",
        f"  Predicted : {r['n_pred_loci']} loci ({r['n_pred_loci_multi']} multi-exon) | "
        f"{r['n_pred_tx']} transcripts | {r['pred_bases']:,} bases",
        f"              {r['n_pred_exons']} exons | {r['n_pred_introns']} introns | "
        f"{r['n_pred_ichains']} intron chains",
        "",
    ]
    for title, key in [
        ("Base level (nucleotides)", "base"),
        ("Exon level (start/end exact)", "exon"),
        ("Intron level (splice junctions)", "intron"),
        ("Intron chain level (= transcript interior match)", "ichain"),
        ("Locus level [intron-chain based, multi-exon only]", "locus_ic"),
    ]:
        lines += _row(title, r[key])
    return "\n".join(lines)


def format_splice_report(r):
    """Format the splice site analysis results (Section 3) as a string."""
    total = r["total"]
    if total == 0:
        return "\n".join(
            ["=" * 62, "SECTION 3 – Splice Site Analysis", "  No introns found.", ""]
        )
    lines = [
        "=" * 62,
        "SECTION 3 – Splice Site Analysis",
        "  Intron donor/acceptor dinucleotides in predicted transcripts.",
        "=" * 62,
        "",
        f"  Total introns analysed : {total}",
        f"  GT-AG (canonical)      : {r['gt_ag']}  ({r['gt_ag'] / total * 100:.2f}%)",
        f"  GC-AG (semi-canonical) : {r['gc_ag']}  ({r['gc_ag'] / total * 100:.2f}%)",
        f"  Other (non-canonical)  : {r['other']}  ({r['other'] / total * 100:.2f}%)",
    ]
    if r["skipped_chroms"]:
        lines.append(
            f"  Skipped chromosomes    : {', '.join(sorted(r['skipped_chroms']))}"
        )
    lines.append("")
    return "\n".join(lines)


def format_busco_report(busco):
    """Format the BUSCO evaluation results (Section 4) as a string."""
    lines = [
        "=" * 62,
        "SECTION 4 – BUSCO Evaluation",
        "=" * 62,
        "",
    ]
    if not busco["lines"]:
        lines += ["  BUSCO did not run or summary not found.", ""]
    else:
        lines += [line.rstrip() for line in busco["lines"]]
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unified GFF evaluation: CDS, gffcompare, splice sites, BUSCO."
    )
    parser.add_argument("--ref", required=True, help="Reference GFF3")
    parser.add_argument("--pred", required=True, help="Prediction GFF3")
    parser.add_argument(
        "--fasta", default=None, help="Genome FASTA (required for splice site & BUSCO)"
    )
    parser.add_argument(
        "--lineage",
        default="embryophyta_odb10",
        help="BUSCO lineage dataset (default: embryophyta_odb10)",
    )
    parser.add_argument(
        "--cpu", type=int, default=32, help="Number of CPUs for BUSCO (default: 32)"
    )
    parser.add_argument(
        "--busco-out",
        default="busco_eval",
        help="BUSCO output directory name (default: busco_eval)",
    )
    parser.add_argument(
        "--busco-env",
        default="busco-5.5.0",
        help="Conda/mamba env name used for BUSCO auto-detection (default: busco-5.5.0)",
    )
    parser.add_argument(
        "--busco-cmd",
        default=os.environ.get("BUSCO_CMD"),
        help=(
            "Explicit BUSCO command prefix, e.g. 'busco' or "
            "'conda run -n busco-5.5.0 busco'. If set, overrides auto-detection. "
            "Can also be set via BUSCO_CMD environment variable."
        ),
    )
    parser.add_argument(
        "--busco-activate-script",
        default=os.environ.get("BUSCO_ACTIVATE_SCRIPT"),
        help=(
            "Optional shell activate script for HPC/site-specific BUSCO fallback, "
            "e.g. /programs/miniconda3/bin/activate. Can also be set via "
            "BUSCO_ACTIVATE_SCRIPT."
        ),
    )
    parser.add_argument(
        "--skip-busco",
        action="store_true",
        help="Skip BUSCO section even when --fasta is provided.",
    )
    parser.add_argument(
        "--auto-install-busco",
        action="store_true",
        help=(
            "If BUSCO is missing, attempt to create env --busco-env with "
            "conda/mamba and continue automatically."
        ),
    )
    parser.add_argument(
        "--output", default=None, help="Output report file (default: stdout)"
    )
    parser.add_argument(
        "--fix-busco-env",
        action="store_true",
        help="If BUSCO is broken, auto-repair the conda env and retry once.",
    )
    args = parser.parse_args()

    REF_FTYPES = {"gene", "mRNA", "CDS", "exon", "five_prime_UTR", "three_prime_UTR"}
    PRED_FTYPES = {"gene", "mRNA", "CDS", "five_prime_UTR", "three_prime_UTR"}

    print("[INFO] Parsing reference GFF ...", file=sys.stderr)
    ref_records = parse_gff(args.ref, REF_FTYPES)
    print("[INFO] Parsing prediction GFF ...", file=sys.stderr)
    pred_records = parse_gff(args.pred, PRED_FTYPES)

    # ── Section 1: CDS ──
    print("[INFO] Section 1: CDS-based evaluation ...", file=sys.stderr)
    ref_cds_genes, ref_cds_by_chrom = build_ref_cds(ref_records)
    pred_cds_genes = _build_gene_tx(pred_records)
    cds_res = eval_cds(ref_cds_genes, ref_cds_by_chrom, pred_cds_genes)

    # ── Section 2: gffcompare-style ──
    print("[INFO] Section 2: gffcompare-style evaluation ...", file=sys.stderr)
    ref_exon_genes = build_ref_exon(ref_records)
    pred_exon_genes = build_pred_exon(pred_records)
    gc_res = eval_gffcompare(ref_exon_genes, pred_exon_genes)

    # ── Sections 3 & 4: need genome FASTA ──
    genome = None
    if args.fasta:
        genome = load_fasta(args.fasta)

    # ── Section 3: splice sites ──
    splice_res = None
    if genome:
        print("[INFO] Section 3: Splice site analysis ...", file=sys.stderr)
        splice_res = eval_splice_sites(pred_exon_genes, genome)
    else:
        print("[INFO] Section 3: Skipped (no --fasta provided).", file=sys.stderr)

    # ── Section 4: BUSCO ──
    busco_res = None
    if genome and not args.skip_busco:
        print("[INFO] Section 4: Running BUSCO ...", file=sys.stderr)
        out_dir = os.path.dirname(os.path.abspath(args.output)) if args.output else None
        summary_path, busco_dir = run_busco(
            pred_exon_genes,
            genome,
            args.lineage,
            args.cpu,
            args.busco_out,
            output_dir=out_dir,
            conda_env=args.busco_env,
            busco_cmd=args.busco_cmd,
            busco_activate_script=args.busco_activate_script,
            auto_install_busco=args.auto_install_busco,
        )
        if summary_path is None and args.fix_busco_env:
            print(
                "[INFO] Attempting to auto-repair BUSCO environment and retry...",
                file=sys.stderr,
            )
            summary_path, busco_dir = run_busco(
                pred_exon_genes,
                genome,
                args.lineage,
                args.cpu,
                args.busco_out,
                output_dir=out_dir,
                conda_env=args.busco_env,
                busco_cmd=args.busco_cmd,
                busco_activate_script=args.busco_activate_script,
                auto_install_busco=True,
            )
        busco_res = parse_busco_summary(summary_path, busco_dir)
    elif args.skip_busco:
        print("[INFO] Section 4: Skipped (--skip-busco).", file=sys.stderr)
    else:
        print("[INFO] Section 4: Skipped (no --fasta provided).", file=sys.stderr)

    # ── Section 5: site-level breakdown ──
    # TIS/TTS from CDS-based genes (exact translation start/stop positions).
    # Donor/Acceptor from exon-based genes (full exon boundaries incl. UTRs).
    print("[INFO] Section 5: Site-level error breakdown ...", file=sys.stderr)
    site_res = eval_sites(
        ref_cds_genes, pred_cds_genes, ref_exon_genes, pred_exon_genes
    )

    # ── Assemble report ──
    parts = [
        format_cds_report(cds_res),
        format_gffcompare_report(gc_res),
        format_splice_report(splice_res)
        if splice_res
        else "=" * 62
        + "\nSECTION 3 – Splice Site Analysis\n  Skipped (provide --fasta).\n",
        format_busco_report(busco_res)
        if busco_res is not None
        else "=" * 62 + "\nSECTION 4 – BUSCO\n  Skipped (provide --fasta).\n",
        format_site_report(site_res),
    ]
    report = "\n\n".join(parts)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report + "\n")
        print(f"[INFO] Report written to: {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
