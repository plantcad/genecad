import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
from numcodecs import get_codec

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.frame_crf import FrameStateGraph, encode_sequence, reverse_complement_codes  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

rows = [
    r
    for r in csv.DictReader(
        open(
            ROOT
            / "genecad_result/experiments/short_first_exon_validation/cross_species_decoder_redecode/per_locus.tsv"
        ),
        delimiter="\t",
    )
    if r["variant"] == "run0" and r["species"] == "Athaliana"
]
seq = "".join(
    line.strip()
    for line in open(ROOT / "data/Athaliana_TAIR12_chr4.fa")
    if not line.startswith(">")
)

matrix = None
for n in ast.parse((ROOT / "src/modeling.py").read_text()).body:
    if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "TOKEN_TRANSITION_PROBS_1"
        for t in n.targets
    ):
        matrix = np.array(ast.literal_eval(n.value))


def meta(p):
    return json.loads((p / ".zarray").read_text())


def chunk(p, i):
    m = meta(p)
    assert m["filters"] is None and m["order"] == "C"
    key = str(i) + (".0" if len(m["shape"]) == 2 else "")
    raw = get_codec(m["compressor"]).decode((p / key).read_bytes())
    return np.frombuffer(raw, dtype=m["dtype"]).reshape(m["chunks"])


base = (
    ROOT
    / "genecad_result/predictions/Athaliana_TAIR12_chr4_finetuned/OZ408686.1/predictions_OZ408686.1"
)
cache = {}
for strand in ["positive", "negative"]:
    relevant = [r for r in rows if (r["strand"] == "+") == (strand == "positive")]
    for r in relevant:
        lo, hi = int(r["window_start"]), int(r["window_end"])
        cache[r["transcript_id"]] = (
            np.empty((hi - lo, 5), dtype=np.float32),
            np.zeros(hi - lo, dtype=bool),
        )
    for store in sorted(base.glob("predictions.*.zarr")):
        p = store / strand / "sequence"
        m = meta(p)
        step = m["chunks"][0]
        for i in range((m["shape"][0] + step - 1) // step):
            coords = chunk(p, i)[: min(step, m["shape"][0] - i * step)]
            needed = []
            for r in relevant:
                lo, hi = int(r["window_start"]), int(r["window_end"])
                keep = (coords >= lo) & (coords < hi)
                if keep.any():
                    needed.append((r, keep))
            if not needed:
                continue
            logits = chunk(store / strand / "feature_logits", i)[: len(coords)]
            for r, keep in needed:
                a, seen = cache[r["transcript_id"]]
                index = coords[keep] - int(r["window_start"])
                assert not seen[index].any()
                a[index] = logits[keep]
                seen[index] = True

results = []
graph = FrameStateGraph(min_coding_run_length=0, exon_length_strictness=0)
for r in rows:
    lo, hi = int(r["window_start"]), int(r["window_end"])
    logits, seen = cache[r["transcript_id"]]
    assert seen.all(), (r["transcript_id"], seen.sum(), len(seen))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    codes = encode_sequence(seq[lo:hi])
    original = json.loads(r["original_cds_chain"])
    reference = json.loads(r["reference_cds_chain"])
    for mode in (
        ["correct", "unflipped_logits", "forward_bases"]
        if r["strand"] == "-"
        else ["correct"]
    ):
        if r["strand"] == "-":
            p = probs if mode == "unflipped_logits" else probs[::-1].copy()
            b = codes if mode == "forward_bases" else reverse_complement_codes(codes)
            labels = graph.decode(p, b, matrix)[::-1]
        else:
            labels = graph.decode(probs, codes, matrix)
        # Candidate genes separated by intergenic runs; choose max CDS base overlap.
        genic = labels != 0
        starts = np.flatnonzero(genic & ~np.r_[False, genic[:-1]])
        ends = np.flatnonzero(genic & ~np.r_[genic[1:], False]) + 1
        candidates = []
        for s, e in zip(starts, ends):
            cds = labels[s:e] == 3
            cs = np.flatnonzero(cds & ~np.r_[False, cds[:-1]]) + s
            ce = np.flatnonzero(cds & ~np.r_[cds[1:], False]) + s + 1
            chain = [[int(a + lo + 1), int(b + lo)] for a, b in zip(cs, ce)]
            overlap = sum(
                max(0, min(b, d) - max(a, c) + 1) for a, b in chain for c, d in original
            )
            if overlap:
                candidates.append((overlap, chain))
        chain = max(candidates, key=lambda x: x[0])[1] if candidates else []
        out = dict(
            transcript_id=r["transcript_id"],
            strand=r["strand"],
            mode=mode,
            original=original,
            reference=reference,
            historical=json.loads(r["decoded_cds_chain"]),
            replayed=chain,
            match_original=chain == original,
            match_reference=chain == reference,
            match_historical=chain == json.loads(r["decoded_cds_chain"]),
        )
        results.append(out)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in out.items()
                    if k not in ("original", "reference", "historical", "replayed")
                }
            ),
            flush=True,
        )

(OUT / "replay_results.json").write_text(json.dumps(results, indent=2))
