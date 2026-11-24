"""Write a copy of a raw GFF3 whose `longest` attribute marks the D4 representative isoform.

D4: per gene, the mRNA with the longest total CDS; ties -> lexicographically smallest mRNA ID.
GeneCAD's extract step treats `longest=1` as the canonical transcript, so this is the only
change made: every mRNA gets longest=1 (representative) or longest=0; all other lines and
attributes are copied unchanged. mRNAs without CDS keep longest=0.

Usage: tag_representative_isoform.py <raw.gff3> <out.gff3> <report.json>
"""

import collections
import json
import re
import sys

src, dst, report = sys.argv[1:4]

tx_gene, cds_len = {}, collections.Counter()
with open(src) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        if f[2] == "mRNA":
            a = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
            tx_gene[a["ID"]] = a["Parent"].split(",")[0]
        elif f[2] == "CDS":
            p = re.search(r"Parent=([^;]+)", f[8]).group(1).split(",")[0]
            cds_len[p] += int(f[4]) - int(f[3]) + 1

by_gene = collections.defaultdict(list)
for t, g in tx_gene.items():
    if cds_len[t] > 0:
        by_gene[g].append(t)
rep = {sorted(ts, key=lambda t: (-cds_len[t], t))[0] for ts in by_gene.values()}

changed = n_mrna = 0
with open(src) as fh, open(dst, "w") as out:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) >= 9 and f[2] == "mRNA":
            n_mrna += 1
            tid = re.search(r"ID=([^;]+)", f[8]).group(1)
            new = "1" if tid in rep else "0"
            attrs = f[8]
            old = re.search(r"(?:^|;)longest=([^;]*)", attrs)
            if old:
                changed += old.group(1) != new
                attrs = re.sub(
                    r"(^|;)longest=[^;]*", rf"\g<1>longest={new}", attrs, count=1
                )
            else:
                changed += new == "1"
                attrs = attrs.rstrip(";") + f";longest={new}"
            f[8] = attrs
            line = "\t".join(f) + "\n"
        out.write(line)

json.dump(
    {
        "source": src,
        "output": dst,
        "genes_with_cds": len(by_gene),
        "representatives": len(rep),
        "mrna": n_mrna,
        "mrna_longest_flag_changed_vs_source": changed,
    },
    open(report, "w"),
    indent=1,
)
print(open(report).read())
