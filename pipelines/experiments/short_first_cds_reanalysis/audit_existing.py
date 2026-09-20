#!/usr/bin/env python3
"""Read-only stdlib audit; run python reanalyse.py [SOURCE.tsv] [OUTPUT_DIR]."""
import csv,json,sys,hashlib,collections,math,statistics
from pathlib import Path
source=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[3]/'genecad_result/experiments/short_first_exon_validation/cross_species_decoder_redecode/per_locus.tsv'
out=Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).parent
out.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader(source.open(),delimiter='\t'))
def chain(row,key): return sorted(map(tuple,json.loads(row[key])),reverse=row['strand']=='-')
def three_end(seg,strand): return seg[1] if strand=='+' else seg[0]
def five_end(seg,strand): return seg[0] if strand=='+' else seg[1]
def tis_trim(raw,ref,strand):
 # New start inside an existing original CDS segment; retain every downstream CDS segment unchanged.
 for k in range(len(raw)):
  if ref[1:]==raw[k+1:] and three_end(ref[0],strand)==three_end(raw[k],strand) and raw[k][0]<=five_end(ref[0],strand)<=raw[k][1]:
   return k
 return None
base=[r for r in rows if r['variant']=='run0']
assert len({(r['species'],r['transcript_id']) for r in base})==len(base)
results=[]
for r in base:
 raw,ref,dec=[chain(r,k) for k in ['original_cds_chain','reference_cds_chain','decoded_cds_chain']]
 assert raw and ref
 strand=r['strand']; correct=raw==ref
 suffix=0
 while suffix<min(len(raw),len(ref)) and raw[-suffix-1]==ref[-suffix-1]: suffix+=1
 trim=tis_trim(raw,ref,strand)
 if correct: cl='exact_chain'
 elif raw[1:]==ref[1:] and three_end(raw[0],strand)==three_end(ref[0],strand): cl='first_segment_TIS_boundary_only'
 elif trim is not None: cl='downstream_TIS_trim_existing_CDS_path'
 elif raw[1:]==ref[1:]: cl='first_CDS_segment_replacement_only'
 elif suffix: cl='multiple_prefix_segments_differ_shared_exact_suffix'
 else: cl='no_shared_exact_terminal_CDS_segment'
 vals={k:float(r['oracle_reference_minus_raw_'+k]) for k in ['emission','transition','total']}
 item={k:r[k] for k in ['species','transcript_id','reference_id','strand','reference_pass']}
 item.update(raw_correct=int(correct),decoded_empty=int(not dec),decoded_correct=int(dec==ref),chain_class=cl,raw_CDS_segments=len(raw),reference_CDS_segments=len(ref),shared_exact_suffix_segments=suffix,trim_removed_segments=trim if trim is not None else '',oracle_emission=vals['emission'],oracle_transition=vals['transition'],oracle_total=vals['total'],oracle_total_positive=int(vals['total']>1e-6),oracle_total_zero=int(abs(vals['total'])<=1e-6),recorded_transition=r['chain_transition'])
 results.append(item)
with (out/'locus_audit.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]),delimiter='\t');w.writeheader();w.writerows(results)
def group_stats(rs):
 return {'n':len(rs),'raw_correct':sum(r['raw_correct'] for r in rs),'raw_errors':sum(not r['raw_correct'] for r in rs),'decoded_empty':sum(r['decoded_empty'] for r in rs),'empty_among_correct':sum(r['decoded_empty'] and r['raw_correct'] for r in rs),'empty_among_errors':sum(r['decoded_empty'] and not r['raw_correct'] for r in rs),'correct_to_wrong':sum(r['raw_correct'] and not r['decoded_correct'] for r in rs),'wrong_to_correct':sum(not r['raw_correct'] and r['decoded_correct'] for r in rs),'classes':dict(collections.Counter(r['chain_class'] for r in rs))}
report={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'all_rows':len(rows),'variant_counts':dict(collections.Counter(r['variant'] for r in rows)),'run0':group_stats(results),'by_strand':{s:group_stats([r for r in results if r['strand']==s]) for s in ['+','-']},'by_species':{s:group_stats([r for r in results if r['species']==s]) for s in sorted({r['species'] for r in results})},'oracle_by_correctness':{}}
for c in [0,1]:
 rs=[r for r in results if r['raw_correct']==c]
 report['oracle_by_correctness']['correct' if c else 'error']={k:{'positive':sum(r['oracle_'+k]>1e-6 for r in rs),'negative':sum(r['oracle_'+k]<-1e-6 for r in rs),'zero':sum(abs(r['oracle_'+k])<=1e-6 for r in rs),'median':statistics.median(r['oracle_'+k] for r in rs),'min':min(r['oracle_'+k] for r in rs),'max':max(r['oracle_'+k] for r in rs)} for k in ['emission','transition','total']}
report['by_variant_strand']={}
for variant in sorted({r['variant'] for r in rows}):
 for strand in ['+','-']:
  subset=[r for r in rows if r['variant']==variant and r['strand']==strand]
  parsed=[(chain(r,'original_cds_chain'),chain(r,'reference_cds_chain'),chain(r,'decoded_cds_chain')) for r in subset]
  report['by_variant_strand'][variant+' '+strand]={'n':len(parsed),'empty':sum(not d for a,b,d in parsed),'raw_chain_replayed':sum(a==d for a,b,d in parsed),'correct_to_wrong':sum(a==b and d!=b for a,b,d in parsed),'wrong_to_correct':sum(a!=b and d==b for a,b,d in parsed)}
report['score_additivity_max_error']=max(abs(r['oracle_emission']+r['oracle_transition']-r['oracle_total']) for r in results)
report['by_class']={cl:group_stats([r for r in results if r['chain_class']==cl]) for cl in sorted({r['chain_class'] for r in results})}
report['definitions']={
'exact_chain':'All sorted CDS intervals identical; UTR/exon envelopes not represented.',
'first_segment_TIS_boundary_only':'Same number of CDS segments; every segment after first identical in transcription order; first CDS segment has same 3-prime endpoint, only its TIS boundary differs. Upstream extension is allowed geometrically; physical transcript-exon feasibility not established.',
'downstream_TIS_trim_existing_CDS_path':'Reference can be obtained by dropping zero or more leading original CDS segments then moving start inside next original CDS segment, preserving its 3-prime endpoint and all following CDS intervals. Classification does not test ORF, reading-frame, splice motifs, or transcript exon envelopes.',
'first_CDS_segment_replacement_only':'All segments after first identical; first segment differs at its 3-prime endpoint, so TIS relocation alone cannot suffice.',
'multiple_prefix_segments_differ_shared_exact_suffix':'Not above classes. One or more exact CDS segments shared at 3-prime tail, but more than first-segment-only change or existing-CDS-path start truncation required. This does not prove global decoder nonrepresentability.',
'no_shared_exact_terminal_CDS_segment':'No exact last CDS interval shared. This can include a changed terminal segment start OR stop, not necessarily a changed stop.'}
(out/'audit_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['run0','by_strand','oracle_by_correctness']},indent=2))
