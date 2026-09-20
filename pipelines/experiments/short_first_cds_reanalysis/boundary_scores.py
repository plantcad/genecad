"""Read sparse legacy Zarr v2 chunks without importing zarr; no model execution.
Run from repository root: .venv/bin/python pipelines/experiments/short_first_cds_reanalysis/boundary_scores.py [OUTPUT_DIR]
CDS chain coordinates are GFF 1-based inclusive; sequence cache coordinates 0-based.
"""
from pathlib import Path
import csv,json,collections,sys, numpy as np, numcodecs
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent; OUT.mkdir(parents=True,exist_ok=True); INP=ROOT/'genecad_result/experiments/short_first_exon_validation/cross_species_decoder_redecode/per_locus.tsv'
rows=list(csv.DictReader(INP.open(),delimiter='\t')); seen=set(); cases=[]
for r in rows:
 key=(r['species'],r['transcript_id'])
 if key in seen: continue
 seen.add(key)
 o=json.loads(r['original_cds_chain']); t=json.loads(r['reference_cds_chain'])
 if len(o)<2 or len(t)<2: continue
 first=lambda c: sorted(c)[0] if r['strand']=='+' else sorted(c)[-1]
 donor=lambda c: first(c)[1] if r['strand']=='+' else first(c)[0]-2
 a,b=donor(o),donor(t)
 if a==b and o!=t: continue # controls restricted to exact full CDS chains
 cases.append(dict(species=r['species'],transcript_id=r['transcript_id'],seqid=r['seqid'],strand=r['strand'],reference_pass=r['reference_pass'],group='changed_donor' if a!=b else 'exact_chain_control',original_index0=a,reference_index0=b))
def meta(p):return json.loads((p/'.zarray').read_text())
def decode(p,k,m):
 v=(p/k).read_bytes()
 if m['compressor']:v=numcodecs.get_codec(m['compressor']).decode(v)
 for f in reversed(m['filters'] or []):v=numcodecs.get_codec(f).decode(v)
 return np.frombuffer(v,dtype=m['dtype']) if isinstance(v,(bytes,bytearray,memoryview)) else np.asarray(v)
def names(p):
 m=meta(p);return decode(p,'0',m).tolist()
def lse(a):m=np.max(a);return float(m+np.log(np.exp(a-m).sum()))
groups=collections.defaultdict(list)
for c in cases:groups[(c['species'],c['seqid'],c['strand'])].append(c)
provenance=[]
for (species,seqid,strand),cs in groups.items():
 targets=set(v for c in cs for v in [c['original_index0'],c['reference_index0']]); values={}
 base=ROOT/f'genecad_result/predictions/{species}_full_finetuned/{seqid}/predictions_{seqid}'
 for store in sorted(base.glob('predictions.*.zarr')):
  p=store/('positive' if strand=='+' else 'negative')
  if not (p/'token_logits/.zarray').exists():continue
  tn=names(p/'token');fn=names(p/'feature');sm=meta(p/'sequence');tm=meta(p/'token_logits');fm=meta(p/'feature_logits')
  assert len(tn)==17 and tn[1:5]==['B-intron','I-intron','L-intron','U-intron'],tn
  assert tm['chunks'][0]==sm['chunks'][0]==fm['chunks'][0]
  prov=dict(path=str(p.relative_to(ROOT)),token_names=tn,feature_names=fn,shape=tm['shape'],chunks=tm['chunks'],coordinate_chunks=0,logit_chunks=0,max_feature_aggregation_error=0)
  for k in sorted((x.name for x in (p/'sequence').iterdir() if x.name.isdigit()),key=int):
   coords=decode(p/'sequence',k,sm);prov['coordinate_chunks']+=1
   wanted=[t for t in targets if coords.min()<=t<=coords.max()]
   if not wanted:continue
   match={t:np.flatnonzero(coords==t) for t in wanted};match={t:j for t,j in match.items() if len(j)}
   if not match:continue
   token=decode(p/'token_logits',k+'.0',tm).reshape(-1,17);feat=decode(p/'feature_logits',k+'.0',fm).reshape(-1,5);prov['logit_chunks']+=1
   for t,js in match.items():
    assert len(js)==1
    j=js[0];a=token[j].astype(float);f=feat[j].astype(float)
    agg=np.array([a[0]]+[lse(a[n:n+4]) for n in [1,5,9,13]])
    prov['max_feature_aggregation_error']=max(prov['max_feature_aggregation_error'],float(abs(agg-f).max()))
    values[t]=dict(boundary_conditional_logp=float(a[1]-lse(a[1:5])),intron_feature_logp=float(f[1]-lse(f)),boundary_joint_logp=float(a[1]-lse(a)),source=str(p.relative_to(ROOT)),chunk=k,chunk_offset=int(j))
  provenance.append(prov)
 for c in cs:
  for role in ['original','reference']:
   value=values.get(c[role+'_index0'])
   c[role+'_found']=value is not None
   if value:c.update({role+'_'+k:v for k,v in value.items()})
  if c['original_found'] and c['reference_found']:
   for metric in ['boundary_conditional_logp','intron_feature_logp','boundary_joint_logp']:
    c['delta_'+metric]=c['reference_'+metric]-c['original_'+metric]
 print(species,seqid,strand,'cases',len(cs),'found',len(values),'/',len(targets),flush=True)
keys=list(dict.fromkeys(k for c in cases for k in c))
with (OUT/'per_boundary.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys,delimiter='\t');w.writeheader();w.writerows(cases)
summary=[]
for species in ['ALL']+sorted(set(c['species'] for c in cases)):
 for group in ['changed_donor','exact_chain_control']:
  allc=[c for c in cases if (species=='ALL' or c['species']==species) and c['group']==group];cc=[c for c in allc if c['original_found'] and c['reference_found']]
  d=dict(species=species,group=group,n_total=len(allc),n_complete=len(cc),n_missing=len(allc)-len(cc),n_reference_pass=sum(c['reference_pass']=='True' for c in cc))
  for metric in ['boundary_conditional_logp','intron_feature_logp','boundary_joint_logp']:
   v=np.array([c['delta_'+metric] for c in cc]);d[metric]=dict(reference_higher=int((v>0).sum()),tie=int((v==0).sum()),original_higher=int((v<0).sum()),median_delta=float(np.median(v)) if len(v) else None)
  if cc:d['median_reference_boundary_conditional_probability']=float(np.median([np.exp(c['reference_boundary_conditional_logp']) for c in cc]))
  summary.append(d)
(OUT/'boundary_summary.json').write_text(json.dumps(dict(input=str(INP),unique_cohort_loci=len(seen),results=summary,provenance=provenance),indent=2))
