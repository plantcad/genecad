
Example GFF parsing:

```python
import pandas as pd
import pprint
from src.gff_parser import GFFExaminer, parse as parse_gff

# See https://biopython.org/wiki/GFF_Parsing
path = "/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/training_data/gff/Athaliana_447_Araport11.gene.gff3"
# path = "/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/training_data/gff/Osativa_323_v7.0.gene.gff3"
examiner = GFFExaminer()
in_handle = open(path)
pprint.pprint(examiner.parent_child_map(in_handle))
in_handle.close()

examiner = GFFExaminer()
in_handle = open(path)
pprint.pprint(examiner.available_limits(in_handle))
in_handle.close()


in_handle = open(path)
records = []
for rec in parse_gff(in_handle):
    records.append(rec)
in_handle.close()

# Get one gene and show whats in it:

# For arabidopsis, Chr1 gene 1 is on reverse strand while gene 0 is forward
i, j, k = 0, 1, 0 
records[i]
records[i].features[j]
records[i].features[j].sub_features
records[i].features[j].sub_features[0].qualifiers
records[i].features[j].sub_features[1].qualifiers
records[i].features[j].sub_features[k].sub_features

# Get one chromosome
In [49]: records[i]
Out[49]: SeqRecord(seq=Seq(None, length=43250710), id='Chr1', name='<unknown name>', description='<unknown description>', dbxrefs=[])

# Get one gene on chromosome
In [17]: records[i].features[j]
Out[17]: SeqFeature(SimpleLocation(ExactPosition(6787), ExactPosition(9130), strand=-1), type='gene', id='AT1G01020.Araport11.447', qualifiers=...)

# Show all transcripts for gene
In [18]: records[i].features[j].sub_features
Out[18]: 
[SeqFeature(SimpleLocation(ExactPosition(6787), ExactPosition(9130), strand=-1), type='mRNA', id='AT1G01020.1.Araport11.447', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(6787), ExactPosition(9130), strand=-1), type='mRNA', id='AT1G01020.4.Araport11.447', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(6787), ExactPosition(9130), strand=-1), type='mRNA', id='AT1G01020.3.Araport11.447', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(6787), ExactPosition(9130), strand=-1), type='mRNA', id='AT1G01020.5.Araport11.447', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(6787), ExactPosition(8737), strand=-1), type='mRNA', id='AT1G01020.2.Araport11.447', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(6787), ExactPosition(8737), strand=-1), type='mRNA', id='AT1G01020.6.Araport11.447', qualifiers=...)]

# Choose transcript based on `longest` being "1"
 In [20]: records[i].features[j].sub_features[0].qualifiers
Out[20]: 
{'ID': ['AT1G01020.1.Araport11.447'],
 'Name': ['AT1G01020.1'],
 'pacid': ['37399351'],
 'longest': ['1'],
 'geneName': ['ARV1'],
 'Parent': ['AT1G01020.Araport11.447'],
 'source': ['phytozomev12']}

In [21]: records[i].features[j].sub_features[1].qualifiers
Out[21]: 
{'ID': ['AT1G01020.4.Araport11.447'],
 'Name': ['AT1G01020.4'],
 'pacid': ['37399352'],
 'longest': ['0'],
 'geneName': ['ARV1'],
 'Parent': ['AT1G01020.Araport11.447'],
 'source': ['phytozomev12']}

# Then the transcript features look like this:
In [16]: records[i].features[j-1].sub_features[k].sub_features
Out[16]: 
[SeqFeature(SimpleLocation(ExactPosition(3630), ExactPosition(3759), strand=1), type='five_prime_UTR', id='AT1G01010.1.Araport11.447.five_prime_UTR.1', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(3759), ExactPosition(3913), strand=1), type='CDS', id='AT1G01010.1.Araport11.447.CDS.1', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(3995), ExactPosition(4276), strand=1), type='CDS', id='AT1G01010.1.Araport11.447.CDS.2', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(4485), ExactPosition(4605), strand=1), type='CDS', id='AT1G01010.1.Araport11.447.CDS.3', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(4705), ExactPosition(5095), strand=1), type='CDS', id='AT1G01010.1.Araport11.447.CDS.4', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(5173), ExactPosition(5326), strand=1), type='CDS', id='AT1G01010.1.Araport11.447.CDS.5', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(5438), ExactPosition(5630), strand=1), type='CDS', id='AT1G01010.1.Araport11.447.CDS.6', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(5630), ExactPosition(5899), strand=1), type='three_prime_UTR', id='AT1G01010.1.Araport11.447.three_prime_UTR.1', qualifiers=...)]

# On the reverse strand, features will look like this:
[SeqFeature(SimpleLocation(ExactPosition(8570), ExactPosition(8666), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.1', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(8666), ExactPosition(9130), strand=-1), type='five_prime_UTR', id='AT1G01020.1.Araport11.447.five_prime_UTR.1', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(8416), ExactPosition(8464), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.2', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(8235), ExactPosition(8325), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.3', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(7941), ExactPosition(7987), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.4', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(7761), ExactPosition(7835), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.5', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(7563), ExactPosition(7649), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.6', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(7383), ExactPosition(7450), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.7', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(7156), ExactPosition(7232), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.8', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(6787), ExactPosition(6914), strand=-1), type='three_prime_UTR', id='AT1G01020.1.Araport11.447.three_prime_UTR.1', qualifiers=...),
 SeqFeature(SimpleLocation(ExactPosition(6914), ExactPosition(7069), strand=-1), type='CDS', id='AT1G01020.1.Araport11.447.CDS.9', qualifiers=...)]
```

Note that the ExactPosition values in the reverse strand are inclusive on the right side, and exclusive on the left, e.g. the values in the raw GFF look like this for the `AT1G01020.1.Araport11.447.CDS.9` feature above (`ExactPosition(6914)` -> 6915 in GFF, `ExactPosition(7069)` -> 7069 in GFF):

```
(ml-rel) i615-042[gg](1007)$ cat Athaliana_447_Araport11.gene.gff3 | grep -i 'AT1G01020.1.Araport11.447.CDS.9'
Chr1	phytozomev12	CDS	6915	7069	.	-	2	ID=AT1G01020.1.Araport11.447.CDS.9;Parent=AT1G01020.1.Araport11.447;pacid=37399351
```

Whereas on the forward string, positions are exclusive on the right instead.  E.g. for the feature `LOC_Os01g01010.1.MSUv7.0.three_prime_UTR.2`:

 
----

The "parent_child_map" and "available_limits" for each GFF:

```python
# from path = "/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/training_data/gff/Osativa_323_v7.0.gene.gff3"
{('phytozomev11', 'gene'): [('phytozomev11', 'mRNA')],
 ('phytozomev11', 'mRNA'): [('phytozomev11', 'CDS'),
                            ('phytozomev11', 'five_prime_UTR'),
                            ('phytozomev11', 'three_prime_UTR')]}

{'gff_id': {('Chr1',): 57508,
            ('Chr10',): 22462,
            ('Chr11',): 26079,
            ('Chr12',): 25516,
            ('Chr2',): 47003,
            ('Chr3',): 51646,
            ('Chr4',): 36630,
            ('Chr5',): 34680,
            ('Chr6',): 33222,
            ('Chr7',): 31548,
            ('Chr8',): 28666,
            ('Chr9',): 23062,
            ('ChrSy',): 505,
            ('ChrUn',): 572},
 'gff_source': {('phytozomev11',): 419099},
 'gff_source_type': {('phytozomev11', 'CDS'): 239565,
                     ('phytozomev11', 'five_prime_UTR'): 43555,
                     ('phytozomev11', 'gene'): 42189,
                     ('phytozomev11', 'mRNA'): 52424,
                     ('phytozomev11', 'three_prime_UTR'): 41366},
 'gff_type': {('CDS',): 239565,
              ('five_prime_UTR',): 43555,
              ('gene',): 42189,
              ('mRNA',): 52424,
              ('three_prime_UTR',): 41366}}
```

```python
# from path = "/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/training_data/gff/Athaliana_447_Araport11.gene.gff3"
{('phytozomev12', 'gene'): [('phytozomev12', 'mRNA')],
 ('phytozomev12', 'mRNA'): [('phytozomev12', 'CDS'),
                            ('phytozomev12', 'five_prime_UTR'),
                            ('phytozomev12', 'three_prime_UTR')]}

{'gff_id': {('Chr1',): 125137,
            ('Chr2',): 72358,
            ('Chr3',): 90387,
            ('Chr4',): 71910,
            ('Chr5',): 106864,
            ('ChrC',): 280,
            ('ChrM',): 388},
 'gff_source': {('phytozomev12',): 467324},
 'gff_source_type': {('phytozomev12', 'CDS'): 286464,
                     ('phytozomev12', 'five_prime_UTR'): 56414,
                     ('phytozomev12', 'gene'): 27655,
                     ('phytozomev12', 'mRNA'): 48456,
                     ('phytozomev12', 'three_prime_UTR'): 48335},
 'gff_type': {('CDS',): 286464,
              ('five_prime_UTR',): 56414,
              ('gene',): 27655,
              ('mRNA',): 48456,
              ('three_prime_UTR',): 48335}}
```

