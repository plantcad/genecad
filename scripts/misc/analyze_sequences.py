import numpy as np
import xarray as xr
import pandas as pd
from src.dataset import open_datatree

xr.set_options(display_width=128)
tap = lambda df, fn: fn(df) or df # noqa: E731
leaves = lambda dt: [dt.ds for dt in dt.subtree if dt.is_leaf] # noqa: E731


dt = open_datatree("/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/transform/sequences.zarr", consolidated=False)
print(dt)
# <xarray.DataTree>
# Group: /
# ├── Group: /Osativa
# │   ├── Group: /Osativa/Chr1
# │   │       Dimensions:             (feature: 3, strand: 2, sequence: 43250710, region: 7, reason: 2)
# │   │       Coordinates:
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * sequence            (sequence) int64 346MB 0 1 2 3 4 5 6 ... 43250703 43250704 43250705 43250706 43250707 43250708 43250709
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           feature_labels      (strand, sequence, feature) int8 260MB 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# │   │           region_labels       (strand, sequence, region) int8 606MB ...
# │   │           label_masks         (strand, sequence, reason) int8 173MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 692MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 87MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'A' b'C' b'G' b'T' b'C' b'A' b'T'
# │   │           sequence_masks      (strand, sequence) bool 87MB ...
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr1
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr5
# │   │       Dimensions:             (strand: 2, sequence: 29915194, feature: 3, reason: 2, region: 7)
# │   │       Coordinates:
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * sequence            (sequence) int64 239MB 0 1 2 3 4 5 6 ... 29915187 29915188 29915189 29915190 29915191 29915192 29915193
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           feature_labels      (strand, sequence, feature) int8 179MB ...
# │   │           sequence_masks      (strand, sequence) bool 60MB ...
# │   │           label_masks         (strand, sequence, reason) int8 120MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 479MB ...
# │   │           region_labels       (strand, sequence, region) int8 419MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 60MB b'g' b'g' b'a' b't' b't' b't' b'g' ... b'C' b'T' b'a' b'c' b'c' b'a' b't'
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr5
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr11
# │   │       Dimensions:             (feature: 3, strand: 2, sequence: 29014744, reason: 2, region: 7)
# │   │       Coordinates:
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * sequence            (sequence) int64 232MB 0 1 2 3 4 5 6 ... 29014737 29014738 29014739 29014740 29014741 29014742 29014743
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           label_masks         (strand, sequence, reason) int8 116MB ...
# │   │           feature_labels      (strand, sequence, feature) int8 174MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 464MB ...
# │   │           sequence_masks      (strand, sequence) bool 58MB ...
# │   │           region_labels       (strand, sequence, region) int8 406MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 58MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'T' b'A' b'G' b'C' b'T' b'G' b'A'
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr11
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr10
# │   │       Dimensions:             (feature: 3, strand: 2, sequence: 23206044, reason: 2, region: 7)
# │   │       Coordinates:
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * sequence            (sequence) int64 186MB 0 1 2 3 4 5 6 ... 23206037 23206038 23206039 23206040 23206041 23206042 23206043
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           label_masks         (strand, sequence, reason) int8 93MB ...
# │   │           sequence_masks      (strand, sequence) bool 46MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 46MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'T' b'T' b'G' b'A' b'T' b'G' b'C'
# │   │           feature_labels      (strand, sequence, feature) int8 139MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 371MB ...
# │   │           region_labels       (strand, sequence, region) int8 325MB ...
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr10
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr7
# │   │       Dimensions:             (feature: 3, strand: 2, sequence: 29680274, reason: 2, region: 7)
# │   │       Coordinates:
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * sequence            (sequence) int64 237MB 0 1 2 3 4 5 6 ... 29680267 29680268 29680269 29680270 29680271 29680272 29680273
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           label_masks         (strand, sequence, reason) int8 119MB ...
# │   │           feature_labels      (strand, sequence, feature) int8 178MB ...
# │   │           region_labels       (strand, sequence, region) int8 416MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 475MB ...
# │   │           sequence_masks      (strand, sequence) bool 59MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 59MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'A' b'G' b'C' b'T' b'C' b'A' b'T'
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr7
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr3
# │   │       Dimensions:             (strand: 2, sequence: 36405403, reason: 2, feature: 3, region: 7)
# │   │       Coordinates:
# │   │         * sequence            (sequence) int64 291MB 0 1 2 3 4 5 6 ... 36405396 36405397 36405398 36405399 36405400 36405401 36405402
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           label_masks         (strand, sequence, reason) int8 146MB ...
# │   │           feature_labels      (strand, sequence, feature) int8 218MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 73MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'A' b'C' b'G' b'A' b'C' b'A' b'T'
# │   │           sequence_masks      (strand, sequence) bool 73MB ...
# │   │           region_labels       (strand, sequence, region) int8 510MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 582MB ...
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr3
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr2
# │   │       Dimensions:             (strand: 2, sequence: 35930235, feature: 3, reason: 2, region: 7)
# │   │       Coordinates:
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * sequence            (sequence) int64 287MB 0 1 2 3 4 5 6 ... 35930228 35930229 35930230 35930231 35930232 35930233 35930234
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           feature_labels      (strand, sequence, feature) int8 216MB ...
# │   │           sequence_masks      (strand, sequence) bool 72MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 72MB b't' b'g' b'g' b'g' b'a' b't' b't' ... b'C' b'A' b'C' b'T' b'T' b'A' b'A'
# │   │           label_masks         (strand, sequence, reason) int8 144MB ...
# │   │           region_labels       (strand, sequence, region) int8 503MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 575MB ...
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr2
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr6
# │   │       Dimensions:             (strand: 2, sequence: 31231937, feature: 3, region: 7, reason: 2)
# │   │       Coordinates:
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * sequence            (sequence) int64 250MB 0 1 2 3 4 5 6 ... 31231930 31231931 31231932 31231933 31231934 31231935 31231936
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           sequence_input_ids  (strand, sequence) int64 500MB ...
# │   │           region_labels       (strand, sequence, region) int8 437MB ...
# │   │           label_masks         (strand, sequence, reason) int8 125MB ...
# │   │           feature_labels      (strand, sequence, feature) int8 187MB ...
# │   │           sequence_masks      (strand, sequence) bool 62MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 62MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'G' b'C' b'C' b'A' b'T' b'G' b'A'
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr6
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr8
# │   │       Dimensions:             (reason: 2, strand: 2, sequence: 28440338, feature: 3, region: 7)
# │   │       Coordinates:
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * sequence            (sequence) int64 228MB 0 1 2 3 4 5 6 ... 28440331 28440332 28440333 28440334 28440335 28440336 28440337
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           feature_labels      (strand, sequence, feature) int8 171MB ...
# │   │           label_masks         (strand, sequence, reason) int8 114MB ...
# │   │           region_labels       (strand, sequence, region) int8 398MB ...
# │   │           sequence_masks      (strand, sequence) bool 57MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 455MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 57MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'C' b'C' b'G' b'C' b'C' b'A' b'T'
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr8
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   ├── Group: /Osativa/Chr4
# │   │       Dimensions:             (strand: 2, sequence: 35463100, feature: 3, reason: 2, region: 7)
# │   │       Coordinates:
# │   │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │   │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │   │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │   │         * sequence            (sequence) int64 284MB 0 1 2 3 4 5 6 ... 35463093 35463094 35463095 35463096 35463097 35463098 35463099
# │   │         * strand              (strand) <U8 64B 'negative' 'positive'
# │   │       Data variables:
# │   │           feature_labels      (strand, sequence, feature) int8 213MB ...
# │   │           label_masks         (strand, sequence, reason) int8 142MB ...
# │   │           sequence_masks      (strand, sequence) bool 71MB ...
# │   │           sequence_input_ids  (strand, sequence) int64 567MB ...
# │   │           region_labels       (strand, sequence, region) int8 496MB ...
# │   │           sequence_tokens     (strand, sequence) |S1 71MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'T' b'C' b'C' b'A' b'T' b'A' b'A'
# │   │       Attributes:
# │   │           species_id:     Osativa
# │   │           chromosome_id:  Chr4
# │   │           filename:       Osativa_323_v7.0.gene.gff3
# │   └── Group: /Osativa/Chr9
# │           Dimensions:             (strand: 2, sequence: 22939188, feature: 3, reason: 2, region: 7)
# │           Coordinates:
# │             * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
# │             * sequence            (sequence) int64 184MB 0 1 2 3 4 5 6 ... 22939181 22939182 22939183 22939184 22939185 22939186 22939187
# │             * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
# │             * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
# │             * strand              (strand) <U8 64B 'negative' 'positive'
# │           Data variables:
# │               sequence_masks      (strand, sequence) bool 46MB ...
# │               sequence_input_ids  (strand, sequence) int64 367MB ...
# │               label_masks         (strand, sequence, reason) int8 92MB ...
# │               feature_labels      (strand, sequence, feature) int8 138MB ...
# │               region_labels       (strand, sequence, region) int8 321MB ...
# │               sequence_tokens     (strand, sequence) |S1 46MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'T' b'C' b'T' b'T' b'C' b'A' b'A'
# │           Attributes:
# │               species_id:     Osativa
# │               chromosome_id:  Chr9
# │               filename:       Osativa_323_v7.0.gene.gff3
# └── Group: /Athaliana
#     ├── Group: /Athaliana/Chr2
#     │       Dimensions:             (strand: 2, sequence: 19696821, feature: 3, reason: 2, region: 7)
#     │       Coordinates:
#     │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
#     │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
#     │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
#     │         * sequence            (sequence) int64 158MB 0 1 2 3 4 5 6 ... 19696814 19696815 19696816 19696817 19696818 19696819 19696820
#     │         * strand              (strand) <U8 64B 'negative' 'positive'
#     │       Data variables:
#     │           feature_labels      (strand, sequence, feature) int8 118MB ...
#     │           sequence_input_ids  (strand, sequence) int64 315MB ...
#     │           label_masks         (strand, sequence, reason) int8 79MB ...
#     │           sequence_masks      (strand, sequence) bool 39MB ...
#     │           sequence_tokens     (strand, sequence) |S1 39MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'T' b'T' b'C' b'A' b'C' b'A' b'C'
#     │           region_labels       (strand, sequence, region) int8 276MB ...
#     │       Attributes:
#     │           species_id:     Athaliana
#     │           chromosome_id:  Chr2
#     │           filename:       Athaliana_447_Araport11.gene.gff3
#     ├── Group: /Athaliana/Chr1
#     │       Dimensions:             (reason: 2, feature: 3, strand: 2, sequence: 30425192, region: 7)
#     │       Coordinates:
#     │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
#     │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
#     │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
#     │         * sequence            (sequence) int64 243MB 0 1 2 3 4 5 6 ... 30425185 30425186 30425187 30425188 30425189 30425190 30425191
#     │         * strand              (strand) <U8 64B 'negative' 'positive'
#     │       Data variables:
#     │           feature_labels      (strand, sequence, feature) int8 183MB ...
#     │           sequence_input_ids  (strand, sequence) int64 487MB ...
#     │           region_labels       (strand, sequence, region) int8 426MB ...
#     │           sequence_masks      (strand, sequence) bool 61MB ...
#     │           label_masks         (strand, sequence, reason) int8 122MB ...
#     │           sequence_tokens     (strand, sequence) |S1 61MB b'g' b'g' b'g' b'a' b't' b't' b't' ... b'C' b'T' b'G' b'C' b'T' b'G' b'A'
#     │       Attributes:
#     │           species_id:     Athaliana
#     │           chromosome_id:  Chr1
#     │           filename:       Athaliana_447_Araport11.gene.gff3
#     ├── Group: /Athaliana/Chr3
#     │       Dimensions:             (strand: 2, sequence: 23459804, feature: 3, region: 7, reason: 2)
#     │       Coordinates:
#     │         * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
#     │         * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
#     │         * sequence            (sequence) int64 188MB 0 1 2 3 4 5 6 ... 23459797 23459798 23459799 23459800 23459801 23459802 23459803
#     │         * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
#     │         * strand              (strand) <U8 64B 'negative' 'positive'
#     │       Data variables:
#     │           feature_labels      (strand, sequence, feature) int8 141MB ...
#     │           label_masks         (strand, sequence, reason) int8 94MB ...
#     │           sequence_input_ids  (strand, sequence) int64 375MB ...
#     │           sequence_masks      (strand, sequence) bool 47MB ...
#     │           sequence_tokens     (strand, sequence) |S1 47MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'T' b'a' b'c' b'c' b'c' b't' b'a'
#     │           region_labels       (strand, sequence, region) int8 328MB ...
#     │       Attributes:
#     │           species_id:     Athaliana
#     │           chromosome_id:  Chr3
#     │           filename:       Athaliana_447_Araport11.gene.gff3
#     └── Group: /Athaliana/Chr4
#             Dimensions:             (strand: 2, sequence: 18584524, reason: 2, region: 7, feature: 3)
#             Coordinates:
#               * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
#               * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
#               * sequence            (sequence) int64 149MB 0 1 2 3 4 5 6 ... 18584517 18584518 18584519 18584520 18584521 18584522 18584523
#               * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
#               * strand              (strand) <U8 64B 'negative' 'positive'
#             Data variables:
#                 label_masks         (strand, sequence, reason) int8 74MB ...
#                 feature_labels      (strand, sequence, feature) int8 112MB ...
#                 region_labels       (strand, sequence, region) int8 260MB ...
#                 sequence_input_ids  (strand, sequence) int64 297MB ...
#                 sequence_tokens     (strand, sequence) |S1 37MB b'N' b'N' b'N' b'N' b'N' b'N' b'N' ... b'G' b'A' b'T' b'A' b'T' b'A' b'A'
#                 sequence_masks      (strand, sequence) bool 37MB ...
#             Attributes:
#                 species_id:     Athaliana
#                 chromosome_id:  Chr4
#                 filename:       Athaliana_447_Araport11.gene.gff3

(
    # Select a specific species + chromosome
    dt["Athaliana"]["Chr1"]
    # Slice to the first 100 coordinates in the sequence dimension (i.e. 100 bp)
    .isel(sequence=slice(100))
    # Load into memory
    .ds.compute()
)
# <xarray.Dataset> Size: 6kB
# Dimensions:             (strand: 2, sequence: 100, feature: 3, region: 7, reason: 2)
# Coordinates:
#   * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
#   * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
#   * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
#   * sequence            (sequence) int64 800B 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 ... 86 87 88 89 90 91 92 93 94 95 96 97 98 99
#   * strand              (strand) <U8 64B 'negative' 'positive'
# Data variables:
#     feature_labels      (strand, sequence, feature) int8 600B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#     sequence_input_ids  (strand, sequence) int64 2kB 5 5 5 3 6 6 6 5 5 5 3 6 6 6 5 5 5 3 ... 6 4 4 4 6 3 3 3 6 3 4 4 6 3 3 6 6 4
#     region_labels       (strand, sequence, region) int8 1kB 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#     sequence_masks      (strand, sequence) bool 200B False False False False False False False ... True True True True True True
#     label_masks         (strand, sequence, reason) int8 400B 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#     sequence_tokens     (strand, sequence) |S1 200B b'g' b'g' b'g' b'a' b't' b't' b't' ... b'C' b'T' b'A' b'A' b'T' b'T' b'C'
# Attributes:
#     species_id:     Athaliana
#     chromosome_id:  Chr1
#     filename:       Athaliana_447_Araport11.gene.gff3

dt["Athaliana"]["Chr1"]["sequence_tokens"]
# <xarray.DataArray 'sequence_tokens' (strand: 2, sequence: 30425192)> Size: 61MB
# array([[b'g', b'g', b'g', ..., b'A', b'C', b'T'],
#        [b'c', b'c', b'c', ..., b'T', b'G', b'A']], dtype='|S1')
# Coordinates:
#   * sequence  (sequence) int64 243MB 0 1 2 3 4 5 6 7 ... 30425184 30425185 30425186 30425187 30425188 30425189 30425190 30425191
#   * strand    (strand) <U8 64B 'negative' 'positive'

(
    dt["Athaliana"]["Chr1"]
    .sel(sequence=slice(1000, 1010))
    .sequence_tokens
    # Rename "sequence" to dimension to "pos" for more clarity in this example
    .rename(sequence="pos")
    # Show the DataArray before conversion to Pandas
    .pipe(tap, lambda data_array: print(data_array))
    # Convert 2D tokens DataArray to 1D Series
    .to_series()
    # Convert 1-byte tokens to 4-byte fixed-length unicode chars
    .str.decode("ascii")
    .unstack("strand") # Unstack from series to dataframe
    # Show tokens as a table with corresponding strand and position
    .pipe(tap, lambda data_frame: print(data_frame.to_markdown(index=True)))
); # noqa: E703
# <xarray.DataArray 'sequence_tokens' (strand: 2, pos: 11)> Size: 22B
# [22 values with dtype=|S1]
# Coordinates:
#   * pos      (pos) int64 88B 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010
#   * strand   (strand) <U8 64B 'negative' 'positive'
# |   pos | negative   | positive   |
# |------:|:-----------|:-----------|
# |  1000 | C          | G          |
# |  1001 | C          | G          |
# |  1002 | A          | T          |
# |  1003 | A          | T          |
# |  1004 | T          | A          |
# |  1005 | A          | T          |
# |  1006 | T          | A          |
# |  1007 | T          | A          |
# |  1008 | T          | A          |
# |  1009 | A          | T          |
# |  1010 | A          | T          |


(
    dt["Athaliana"]["Chr1"].ds
    # Select positive strand sequence and annotations
    .sel(strand="positive", drop=True)
    # Choose annotations starting at where the *most* annotations are present
    # in the sequence dimensions (around 3480 in this case)
    .pipe(lambda ds: ds.isel(sequence=slice(
        (locus := ds.region_labels.sum(dim="region").argmax().item()) - (step := 150),
        locus + step * 18, step # take relatively large steps to see changing annotations
    )))[["sequence_tokens", "region_labels"]]
    .to_dataframe()
    .set_index("sequence_tokens", append=True)
    .unstack("region")
    .pipe(lambda df: df[df.sum(axis="rows").sort_values(ascending=False).index])
    .applymap(lambda x: "█" if x else "│")
)
#                          region_labels                                                                      
# region                            gene transcript coding_sequence exon intron three_prime_utr five_prime_utr
# sequence sequence_tokens                                                                                    
# 3480     b'C'                        │          │               │    │      │               │              │
# 3630     b'A'                        █          █               │    █      │               │              █
# 3780     b'G'                        █          █               █    █      │               │              │
# 3930     b'T'                        █          █               █    │      █               │              │
# 4080     b'A'                        █          █               █    █      │               │              │
# 4230     b'T'                        █          █               █    █      │               │              │
# 4380     b'T'                        █          █               █    │      █               │              │
# 4530     b'G'                        █          █               █    █      │               │              │
# 4680     b'T'                        █          █               █    │      █               │              │
# 4830     b'A'                        █          █               █    █      │               │              │
# 4980     b'G'                        █          █               █    █      │               │              │
# 5130     b'T'                        █          █               █    │      █               │              │
# 5280     b'C'                        █          █               █    █      │               │              │
# 5430     b'G'                        █          █               █    │      █               │              │
# 5580     b'G'                        █          █               █    █      │               │              │
# 5730     b'G'                        █          █               │    █      │               █              │
# 5880     b'A'                        █          █               │    █      │               █              │
# 6030     b'A'                        │          │               │    │      │               │              │
# 6180     b'G'                        │          │               │    │      │               │              │


(
    dt
    # Select the first 100bp of the first 2 chromosomes for all species (2 in this case)
    .match("*/Chr[1,2]").isel(sequence=slice(100))
    .pipe(lambda dt: xr.combine_nested(
        # Generate species x chromosome dataset grid
        [
            [
                dt[species][chromosome].ds
                .expand_dims(dim=["species", "chromosome"])
                .assign_coords(
                    species=[species],
                    chromosome=[chromosome],
                )
                for chromosome in set([dt.attrs["chromosome_id"] for dt in leaves(dt)])
            ]
            for species in set([dt.attrs["species_id"] for dt in leaves(dt)])
        ], 
        # Concatenate along new dimensions simultaneously
        concat_dim=["species", "chromosome"]
    ))
)
# <xarray.Dataset> Size: 19kB
# Dimensions:             (species: 2, chromosome: 2, strand: 2, sequence: 100, feature: 3, region: 7, reason: 2)
# Coordinates:
#   * reason              (reason) <U19 152B 'incomplete_features' 'overlapping_gene'
#   * feature             (feature) <U15 180B 'five_prime_utr' 'cds' 'three_prime_utr'
#   * region              (region) <U15 420B 'gene' 'transcript' 'exon' ... 'five_prime_utr' 'three_prime_utr' 'coding_sequence'
#   * sequence            (sequence) int64 800B 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 ... 86 87 88 89 90 91 92 93 94 95 96 97 98 99
#   * strand              (strand) <U8 64B 'negative' 'positive'
#   * species             (species) <U9 72B 'Athaliana' 'Osativa'
#   * chromosome          (chromosome) <U4 32B 'Chr1' 'Chr2'
# Data variables:
#     feature_labels      (species, chromosome, strand, sequence, feature) int8 2kB 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0
#     sequence_input_ids  (species, chromosome, strand, sequence) int64 6kB 5 5 5 3 6 6 6 5 5 5 3 6 6 ... 3 3 3 4 4 4 6 3 3 3 4 4
#     region_labels       (species, chromosome, strand, sequence, region) int8 6kB 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0
#     sequence_masks      (species, chromosome, strand, sequence) bool 800B False False False False ... False False False False
#     label_masks         (species, chromosome, strand, sequence, reason) int8 2kB 1 1 1 1 1 1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1 1 1 1
#     sequence_tokens     (species, chromosome, strand, sequence) |S1 800B b'g' b'g' b'g' b'a' b't' ... b'a' b'a' b'a' b'c' b'c'


(
    # Get counts of token by species and chromosome (on positive strand only)
    pd.concat([
        (
            ds.sequence_tokens
            .pipe(lambda x: pd.Series(*np.unique(x, return_counts=True)))
            .str.decode("ascii").rename_axis(index="count").rename("token")
            .reset_index() # Convert to data frame w/ ("token", "count")
            .assign(species=ds.attrs["species_id"], chromosome=ds.attrs["chromosome_id"])
        )
        for ds in dt.sel(strand="positive").pipe(leaves)
    ], axis=0, ignore_index=True)
    # Reshape to move tokens into columns
    .pivot(index=["species", "chromosome"], columns="token", values="count")
    .fillna(0).astype(int)
    # Sort tokens by frequency
    .pipe(lambda df: df[df.mean(axis="rows").sort_values(ascending=False).index])
    .pipe(lambda df: df.div(df.sum(axis="columns"), axis="rows"))
    # Display frequencies as percentages
    .applymap(lambda x: "" if x == 0 else (pct if (pct := f"{x:.1%}") != "0.0%" else "<.1%"))
)
# token                     T      A      G      C     a     t     c     g     N     n     W     Y     M     K     R     S     D
# species   chromosome                                                                                                          
# Athaliana Chr1        29.5%  29.4%  16.7%  16.6%  2.5%  2.4%  1.2%  1.1%  0.5%  <.1%  <.1%  <.1%  <.1%  <.1%  <.1%  <.1%      
#           Chr2        29.8%  29.7%  16.7%  16.8%  2.4%  2.2%  1.2%  1.1%  <.1%  <.1%  <.1%  <.1%  <.1%  <.1%  <.1%  <.1%      
#           Chr3        29.3%  29.3%  16.9%  16.9%  2.6%  2.4%  1.3%  1.3%  <.1%        <.1%  <.1%  <.1%        <.1%  <.1%      
#           Chr4        29.3%  29.2%  16.7%  16.8%  2.7%  2.5%  1.4%  1.3%  <.1%  <.1%              <.1%                    <.1%
# Osativa   Chr1        21.4%  21.1%  16.5%  16.4%  7.1%  6.7%  5.5%  5.4%  <.1%                                                
#           Chr10       20.3%  19.7%  15.3%  15.2%  8.4%  8.0%  6.6%  6.4%  0.1%  <.1%                                          
#           Chr11       20.5%  20.0%  15.1%  15.0%  8.5%  8.0%  6.4%  6.3%  <.1%  <.1%                                          
#           Chr2        21.9%  21.5%  16.6%  16.5%  6.8%  6.4%  5.2%  5.1%  <.1%  <.1%                                          
#           Chr3        22.2%  21.8%  17.1%  17.0%  6.4%  6.0%  4.8%  4.7%  <.1%  <.1%                                          
#           Chr4        19.9%  19.5%  15.4%  15.3%  8.4%  7.9%  6.8%  6.7%  <.1%  <.1%                                          
#           Chr5        20.3%  19.8%  15.4%  15.3%  8.2%  7.7%  6.7%  6.5%  <.1%                                                
#           Chr6        20.6%  20.2%  15.6%  15.5%  8.0%  7.6%  6.3%  6.2%  0.1%                                                
#           Chr7        20.5%  20.0%  15.5%  15.4%  8.2%  7.8%  6.3%  6.3%  <.1%                                                
#           Chr8        20.3%  19.8%  15.2%  15.1%  8.5%  8.0%  6.6%  6.5%  <.1%                                                
#           Chr9        20.7%  20.2%  15.6%  15.5%  8.0%  7.5%  6.2%  6.1%  <.1%  <.1%                                          


from numba import njit # noqa: E402

@njit
def cumulative_sum(values: np.ndarray) -> np.ndarray:
    sum = 0
    result = np.zeros(len(values), dtype=np.int32)
    for i, value in enumerate(values):
        sum += value
        result[i] = sum
    return result

(
    xr.DataArray(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ],
        dims=["x", "y"],
    )
    .pipe(tap, lambda da: print(da))
    .pipe(lambda da: np.apply_along_axis(cumulative_sum, da.dims.index("y"), da))
)
# <xarray.DataArray (x: 2, y: 5)> Size: 80B
# array([[ 1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10]])
# Dimensions without coordinates: x, y
# Out[32]: 
# <xarray.DataArray (x: 2, y: 5)> Size: 40B
# array([[ 1,  3,  6, 10, 15],
#        [ 6, 13, 21, 30, 40]], dtype=int32)
# Dimensions without coordinates: x, y



