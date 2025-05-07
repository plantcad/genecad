from src.analysis import get_token_class_weights
from src.modeling import TOKEN_CLASS_NAMES

path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/prep/embedding_dataset"
dataset, weights = get_token_class_weights(path, split="train", class_names=TOKEN_CLASS_NAMES)
# > dataset
# <xarray.Dataset> Size: 7GB
# Dimensions:   (sample: 106691, sequence: 8192)
# Coordinates:
#     batch_id  (sample) int64 854kB 0 0 0 0 0 0 ... 1663 1663 1663 1663 1663 1663
# Dimensions without coordinates: sample, sequence
# Data variables:
#     labels    (sample, sequence) int64 7GB 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0

# > dataset.labels.size
# 874,012,672

# > weights
#     label_index         label_name  label_count    label_freq  label_freq_clip  label_weight
# 0             0         intergenic  541595819.0  7.531898e-01         0.753190  3.272548e-07
# 1             1           B-intron     246090.0  3.422340e-04         0.000342  7.202237e-04
# 2             2           I-intron   77018329.0  1.071083e-01         0.107108  2.301269e-06
# 3             3           L-intron     246373.0  3.426275e-04         0.000343  7.193964e-04
# 4             4           U-intron          NaN  2.781372e-08         0.000001  2.464850e-01
# 5             5   B-five_prime_utr      64969.0  9.035149e-05         0.000090  2.728068e-03
# 6             6   I-five_prime_utr   11744283.0  1.633261e-02         0.016333  1.509159e-05
# 7             7   L-five_prime_utr      65203.0  9.067691e-05         0.000091  2.718278e-03
# 8             8   U-five_prime_utr        176.0  2.447608e-07         0.000001  2.464850e-01
# 9             9              B-cds     284811.0  3.960827e-04         0.000396  6.223069e-04
# 10           10              I-cds   68067808.0  9.466096e-02         0.094661  2.603872e-06
# 11           11              L-cds     284847.0  3.961328e-04         0.000396  6.222283e-04
# 12           12              U-cds         20.0  2.781372e-08         0.000001  2.464850e-01
# 13           13  B-three_prime_utr      60343.0  8.391818e-05         0.000084  2.937207e-03
# 14           14  I-three_prime_utr   19330430.0  2.688256e-02         0.026883  9.168956e-06
# 15           15  L-three_prime_utr      59823.0  8.319502e-05         0.000083  2.962738e-03
# 16           16  U-three_prime_utr        149.0  2.072122e-07         0.000001  2.464850e-01

# > int(weights["label_count"].sum())
# 719,069,473

# > weights.sort_values("label_index")["label_weight"].values
# array([3.27254844e-07, 7.20223719e-04, 2.30126851e-06, 7.19396424e-04,
#        2.46485015e-01, 2.72806808e-03, 1.50915858e-05, 2.71827761e-03,
#        2.46485015e-01, 6.22306916e-04, 2.60387194e-06, 6.22228267e-04,
#        2.46485015e-01, 2.93720655e-03, 9.16895564e-06, 2.96273766e-03,
#        2.46485015e-01])