import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

input_file = "/Users/ahb232/Desktop/machine_learning/zero_shot_assessment/model_8192_l24/scores/Zmays_833_Zm-B73_scores.tsv"
training_data = pd.read_csv(input_file, sep="\t")


def to_average(x):
    y = [float(y) for y in x.split(",")]
    return sum(y) / len(y)


training_single_exon = training_data[training_data["donor"].isna()]
training_multi_exon = training_data[training_data["donor"].notna()]

multi_exon_x = np.array(
    [
        list(training_multi_exon["TIS"]),
        list(training_multi_exon["TTS"]),
        [to_average(x) for x in training_multi_exon["donor"]],
        [to_average(x) for x in training_multi_exon["acceptor"]],
    ]
).transpose()
multi_exon_y = np.array([1 if x else 0 for x in training_multi_exon["classical"]])

single_exon_x = np.array(
    [list(training_single_exon["TIS"]), list(training_single_exon["TTS"])]
).transpose()
single_exon_y = np.array([1 if x else 0 for x in training_single_exon["classical"]])


def fit_model(train_x, train_y, positive_rate=0.75):
    # Adapted from https://github.com/hkiyomaru/pu-learning
    _clf = LogisticRegression().fit(train_x, train_y)

    train_x_labeled = train_x[train_y == 1]
    train_x_unlabeled = train_x[train_y == 0]

    c = train_x_labeled.shape[0] / (
        train_x_labeled.shape[0] + train_x_unlabeled.shape[0] * positive_rate
    )

    train_ss_prob_unlabeled = _clf.predict_proba(train_x_unlabeled)[:, 1]

    new_train_x = []
    new_train_y = []
    sample_weight = []

    # Labeled data is used as positive (y=1)
    for x_labeled in train_x_labeled:
        new_train_x.append(x_labeled)
        new_train_y.append(1)
        sample_weight.append(1)

    # Unlabeled data is used as positive (y=1)
    for x_unlabeled, train_s_prob_unlabeled in zip(
        train_x_unlabeled, train_ss_prob_unlabeled
    ):
        new_train_x.append(x_unlabeled)
        new_train_y.append(1)
        w_pos = ((1 - c) / c) * (train_s_prob_unlabeled / (1 - train_s_prob_unlabeled))
        sample_weight.append(w_pos)

    # Unlabeled data is used as negative as well (y=0)
    for x_unlabeled, train_s_prob_unlabeled in zip(
        train_x_unlabeled, train_ss_prob_unlabeled
    ):
        new_train_x.append(x_unlabeled)
        new_train_y.append(0)
        w_pos = ((1 - c) / c) * (train_s_prob_unlabeled / (1 - train_s_prob_unlabeled))
        w_neg = 1 - w_pos
        sample_weight.append(w_neg)

    clf = LogisticRegression().fit(
        new_train_x, new_train_y, sample_weight=sample_weight
    )

    return clf


## Train for multi-exon transcripts

test_indices = np.random.choice(np.arange(multi_exon_y.shape[0]), 6260, replace=False)
test_x = multi_exon_x[test_indices,]
train_x = np.delete(multi_exon_x, test_indices, axis=0)
test_y = multi_exon_y[test_indices]
train_y = np.delete(multi_exon_y, test_indices)

multi_exon_model = fit_model(train_x, train_y)

test_y_hat = multi_exon_model.predict(test_x)
test_y_prob = multi_exon_model.predict_proba(test_x)[:, 1]

recall_score(test_y, test_y_hat)

sum(np.logical_and(test_y_hat, test_y))

sum(test_y_hat) / len(test_y_hat)

## Train for single exon transcripts

test_indices = np.random.choice(np.arange(single_exon_y.shape[0]), 994, replace=False)
test_x = single_exon_x[test_indices,]
train_x = np.delete(single_exon_x, test_indices, axis=0)
test_y = single_exon_y[test_indices]
train_y = np.delete(single_exon_y, test_indices)

single_exon_model = fit_model(train_x, train_y)

test_y_hat = single_exon_model.predict(test_x)
test_y_prob = single_exon_model.predict_proba(test_x)[:, 1]

recall_score(test_y, test_y_hat)

sum(np.logical_and(test_y_hat, test_y))

sum(test_y_hat) / len(test_y_hat)

# test
input_file = "/Users/ahb232/Desktop/machine_learning/zero_shot_assessment/model_8192_l24/scores/Slycopersicum_all_scores_pc2.tsv"
validation_data = pd.read_csv(input_file, sep="\t")

valid_single_exon_df = validation_data[validation_data["donor"].isna()]
valid_multi_exon_df = validation_data[validation_data["donor"].notna()]

valid_multi_exon_x = np.array(
    [
        list(valid_multi_exon_df["TIS"]),
        list(valid_multi_exon_df["TTS"]),
        [to_average(x) for x in valid_multi_exon_df["donor"]],
        [to_average(x) for x in valid_multi_exon_df["acceptor"]],
    ]
).transpose()
valid_multi_exon_y = np.array([1 if x else 0 for x in valid_multi_exon_df["classical"]])

valid_single_exon_x = np.array(
    [list(valid_single_exon_df["TIS"]), list(valid_single_exon_df["TTS"])]
).transpose()
valid_single_exon_y = np.array(
    [1 if x else 0 for x in valid_single_exon_df["classical"]]
)

valid_multi_exon = multi_exon_model.predict(valid_multi_exon_x)
valid_single_exon = single_exon_model.predict(valid_single_exon_x)

recall_score(valid_multi_exon_y, valid_multi_exon)

recall_score(valid_single_exon_y, valid_single_exon)

sum(valid_multi_exon) / len(valid_multi_exon)

sum(valid_single_exon) / len(valid_single_exon)
