import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()


def to_average(x):
    y = [float(y) for y in x.split(",")]
    return sum(y) / len(y)


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


def validate(test_file, multi_exon_model, single_exon_model):
    validation_data = pd.read_csv(test_file, sep="\t")

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
    valid_multi_exon_y = np.array(
        [1 if x else 0 for x in valid_multi_exon_df["validated"]]
    )

    valid_single_exon_x = np.array(
        [list(valid_single_exon_df["TIS"]), list(valid_single_exon_df["TTS"])]
    ).transpose()
    valid_single_exon_y = np.array(
        [1 if x else 0 for x in valid_single_exon_df["validated"]]
    )

    valid_multi_exon = multi_exon_model.predict(valid_multi_exon_x)
    valid_single_exon = single_exon_model.predict(valid_single_exon_x)

    logger.info(
        "Multi-exon model recall on validation: "
        + str(recall_score(valid_multi_exon_y, valid_multi_exon))
    )
    logger.info(
        "Multi-exon model proportion of positive predictions on validation: "
        + str(sum(valid_multi_exon) / len(valid_multi_exon))
    )

    logger.info(
        "Single-exon model recall on validation: "
        + str(recall_score(valid_single_exon_y, valid_single_exon))
    )
    logger.info(
        "Single-exon model proportion of positive predictions on validation: "
        + str(sum(valid_single_exon) / len(valid_single_exon))
    )


def train(train_file, test_prop, estimated_positive_prop, rng):
    training_data = pd.read_csv(train_file, sep="\t")

    # Need separate models for with/without splice sites
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
    multi_exon_y = np.array([1 if x else 0 for x in training_multi_exon["validated"]])

    single_exon_x = np.array(
        [list(training_single_exon["TIS"]), list(training_single_exon["TTS"])]
    ).transpose()
    single_exon_y = np.array([1 if x else 0 for x in training_single_exon["validated"]])

    me_test_prop = int(multi_exon_y.shape[0] * test_prop)
    se_test_prop = int(single_exon_y.shape[0] * test_prop)

    ## Train for multi-exon transcripts

    test_indices = rng.choice(
        np.arange(multi_exon_y.shape[0]), me_test_prop, replace=False
    )

    test_x = multi_exon_x[test_indices,]
    train_x = np.delete(multi_exon_x, test_indices, axis=0)
    test_y = multi_exon_y[test_indices]
    train_y = np.delete(multi_exon_y, test_indices)

    multi_exon_model = fit_model(train_x, train_y, estimated_positive_prop)

    # test on held-out data
    test_y_hat = multi_exon_model.predict(test_x)

    logger.info("Multi-exon model recall: " + str(recall_score(test_y, test_y_hat)))
    logger.info(
        "Multi-exon model proportion of positive predictions: "
        + str(sum(test_y_hat) / len(test_y_hat))
    )

    ## Train for single exon transcripts

    test_indices = np.random.choice(
        np.arange(single_exon_y.shape[0]), se_test_prop, replace=False
    )
    test_x = single_exon_x[test_indices,]
    train_x = np.delete(single_exon_x, test_indices, axis=0)
    test_y = single_exon_y[test_indices]
    train_y = np.delete(single_exon_y, test_indices)

    single_exon_model = fit_model(train_x, train_y, estimated_positive_prop)

    # test on held-out data
    test_y_hat = single_exon_model.predict(test_x)

    logger.info("Single-exon model recall: " + str(recall_score(test_y, test_y_hat)))
    logger.info(
        "Single-exon model proportion of positive predictions: "
        + str(sum(test_y_hat) / len(test_y_hat))
    )

    return multi_exon_model, single_exon_model


def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Train MMLR Classifier using Positive-Unlabeled Learning"
    )
    parser.add_argument(
        "--training-table",
        "-t",
        type=str,
        required=True,
        help="tsv of Masked Motif scores for training and testing. tsv must contain the following"
        " columns: TIS, TTS, donor, acceptor, and validated. values in columns donor and acceptor"
        " may be a single float, or a comma-separated list of floats. The validated column should "
        "be a true/false value representing whether the transcript in question has experimental "
        "validation for its existence. Un-validated transcripts are treated as unlabeled, not as"
        " erroneous.",
    )
    parser.add_argument(
        "--estimated-positive-rate",
        "-p",
        type=float,
        default=0.75,
        help="estimated proportion of total transcripts that are true protein-coding genes",
    )
    parser.add_argument(
        "--test-proportion",
        type=float,
        default=0.25,
        help="proportion of training data to reserve for testing",
    )
    parser.add_argument(
        "--validation-table",
        "-v",
        type=str,
        default=None,
        help="Optional, tsv of Masked Motif scores for validation of MMLR. Format is "
        "the same as training table.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=str,
        required=True,
        help="path to output json of model weights",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    multi_exon_model, single_exon_model = train(
        args.training_table, args.test_proportion, args.estimated_positive_rate, rng
    )

    if args.validation_table is not None:
        validate(args.validation_table, multi_exon_model, single_exon_model)

    mem_coef = multi_exon_model.coef_[0]
    mem_intercept = multi_exon_model.intercept_[0]
    logger.info(
        f"Multi-exon model: y = sig({mem_coef[0]} * TIS + {mem_coef[1]} * TTS + {mem_coef[2]} * Donor + "
        f"{mem_coef[3]} * Acceptor + {mem_intercept})"
    )

    sem_coef = single_exon_model.coef_[0]
    sem_intercept = single_exon_model.intercept_[0]
    logger.info(
        f"Single-exon model: y = sig({sem_coef[0]} * TIS + {sem_coef[1]} * TTS + {sem_intercept})"
    )

    # Save parameters to dictionary
    out_dict = {}

    out_dict["multi_intercept"] = mem_intercept
    out_dict["multi_TIS"] = mem_coef[0]
    out_dict["multi_TTS"] = mem_coef[1]
    out_dict["multi_Donor"] = mem_coef[2]
    out_dict["multi_Acceptor"] = mem_coef[3]
    out_dict["single_intercept"] = sem_intercept
    out_dict["single_TIS"] = sem_coef[0]
    out_dict["single_TTS"] = sem_coef[1]

    with open(args.output_json, "w") as file:
        json.dump(out_dict, file)


if __name__ == "__main__":
    main()
