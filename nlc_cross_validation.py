#!/usr/bin/env python

"""Train and test a k-fold Natural Language Classifier (NLC) model.
This creates partition files in the specified experiment directory.
If the directory already exists, train the partition files already there."""

import argparse
import glob
import json
import logging
import os

import pandas
import sklearn.cross_validation
from watson_developer_cloud import NaturalLanguageClassifierV1 as NaturalLanguageClassifier

logger = logging.getLogger(__name__)

QUESTION = "Question"
ANSWER_ID = "AnswerId"
PREDICTED_ANSWER_ID = "PredictedAnswerId"
CONFIDENCE = "Confidence"


def create_data_partitions(directory, truth_file, folds):
    """Create a new directory to hold the data partitions and experimental results.

    This creates training and test partition files called train.nnn.csv and test.nnn.csv with rows sampled from the
    ground truth file.

    :param directory: directory to create
    :param truth_file: ground truth csv file with Question and AnswerId columns
    :param folds: number of cross-validation folds
    """
    os.mkdir(directory)
    truth = pandas.read_csv(truth_file, encoding="utf-8")
    logger.info("%s: %d-fold cross-validation of %d instances" % (directory, folds, len(truth)))
    for i, (train_indexes, test_indexes) in enumerate(sklearn.cross_validation.KFold(len(truth), folds, shuffle=True)):
        train = truth.iloc[train_indexes]
        train.to_csv(os.path.join(directory, "train.%03d.csv" % i), header=False, index=False, encoding="utf-8")
        test = truth.iloc[test_indexes]
        test.to_csv(os.path.join(directory, "test.%03d.csv" % i), index=False, encoding="utf-8")


def train_nlc_models(nlc, directory, experiment_name):
    if experiment_name is None:
        experiment_name = os.path.basename(directory)
    # Enumerate all train.*.csv files in the directory.
    for training_filename in glob.glob(os.path.join(directory, "train.*.csv")):
        # Start training NLC models for each.
        n = os.path.basename(training_filename).split(".")[1]
        model_name = "%s %s" % (experiment_name, n)
        response = os.path.join(directory, "response.%s.json" % n)
        if not os.path.isfile(response):
            logger.info("Train model %s" % model_name)
            with open(training_filename, "rb") as training_file:
                r = nlc.create(training_data=training_file, name=model_name)
            logger.debug(json.dumps(r, indent=2))
            with open(response, "w") as f:
                json.dump(r, f, indent=2)


def nlc_model_status(nlc, directory):
    for model, classifier_id in models_from_directory(directory):
        status = nlc.status(classifier_id)
        print("Model %s\t%s: %s" % (model, status["status"], status["status_description"]))


def test_nlc_models(nlc, directory, random):
    results_filenames = []
    for model_id, classifier_id in models_from_directory(directory):
        results_filename = os.path.join(directory, "results.%s.csv" % model_id)
        results_filenames.append(results_filename)
        if not os.path.isfile(results_filename):
            logger.info("Test model %s" % model_id)
            test_filename = os.path.join(directory, "test.%s.csv" % model_id)
            test = pandas.read_csv(test_filename, encoding="utf-8")
            if random is None:
                predicted_answers = pandas.DataFrame(columns=(PREDICTED_ANSWER_ID, CONFIDENCE))
                n = len(test[QUESTION])
                for i, question in enumerate(test[QUESTION]):
                    logger.debug("Model %s (%d/%d): %s" % (model_id, i, n, question))
                    answer_id, confidence = top_nlc_answer(nlc, classifier_id, question)
                    predicted_answers = predicted_answers.append(
                        {PREDICTED_ANSWER_ID: answer_id, CONFIDENCE: confidence}, ignore_index=True)
                test = test.join(predicted_answers)
            else:
                test[PREDICTED_ANSWER_ID] = random[ANSWER_ID].sample(len(test))
            test.set_index(QUESTION).to_csv(results_filename, encoding="utf-8")
        else:
            logger.info("Results already exist for model %s" % model_id)
        all_results = pandas.concat([pandas.read_csv(filename, encoding="utf-8").set_index(QUESTION)
                                     for filename in results_filenames])
        # noinspection PyUnresolvedReferences
        all_results.to_csv(os.path.join(directory, "results.csv"), encoding="utf-8")


def top_nlc_answer(nlc, classifier_id, question):
    classification = nlc.classify(classifier_id, question)
    return classification["classes"][0]["class_name"], classification["classes"][0]["confidence"]


def models_from_directory(directory):
    models = []
    for filename in glob.glob(os.path.join(directory, "response.*.json")):
        with open(filename) as f:
            classifier_id = json.load(f)["classifier_id"]
            models.append((os.path.basename(filename).split(".")[1], classifier_id))
    return sorted(models)


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=str, help="experiment directory")
    parser.add_argument("username", type=str, help="NLC username")
    parser.add_argument("password", type=str, help="NLC password")
    parser.add_argument("--url", type=str,
                        default="https://gateway-s.watsonplatform.net/natural-language-classifier/api",
                        help="NLC url")
    parser.add_argument("--log", type=str, default="ERROR", help="logging level")

    subparsers = parser.add_subparsers(dest="system", help="train, status, or test")

    train_parser = subparsers.add_parser("train", help="train NLC cross-validation models")
    train_parser.add_argument("truth", type=argparse.FileType(), help="truth file")
    train_parser.add_argument("--experiment-name", type=str,
                              help="experiment name, defaults to the experiment directory")
    train_parser.add_argument("--folds", type=int, default=5, help="number of cross validation folds, default 5")

    status_parser = subparsers.add_parser("status", help="return NLC model status")

    test_parser = subparsers.add_parser("test", help="apply models to test data")
    test_parser.add_argument("--random", metavar="truth file", type=argparse.FileType(),
                             help="randomly assign answers from the ground truth file")

    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(message)s")
    nlc = NaturalLanguageClassifier(url=args.url, username=args.username, password=args.password)

    if args.system == "train":
        if not os.path.isdir(args.directory):
            create_data_partitions(args.directory, args.truth, args.folds)
        train_nlc_models(nlc, args.directory, args.experiment_name)
    elif args.system == "status":
        nlc_model_status(nlc, args.directory)
    else:  # args.system == "test"
        if args.random:
            args.random = pandas.read_csv(args.random, encoding="utf-8")
        test_nlc_models(nlc, args.directory, args.random)
