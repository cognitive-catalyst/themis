#!/usr/bin/env python

import argparse
import os

"""Display a precision curve for multiple systems"""

import matplotlib.pyplot as plt
import pandas

ANSWER_ID = "AnswerId"
CONFIDENCE = "Confidence"
BIN = "Bin"


def precision_curve_plot(truth, predictions, points, labels, title):
    for prediction, label in zip(predictions, labels):
        questions_attempted, precision = precision_curve(truth, pandas.read_csv(prediction, encoding="utf-8"), points)
        plt.plot(questions_attempted, precision, label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel("Questions Attempted")
    plt.ylabel("Precision")
    return plt


def precision_curve(truth, predictions, bins):
    in_purview = pandas.merge(truth, predictions, on="Question")
    in_purview[BIN] = pandas.qcut(in_purview[CONFIDENCE], bins, labels=False)
    correct = in_purview[in_purview[ANSWER_ID] == in_purview["PredictedAnswerId"]]
    questions_attempted = []
    precision = []
    for t in xrange(bins):
        questions_attempted.append(ratio(sum(in_purview[BIN] >= t), len(in_purview)))
        precision.append(ratio(sum(correct[BIN] >= t), sum(in_purview[BIN] >= t)))
    return questions_attempted, precision


def ratio(n, d):
    if n == 0 and d == 0:
        return 0
    else:
        return n / float(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth", type=argparse.FileType(), help="truth")
    parser.add_argument("predictions", nargs="+", type=argparse.FileType(),
                        help="prediction file containing Question, PredictedAnswerId, and Confidence columns")
    parser.add_argument("--title", type=str, default="Precision Curve", help="graph title")
    parser.add_argument("--points", type=int, default=100, help="number of x-axis points in the graph")
    parser.add_argument("--labels", nargs="+", type=str, help="prediction labels")
    args = parser.parse_args()

    if not len(args.labels) == len(args.predictions):
        parser.print_usage()
        parser.error("Number of prediction labels must equal number of labels")
    truth = pandas.read_csv(args.truth, encoding="utf-8")
    if args.labels is None:
        args.labels = [os.path.basename(prediction) for prediction in args.predictions]
    precision_curve_plot(truth, args.predictions, args.points, args.labels, args.title).show()
