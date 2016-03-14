#!/usr/bin/env python

"""Create a ROC curve from judged question/answer pairs"""

import argparse

import matplotlib.pyplot as plt
import numpy
import pandas

QUESTION = "Question"
FREQUENCY = "Frequency"
ANSWER_ID = "AnswerId"
CONFIDENCE = "Confidence"
IN_PURVIEW = "InPurview"
JUDGEMENT = "Judgement"
CORRECT = "Correct"


def roc_curve(judgements):
    ts = confidence_thresholds(judgements, True)
    true_positive_rates = [true_positive_rate(judgements, t) for t in ts]
    false_positive_rates = [false_positive_rate(judgements, t) for t in ts]
    return true_positive_rates, false_positive_rates, ts


def true_positive_rate(judgements, t):
    correct = judgements[judgements[CORRECT]]
    true_positive = sum(correct[correct[CONFIDENCE] >= t][FREQUENCY])
    in_purview = sum(judgements[judgements[IN_PURVIEW]][FREQUENCY])
    return true_positive / float(in_purview)


def false_positive_rate(judgements, t):
    out_of_purview_questions = judgements[~judgements[IN_PURVIEW]]
    false_positive = sum(out_of_purview_questions[out_of_purview_questions[CONFIDENCE] >= t][FREQUENCY])
    out_of_purview = sum(judgements[~judgements[IN_PURVIEW]][FREQUENCY])
    return false_positive / float(out_of_purview)


def precision_curve(judgements):
    ts = confidence_thresholds(judgements, False)
    precision_values = [precision(judgements, t) for t in ts]
    attempted_values = [questions_attempted(judgements, t) for t in ts]
    return precision_values, attempted_values, ts


def precision(judgements, t):
    s = judgements[judgements[CONFIDENCE] >= t]
    correct = sum(s[s[CORRECT]][FREQUENCY])
    in_purview = sum(s[s[IN_PURVIEW]][FREQUENCY])
    return correct / float(in_purview)


def questions_attempted(judgements, t):
    s = judgements[judgements[CONFIDENCE] >= t]
    in_purview_attempted = sum(s[s[IN_PURVIEW]][FREQUENCY])
    total_in_purview = sum(judgements[judgements[IN_PURVIEW]][FREQUENCY])
    return in_purview_attempted / float(total_in_purview)


def confidence_thresholds(judgements, add_max):
    ts = judgements[CONFIDENCE].sort_values(ascending=False).unique()
    if add_max:
        ts = numpy.insert(ts, 0, numpy.Infinity)
    return ts


def plot_curve(p):
    x_label = p.columns[0]
    y_label = p.columns[1]
    plt.plot(p[x_label], p[y_label])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


class CsvFileType(object):
    def __init__(self, columns=None, rename=None):
        self.columns = columns
        self.rename = rename

    def __call__(self, filename):
        csv = pandas.read_csv(filename, usecols=self.columns, encoding="utf-8")
        if self.rename is not None:
            csv = csv.rename(columns=self.rename)
        return csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judgements", type=CsvFileType(),
                        help="judgements file created by annotation_assist_results.py")
    parser.add_argument("--correct", type=int, default=50, help="cutoff value for a correct score, default 50")
    parser.add_argument("--plot", action="store_true", help="draw the ROC curve")
    subparsers = parser.add_subparsers(dest="curve_type", help="type of curve to draw")
    subparsers.add_parser("roc", help="ROC curve")
    subparsers.add_parser("precision", help="precision curve")
    args = parser.parse_args()

    args.judgements[CORRECT] = args.judgements[JUDGEMENT] >= args.correct

    if args.curve_type == "roc":
        tpr, fpr, ts = roc_curve(args.judgements)
        p = pandas.DataFrame.from_dict({"TPR": tpr, "FPR": fpr, "Confidence": ts})
        p = p[["FPR", "TPR", "Confidence"]]
    else:
        p, a, ts = precision_curve(args.judgements)
        p = pandas.DataFrame.from_dict({"Precision": p, "Questions Attempted": a, "Confidence": ts})
        p = p[["Questions Attempted", "Precision", "Confidence"]]
    print(p.to_csv(index=False))
    if args.plot:
        plot_curve(p)
