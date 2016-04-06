import matplotlib.pyplot as plt
import numpy
import pandas

from themis import CORRECT, IN_PURVIEW, CONFIDENCE, FREQUENCY

THRESHOLD = "Threshold"
TRUE_POSITIVE_RATE = "True Positive Rate"
FALSE_POSITIVE_RATE = "False Positive Rate"
PRECISION = "Precision"
ATTEMPTED = "Attempted"


def roc_curve(judgments):
    """
    Generate points for a receiver operating characteristic (ROC) curve.

    :param judgments: confidence, in purview, correct, and frequency information
    :type judgments: pandas.DataFrame
    :return: true positive rate, false positive rate, and confidence thresholds
    :rtype: pandas.DataFrame
    """
    ts = confidence_thresholds(judgments, True)
    true_positive_rates = [true_positive_rate(judgments, t) for t in ts]
    false_positive_rates = [false_positive_rate(judgments, t) for t in ts]
    plot = pandas.DataFrame.from_dict(
        {THRESHOLD: ts, TRUE_POSITIVE_RATE: true_positive_rates, FALSE_POSITIVE_RATE: false_positive_rates})
    return plot[[THRESHOLD, FALSE_POSITIVE_RATE, TRUE_POSITIVE_RATE]].set_index(THRESHOLD)


def true_positive_rate(judgments, t):
    correct = judgments[judgments[CORRECT]]
    true_positive = sum(correct[correct[CONFIDENCE] >= t][FREQUENCY])
    in_purview = sum(judgments[judgments[IN_PURVIEW]][FREQUENCY])
    return true_positive / float(in_purview)


def false_positive_rate(judgments, t):
    out_of_purview_questions = judgments[~judgments[IN_PURVIEW]]
    false_positive = sum(out_of_purview_questions[out_of_purview_questions[CONFIDENCE] >= t][FREQUENCY])
    out_of_purview = sum(judgments[~judgments[IN_PURVIEW]][FREQUENCY])
    return false_positive / float(out_of_purview)


def precision_curve(judgments):
    """
    Generate points for a precision curve.

    :param judgments: confidence, in purview, correct, and frequency information
    :type judgments: pandas.DataFrame
    :return: questions attempted, precision, and confidence thresholds
    :rtype: pandas.DataFrame
    """
    ts = confidence_thresholds(judgments, False)
    precision_values = [precision(judgments, t) for t in ts]
    attempted_values = [questions_attempted(judgments, t) for t in ts]
    plot = pandas.DataFrame.from_dict({THRESHOLD: ts, PRECISION: precision_values, ATTEMPTED: attempted_values})
    return plot[[THRESHOLD, ATTEMPTED, PRECISION]].set_index(THRESHOLD)


def precision(judgments, t):
    s = judgments[judgments[CONFIDENCE] >= t]
    correct = sum(s[s[CORRECT]][FREQUENCY])
    in_purview = sum(s[s[IN_PURVIEW]][FREQUENCY])
    return correct / float(in_purview)


def questions_attempted(judgments, t):
    s = judgments[judgments[CONFIDENCE] >= t]
    in_purview_attempted = sum(s[s[IN_PURVIEW]][FREQUENCY])
    total_in_purview = sum(judgments[judgments[IN_PURVIEW]][FREQUENCY])
    return in_purview_attempted / float(total_in_purview)


def confidence_thresholds(judgments, add_max):
    ts = judgments[CONFIDENCE].sort_values(ascending=False).unique()
    if add_max:
        ts = numpy.insert(ts, 0, numpy.Infinity)
    return ts


def plot_curves(curves, labels):
    x_label = curves[0].columns[1]
    y_label = curves[0].columns[2]
    for curve, label in zip(curves, labels):
        plt.plot(curve[x_label], curve[y_label], label=label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
