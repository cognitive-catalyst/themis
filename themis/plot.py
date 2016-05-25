import matplotlib.pyplot as plt
import numpy
import pandas

from themis import CORRECT, IN_PURVIEW, CONFIDENCE, FREQUENCY, CsvFileType, QUESTION, logger
from themis.analyze import SYSTEM, drop_missing

THRESHOLD = "Threshold"
TRUE_POSITIVE_RATE = "True Positive Rate"
FALSE_POSITIVE_RATE = "False Positive Rate"
PRECISION = "Precision"
ATTEMPTED = "Attempted"


def generate_curves(curve_type, collated):
    """
    Generate curves of the same type for multiple systems.

    :param collated: questions, answers, judgments, confidences, and frequencies across systems
    :type collated: list of pandas.DataFrame
    :param curve_type: 'precision' or 'roc'
    :type curve_type: str
    :return: mapping of system labels to curve data
    :rtype: {str : pandas.DataFrame}
    """
    collated = pandas.concat(collated)
    collated = drop_missing(collated)
    ds = collated.duplicated(subset=(SYSTEM, QUESTION))
    if any(ds):
        logger.error("Duplicate answers for %s" % ", ".join(collated[ds][SYSTEM].drop_duplicates()))
        raise ValueError("Cannot have multiple answers to the same question from a single system")
    try:
        curve = {"precision": precision_curve, "roc": roc_curve}[curve_type]
    except KeyError:
        raise ValueError("Invalid curve type %s" % curve_type)
    curves = {}
    for label, data in collated.groupby(SYSTEM):
        curves[label] = {"precision": PrecisionCurveFileType,
                         "roc": ROCCurveFileType}[curve_type].output_format(curve(data))
    return curves


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
    curve = pandas.DataFrame.from_dict({THRESHOLD: ts,
                                        TRUE_POSITIVE_RATE: true_positive_rates,
                                        FALSE_POSITIVE_RATE: false_positive_rates})
    return curve


def true_positive_rate(judgments, t):
    correct = judgments[judgments[CORRECT]]
    true_positive = sum(correct[correct[CONFIDENCE] >= t][FREQUENCY])
    in_purview = sum(judgments[judgments[IN_PURVIEW]][FREQUENCY])
    return true_positive / float(in_purview)


def false_positive_rate(judgments, t):
    out_of_purview_questions = judgments[judgments[IN_PURVIEW] == False]
    false_positive = sum(out_of_purview_questions[out_of_purview_questions[CONFIDENCE] >= t][FREQUENCY])
    out_of_purview = sum(out_of_purview_questions[FREQUENCY])
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
    precision_values = [(t, precision(judgments, t)) for t in ts]
    attempted_values = [(t, questions_attempted(judgments, t)) for t in ts]
    # Plot those threshold values that have both x and y values.
    xs = [(pv[0], pv[1], av[1]) for pv, av in zip(precision_values, attempted_values)
          if pv[1] is not None and av[1] is not None]
    ts, precision_values, attempted_values = zip(*xs)  # zip(*x) is the inverse of zip(x)
    curve = pandas.DataFrame.from_dict({THRESHOLD: ts, PRECISION: precision_values, ATTEMPTED: attempted_values})
    return curve


def precision(judgments, t):
    s = judgments[judgments[CONFIDENCE] >= t]
    correct = sum(s[s[CORRECT]][FREQUENCY])
    in_purview = sum(s[s[IN_PURVIEW]][FREQUENCY])
    try:
        return correct / float(in_purview)
    except ZeroDivisionError:
        logger.warning("No in-purview questions at threshold level %0.3f" % t)
        return None


def questions_attempted(judgments, t):
    s = judgments[judgments[CONFIDENCE] >= t]
    in_purview_attempted = sum(s[s[IN_PURVIEW]][FREQUENCY])
    total_in_purview = sum(judgments[judgments[IN_PURVIEW]][FREQUENCY])
    try:
        return in_purview_attempted / float(total_in_purview)
    except ZeroDivisionError:
        logger.warning("No in-purview questions attempted at threshold level %0.3f" % t)
        return None


def confidence_thresholds(judgments, add_max):
    ts = judgments[CONFIDENCE].sort_values(ascending=False).unique()
    if add_max:
        ts = numpy.insert(ts, 0, numpy.Infinity)
    return ts


def plot_curves(curves, curve_type):
    x_label = curves.values()[0].columns[0]
    y_label = curves.values()[0].columns[1]
    for label, curve in curves.items():
        plt.plot(curve[x_label], curve[y_label], label=label)
    plt.legend(loc={"precision": 3, "roc": 1}[curve_type])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


class PrecisionCurveFileType(CsvFileType):
    columns = [THRESHOLD, ATTEMPTED, PRECISION]

    def __init__(self):
        super(self.__class__, self).__init__(self.__class__.columns)

    @classmethod
    def output_format(cls, curve):
        curve = curve[cls.columns]
        curve = curve.sort_values(THRESHOLD)
        return curve.set_index(THRESHOLD)


class ROCCurveFileType(CsvFileType):
    columns = [THRESHOLD, FALSE_POSITIVE_RATE, TRUE_POSITIVE_RATE]

    def __init__(self):
        super(self.__class__, self).__init__(self.__class__.columns)

    @classmethod
    def output_format(cls, curve):
        curve = curve[cls.columns]
        curve = curve.sort_values(THRESHOLD)
        return curve.set_index(THRESHOLD)
