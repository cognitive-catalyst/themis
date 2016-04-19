import matplotlib.pyplot as plt
import numpy
import pandas

from themis import CORRECT, IN_PURVIEW, CONFIDENCE, FREQUENCY, QUESTION, ANSWER
from themis.judge import strip_newlines_from_answer_text

THRESHOLD = "Threshold"
TRUE_POSITIVE_RATE = "True Positive Rate"
FALSE_POSITIVE_RATE = "False Positive Rate"
PRECISION = "Precision"
ATTEMPTED = "Attempted"


def generate_curves(curve_type, labeled_qa_pairs, judgments, frequency):
    """
    Generate curves of the same type for multiple systems.

    :param curve_type: 'precision' or 'roc'
    :type curve_type: str
    :param labeled_qa_pairs: mapping of system labels to Q&A pairs
    :type labeled_qa_pairs: {str : pandas.DataFrame}
    :param judgments: Q&A pairs with purview and correctness judgments
    :type judgments: pandas.DataFrame
    :param frequency: questions and their frequencies
    :type frequency: pandas.DataFrame
    :return: mapping of system labels to plot data
    :rtype: {str : pandas.DataFrame}
    """
    curves = {}
    for label, answers in labeled_qa_pairs:
        data = add_judgments_and_frequencies_to_qa_pairs(answers, judgments, frequency)
        if curve_type == "precision":
            curves[label] = precision_curve(data)
        elif curve_type == "roc":
            curves[label] = roc_curve(data)
        else:
            raise ValueError("Invalid curve type %s" % curve_type)
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


def plot_curves(curves):
    x_label = curves.values()[0].columns[0]
    y_label = curves.values()[0].columns[1]
    for label, curve in curves.items():
        plt.plot(curve[x_label], curve[y_label], label=label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def add_judgments_and_frequencies_to_qa_pairs(qa_pairs, judgments, question_frequencies):
    """
    Collate system answer confidences and annotator judgments by question/answer pair.
    Add to each pair the question frequency.

    Though you expect the set of question/answer pairs in the system answers and judgments to not be disjoint, it may
    be the case that neither is a subset of the other. If annotation is incomplete, there may be Q/A pairs in the
    system answers that haven't been annotated yet. If multiple systems are being judged, there may be Q/A pairs in the
    judgements that don't appear in the system answers.

    :param qa_pairs: question, answer, and confidence provided by a Q&A system
    :type qa_pairs: pandas.DataFrame
    :param judgments: question, answer, in purview, and judgement provided by annotators
    :type judgments: pandas.DataFrame
    :param question_frequencies: question and question frequency in the test set
    :type question_frequencies: pandas.DataFrame
    :return: question and answer pairs with confidence, in purview, judgement and question frequency
    :rtype: pandas.DataFrame
    """
    qa_pairs = strip_newlines_from_answer_text(qa_pairs)
    qa_pairs = pandas.merge(qa_pairs, judgments, on=(QUESTION, ANSWER))
    return pandas.merge(qa_pairs, question_frequencies, on=QUESTION)
