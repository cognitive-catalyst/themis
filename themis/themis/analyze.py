import matplotlib.pyplot as plt
import numpy
import pandas

from themis import QUESTION, CORRECT, CsvFileType, IN_PURVIEW, ANSWER, CONFIDENCE, FREQUENCY

THRESHOLD = "Threshold"
TRUE_POSITIVE_RATE = "True Positive Rate"
FALSE_POSITIVE_RATE = "False Positive Rate"
PRECISION = "Precision"
ATTEMPTED = "Attempted"

# Annotation Assist column names
QUESTION_TEXT = "Question_Text"
IS_IN_PURVIEW = "Is_In_Purview"
SYSTEM_ANSWER = "System_Answer"
ANNOTATION_SCORE = "Annotation_Score"


def roc_curve(judgements):
    """
    Plot a receiver operating characteristic (ROC) curve.

    :param judgements: data frame with confidence, in purview, correct, and frequency information
    :return: true positive rate, false positive rate, confidence thresholds
    """
    ts = confidence_thresholds(judgements, True)
    true_positive_rates = [true_positive_rate(judgements, t) for t in ts]
    false_positive_rates = [false_positive_rate(judgements, t) for t in ts]
    plot = pandas.DataFrame.from_dict(
        {THRESHOLD: ts, TRUE_POSITIVE_RATE: true_positive_rates, FALSE_POSITIVE_RATE: false_positive_rates})
    return plot[[THRESHOLD, FALSE_POSITIVE_RATE, TRUE_POSITIVE_RATE]].set_index(THRESHOLD)


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
    plot = pandas.DataFrame.from_dict({THRESHOLD: ts, PRECISION: precision_values, ATTEMPTED: attempted_values})
    return plot[[THRESHOLD, ATTEMPTED, PRECISION]].set_index(THRESHOLD)


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


def plot_curve(xs, ys, x_label, y_label):
    plt.plot(xs, ys)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def add_judgements_and_frequencies_to_qa_pairs(qa_pairs, judgements, question_frequencies):
    # The Annotation Assist tool strips newlines, so remove them from the answer text in the system output as well.
    qa_pairs[ANSWER] = qa_pairs[ANSWER].str.replace("\n", "")
    qa_pairs = pandas.merge(qa_pairs, judgements, on=(QUESTION, ANSWER)).dropna()
    return pandas.merge(qa_pairs, question_frequencies, on=QUESTION)


class AnnotationAssistFileType(CsvFileType):
    """
    Read the file produced by the `Annotation Assist <https://github.com/cognitive-catalyst/annotation-assist>` tool.
    """

    def __init__(self):
        super(self.__class__, self).__init__([QUESTION_TEXT, IS_IN_PURVIEW, SYSTEM_ANSWER, ANNOTATION_SCORE],
                                             {QUESTION_TEXT: QUESTION, IS_IN_PURVIEW: IN_PURVIEW,
                                              SYSTEM_ANSWER: ANSWER})

    def __call__(self, filename):
        annotation_assist = super(self.__class__, self).__call__(filename)
        annotation_assist[IN_PURVIEW] = annotation_assist[IN_PURVIEW].astype("bool")
        return annotation_assist[[QUESTION, ANSWER, IN_PURVIEW, ANNOTATION_SCORE]]


def mark_annotation_assist_correct(annotation_assist, judgement_threshold):
    """
    Convert the annotation score column to a boolean correct column by applying a threshold.

    :param annotation_assist: Annotation Assist file
    :param judgement_threshold: threshold above which an answer is deemed correct
    :return: dataframe with a boolean Correct column
    """
    annotation_assist[CORRECT] = annotation_assist[ANNOTATION_SCORE] >= judgement_threshold
    return annotation_assist.drop(ANNOTATION_SCORE, axis="columns")
