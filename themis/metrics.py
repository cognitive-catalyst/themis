import numpy

from themis import CONFIDENCE, CORRECT, FREQUENCY, IN_PURVIEW, logger


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


def precision_grounded_confidence(ts, ps, qas, confidence, method='precision_only'):
    # lookup the associated precision & QA for the confidence using the closest threshold value (in case of mismatches)
    t_index = (numpy.abs(ts - confidence)).argmin()
    precision_t = ps[t_index]
    qa_t = qas[t_index]
    if method == 'inverse_qa_p_corrected':
        return (1 - qa_t) * precision_t
    elif method == 'inverse_qa':
        return (1 - qa_t)
    elif method == 'precision_only':
        return precision_t
    else:
        raise ValueError("Invalid method choice for precision_grounded_confidence.")


def __standardize_confidence(system, method='rank'):
    """
    Takes a dataframe of a SINGLE SYSTEM with associated CONFIDENCE scores and standardizes the confidence
    values using the percentile in the list as the new confidence.

    :param system: dataframe containing rows of a single system that has a CONFIDENCE column present.
    :return: a Series containing the standardized confidence.
    :rtype pandas.Series
    """
    if method == 'precision':
        ts = confidence_thresholds(system, False)
        ps = [precision(system, t) for t in ts]
        qas = [questions_attempted(system, t) for t in ts]
        return system.apply(lambda x: precision_grounded_confidence(ts, ps, qas, x[CONFIDENCE],
                                                                   method='precision_only'), axis=1)
    elif method == 'rank':
        return system[CONFIDENCE].rank(pct=True)
    else:
        raise ValueError("Invalid method choice for standardize_confidence.")
