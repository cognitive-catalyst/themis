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