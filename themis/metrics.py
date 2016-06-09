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
