import pandas

from themis import ANSWER, ANSWER_ID, CONFIDENCE, logger
from themis.question import DATE_TIME, USER_EXPERIENCE


def filter_corpus(corpus, max_size):
    """
    Remove corpus entries above a specified size.

    :param corpus: corpus with Answer Id and Answer columns
    :type corpus: pandas.DataFrame
    :param max_size: maximum allowed Answer size in characters
    :type max_size: int
    :return: corpus with oversize answers removed
    :rtype: pandas.DataFrame
    """
    if max_size is not None:
        filtered = corpus[corpus[ANSWER].str.len() <= max_size]
        n = len(corpus)
        if n:
            m = n - len(filtered)
            logger.info("Filtered %d of %d answers over size %d (%0.3f%%)" % (m, n, max_size, 100.0 * m / n))
        corpus = filtered
    return corpus.set_index(ANSWER_ID)


def filter_usage_log_by_date(usage_log, before, after):
    """
    Only retain questions that were asked within a specified time window.

    :param usage_log: QuestionsData.csv report log
    :type usage_log: pandas.DataFrame
    :param before: only use questions from before this date
    :type before: pandas.datetime
    :param after: only use questions from after this date
    :type after: pandas.datetime
    :return: usage log with questions in the specified time span
    :rtype: pandas.DataFrame
    """
    n = len(usage_log)
    if after is not None:
        usage_log = usage_log[usage_log[DATE_TIME] >= after]
    if before is not None:
        usage_log = usage_log[usage_log[DATE_TIME] <= before]
    if n:
        logger.info("Filtered %d questions by date" % (n - len(usage_log)))
    return usage_log


def filter_usage_log_by_user_experience(usage_log, disallowed):
    """
    Only retain questions whose 'user experience' value does not appear on a blacklist.

    :param usage_log: QuestionsData.csv report log
    :type usage_log: pandas.DataFrame
    :param disallowed: set of disallowed 'user experience' values
    :type disallowed: enumerable set of str
    :return: usage log with questions removed
    :rtype: pandas.DataFrame
"""
    if USER_EXPERIENCE in usage_log.columns:
        n = len(usage_log)
        usage_log = usage_log[~usage_log[USER_EXPERIENCE].isin(disallowed)]
        logger.info("Removed %d questions with user experience '%s'" % ((n - len(usage_log)), ",".join(disallowed)))
    else:
        logger.warn("No {0} field in Usage Log".format(USER_EXPERIENCE))
    return usage_log


def deakin(usage_log):
    low_confidence_response = usage_log[ANSWER].str.contains(
        "Here's Watson's response, but remember it's best to use full sentences.")
    logger.info("Removed %d questions with low confidence responses" % sum(low_confidence_response))
    usage_log = usage_log[~low_confidence_response]
    usage_log = filter_usage_log_by_user_experience(usage_log, ["Dialog Response"])
    usage_log = fix_confidence_ranges(usage_log)
    return usage_log


def fix_confidence_ranges(usage_log):
    """
    Scale all confidence values between 0 and 1.

    The top answer confidence value in the WEA logs ranges either from 0-1 or 0-100 depending on the value in the user
    experience column.

    :param usage_log: user interaction logs from QuestionsData.csv XMGR report
    :type usage_log: pandas.DataFrame
    :return: logs with all confidence values scaled between 0 and 1
    :rtype: pandas.DataFrame
    """
    # groupby drops null values, so rewrite these as "NA".
    usage_log.loc[usage_log[USER_EXPERIENCE].isnull(), USER_EXPERIENCE] = "NA"
    m = usage_log.groupby(USER_EXPERIENCE)[CONFIDENCE].max()
    m[m > 1] = 100
    for user_experience in m.index:
        index = usage_log[USER_EXPERIENCE] == user_experience
        usage_log.loc[index, CONFIDENCE] = usage_log[index][CONFIDENCE].div(m[user_experience])
    return usage_log
