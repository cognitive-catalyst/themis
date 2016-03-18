import pandas

from themis import ANSWER, QUESTION, ANSWER_ID, CONFIDENCE, logger, FREQUENCY


def create_test_set_from_wea_logs(wea_logs, corpus, n):
    """
    Extract question text and the frequency with which a question was asked from the XMGR QuestionsData.csv report log,
    ignoring questions that were not answered.

    Optionally sample of a set of questions. The sampled question frequency will be drawn from the same distribution as
    the original one in the logs.

    :param wea_logs: DataFrame of QuestionsData.csv report log
    :param corpus: DataFrame mapping answer text to answer id
    :param n: number of questions to sample
    :return: DataFrame with Question, Frequency columns
    """
    wea_logs = wea_logs[~wea_logs["UserExperience"].isin(["DIALOG", "Dialog Response"])]
    wea_logs = add_answer_ids(wea_logs, corpus)
    # Frequency is the number of times the question appeared in the report log.
    test_set = pandas.merge(wea_logs.drop_duplicates(QUESTION),
                            wea_logs.groupby(QUESTION).size().to_frame(FREQUENCY).reset_index())
    if n is not None:
        test_set = test_set.sample(n=n, weights=test_set[FREQUENCY])
    logger.info("Test set with %d unique questions" % len(test_set))
    return test_set.sort_values([FREQUENCY, QUESTION], ascending=[False, True]).set_index(QUESTION)


def wea_test(test_set, corpus, wea_logs):
    wea_logs = add_answer_ids(wea_logs, corpus)
    wea_logs = fix_confidence_ranges(wea_logs)
    test_set = pandas.merge(test_set, wea_logs, on=QUESTION)
    missing_answers = test_set[test_set[ANSWER_ID].isnull()]
    if len(missing_answers):
        logger.warning("%d questions without answers" % len(missing_answers))
    return test_set[[QUESTION, ANSWER_ID, CONFIDENCE]].sort_values(QUESTION).set_index(QUESTION)


def add_answer_ids(wea_logs, corpus):
    wea_logs = pandas.merge(wea_logs, corpus, on=ANSWER, how="left")
    n = len(wea_logs.drop_duplicates(QUESTION))
    logger.info("%d unique questions in logs" % n)
    m = wea_logs[wea_logs[ANSWER_ID].isnull()].drop_duplicates(QUESTION)
    if not m.empty:
        m = len(m)
        logger.warning("%d questions without answers in corpus (%0.4f)" % (m, m / float(n)))
    return wea_logs.dropna(subset=[ANSWER_ID])


def fix_confidence_ranges(wea_logs):
    """
    Scale all confidence values between 0 and 1.

    The top answer confidence value in the WEA logs ranges either from 0-1 or 0-100 depending on the value in the user
    experience column.

    :param wea_logs: user interaction logs from QuestionsData.csv XMGR report
    :return: logs with all confidence values scaled between 0 and 1
    """
    # groupby drops null values, so rewrite these as "NA".
    wea_logs.loc[wea_logs["UserExperience"].isnull(), "UserExperience"] = "NA"
    m = wea_logs.groupby("UserExperience")[CONFIDENCE].max()
    m[m > 1] = 100
    for user_experience in m.index:
        index = wea_logs["UserExperience"] == user_experience
        wea_logs.loc[index, CONFIDENCE] = wea_logs[index][CONFIDENCE].div(m[user_experience])
    return wea_logs
