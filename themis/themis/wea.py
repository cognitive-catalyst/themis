import pandas

from themis import logger, QUESTION, CONFIDENCE, FREQUENCY, ANSWER

QUESTION_TEXT = "QuestionText"
TOP_ANSWER_TEXT = "TopAnswerText"
USER_EXPERIENCE = "UserExperience"
TOP_ANSWER_CONFIDENCE = "TopAnswerConfidence"


def create_test_set_from_wea_logs(wea_logs, n):
    """
    Extract question text and the frequency with which a question was asked from the XMGR QuestionsData.csv report log,
    ignoring questions that were handled solely by dialog.

    This also ignores answers that begin "Here's Watson's response, but remember it's best to use full sentences.",
    because WEA does not log what the actual answer was for these.

    Optionally sample of a set of questions. The sampled question frequency will be drawn from the same distribution as
    the original one in the logs.

    :param wea_logs: DataFrame of QuestionsData.csv report log
    :param n: number of questions to sample
    :return: DataFrame with Question, Frequency columns
    """
    wea_logs = wea_logs[~wea_logs[USER_EXPERIENCE].isin(["DIALOG", "Dialog Response"])]
    wea_logs = wea_logs[
        ~wea_logs[ANSWER].str.contains("Here's Watson's response, but remember it's best to use full sentences.")]
    # Frequency is the number of times the question appeared in the report log.
    test_set = pandas.merge(wea_logs.drop_duplicates(QUESTION),
                            wea_logs.groupby(QUESTION).size().to_frame(FREQUENCY).reset_index())
    if n is not None:
        test_set = test_set.sample(n=n, weights=test_set[FREQUENCY])
    logger.info("Test set with %d unique questions" % len(test_set))
    test_set = test_set[[FREQUENCY, QUESTION]].sort_values([FREQUENCY, QUESTION], ascending=[False, True])
    return test_set.set_index(QUESTION)


def wea_test(test_set, wea_logs):
    wea_logs = fix_confidence_ranges(wea_logs)
    test_set = pandas.merge(test_set, wea_logs, on=QUESTION)
    missing_answers = test_set[test_set[ANSWER].isnull()]
    if len(missing_answers):
        logger.warning("%d questions without answers" % len(missing_answers))
    return test_set[[QUESTION, ANSWER, CONFIDENCE]].sort_values(QUESTION).set_index(QUESTION)


def fix_confidence_ranges(wea_logs):
    """
    Scale all confidence values between 0 and 1.

    The top answer confidence value in the WEA logs ranges either from 0-1 or 0-100 depending on the value in the user
    experience column.

    :param wea_logs: user interaction logs from QuestionsData.csv XMGR report
    :return: logs with all confidence values scaled between 0 and 1
    """
    # groupby drops null values, so rewrite these as "NA".
    wea_logs.loc[wea_logs[USER_EXPERIENCE].isnull(), USER_EXPERIENCE] = "NA"
    m = wea_logs.groupby(USER_EXPERIENCE)[CONFIDENCE].max()
    m[m > 1] = 100
    for user_experience in m.index:
        index = wea_logs[USER_EXPERIENCE] == user_experience
        wea_logs.loc[index, CONFIDENCE] = wea_logs[index][CONFIDENCE].div(m[user_experience])
    return wea_logs
