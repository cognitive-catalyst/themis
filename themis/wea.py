import re

import pandas

from themis import logger, QUESTION, CONFIDENCE, FREQUENCY, ANSWER, ANSWER_ID, CsvFileType

# Column headers in WEA logs
QUESTION_TEXT = "QuestionText"
TOP_ANSWER_TEXT = "TopAnswerText"
USER_EXPERIENCE = "UserExperience"
TOP_ANSWER_CONFIDENCE = "TopAnswerConfidence"
DATE_TIME = "DateTime"


def create_test_set_from_wea_logs(wea_logs, before, after, n):
    """
    Extract question text and the frequency with which a question was asked from the XMGR QuestionsData.csv report log,
    ignoring questions that were handled solely by dialog.

    This also ignores answers that begin "Here's Watson's response, but remember it's best to use full sentences.",
    because WEA does not log what the actual answer was for these.

    Optionally sample of a set of questions. The sampled question frequency will be drawn from the same distribution as
    the original one in the logs.

    :param wea_logs: QuestionsData.csv report log
    :type wea_logs: pandas.DataFrame
    :param before: only use questions from before this date
    :type before:
    :param after: only use questions from after this date
    :type after:
    :param n: number of questions to sample, use all questions if None
    :type n: int
    :return: questions and their frequencies
    :rtype: pandas.DataFrame
    """
    wea_logs = wea_logs[~wea_logs[USER_EXPERIENCE].isin(["DIALOG", "Dialog Response"])]
    wea_logs = wea_logs[
        ~wea_logs[ANSWER].str.contains("Here's Watson's response, but remember it's best to use full sentences.")]
    if after is not None:
        wea_logs = wea_logs[wea_logs[DATE_TIME] >= after]
    if before is not None:
        wea_logs = wea_logs[wea_logs[DATE_TIME] <= before]
    # Frequency is the number of times the question appeared in the report log.
    test_set = pandas.merge(wea_logs.drop_duplicates(QUESTION),
                            wea_logs.groupby(QUESTION).size().to_frame(FREQUENCY).reset_index())
    if n is not None:
        test_set = test_set.sample(n=n, weights=test_set[FREQUENCY])
    logger.info("Test set with %d unique questions" % len(test_set))
    test_set = test_set[[FREQUENCY, QUESTION]].sort_values([FREQUENCY, QUESTION], ascending=[False, True])
    return test_set.set_index(QUESTION)


def wea_test(test_set, wea_logs):
    """
    Get answers returned by WEA to questions by looking them up in the usage logs.

    :param test_set: Question and Frequency information
    :type test_set: pandas.DataFrame
    :param wea_logs: user interaction logs from QuestionsData.csv XMGR report
    :type wea_logs: pandas.DataFrame
    :return: Question, Answer, and Confidence
    :rtype: pandas.DataFrame
    """
    wea_logs = fix_confidence_ranges(wea_logs)
    wea_logs = wea_logs.drop_duplicates(QUESTION)
    answers = pandas.merge(test_set, wea_logs, on=QUESTION)
    missing_answers = answers[answers[ANSWER].isnull()]
    if len(missing_answers):
        logger.warning("%d questions without answers" % len(missing_answers))
    logger.info("Answered %d questions" % len(answers))
    return answers[[QUESTION, ANSWER, CONFIDENCE]].sort_values(QUESTION).set_index(QUESTION)


def fix_confidence_ranges(wea_logs):
    """
    Scale all confidence values between 0 and 1.

    The top answer confidence value in the WEA logs ranges either from 0-1 or 0-100 depending on the value in the user
    experience column.

    :param wea_logs: user interaction logs from QuestionsData.csv XMGR report
    :type wea_logs: pandas.DataFrame
    :return: logs with all confidence values scaled between 0 and 1
    :rtype: pandas.DataFrame
    """
    # groupby drops null values, so rewrite these as "NA".
    wea_logs.loc[wea_logs[USER_EXPERIENCE].isnull(), USER_EXPERIENCE] = "NA"
    m = wea_logs.groupby(USER_EXPERIENCE)[CONFIDENCE].max()
    m[m > 1] = 100
    for user_experience in m.index:
        index = wea_logs[USER_EXPERIENCE] == user_experience
        wea_logs.loc[index, CONFIDENCE] = wea_logs[index][CONFIDENCE].div(m[user_experience])
    return wea_logs


def augment_system_logs(wea_logs, annotation_assist):
    """
    Add In Purview and Annotation Score information to system usage logs

    :param wea_logs: user interaction logs from QuestionsData.csv XMGR report
    :type wea_logs: pandas.DataFrame
    :param annotation_assist: Annotation Assist judgments
    :type annotation_assist: pandas.DataFrame
    :return: user interaction logs with additional columns
    :rtype: pandas.DataFrame
    """
    wea_logs[ANSWER] = wea_logs[ANSWER].str.replace("\n", "")
    augmented = pandas.merge(wea_logs, annotation_assist, on=(QUESTION, ANSWER), how="left")
    n = len(wea_logs[[QUESTION, ANSWER]].drop_duplicates())
    m = len(annotation_assist)
    logger.info("%d unique question/answer pairs, %d judgments (%0.3f)" % (n, m, m / float(n)))
    return augmented.rename(columns={QUESTION: QUESTION_TEXT, ANSWER: TOP_ANSWER_TEXT})


def filter_corpus(corpus, max_size):
    """
    Remove corpus entries above a specified size

    :param corpus: corpus with Answer Id and Answer columns
    :type corpus: pandas.DataFrame
    :param max_size: maximum allowed Answer size in characters
    :type max_size: int
    :return: corpus with oversize answers removed
    :rtype: pandas.DataFrame
    """
    if max_size is not None:
        filtered = corpus[corpus[ANSWER].str.len() <= max_size]
        logger.info("Filtered %d answers over size %d" % (len(corpus) - len(filtered), max_size))
        corpus = filtered
    return corpus.set_index(ANSWER_ID)


class WeaLogFileType(CsvFileType):
    """
    Read the QuestionsData.csv file in the XMGR usage report logs.
    """

    WEA_DATE_FORMAT = re.compile(
        r"(?P<month>\d\d)(?P<day>\d\d)(?P<year>\d\d\d\d):(?P<hour>\d\d)(?P<min>\d\d)(?P<sec>\d\d):UTC")

    def __init__(self):
        super(self.__class__, self).__init__(
            [DATE_TIME, QUESTION_TEXT, TOP_ANSWER_TEXT, TOP_ANSWER_CONFIDENCE, USER_EXPERIENCE],
            {QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER,
             TOP_ANSWER_CONFIDENCE: CONFIDENCE})

    def __call__(self, filename):
        wea_logs = super(self.__class__, self).__call__(filename)
        wea_logs[DATE_TIME] = pandas.to_datetime(wea_logs[DATE_TIME].apply(self.standard_date_format))
        return wea_logs

    @staticmethod
    def standard_date_format(s):
        """
        Convert from WEA's idiosyncratic string date format to the ISO standard.

        :param s: WEA date
        :type s: str
        :return: standard date
        :rtype: str
        """
        m = WeaLogFileType.WEA_DATE_FORMAT.match(s).groupdict()
        return "%s-%s-%sT%s:%s:%sZ" % (m['year'], m['month'], m['day'], m['hour'], m['min'], m['sec'])
