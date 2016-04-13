import re

import pandas

from themis import QUESTION, CONFIDENCE, ANSWER
from themis import logger, CsvFileType

# Column headers in WEA logs
QUESTION_TEXT = "QuestionText"
TOP_ANSWER_TEXT = "TopAnswerText"
USER_EXPERIENCE = "UserExperience"
TOP_ANSWER_CONFIDENCE = "TopAnswerConfidence"
DATE_TIME = "DateTime"


def get_answers_from_usage_log(questions, usage_log):
    """
    Get answers returned by WEA to questions by looking them up in the usage log.

    :param questions: questions to look up in the usage logs
    :type questions: pandas.DataFrame
    :param usage_log: user interaction logs from QuestionsData.csv XMGR report
    :type usage_log: pandas.DataFrame
    :return: Question, Answer, and Confidence
    :rtype: pandas.DataFrame
    """
    usage_log = usage_log.drop_duplicates(QUESTION)
    answers = pandas.merge(questions, usage_log, on=QUESTION)
    missing_answers = answers[answers[ANSWER].isnull()]
    if len(missing_answers):
        logger.warning("%d questions without answers" % len(missing_answers))
    logger.info("Answered %d questions" % len(answers))
    answers = answers[[QUESTION, ANSWER, CONFIDENCE]].sort_values([QUESTION, CONFIDENCE], ascending=[True, False])
    return answers.set_index(QUESTION)


def augment_usage_log(usage_log, annotation_assist):
    """
    Add In Purview and Annotation Score information to system usage log.

    :param usage_log: user interaction logs from QuestionsData.csv XMGR report
    :type usage_log: pandas.DataFrame
    :param annotation_assist: Annotation Assist judgments
    :type annotation_assist: pandas.DataFrame
    :return: user interaction logs with additional columns
    :rtype: pandas.DataFrame
    """
    usage_log[ANSWER] = usage_log[ANSWER].str.replace("\n", "")
    augmented = pandas.merge(usage_log, annotation_assist, on=(QUESTION, ANSWER), how="left")
    n = len(usage_log[[QUESTION, ANSWER]].drop_duplicates())
    if n:
        m = len(annotation_assist)
        logger.info("%d unique question/answer pairs, %d judgments (%0.3f%%)" % (n, m, 100.0 * m / n))
    return augmented.rename(columns={QUESTION: QUESTION_TEXT, ANSWER: TOP_ANSWER_TEXT})


class UsageLogFileType(CsvFileType):
    """
    Read the QuestionsData.csv file in the usage log.
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
        m = UsageLogFileType.WEA_DATE_FORMAT.match(s).groupdict()
        return "%s-%s-%sT%s:%s:%sZ" % (m['year'], m['month'], m['day'], m['hour'], m['min'], m['sec'])
