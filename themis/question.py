import re

import pandas

from themis import ANSWER, CONFIDENCE, FREQUENCY, QUESTION, CsvFileType, logger

# Column headers in usage log
QUESTION_TEXT = "QuestionText"
TOP_ANSWER_TEXT = "TopAnswerText"
USER_EXPERIENCE = "UserExperience"
TOP_ANSWER_CONFIDENCE = "TopAnswerConfidence"
DATE_TIME = "DateTime"


def extract_question_answer_pairs_from_usage_logs(usage_log):
    """
    Extract questions and answers from usage logs, adding question frequency information.

    We are assuming here that a given question always elicits the same answer. Print a warning if this is not the case
    and drop answers to make the answers unique. It is arbitrary which answer is dropped.

    :param usage_log: QuestionsData.csv usage log
    :type usage_log: pandas.DatFrame
    :return: Q&A pairs with question frequency information
    :rtype: pandas.DatFrame
    """
    frequency = question_frequency(usage_log)
    qa_pairs = usage_log.drop_duplicates(subset=(QUESTION, ANSWER))
    m = sum(qa_pairs.duplicated(QUESTION))
    if m:
        n = len(frequency)
        logger.warning("%d questions of %d have multiple answers (%0.3f%%), only keeping one answer per question" %
                       (m, n, 100.0 * m / n))
        qa_pairs = qa_pairs.drop_duplicates(QUESTION)
    qa_pairs = pandas.merge(qa_pairs, frequency, on=QUESTION)
    logger.info("%d question/answer pairs" % len(qa_pairs))
    return qa_pairs


def question_frequency(usage_log):
    """
    Count the number of times each question appears in the usage log.

    :param usage_log: QuestionsData.csv report log
    :type usage_log: pandas.DataFrame
    :return: table of question and frequency
    :rtype: pandas.DataFrame
    """
    questions = pandas.merge(usage_log.drop_duplicates(QUESTION),
                             usage_log.groupby(QUESTION).size().to_frame(FREQUENCY).reset_index())
    questions = questions[[FREQUENCY, QUESTION]].sort_values([FREQUENCY, QUESTION], ascending=[False, True])
    return questions


class QuestionFrequencyFileType(CsvFileType):
    columns = [QUESTION, FREQUENCY]

    def __init__(self):
        super(self.__class__, self).__init__(QuestionFrequencyFileType.columns)

    @staticmethod
    def output_format(question_frequency):
        question_frequency = question_frequency[QuestionFrequencyFileType.columns]
        question_frequency = question_frequency.sort_values(FREQUENCY, ascending=False)
        return question_frequency.set_index(QUESTION)


class UsageLogFileType(CsvFileType):
    """
    Read the QuestionsData.csv file in the usage log.
    """

    WEA_DATE_FORMAT = re.compile(
        r"(?P<month>\d\d)(?P<day>\d\d)(?P<year>\d\d\d\d):(?P<hour>\d\d)(?P<min>\d\d)(?P<sec>\d\d):UTC")

    canonical_cols = [QUESTION_TEXT, TOP_ANSWER_TEXT, TOP_ANSWER_CONFIDENCE]
    full_cols = [DATE_TIME, QUESTION_TEXT, TOP_ANSWER_TEXT, TOP_ANSWER_CONFIDENCE, USER_EXPERIENCE]

    def __init__(self, columns=full_cols):
        # super(self.__class__, self).__init__(
        #     [DATE_TIME, QUESTION_TEXT, TOP_ANSWER_TEXT, TOP_ANSWER_CONFIDENCE, USER_EXPERIENCE],
        #     {QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER, TOP_ANSWER_CONFIDENCE: CONFIDENCE})

        super(self.__class__, self).__init__(
            columns,
            {QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER, TOP_ANSWER_CONFIDENCE: CONFIDENCE})

    def __call__(self, filename):
        try:
            usage_log = super(self.__class__, self).__call__(filename)
            usage_log[DATE_TIME] = pandas.to_datetime(usage_log[DATE_TIME].apply(self.standard_date_format))
        except ValueError:
            self.__init__(UsageLogFileType.canonical_cols)
            usage_log = super(self.__class__, self).__call__(filename)
        return usage_log

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


class QAPairFileType(CsvFileType):
    columns = [QUESTION, ANSWER, CONFIDENCE, USER_EXPERIENCE, FREQUENCY, DATE_TIME]

    def __init__(self):
        super(self.__class__, self).__init__(QAPairFileType.columns)

    @staticmethod
    def output_format(qa_pairs):
        columns = list(set(qa_pairs.columns).intersection(QAPairFileType.columns))
        qa_pairs = qa_pairs[columns]
        qa_pairs = qa_pairs.sort_values([FREQUENCY, CONFIDENCE, QUESTION], ascending=(False, False, True))
        return qa_pairs.set_index([QUESTION, ANSWER])
