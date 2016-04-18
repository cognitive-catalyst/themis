import re

import pandas

from themis import QUESTION, CONFIDENCE, ANSWER, FREQUENCY
from themis import logger, CsvFileType

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


def sample_questions(qa_pairs, sample_size):
    """
    Sample questions by frequency.

    :param qa_pairs: question/answer pairs with question frequencies
    :type qa_pairs: pandas.DataFrame
    :param sample_size: number of questions to sample
    :type sample_size: int
    :return: sample of unique questions and their frequencies
    :rtype: pandas.DataFrame
    """
    qa_pairs = qa_pairs[[QUESTION, FREQUENCY]].drop_duplicates(QUESTION)
    sample = qa_pairs.sample(sample_size, weights=FREQUENCY)
    sample = sample.sort_values([FREQUENCY, QUESTION], ascending=[False, True])
    return sample.set_index(QUESTION)


def get_answers_from_usage_log(questions, qa_pairs_from_logs):
    """
    Get answers returned by WEA to questions by looking them up in the usage log.

    Each question in the Q&A pairs must have a unique answer.

    :param questions: questions to look up in the usage logs
    :type questions: pandas.DataFrame
    :param qa_pairs_from_logs: question/answer pairs extracted from user logs
    :type qa_pairs_from_logs: pandas.DataFrame
    :return: Question, Answer, and Confidence
    :rtype: pandas.DataFrame
    """
    answers = pandas.merge(questions, qa_pairs_from_logs, on=QUESTION, how="left")
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
            {QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER, TOP_ANSWER_CONFIDENCE: CONFIDENCE})

    def __call__(self, filename):
        usage_log = super(self.__class__, self).__call__(filename)
        usage_log[DATE_TIME] = pandas.to_datetime(usage_log[DATE_TIME].apply(self.standard_date_format))
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
        qa_pairs = qa_pairs[QAPairFileType.columns]
        qa_pairs = qa_pairs.sort_values([FREQUENCY, CONFIDENCE, QUESTION], ascending=(False, False, True))
        return qa_pairs.set_index([QUESTION, ANSWER])
