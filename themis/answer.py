import re

import pandas
# noinspection PyPackageRequirements
import solr
from themis import logger, DataFrameCheckpoint, percent_complete_message
from themis import QUESTION, ANSWER, CONFIDENCE


def answer_questions(system, questions, output_filename, checkpoint_frequency):
    """
    Use a Q&A system to provide answers to a test set of questions

    :param system: Q&A system
    :type system: object that exports an ask method
    :param questions: questions to ask
    :type questions: set
    :param output_filename: name of file to which write questions, answers, and confidences
    :type output_filename: str
    :param checkpoint_frequency: how often to write intermediary results to the output file
    :type checkpoint_frequency: int
    """
    logger.info("Get answers to %d questions from %s" % (len(questions), system))
    answers = DataFrameCheckpoint(output_filename, [QUESTION, ANSWER, CONFIDENCE], checkpoint_frequency)
    try:
        if answers.recovered:
            logger.info("Recovered %d answers from %s" % (len(answers.recovered), output_filename))
        questions = sorted(questions - answers.recovered)
        n = len(answers.recovered) + len(questions)
        for i, question in enumerate(questions, len(answers.recovered) + 1):
            if i is 1 or i == n or i % checkpoint_frequency is 0:
                logger.info(percent_complete_message("Question", i, n))
            answer, confidence = system.ask(question)
            logger.debug("%s\t%s\t%s" % (question, answer, confidence))
            answers.write(question, answer, confidence)
    finally:
        answers.close()


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


class Solr(object):
    # TODO Missing the full reserved set: + - && || ! ( ) { } [ ] ^ " ~ * ? : \
    SOLR_CHARS = re.compile(r"""([\+\-!\[\](){}^"~*?:\\])""")

    def __init__(self, url):
        self.url = url
        self.connection = solr.SolrConnection(self.url)

    def __repr__(self):
        return "Solr: %s" % self.url

    def ask(self, question):
        question = self.escape_solr_query(question)
        logger.debug(question)
        r = self.connection.query(question).results
        n = len(r)
        logger.debug("%d results" % n)
        if n:
            answer = r[0][ANSWER][0]
            confidence = r[0]["score"]
        else:
            answer = None
            confidence = None
        return answer, confidence

    def escape_solr_query(self, s):
        s = s.replace("/", "\\/")
        return re.sub(self.SOLR_CHARS, lambda m: "\%s" % m.group(1), s)
