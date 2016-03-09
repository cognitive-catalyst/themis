#!/usr/bin/env python

"""Answer a set of questions using one of the Q&A systems.
This takes a CSV file with a Questions column as input and writes a CSV with AnswerId and Confidence columns added.
For the Solr and NLC classifiers, intermediary results are written to the output file at a specified interval.
If the job fails you can restart it with that output file and it will pick up where it left off.
"""
import argparse
import logging
import re

import pandas
import solr
from watson_developer_cloud import NaturalLanguageClassifierV1 as NaturalLanguageClassifier

logger = logging.getLogger(__name__)

QUESTION = "Question"
QUESTION_TEXT = "QuestionText"
ANSWER_ID = "AnswerId"
TOP_ANSWER_CONFIDENCE = "TopAnswerConfidence"
CONFIDENCE = "Confidence"


def wea_answers(questions, output_filename, wea_logs_file):
    logger.info("Get answers to %d questions from WEA logs %s" % (len(questions), wea_logs_file.name))
    answers = pandas.read_csv(wea_logs_file, usecols=[QUESTION_TEXT, ANSWER_ID, TOP_ANSWER_CONFIDENCE],
                              encoding="utf-8").drop_duplicates(QUESTION_TEXT)
    answers = answers[answers[QUESTION_TEXT].isin(questions)]
    # TODO What if a question is not in the WEA log file?
    answers.rename(columns={QUESTION_TEXT: QUESTION, TOP_ANSWER_CONFIDENCE: CONFIDENCE}, inplace=True)
    answers.sort([QUESTION], inplace=True)
    answers = answers[[QUESTION, ANSWER_ID, CONFIDENCE]]
    answers.to_csv(output_filename, index=False, encoding="utf-8")


def answer_questions(system, questions, output_filename, interval):
    logger.info("Get answers to %d questions from %s" % (len(questions), system))
    answers = DataFrameCheckpoint(output_filename, [QUESTION, ANSWER_ID, CONFIDENCE], interval)
    try:
        questions = sorted(set(questions) - answers.recovered)
        n = len(answers.recovered) + len(questions)
        for i, question in enumerate(questions, len(answers.recovered) + 1):
            if i is 1 or i % interval is 0:
                logger.info("Question %d of %d" % (i, n))
            answer_id, confidence = system.ask(question)
            logger.debug("%s\t%s\t%s" % (question, answer_id, confidence))
            answers.write(question, answer_id, confidence)
    finally:
        answers.close()


class NLC(object):
    def __init__(self, url, username, password, classifier_id):
        self.nlc = NaturalLanguageClassifier(url=url, username=username, password=password)
        self.classifier_id = classifier_id

    def __repr__(self):
        return "NLC: %s" % self.classifier_id

    def ask(self, question):
        classification = self.nlc.classify(self.classifier_id, question)
        return classification["classes"][0]["class_name"], classification["classes"][0]["confidence"]


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
            answer_id = r[0]["id"]
            confidence = r[0]["score"]
        else:
            answer_id = None
            confidence = None
        return answer_id, confidence

    def escape_solr_query(self, s):
        s = s.replace("/", "\\/")
        return re.sub(self.SOLR_CHARS, lambda m: "\%s" % m.group(1), s)


class DataFrameCheckpoint(object):
    def __init__(self, output_filename, columns, interval=None):
        try:
            recovered = pandas.read_csv(open(output_filename), usecols=[0], encoding="utf-8")
            self.recovered = set(recovered[recovered.columns[0]])
            self.need_header = False
            logger.debug("Recovered %d items from disk" % len(self.recovered))
        except IOError:
            self.recovered = set()
            self.need_header = True
        except ValueError:
            raise Exception("Cannot recover data from %s" % output_filename)
        self.output_file = open(output_filename, "a")
        self.columns = columns
        self.buffer = pandas.DataFrame(columns=self.columns)
        self.interval = interval

    def write(self, *values):
        self.buffer = self.buffer.append(dict(zip(self.buffer.columns, values)), ignore_index=True)
        if self.interval is not None and len(self.buffer) % self.interval is 0:
            self.flush()

    def close(self):
        self.flush()
        self.output_file.close()

    def flush(self):
        logger.debug("Flush %d items to disk" % len(self.buffer))
        self.buffer.to_csv(self.output_file, header=self.need_header, index=False, encoding="utf-8")
        self.output_file.flush()
        self.buffer = pandas.DataFrame(columns=self.columns)
        self.need_header = False


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--max-questions", metavar="N", type=int, help="only ask the first N questions")
    parser.add_argument("--log", type=str, default="ERROR", help="logging level")

    filenames_parser = argparse.ArgumentParser(add_help=False)
    filenames_parser.add_argument("questions_file", metavar="questions", type=argparse.FileType(),
                                  help="questions file")
    filenames_parser.add_argument("output_filename", metavar="output", type=str, help="output file")
    interval_parser = argparse.ArgumentParser(add_help=False)
    interval_parser.add_argument("--interval", type=int, default=100,
                                 help="checkpointing interval, default 100 questions")

    subparsers = parser.add_subparsers(dest="system", help="Q&A system to use")

    wea_parser = subparsers.add_parser("wea", help="Answer questions with WEA logs", parents=[filenames_parser])
    wea_parser.add_argument("wea_logs_file", metavar="wea", type=argparse.FileType(),
                            help="WEA logs with answer id column")

    nlc_parser = subparsers.add_parser("nlc", help="Answer questions with NLC",
                                       parents=[filenames_parser, interval_parser])
    nlc_parser.add_argument("username", type=str, help="NLC username")
    nlc_parser.add_argument("password", type=str, help="NLC password")
    nlc_parser.add_argument("classifier_id", type=str, help="NLC classifier id")
    nlc_parser.add_argument("--url", type=str,
                            default="https://gateway-s.watsonplatform.net/natural-language-classifier/api",
                            help="NLC url")

    solr_parser = subparsers.add_parser("solr", help="Answer questions with Solr",
                                        parents=[filenames_parser, interval_parser])
    solr_parser.add_argument("url", type=str, help="solr URL")
    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(message)s")

    questions = pandas.read_csv(args.questions_file, encoding="utf-8", nrows=args.max_questions)[QUESTION]
    if args.system == "wea":
        wea_answers(questions, args.output_filename, args.wea_logs_file)
    else:
        if args.system == "solr":
            system = Solr(args.url)
        else:
            system = NLC(args.url, args.username, args.password, args.classifier_id)
        answer_questions(system, questions, args.output_filename, args.interval)
