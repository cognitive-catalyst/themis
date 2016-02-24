#!/usr/bin/env python

"""Classify a set of questions using an NLC model"""

import argparse
import logging
import os

import pandas
import requests

logger = logging.getLogger(__name__)

ANSWER_ID = "AnswerId"
CONFIDENCE = "Confidence"
QUESTION = "Question"


def answer_questions(nlc, questions_file, output_filename, max_questions, interval):
    output = DataFrameCheckpoint(output_filename, [QUESTION, ANSWER_ID, CONFIDENCE], interval)
    try:
        questions = sorted(
            set(pandas.read_csv(questions_file, nrows=max_questions, encoding="utf-8")[QUESTION]) - output.recovered)
        n = len(output.recovered) + len(questions)
        for i, question in enumerate(questions, len(output.recovered) + 1):
            if i is 1 or i % interval is 0:
                logger.info("Question %d of %d" % (i, n))
            answer = nlc.classify(question)
            answer_id = answer["classes"][0]["class_name"]
            confidence = answer["classes"][0]["confidence"]
            logger.debug("%s\t%s\t%f" % (question, answer_id, confidence))
            output.write(question, answer_id, confidence)
    finally:
        output.close()


class NLC(object):
    def __init__(self, url, username, password, classifier):
        self.url = url
        self.username = username
        self.password = password
        self.classifier = classifier

    def __repr__(self):
        return "NLC: %s" % self.url

    def classify(self, text):
        url = os.path.join(self.url, self.classifier, "classify")
        r = requests.get(url, auth=(self.username, self.password), params={"text": text})
        logger.debug("GET %s\t%d" % (r.url, r.status_code))
        return r.json()


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
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="""This takes a CSV file with a Questions column as input and writes a CSV
                                     with AnswerId and Confidence columns added. Intermediate results are periodically
                                     saved and if the script fails in the middle it will pick up from where it left
                                     off.""")
    parser.add_argument("questions_file", metavar="questions", type=argparse.FileType(), help="questions file")
    parser.add_argument("output_filename", metavar="output", type=str, help="output file")
    parser.add_argument("username", type=str, help="NLC username")
    parser.add_argument("password", type=str, help="NLC password")
    parser.add_argument("classifier", type=str, help="classifier")
    parser.add_argument("--max-questions", metavar="N", type=int, help="only ask the first N questions")
    parser.add_argument("--interval", type=int, default=100, help="checkpointing interval, default 100 questions")
    parser.add_argument("--url", type=str,
                        default="https://gateway-s.watsonplatform.net/natural-language-classifier/api/v1/classifiers",
                        help="NLC url")
    parser.add_argument('--log', type=str, default="ERROR", help="logging level")
    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(message)s")
    nlc = NLC(args.url, args.username, args.password, args.classifier)
    answer_questions(nlc, args.questions_file, args.output_filename, args.max_questions, args.interval)
