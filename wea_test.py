#!/usr/bin/env python

"""Logged WEA answers and confidences"""
import argparse
import logging

import pandas

logger = logging.getLogger(__name__)

QUESTION = "Question"
QUESTION_TEXT = "QuestionText"
ANSWER_ID = "AnswerId"
TOP_ANSWER_CONFIDENCE = "TopAnswerConfidence"
CONFIDENCE = "Confidence"


def wea_test(questions_data_file):
    test_set = pandas.read_csv(questions_data_file, usecols=[QUESTION_TEXT, ANSWER_ID, TOP_ANSWER_CONFIDENCE],
                               encoding="utf-8")
    logger.info("%d questions" % len(test_set))
    test_set.rename(columns={QUESTION_TEXT: QUESTION, TOP_ANSWER_CONFIDENCE: CONFIDENCE}, inplace=True)
    test_set = test_set[test_set[ANSWER_ID].notnull()]
    # Drop duplicate questions.
    test_set.drop_duplicates(QUESTION, inplace=True)
    test_set.sort(QUESTION, inplace=True)
    logger.info("%d unique questions" % len(test_set))
    return test_set[[QUESTION, ANSWER_ID, CONFIDENCE]]


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wea_logs", metavar="wea", type=argparse.FileType(), help="WEA logs with answer ID column")
    parser.add_argument('--log', type=str, default="ERROR", help="logging level")
    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(message)s")
    wea = wea_test(args.wea_logs)
    print(wea.to_csv(encoding="utf-8", index=False))
