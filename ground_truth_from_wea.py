#!/usr/bin/env python

"""Download ground truth from a WEA instance.
This prints a CSV with Question and Answer Id columns.
Optionally it will write a JSON file detailing all the question information in the XMGR instance."""

import argparse
import json
import logging
import os

import pandas
import requests

logger = logging.getLogger(__name__)


def get_ground_truth_from_xmgr(xmgr):
    # Get all the questions that are not in a REJECTED state.
    all_questions = [question for question in xmgr.get_questions() if not question["state"] == "REJECTED"]
    # Indexed the questions by their question id so that mapped questions can be looked up.
    questions = dict([(question["id"], question) for question in all_questions])
    # Read ground truth from questions.
    answers = {}
    off_topic = 0
    unmapped = 0
    # Answered questions are either mapped to a PAU mapped to another question that is mapped to a PAU.
    for question in questions.values():
        if "predefinedAnswerUnit" in question:
            answers[question["text"]] = question["predefinedAnswerUnit"]
        elif "mappedQuestion" in question:
            answers[question["text"]] = questions[question["mappedQuestion"]["id"]]
        elif question["offTopic"]:
            logger.info("Off-topic: %s" % question["text"])
            off_topic += 1
        else:
            logger.warning("Unmapped: %s" % question["text"])
            unmapped += 1
    ground_truth = pandas.DataFrame.from_dict(
        {"Question": answers.keys(), "AnswerId": answers.values()}).set_index("Question")
    logger.info("%d mapped, %d unmapped, %d off-topic" % (len(ground_truth), unmapped, off_topic))
    return all_questions, ground_truth


class Xmgr(object):
    def __init__(self, project_url, username, password):
        self.project_url = project_url
        self.username = username
        self.password = password

    def __repr__(self):
        return "XMGR: %s" % self.project_url

    def get_questions(self, pagesize=500):
        questions = []
        total = None
        while total is None or len(questions) < total:
            response = self.get('workbench/api/questions', params={"offset": len(questions), "pagesize": pagesize})
            if total is None:
                total = response["total"]
            questions.extend(response["items"])
        logger.debug("%d questions" % len(questions))
        return questions

    def get(self, path, params=None, headers=None):
        url = os.path.join(self.project_url + "/", path)
        r = requests.get(url, auth=(self.username, self.password), params=params, headers=headers)
        logger.debug("GET %s\t%d" % (url, r.status_code))
        return r.json()


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xmgr", type=str, help="XMGR url")
    parser.add_argument("username", type=str, help="XMGR username")
    parser.add_argument("password", type=str, help="XMGR password")
    parser.add_argument('--json', type=argparse.FileType("w"), help="Write ground truth details to JSON file")
    parser.add_argument('--log', type=str, default="ERROR", help="logging level")
    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(message)s")
    xmgr = Xmgr(args.xmgr, args.username, args.password)
    all_questions, ground_truth = get_ground_truth_from_xmgr(xmgr)
    if args.json:
        json.dump(all_questions, args.json, indent=2)
    print(ground_truth.to_csv(encoding="utf-8"))
