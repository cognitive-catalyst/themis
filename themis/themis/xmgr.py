"""Utilities to download information from an Watson Experience Manager (XMGR) project"""
import json
import os

import pandas
import requests

from themis import logger, to_csv, QUESTION, ANSWER_ID


def download_from_xmgr(url, username, password, output_directory):
    """
    Download truth and corpus from an XMGR project

    This creates the following files in the output directory:

    * truth.csv ... Mapping of questions to answer ids
    * truth.json ... all truth mappings retrieved from xmgr

    The output directory is created if it does not exist.

    :param url: project URL
    :param username: username
    :param password: password
    :param output_directory: directory in which to write XMGR files
    """
    try:
        os.makedirs(output_directory)
    except OSError:
        pass
    xmgr = XmgrProject(url, username, password)
    truth_csv = os.path.join(output_directory, "truth.csv")
    if not os.path.isfile(truth_csv):
        logger.info("Get truth from %s" % xmgr)
        all_questions, truth = download_truth(xmgr)
        with open(os.path.join(output_directory, "truth.json"), "w") as f:
            json.dump(all_questions, f, indent=2)
        to_csv(truth_csv, truth)
    logger.info("Get corpus from %s" % xmgr)


def download_truth(xmgr):
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
            off_topic += 1
        else:
            unmapped += 1
    ground_truth = pandas.DataFrame.from_dict(
        {QUESTION: answers.keys(), ANSWER_ID: answers.values()}).set_index(QUESTION)
    logger.info("%d mapped, %d unmapped, %d off-topic" % (len(ground_truth), unmapped, off_topic))
    return all_questions, ground_truth


class XmgrProject(object):
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
        logger.debug("GET %s, Status %d" % (url, r.status_code))
        return r.json()
