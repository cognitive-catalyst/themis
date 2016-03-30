import json
import tempfile

from watson_developer_cloud import NaturalLanguageClassifierV1 as NaturalLanguageClassifier

from themis import logger, to_csv, QUESTION, ANSWER_ID


def classifier_list(url, username, password):
    connection = NaturalLanguageClassifier(url=url, username=username, password=password)
    return connection.list()["classifiers"]


def classifier_status(url, username, password, classifier_ids):
    n = NaturalLanguageClassifier(url=url, username=username, password=password)
    for classifier_id in classifier_ids:
        status = n.status(classifier_id)
        print("%s: %s" % (status["status"], status["status_description"]))


def remove_classifiers(url, username, password, classifier_ids):
    n = NaturalLanguageClassifier(url=url, username=username, password=password)
    for classifier_id in classifier_ids:
        n.remove(classifier_id)


def train_nlc(url, username, password, truth, name):
    logger.info("Train  model %s with %d instances" % (name, len(truth)))
    with tempfile.TemporaryFile() as training_file:
        to_csv(training_file, truth[[QUESTION, ANSWER_ID]], header=False, index=False)
        training_file.seek(0)
        nlc = NaturalLanguageClassifier(url=url, username=username, password=password)
        r = nlc.create(training_data=training_file, name=name)
        logger.info((json.dumps(r, indent=2)))
    return r["classifier_id"]


class NLC(object):
    """
    Wrapper to a Natural Language Classifier via the
    `Watson developer cloud Python SDK <https://github.com/watson-developer-cloud/python-sdk>`.
    """

    def __init__(self, url, username, password, classifier_id):
        self.nlc = NaturalLanguageClassifier(url=url, username=username, password=password)
        self.classifier_id = classifier_id

    def __repr__(self):
        return "NLC: %s" % self.classifier_id

    def ask(self, question):
        classification = self.nlc.classify(self.classifier_id, question)
        return classification["classes"][0]["class_name"], classification["classes"][0]["confidence"]
