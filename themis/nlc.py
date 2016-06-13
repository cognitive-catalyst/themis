import tempfile

from watson_developer_cloud import NaturalLanguageClassifierV1 as NaturalLanguageClassifier

from themis import QUESTION, ANSWER_ID, ANSWER
from themis import logger, to_csv, pretty_print_json


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
    logger.info("Train model %s with %d instances" % (name, len(truth)))
    with tempfile.TemporaryFile() as training_file:
        # NLC cannot handle newlines.
        truth[QUESTION] = truth[QUESTION].str.replace("\n", " ")
        to_csv(training_file, truth[[QUESTION, ANSWER_ID]], header=False, index=False)
        training_file.seek(0)
        nlc = NaturalLanguageClassifier(url=url, username=username, password=password)
        r = nlc.create(training_data=training_file, name=name)
        logger.info(pretty_print_json(r))
    return r["classifier_id"]

#Training for k-fold train file
def nlc_train(url, username, password, train):
    logger.info("Train model %s with %d instances" % (len(train)))
    with tempfile.TemporaryFile() as training_file:
        train[0] = train[0].str.replace("\n", " ")
        #train[3] = train[3].str.replace("\n", " ")
        to_csv(training_file, train[[0, 3]], header=False, index=False)
        training_file.seek(0)
        nlc = NaturalLanguageClassifier(url=url, username=username, password=password)
        r = nlc.create(training_data=training_file)
        logger.info(pretty_print_json(r))
    return r["classifier_id"]

class NLC(object):
    """
    Wrapper to a Natural Language Classifier via the
    `Watson developer cloud Python SDK <https://github.com/watson-developer-cloud/python-sdk>`.
    """

    def __init__(self, url, username, password, classifier_id, corpus):
        self.nlc = NaturalLanguageClassifier(url=url, username=username, password=password)
        self.classifier_id = classifier_id
        self.corpus = corpus

    def __repr__(self):
        return "NLC: %s" % self.classifier_id

    def ask(self, question):
        classification = self.nlc.classify(self.classifier_id, question)
        class_name = classification["classes"][0]["class_name"]
        confidence = classification["classes"][0]["confidence"]
        return self.corpus.loc[class_name][ANSWER], confidence
