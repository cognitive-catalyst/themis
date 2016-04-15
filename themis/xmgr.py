"""Utilities to download information from an Watson Experience Manager (XMGR) project"""
import json
import os

import pandas
import requests

from themis import QUESTION, ANSWER_ID, ANSWER, TITLE, FILENAME, QUESTION_ID
from themis import logger, to_csv, DataFrameCheckpoint, ensure_directory_exists, percent_complete_message, \
    CsvFileType


def download_truth_from_xmgr(xmgr, output_directory):
    """
    Download truth from an XMGR project.

    Truth is a mapping of sets of questions to answer documents. Truth is used to train the WEA model and may be used
    to train an NLC model.

    This function creates two files in the output directory: a raw truth.json that contains all the information
    downloaded from XMGR and a filtered truth.csv file.

    :param xmgr: connection to an XMGR project REST API
    :type xmgr: XmgrProject
    :param output_directory: directory in which to create truth.json and truth.csv
    :type output_directory: str
    """
    ensure_directory_exists(output_directory)
    truth_json = os.path.join(output_directory, "truth.json")
    truth_csv = os.path.join(output_directory, "truth.csv")
    if os.path.isfile(truth_json) and os.path.isfile(truth_csv):
        logger.info("Truth already downloaded")
        return
    if not os.path.isfile(truth_json):
        logger.info("Get questions from %s" % xmgr)
        mapped_questions = [question for question in xmgr.get_questions() if not question["state"] == "REJECTED"]
        with open(truth_json, "w") as f:
            json.dump(mapped_questions, f, indent=2)
    else:
        with open(truth_json) as f:
            mapped_questions = json.load(f)
    logger.info("Build truth from questions")
    truth = get_truth_from_mapped_questions(mapped_questions)
    to_csv(truth_csv, TruthFileType.output_format(truth))


def get_truth_from_mapped_questions(mapped_questions):
    def get_pau_mapping(question):
        if "predefinedAnswerUnit" in question:
            return question["predefinedAnswerUnit"]
        elif "mappedQuestion" in question:
            question_id = question["mappedQuestion"]["id"]
            try:
                mapped_question = questions[question_id]
            except KeyError:
                logger.warning("Question %s mapped to non-existent question %s" % (question["id"], question_id))
                return None
            return get_pau_mapping(mapped_question)
        else:
            return None

    unmapped = 0
    # Index the questions by their question id so that mapped questions can be looked up.
    questions = dict([(question["id"], question) for question in mapped_questions])
    for question in questions.values():
        question[ANSWER_ID] = get_pau_mapping(question)
        if question[ANSWER_ID] is None:
            unmapped += 1
    questions = [q for q in questions.values() if q[ANSWER_ID] is not None]
    question_ids = [q["id"] for q in questions]
    question_text = [q["text"] for q in questions]
    answer_id = [q[ANSWER_ID] for q in questions]
    truth = pandas.DataFrame.from_dict({QUESTION_ID: question_ids, QUESTION: question_text, ANSWER_ID: answer_id})
    logger.info("%d mapped, %d unmapped" % (len(truth), unmapped))
    return truth


def download_corpus_from_xmgr(xmgr, output_directory, checkpoint_frequency, max_docs):
    """
    Download the corpus from an XMGR project

    A corpus is a mapping of answer text to answer Ids. It also contains answer titles and the names of the documents
    from which the answers were extracted.

    This can take a long time to complete, so intermediate results are saved in the directory. If you restart an
    incomplete download it will pick up where it left off.

    :param xmgr: connection to an XMGR project REST API
    :type xmgr: XmgrProject
    :param output_directory: directory into which write the corpus.csv file
    :type output_directory: str
    :checkpoint_frequency: how often to write intermediate results to a checkpoint file
    :type checkpoint_frequency: int
    :param max_docs: maximum number of corpus documents to download, if None, download them all
    :type max_docs: int
    """
    pau_ids_csv = os.path.join(output_directory, "pau_ids.csv")
    corpus_csv = os.path.join(output_directory, "corpus.csv")
    if not os.path.isfile(pau_ids_csv) and os.path.isfile(corpus_csv):
        logger.info("Corpus already downloaded")
        return
    logger.info("Get corpus from %s" % xmgr)
    # Get the documents from XMGR
    document_ids = set(document["id"] for document in xmgr.get_documents())
    total = len(document_ids)
    if max_docs is not None:
        document_ids = set(list(document_ids)[:max_docs])
    # Get the list of all PAUs referenced by the documents, periodically saving intermediate results.
    pau_ids_checkpoint = DataFrameCheckpoint(pau_ids_csv, ["Document Id", "Answer IDs"], checkpoint_frequency)
    try:
        document_ids -= pau_ids_checkpoint.recovered
        if pau_ids_checkpoint.recovered:
            logger.info("Recovered %d document ids from previous run" % len(pau_ids_checkpoint.recovered))
        n = len(document_ids)
        start = len(pau_ids_checkpoint.recovered) + 1
        if n:
            logger.info("Get PAU ids from %d documents" % n)
            for i, document_id in enumerate(document_ids, start):
                if i % checkpoint_frequency == 0 or i == start or i == n:
                    logger.info(percent_complete_message("Get PAU ids from document", i, total))
                pau_ids = xmgr.get_pau_ids_from_document(document_id)
                pau_ids_checkpoint.write(document_id, serialize_pau_ids(pau_ids))
    finally:
        pau_ids_checkpoint.close()
    pau_ids_checkpoint = pandas.read_csv(pau_ids_csv, encoding="utf-8")
    pau_ids = reduce(lambda m, s: m | deserialize_pau_ids(s), pau_ids_checkpoint["Answer IDs"], set())
    total = len(pau_ids)
    logger.info("%d PAUs total" % len(pau_ids))
    # Download the PAUs, periodically saving intermediate results.
    corpus_csv_checkpoint = DataFrameCheckpoint(corpus_csv, [ANSWER_ID, ANSWER, TITLE, FILENAME], checkpoint_frequency)
    try:
        if corpus_csv_checkpoint.recovered:
            logger.info("Recovered %d PAUs from previous run" % len(corpus_csv_checkpoint.recovered))
        pau_ids -= corpus_csv_checkpoint.recovered
        n = len(pau_ids)
        m = 0
        start = len(corpus_csv_checkpoint.recovered) + 1
        for i, pau_id in enumerate(pau_ids, start):
            if i % checkpoint_frequency == 0 or i == start or i == n:
                logger.info(percent_complete_message("Get PAU", i, total))
            pau = xmgr.get_pau(pau_id)
            if pau is not None:
                answer_text = pau["responseMarkup"]
                title = pau["title"]
                filename = pau["sourceName"]
                # Note: the "id" field of a PAU is not equal to the "PAU id" you use to look it up from the REST API.
                corpus_csv_checkpoint.write(pau["id"], answer_text, title, filename)
                m += 1
    finally:
        corpus_csv_checkpoint.close()
    logger.info("%d PAU ids, %d with PAUs (%0.4f)" % (n, m, m / float(n)))
    os.remove(pau_ids_csv)


def serialize_pau_ids(pau_ids):
    return ",".join(sorted(pau_ids))


def deserialize_pau_ids(s):
    return set(s.split(","))


def verify_answer_ids(corpus, truth, output_directory):
    """
    Verify that all the answer IDs in the truth appear in the corpus.

    If they are all present in the corpus, this does nothing. If any are missing, it creates two new files:
    truth.in-corpus.csv and truth.not-in-corpus.csv.

    :param corpus: corpus downloaded from xmgr
    :type corpus: pandas.DataFrame
    :param truth:  truth downloaded from xmgr
    :type truth: pandas.DataFrame
    :param output_directory:
    :type output_directory:
    """
    truth_ids = set(truth[ANSWER_ID])
    corpus_ids = set(corpus[ANSWER_ID])
    d = truth_ids - corpus_ids
    if d:
        print("%d truth answer ids of %d not in corpus (%0.3f)" %
              (len(d), len(truth_ids), len(d) / float(len(truth_ids))))
        non_corpus = truth[truth[ANSWER_ID].isin(d)]
        truth_not_in_corpus_csv = os.path.join(output_directory, "truth.not-in-corpus.csv")
        to_csv(truth_not_in_corpus_csv, non_corpus)
        truth_in_corpus_csv = os.path.join(output_directory, "truth.in-corpus.csv")
        truth = truth[~truth[ANSWER_ID].isin(d)]
        to_csv(truth_in_corpus_csv, truth)
        print("Split truth into %s and %s." % (truth_in_corpus_csv, truth_not_in_corpus_csv))
    else:
        print("All truth answer ids are in the corpus.")


class DownloadCorpusFromXmgrClosure(object):
    def __init__(self, xmgr, output_directory, checkpoint_frequency, max_docs):
        self.xmgr = xmgr
        self.output_directory = output_directory
        self.checkpoint_frequency = checkpoint_frequency
        self.max_docs = max_docs

    def __call__(self):
        download_corpus_from_xmgr(self.xmgr, self.output_directory, self.checkpoint_frequency, self.max_docs)


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

    def get_documents(self):
        return self.get("xmgr/corpus/document")

    def get_pau_ids_from_document(self, document_id):
        trec_document = self.get("xmgr/corpus/wea/trec", {"srcDocId": document_id})
        pau_ids = [item["DOCNO"] for item in trec_document["items"]]
        logger.debug("Document %s, %d PAUs" % (document_id, len(pau_ids)))
        if not len(pau_ids) == len(set(pau_ids)):
            logger.warning("Document %s contains duplicate PAUs" % document_id)
        return set(pau_ids)

    def get_pau(self, pau_id):
        hits = self.get(os.path.join("wcea/api/GroundTruth/paus", pau_id))["hits"]
        if hits:
            pau = hits[0]
        else:
            pau = None
        return pau

    def get(self, path, params=None, headers=None):
        url = os.path.join(self.project_url, path)
        r = requests.get(url, auth=(self.username, self.password), params=params, headers=headers)
        logger.debug("GET %s, Status %d" % (url, r.status_code))
        r.raise_for_status()
        try:
            return r.json()
        except ValueError as e:
            # When it handles an invalid URL, XMGR returns HTTP status 200 with text on a web page describing the
            # error. This web text cannot be parsed as JSON, causing a confusing ValueError to be thrown. Catch this
            # case and raise a more sensible exception.
            if r.status_code == 200 and "The page you were looking for could not be found." in r.text:
                raise ValueError("Invalid URL %s" % url)
            else:
                raise e


class CorpusFileType(CsvFileType):
    def __init__(self):
        super(self.__class__, self).__init__([ANSWER_ID, ANSWER, TITLE, FILENAME])


class TruthFileType(CsvFileType):
    def __init__(self):
        super(self.__class__, self).__init__([QUESTION_ID, QUESTION, ANSWER_ID])

    @staticmethod
    def output_format(truth):
        truth = truth.sort_values(QUESTION_ID)
        truth = truth[[QUESTION_ID, QUESTION, ANSWER_ID]].set_index(QUESTION_ID)
        return truth
