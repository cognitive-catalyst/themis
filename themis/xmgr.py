"""Utilities to download information from an Watson Experience Manager (XMGR) project"""
import json
import os

import pandas
import requests

from themis import QUESTION, ANSWER_ID, ANSWER, TITLE, FILENAME, QUESTION_ID, from_csv, DOCUMENT_ID
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
    document_ids_csv = os.path.join(output_directory, "document_ids.csv")
    corpus_csv = os.path.join(output_directory, "corpus.csv")
    if os.path.isfile(corpus_csv) and not os.path.isfile(document_ids_csv):
        logger.info("Corpus already downloaded")
        return
    document_ids = sorted(set(document["id"] for document in xmgr.get_documents()))
    document_ids = document_ids[:max_docs]
    n = len(document_ids)
    downloaded_document_ids = DataFrameCheckpoint(document_ids_csv, ["Document Id"], checkpoint_frequency)
    corpus = DataFrameCheckpoint(corpus_csv, [ANSWER_ID, ANSWER, TITLE, FILENAME, DOCUMENT_ID])
    try:
        if downloaded_document_ids.recovered:
            logger.info("Recovered PAUs from %d documents from previous run" % len(downloaded_document_ids.recovered))
        document_ids = sorted(set(document_ids) - downloaded_document_ids.recovered)
        m = len(document_ids)
        start = len(downloaded_document_ids.recovered) + 1
        if m:
            for i, document_id in enumerate(document_ids, start):
                if i % checkpoint_frequency == 0 or i == start or i == m:
                    corpus.flush()
                    logger.info(percent_complete_message("Get PAUs from document", i, n))
                paus = xmgr.get_paus_from_document(document_id)
                for pau in paus:
                    corpus.write(pau["id"], pau["responseMarkup"], pau["title"], pau["sourceName"], document_id)
                downloaded_document_ids.write(str(document_id))
    finally:
        downloaded_document_ids.close()
        corpus.close()
    corpus = from_csv(corpus_csv).drop_duplicates(ANSWER_ID)
    to_csv(corpus_csv, corpus)
    docs = len(from_csv(document_ids_csv))
    os.remove(document_ids_csv)
    logger.info("%d documents and %d PAUs in corpus" % (docs, len(corpus)))


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
        print("%d truth answer ids of %d not in corpus (%0.3f%%)" %
              (len(d), len(truth_ids), 100.0 * len(d) / len(truth_ids)))
        non_corpus = truth[truth[ANSWER_ID].isin(d)]
        truth_not_in_corpus_csv = os.path.join(output_directory, "truth.not-in-corpus.csv")
        to_csv(truth_not_in_corpus_csv, TruthFileType.output_format(non_corpus))
        truth_in_corpus_csv = os.path.join(output_directory, "truth.in-corpus.csv")
        truth = truth[~truth[ANSWER_ID].isin(d)]
        to_csv(truth_in_corpus_csv, TruthFileType.output_format(truth))
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

    def get_paus_from_document(self, document_id):
        logger.debug("Get PAUs from document %s" % document_id)
        paus = []
        # Get the TREC ids corresponding to the document Id.
        trec_document = self.get("xmgr/corpus/wea/trec", {"srcDocId": document_id})
        trec_ids = set(item["DOCNO"] for item in trec_document["items"])
        logger.debug("%d TREC IDs in document %s" % (len(trec_ids), document_id))
        # Get the PAUs corresponding to the trec IDs.
        for trec_id in trec_ids:
            paus.extend(self.get(self.urljoin("wcea/api/GroundTruth/paus", trec_id))["hits"])
        return paus

    def get(self, path, params=None, headers=None):
        def debug_msg():
            if params is None:
                s = "GET %s, Status %d" % (url, r.status_code)
            else:
                s = "GET %s, %s, Status %d" % (url, params, r.status_code)
            return s

        url = self.urljoin(self.project_url, path)
        r = requests.get(url, auth=(self.username, self.password), params=params, headers=headers)
        logger.debug(debug_msg())
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

    # Use this because urlparse.urljoin discards path components that contain a "$", which XMGR project paths do.
    @staticmethod
    def urljoin(a, b):
        return "%s/%s" % (a.rstrip("/"), b.lstrip("/"))


class CorpusFileType(CsvFileType):
    def __init__(self):
        super(self.__class__, self).__init__([ANSWER_ID, ANSWER, TITLE, FILENAME, DOCUMENT_ID])


class TruthFileType(CsvFileType):
    def __init__(self):
        super(self.__class__, self).__init__([QUESTION_ID, QUESTION, ANSWER_ID])

    @staticmethod
    def output_format(truth):
        truth = truth.sort_values(QUESTION_ID)
        truth = truth[[QUESTION_ID, QUESTION, ANSWER_ID]].set_index(QUESTION_ID)
        return truth
