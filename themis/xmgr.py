"""Utilities to download information from an Watson Experience Manager (XMGR) project"""
import glob
import json
import os
import re
import xml

import pandas
import requests
import xmltodict

from themis import QUESTION, ANSWER_ID, ANSWER, TITLE, FILENAME, QUESTION_ID, from_csv, DOCUMENT_ID
from themis import logger, to_csv, DataFrameCheckpoint, ensure_directory_exists, percent_complete_message, \
    CsvFileType
from themis.question import QAPairFileType


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
    logger.info("Download corpus from %s" % xmgr)
    document_ids = sorted(set(document["id"] for document in xmgr.get_documents()))
    document_ids = document_ids[:max_docs]
    n = len(document_ids)
    downloaded_document_ids = DataFrameCheckpoint(document_ids_csv, [DOCUMENT_ID, "Paus"], checkpoint_frequency)
    corpus = DataFrameCheckpoint(corpus_csv, CorpusFileType.columns)
    try:
        if downloaded_document_ids.recovered:
            logger.info("Recovered %d documents from previous run" % len(downloaded_document_ids.recovered))
        document_ids = sorted(set(document_ids) - downloaded_document_ids.recovered)
        m = len(document_ids)
        start = len(downloaded_document_ids.recovered) + 1
        if m:
            for i, document_id in enumerate(document_ids, start):
                if i % checkpoint_frequency == 0 or i == start or i == m:
                    corpus.flush()
                    logger.info(percent_complete_message("Get PAUs from document", i, n))
                paus = xmgr.get_paus_from_document(document_id)
                # The document id and number of PAUs are both integers. Cast them to strings, otherwise pandas will
                # write them as floats.
                for pau in paus:
                    corpus.write(pau["id"], pau["responseMarkup"], pau["title"], pau["sourceName"], str(document_id))
                downloaded_document_ids.write(str(document_id), str(len(paus)))
    finally:
        downloaded_document_ids.close()
        corpus.close()
    corpus = from_csv(corpus_csv).drop_duplicates(ANSWER_ID)
    to_csv(corpus_csv, CorpusFileType.output_format(corpus))
    docs = len(from_csv(document_ids_csv))
    os.remove(document_ids_csv)
    logger.info("%d documents and %d PAUs in corpus" % (docs, len(corpus)))


def corpus_from_trec_files(trec_directory):
    """
    Construct a corpus out of a directory of .XML TREC files.

    :param trec_directory: directories containing TREC files
    :type trec_directory: str
    :return: corpus
    :rtype: pandas.DataFrame
    """

    # TREC files contain lots of ampersand characters even though this is an invalid XML character, so if a TREC file
    # doesn't parse try replacing & characters that aren't part of an HTML escape sequence with &amp;.
    def parse_trec():
        with open(trec_filename) as trec_file:
            content = trec_file.read()
            try:
                return xmltodict.parse(content)
            except xml.parsers.expat.ExpatError as e:
                logger.debug("Retry replacing ampersand TREC file %s %s" % (trec_filename, e))
                content = re.sub("&(?!\w+;)", "&amp;", content)
                try:
                    return xmltodict.parse(content)
                except xml.parsers.expat.ExpatError:
                    logger.warning("TREC file %s %s" % (trec_filename, e))
                    return None

    def filename():
        try:
            return metadata["meta:custom:key:fileName"]
        except KeyError:
            return metadata["meta:key:originalfile"]

    corpus = CorpusFileType.create_empty()
    trec_filenames = glob.glob(os.path.join(trec_directory, "*.xml"))
    n = len(trec_filenames)
    logger.info("%d xml files in %s" % (n, trec_directory))
    invalid = 0
    for i, trec_filename in enumerate(trec_filenames, 1):
        if i % 100 == 0 or i == 1 or i == n:
            logger.info(percent_complete_message("Get PAU from TREC document", i, n))
        logger.debug(trec_filename)
        trec = parse_trec()
        if trec is not None:
            metadata = trec["DOC"]["metadata"]
            corpus = corpus.append({
                ANSWER_ID: metadata["meta:key:pauTid"],
                ANSWER: trec["DOC"]["text"],
                TITLE: trec["DOC"]["title"],
                FILENAME: filename(),
                DOCUMENT_ID: metadata["meta:documentid"]}, ignore_index=True)
        else:
            invalid += 1
    if invalid:
        logger.warning("%d of %d xml files invalid (%0.3f%%)" % (invalid, n, 100 * invalid / n))
    logger.info("%d documents and %d PAUs in corpus" % (len(corpus[DOCUMENT_ID].drop_duplicates()), len(corpus)))
    return corpus


def validate_truth_with_corpus(corpus, truth, output_directory):
    """
    Verify that all the answer IDs in the truth appear in the corpus.

    If they are all present in the corpus, this does nothing. If any are missing, it creates two new files:
    truth.in-corpus.csv and truth.not-in-corpus.csv.

    :param corpus: corpus downloaded from xmgr
    :type corpus: pandas.DataFrame
    :param truth:  truth downloaded from xmgr
    :type truth: pandas.DataFrame
    :param output_directory: directory in which to create files
    :type output_directory: str
    """
    missing_answers = ~truth[ANSWER_ID].isin(corpus[ANSWER_ID])
    if any(missing_answers):
        ensure_directory_exists(output_directory)
        missing_truth_answers = truth[missing_answers]
        n = len(truth)
        m = len(missing_truth_answers)
        print("%d truth answers of %d (%0.3f%%) not in the corpus" % (m, n, 100.0 * m / n))
        truth_in_corpus_csv = os.path.join(output_directory, "truth.in-corpus.csv")
        truth_not_in_corpus_csv = os.path.join(output_directory, "truth.not-in-corpus.csv")
        print("Writing truth to %s and %s" % (truth_in_corpus_csv, truth_not_in_corpus_csv))
        to_csv(truth_in_corpus_csv, TruthFileType.output_format(truth[~missing_answers]))
        to_csv(truth_not_in_corpus_csv, TruthFileType.output_format(missing_truth_answers))
    else:
        print("All truth answer ids are in the corpus.")


def validate_answers_with_corpus(corpus, qa_pairs, output_directory):
    """
    Verify that all the answers in the Q&A pairs are present in the corpus.

    If they are all present in the corpus, this does nothing. If any are missing, it creates two new files:
    answers.in-corpus.csv and answers.not-in-corpus.csv.

    :param corpus: corpus downloaded from xmgr
    :type corpus: pandas.DataFrame
    :param qa_pairs: Q&A pairs extracted from the usage logs
    :type qa_pairs: pandas.DataFrame
    :param output_directory: directory in which to create files
    :type output_directory: str
    """
    missing_answers = ~qa_pairs[ANSWER].isin(corpus[ANSWER])
    if any(missing_answers):
        ensure_directory_exists(output_directory)
        missing_answer_qa_pairs = qa_pairs[missing_answers]
        n = len(qa_pairs)
        m = len(missing_answer_qa_pairs)
        print("%d usage log answers of %d (%0.3f%%) not in the corpus" % (m, n, 100.0 * m / n))
        answers_in_corpus_csv = os.path.join(output_directory, "answers.in-corpus.csv")
        answers_not_in_corpus_csv = os.path.join(output_directory, "answers.not-in-corpus.csv")
        print("Writing Q&A pairs to %s and %s" % (answers_not_in_corpus_csv, answers_in_corpus_csv))
        to_csv(answers_in_corpus_csv, QAPairFileType.output_format(qa_pairs[~missing_answers]))
        to_csv(answers_not_in_corpus_csv, QAPairFileType.output_format(missing_answer_qa_pairs))
    else:
        print("All usage log answers are the corpus.")


def examine_truth(corpus, truth):
    """
    Print an HTML file of all the answers in the truth with their corresponding questions

    :param corpus: corpus containing mapping of answer IDs to answers
    :type corpus: pandas.DataFrame
    :param truth: truth containing mapping of questions to answer IDs
    :type truth: pandas.DataFrame
    """
    truth = pandas.merge(truth, corpus, on=ANSWER_ID, how="left")[[QUESTION, ANSWER, ANSWER_ID]]
    print("""
<html>

<head>
    <style>
        div {
            border: 5px solid black;
        }
    </style>
</head>
<body>""")
    for (answer, answer_id), mapping in sorted(truth.groupby((ANSWER, ANSWER_ID)), key=lambda x: (-len(x[1]), x[0][0])):
        questions = "\n".join("<LI>%s</LI>" % q for q in mapping[QUESTION].sort_values())

        s = """
<div>
    <h1>%s (%d questions)</h1>
    %s
    <hr>
    <ol>
        %s
    </ol>
</div>""" % (answer_id, len(mapping), answer, questions)
        print(s.encode("utf-8"))
    print("""
</body>
</html>""")


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
        pau_ids = self.get_pau_ids_in_document(document_id)
        logger.debug("%d TREC IDs in document %s" % (len(pau_ids), document_id))
        for trec_id in pau_ids:
            paus.extend(self.get_paus(trec_id))
        return paus

    def get_pau_ids_in_document(self, document_id):
        document = self.get("xmgr/corpus/wea/trec", {"srcDocId": document_id})
        pau_ids = set(item["DOCNO"] for item in document["items"])
        return pau_ids

    def get_paus(self, i):
        return self.get(self.urljoin("wcea/api/GroundTruth/paus", i))["hits"]

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
    columns = [ANSWER_ID, ANSWER, TITLE, FILENAME, DOCUMENT_ID]

    def __init__(self):
        super(self.__class__, self).__init__(self.__class__.columns)

    @classmethod
    def create_empty(cls):
        return pandas.DataFrame(columns=cls.columns)

    @classmethod
    def output_format(cls, corpus):
        corpus = corpus[cls.columns]
        # Cast integer document IDs to strings so that Pandas does not write them as real numbers.
        corpus[DOCUMENT_ID] = corpus[DOCUMENT_ID].astype("string")
        # Sort by document ID first so that answers from the same document are all grouped together.
        corpus = corpus.sort_values([DOCUMENT_ID, ANSWER_ID]).set_index(ANSWER_ID)
        return corpus


class TruthFileType(CsvFileType):
    def __init__(self):
        super(self.__class__, self).__init__([QUESTION_ID, QUESTION, ANSWER_ID])

    @staticmethod
    def output_format(truth):
        truth = truth.sort_values(QUESTION_ID)
        truth = truth[[QUESTION_ID, QUESTION, ANSWER_ID]].set_index(QUESTION_ID)
        return truth
