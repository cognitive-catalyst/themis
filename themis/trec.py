"""
Parse the TREC XML format file in which XMGR stores PAU information.

This is for when we have file system access to the corpus instead of needing to download it.
"""
import glob
import os

from bs4 import BeautifulSoup

from themis import logger, from_csv, ANSWER_ID, ANSWER, TITLE, FILENAME, DOCUMENT_ID
from themis.checkpoint import DataFrameCheckpoint, get_items
from themis.xmgr import CorpusFileType


def corpus_from_trec(checkpoint_filename, trec_directory, checkpoint_frequency, max_docs):
    trec_filenames = sorted(glob.glob(os.path.join(trec_directory, "*.xml")))[:max_docs]
    checkpoint = get_items("TREC files",
                           trec_filenames,
                           TrecFileCheckpoint(checkpoint_filename, checkpoint_frequency),
                           parse_trec_file,
                           checkpoint_frequency)
    if checkpoint.invalid:
        n = len(trec_filenames)
        logger.warning("%d of %d TREC files are invalid (%0.3f%%)" %
                       (checkpoint.invalid, n, 100 * checkpoint.invalid / n))
    # I'm not sure why I'm getting duplicates after a restart.
    return from_csv(checkpoint_filename).drop_duplicates().drop(TrecFileCheckpoint.TREC_FILENAME, axis="columns")


def parse_trec_file(trec_filename):
    """
    Extract corpus fields from a TREC XML file.

    The TREC files may be mal-formed XML. (For instance they contain disallowed '&', '<', and '>' characters inside
    text, so parse them with the robust Beautiful Soup package, returning None if the file cannot be successfully
    parsed.

    :param trec_filename: name of TREC XML file
    :type trec_filename: str
    :return: labeled fields extracted from the TREC file
    :rtype: dict
    """
    with open(trec_filename) as trec_file:
        parse = BeautifulSoup(trec_file, "lxml")
        try:
            return {
                ANSWER_ID: parse.find("meta:key:pautid").text,
                ANSWER: parse.find("text").text,
                TITLE: parse.find("title").text,
                FILENAME: parse.find("meta:key:originalfile").text,
                DOCUMENT_ID: parse.find("meta:documentid").text
            }
        except AttributeError:
            # If a XML tag is missing, find will return None, which will not have a 'text' attribute.
            return None


class TrecFileCheckpoint(DataFrameCheckpoint):
    """
    A checkpoint that indexes TREC file contents by their file name on the local system.

    It also keeps track of the number of invalid TREC files that were written to it.
    """
    TREC_FILENAME = "TREC Filename"

    def __init__(self, filename, interval):
        self.invalid = 0
        super(self.__class__, self).__init__(filename,
                                             [TrecFileCheckpoint.TREC_FILENAME] + CorpusFileType.columns,
                                             interval)

    def write(self, trec_filename, trec):
        if trec is not None:
            super(self.__class__, self).write(trec_filename,
                                              trec[ANSWER_ID],
                                              trec[ANSWER],
                                              trec[TITLE],
                                              trec[FILENAME],
                                              trec[DOCUMENT_ID])
        else:
            self.invalid += 1
