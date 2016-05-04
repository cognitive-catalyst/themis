from __future__ import print_function

import json
import logging
import os
import sys

import pandas

__version__= "2.3.0"

logger = logging.getLogger(__name__)

QUESTION = "Question"
QUESTION_ID = "Question Id"
ANSWER = "Answer"
ANSWER_ID = "Answer Id"
TITLE = "Title"
FILENAME = "Filename"
DOCUMENT_ID = "Document Id"
CONFIDENCE = "Confidence"
FREQUENCY = "Frequency"
CORRECT = "Correct"
IN_PURVIEW = "In Purview"


def from_csv(file, **kwargs):
    return pandas.read_csv(file, encoding="utf-8", **kwargs)


def to_csv(filename, dataframe, **kwargs):
    dataframe.to_csv(filename, encoding="utf-8", **kwargs)


def print_csv(dataframe, **kwargs):
    print(dataframe.to_csv(encoding="utf-8", **kwargs))


class CsvFileType(object):
    """Pandas CSV file type used with argparse

    This allows you to specify the columns you wish to use and optionally rename them.
    """

    def __init__(self, columns=None, rename=None):
        self.columns = columns
        self.rename = rename

    def __call__(self, filename):
        try:
            csv = from_csv(filename, usecols=self.columns)
            if self.rename is not None:
                csv = csv.rename(columns=self.rename)
            csv.filename = filename
            return csv
        except ValueError as e:
            print("Invalid format for %s: %s" % (filename, e), file=sys.stderr)
            raise e


def percent_complete_message(msg, n, total):
    return "%s %d of %d (%0.3f%%)" % (msg, n, total, 100.0 * n / total)


def pretty_print_json(j):
    return json.dumps(j, indent=2)


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


def ensure_directory_exists(directory):
    try:
        os.makedirs(directory)
    except OSError:
        pass
