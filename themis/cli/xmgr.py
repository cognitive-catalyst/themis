"""
The xmgr command extracts information from the XMGR instance and the usage log.

corpus
    Download corpus from XMGR.

truth
    Download truth from XMGR.

questions
    Extract question set form usage log.

filter
    Filter oversized answer documents out of the corpus.

frequency
    List questions by the frequency with which they appear in the usage log.
"""

import argparse

import pandas

from themis import CsvFileType
from themis.wea import WeaLogFileType


def xmgr_command(subparsers):
    xmgr_shared_arguments = argparse.ArgumentParser(add_help=False)
    xmgr_shared_arguments.add_argument("url", help="XMGR url")
    xmgr_shared_arguments.add_argument("username", help="XMGR username")
    xmgr_shared_arguments.add_argument("password", help="XMGR password")

    parser = subparsers.add_parser("xmgr", help="download information from XMGR")
    subparsers = parser.add_subparsers(title="XMGR", description="extract information from XMGR", help="XMGR actions")
    # Get corpus from XMGR.
    xmgr_corpus = subparsers.add_parser("corpus", parents=[xmgr_shared_arguments], help="download corpus from XMGR")
    xmgr_corpus.add_argument("--checkpoint-frequency", type=int, default=100,
                             help="how often to flush to a checkpoint file")
    xmgr_corpus.add_argument("--max-docs", type=int, help="maximum number of corpus documents to download")
    xmgr_corpus.add_argument("--retries", type=int, help="number of times to retry downloading after an error")
    xmgr_corpus.set_defaults(func=corpus_handler)
    # Get truth from XMGR.
    xmgr_truth = subparsers.add_parser("truth", parents=[xmgr_shared_arguments], help="download truth file from XMGR")
    xmgr_truth.add_argument("--output-directory", type=str, default=".", help="output directory")
    xmgr_truth.set_defaults(func=truth_handler)
    # Extract questions from the usage log.
    xmgr_questions = subparsers.add_parser("questions", help="extract question sets from the usage log")
    xmgr_questions.add_argument("usage_log", type=WeaLogFileType(), help="QuestionsData.csv usage log file from XMGR")
    xmgr_questions.add_argument("--before", type=pandas.to_datetime,
                                help="keep interactions before the specified date")
    xmgr_questions.add_argument("--after", type=pandas.to_datetime, help="keep interactions after the specified date")
    xmgr_questions.add_argument("--sample_size", type=int,
                                help="number of unique questions to sample, default is to use them all")
    xmgr_questions.set_defaults(func=questions_handler)
    # Filter corpus.
    xmgr_filter = subparsers.add_parser("filter", help="filter the corpus downloaded from XMGR")
    xmgr_filter.add_argument("corpus", type=CsvFileType(), help="corpus file created by the xmgr command")
    xmgr_filter.add_argument("--max-size", type=int, help="maximum size of answer text")
    xmgr_filter.set_defaults(func=filter_handler)
    # Get question frequencies from the usage log.
    xmgr_frequency = subparsers.add_parser("frequency", help="get question frequencies from the usage log")
    xmgr_frequency.add_argument("usage_log", type=WeaLogFileType(), help="QuestionsData.csv usage log file from XMGR")
    xmgr_frequency.set_defaults(func=frequencies_handler)


def corpus_handler(args):
    pass


def truth_handler(args):
    pass


def questions_handler(args):
    pass


def filter_handler(args):
    pass


def frequencies_handler(args):
    pass
