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

from themis import CsvFileType, retry, print_csv
from themis.wea import filter_corpus
from themis.xmgr import download_truth_from_xmgr, XmgrProject, DownloadCorpusFromXmgrClosure


def download_command(subparsers):
    xmgr_shared_arguments = argparse.ArgumentParser(add_help=False)
    xmgr_shared_arguments.add_argument("url", help="XMGR url")
    xmgr_shared_arguments.add_argument("username", help="XMGR username")
    xmgr_shared_arguments.add_argument("password", help="XMGR password")

    output_directory_argument = argparse.ArgumentParser(add_help=False)
    output_directory_argument.add_argument("--output-directory", metavar="OUTPUT-DIRECTORY", type=str, default=".",
                                           help="output directory")

    download_parser = subparsers.add_parser("download", help="download information from XMGR")
    subparsers = download_parser.add_subparsers(description="download information from XMGR")
    # download corpus
    xmgr_corpus = subparsers.add_parser("corpus", parents=[xmgr_shared_arguments, output_directory_argument],
                                        help="download corpus")
    xmgr_corpus.add_argument("--checkpoint-frequency", metavar="CHECKPOINT-FREQUENCY", type=int, default=100,
                             help="how often to flush to a checkpoint file")
    xmgr_corpus.add_argument("--max-docs", metavar="MAX-DOCS", type=int,
                             help="maximum number of corpus documents to download")
    xmgr_corpus.add_argument("--retries", type=int, help="number of times to retry downloading after an error")
    xmgr_corpus.set_defaults(func=corpus_handler)
    # download truth
    xmgr_truth = subparsers.add_parser("truth", parents=[xmgr_shared_arguments, output_directory_argument],
                                       help="download truth file")
    xmgr_truth.set_defaults(func=truth_handler)
    # download filter
    xmgr_filter = subparsers.add_parser("filter", help="filter the corpus downloaded from XMGR")
    xmgr_filter.add_argument("corpus", type=CsvFileType(), help="corpus file created by the xmgr command")
    xmgr_filter.add_argument("--max-size", metavar="MAX-SIZE", type=int, help="maximum size of answer text")
    xmgr_filter.set_defaults(func=filter_corpus_handler)


def corpus_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    c = DownloadCorpusFromXmgrClosure(xmgr, args.output_directory, args.checkpoint_frequency, args.max_docs)
    retry(c, args.retries)


def truth_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    download_truth_from_xmgr(xmgr, args.output_directory)


def filter_corpus_handler(args):
    corpus = filter_corpus(args.corpus, args.max_size)
    print_csv(corpus)
