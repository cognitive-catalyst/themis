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

from themis import retry
from themis.xmgr import download_truth_from_xmgr, XmgrProject, DownloadCorpusFromXmgrClosure


def download_command(subparsers):
    xmgr_shared_arguments = argparse.ArgumentParser(add_help=False)
    xmgr_shared_arguments.add_argument("url", help="XMGR url")
    xmgr_shared_arguments.add_argument("username", help="XMGR username")
    xmgr_shared_arguments.add_argument("password", help="XMGR password")
    xmgr_shared_arguments.add_argument("--output-directory", metavar="OUTPUT-DIRECTORY", type=str, default=".",
                                       help="output directory")

    download_parser = subparsers.add_parser("download", help="download information from XMGR")
    subparsers = download_parser.add_subparsers(description="download information from XMGR")
    # Download corpus from XMGR.
    download_corpus = subparsers.add_parser("corpus", parents=[xmgr_shared_arguments], help="download corpus")
    download_corpus.add_argument("--checkpoint-frequency", metavar="CHECKPOINT-FREQUENCY", type=int, default=100,
                                 help="how often to flush to a checkpoint file")
    download_corpus.add_argument("--max-docs", metavar="MAX-DOCS", type=int,
                                 help="maximum number of corpus documents to download")
    download_corpus.add_argument("--retries", type=int, help="number of times to retry downloading after an error")
    download_corpus.set_defaults(func=corpus_handler)
    # Download truth from XMGR.
    download_truth = subparsers.add_parser("truth", parents=[xmgr_shared_arguments], help="download truth file")
    download_truth.set_defaults(func=truth_handler)


def corpus_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    c = DownloadCorpusFromXmgrClosure(xmgr, args.output_directory, args.checkpoint_frequency, args.max_docs)
    retry(c, args.retries)


def truth_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    download_truth_from_xmgr(xmgr, args.output_directory)
