import argparse

from themis import retry, print_csv
from themis.fixup import filter_corpus
from themis.xmgr import download_truth_from_xmgr, XmgrProject, DownloadCorpusFromXmgrClosure, CorpusFileType


def xmgr_command(subparsers):
    xmgr_shared_arguments = argparse.ArgumentParser(add_help=False)
    xmgr_shared_arguments.add_argument("url", help="XMGR url")
    xmgr_shared_arguments.add_argument("username", help="XMGR username")
    xmgr_shared_arguments.add_argument("password", help="XMGR password")
    xmgr_shared_arguments.add_argument("--output-directory", metavar="OUTPUT-DIRECTORY", type=str, default=".",
                                       help="output directory")

    xmgr_parser = subparsers.add_parser("xmgr", help="download information from XMGR")
    subparsers = xmgr_parser.add_subparsers(description="download information from XMGR")
    # Download corpus from XMGR.
    xmgr_corpus = subparsers.add_parser("corpus", parents=[xmgr_shared_arguments], help="download corpus")
    xmgr_corpus.add_argument("--checkpoint-frequency", metavar="CHECKPOINT-FREQUENCY", type=int, default=100,
                             help="how often to flush to a checkpoint file")
    xmgr_corpus.add_argument("--max-docs", metavar="MAX-DOCS", type=int,
                             help="maximum number of corpus documents to download")
    xmgr_corpus.add_argument("--retries", type=int, help="number of times to retry downloading after an error")
    xmgr_corpus.set_defaults(func=corpus_handler)
    # Download truth from XMGR.
    xmgr_truth = subparsers.add_parser("truth", parents=[xmgr_shared_arguments], help="download truth file")
    xmgr_truth.set_defaults(func=truth_handler)
    # Filter corpus.
    xmgr_filter = subparsers.add_parser("filter", help="fix up corpus")
    xmgr_filter.add_argument("corpus", type=CorpusFileType(), help="file downloaded by 'xmgr corpus'")
    xmgr_filter.add_argument("--max-size", metavar="MAX-SIZE", type=int,
                             help="maximum size of answer text in characters")
    xmgr_filter.set_defaults(func=fixup_corpus_handler)


def corpus_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    c = DownloadCorpusFromXmgrClosure(xmgr, args.output_directory, args.checkpoint_frequency, args.max_docs)
    retry(c, args.retries)


def truth_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    download_truth_from_xmgr(xmgr, args.output_directory)


def fixup_corpus_handler(args):
    corpus = filter_corpus(args.corpus, args.max_size)
    print_csv(corpus)
