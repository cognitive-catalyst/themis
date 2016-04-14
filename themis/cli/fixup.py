from themis import print_csv
from themis.fixup import filter_corpus
from themis.xmgr import CorpusFileType


def fixup_command(subparsers):
    fixup_parser = subparsers.add_parser("fixup", help="fix up downloaded files")
    subparsers = fixup_parser.add_subparsers(description="fix up downloaded files")
    # Fixup corpus.
    fixup_corpus = subparsers.add_parser("corpus", help="fix up file downloaded by 'download corpus'")
    fixup_corpus.add_argument("corpus", type=CorpusFileType(), help="corpus file")
    fixup_corpus.add_argument("--max-size", metavar="MAX-SIZE", type=int,
                              help="maximum size of answer text in characters")
    fixup_corpus.set_defaults(func=fixup_corpus_handler)


def fixup_corpus_handler(args):
    corpus = filter_corpus(args.corpus, args.max_size)
    print_csv(corpus)
