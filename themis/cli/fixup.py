import pandas

from themis import print_csv
from themis.fixup import filter_corpus, filter_usage_log_by_date, filter_usage_log_by_user_experience, deakin
from themis.usage_log import UsageLogFileType
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
    # Fixup usage log.
    fixup_usage_log = subparsers.add_parser("usage", help="fix up usage log")
    fixup_usage_log.add_argument("usage_log", metavar="usage-log", type=UsageLogFileType(),
                                 help="QuestionsData.csv usage log file from XMGR")
    fixup_usage_log.add_argument("--before", metavar="DATE", type=pandas.to_datetime,
                                 help="keep interactions before the specified date")
    fixup_usage_log.add_argument("--after", metavar="DATE", type=pandas.to_datetime,
                                 help="keep interactions after the specified date")
    fixup_usage_log.add_argument("--user-experience", nargs="+",
                                 help="disallowed User Experience values (DIALOG is always disallowed)")
    fixup_usage_log.add_argument("--deakin", action="store_true", help="fixups specific to the Deakin system")
    fixup_usage_log.set_defaults(func=fixup_usage_log_handler)


def fixup_corpus_handler(args):
    corpus = filter_corpus(args.corpus, args.max_size)
    print_csv(corpus)


def fixup_usage_log_handler(args):
    usage_log = args.usage_log
    if args.before or args.after:
        usage_log = filter_usage_log_by_date(usage_log, args.before, args.after)
    user_experience = set(args.user_experience) | {"DIALOG"}  # DIALOG is always disallowed
    usage_log = filter_usage_log_by_user_experience(usage_log, user_experience)
    if args.deakin:
        usage_log = deakin(usage_log)
    print_csv(usage_log, index=False)
