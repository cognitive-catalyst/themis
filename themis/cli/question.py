import pandas

from themis import print_csv, logger
from themis.fixup import filter_usage_log_by_date, filter_usage_log_by_user_experience, deakin
from themis.usage_log import UsageLogFileType, QAPairFileType
from themis.usage_log import extract_question_answer_pairs_from_usage_logs, sample_questions


def question_command(subparsers):
    question_parser = subparsers.add_parser("question", help="get questions to ask a Q&A system")
    subparsers = question_parser.add_subparsers(description="get questions to ask a Q&A system")
    # Extract questions from usage logs.
    question_extract = subparsers.add_parser("extract", help="extract question/answer pairs from usage logs")
    question_extract.add_argument("usage_log", metavar="usage-log", type=UsageLogFileType(),
                                  help="QuestionsData.csv usage log file from XMGR")
    question_extract.add_argument("--before", metavar="DATE", type=pandas.to_datetime,
                                  help="keep interactions before the specified date")
    question_extract.add_argument("--after", metavar="DATE", type=pandas.to_datetime,
                                  help="keep interactions after the specified date")
    question_extract.add_argument("--user-experience", nargs="+", default=set(),
                                  help="disallowed User Experience values (DIALOG is always disallowed)")
    question_extract.add_argument("--deakin", action="store_true", help="fixups specific to the Deakin system")
    question_extract.set_defaults(func=extract_handler)
    # Sample questions by frequency.
    question_sample = subparsers.add_parser("sample", help="sample questions")
    question_sample.add_argument("questions", type=QAPairFileType(),
                                 help="question/answer pairs extracted from usage log by the 'question extract' command")
    question_sample.add_argument("sample_size", metavar="sample-size", type=int,
                                 help="number of unique questions to sample")
    question_sample.set_defaults(func=sample_handler)


def extract_handler(args):
    # Do custom fixup of usage logs.
    usage_log = args.usage_log
    n = len(usage_log)
    if args.before or args.after:
        usage_log = filter_usage_log_by_date(usage_log, args.before, args.after)
    user_experience = set(args.user_experience) | {"DIALOG"}  # DIALOG is always disallowed
    usage_log = filter_usage_log_by_user_experience(usage_log, user_experience)
    if args.deakin:
        usage_log = deakin(usage_log)
    m = n - len(usage_log)
    if n:
        logger.info("Removed %d of %d questions (%0.3f%%)" % (m, n, 100.0 * m / n))
    # Extract Q&A pairs from fixed up usage logs.
    qa_pairs = extract_question_answer_pairs_from_usage_logs(usage_log)
    print_csv(QAPairFileType.output_format(qa_pairs))


def sample_handler(args):
    sample = sample_questions(args.questions, args.sample_size)
    print_csv(sample)
