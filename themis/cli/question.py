import argparse

import pandas

from themis import print_csv
from themis.wea import WeaLogFileType
from themis.xmgr import filter_usage_log_by_date, create_question_set_from_usage_logs, question_frequency


def question_command(subparsers):
    questions_shared_arguments = argparse.ArgumentParser(add_help=False)
    questions_shared_arguments.add_argument("usage_log", metavar="usage-log", type=WeaLogFileType(),
                                            help="QuestionsData.csv usage log file from XMGR")
    questions_shared_arguments.add_argument("--before", type=pandas.to_datetime,
                                            help="keep interactions before the specified date")
    questions_shared_arguments.add_argument("--after", type=pandas.to_datetime,
                                            help="keep interactions after the specified date")

    question_parser = subparsers.add_parser("question", help="extract questions from usage logs")
    subparsers = question_parser.add_subparsers(description="extract questions from usage logs")
    # question sample
    question_sample = subparsers.add_parser("sample", parents=[questions_shared_arguments],
                                            help="samples questions from usage logs")
    question_sample.add_argument("--sample-size", metavar="SAMPLE-SIZE", type=int,
                                 help="number of unique questions to sample, default is to use them all")
    question_sample.set_defaults(func=extract_handler)
    # question frequency
    question_frequencies = subparsers.add_parser("frequency", parents=[questions_shared_arguments],
                                                 help="get question frequencies from usage logs")
    question_frequencies.set_defaults(func=frequencies_handler)


def extract_handler(args):
    usage_log = filter_usage_log_by_date(args.usage_log, args.before, args.after)
    questions = create_question_set_from_usage_logs(usage_log, args.sample_size)
    print_csv(questions)


def frequencies_handler(args):
    usage_log = filter_usage_log_by_date(args.usage_log, args.before, args.after)
    frequency = question_frequency(usage_log)
    print_csv(frequency)
