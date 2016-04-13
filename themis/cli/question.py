import argparse

from themis import print_csv
from themis.wea import WeaLogFileType
from themis.xmgr import create_question_set_from_usage_logs, question_frequency


def question_command(subparsers):
    questions_shared_arguments = argparse.ArgumentParser(add_help=False)
    questions_shared_arguments.add_argument("usage_log", metavar="usage-log", type=WeaLogFileType(),
                                            help="QuestionsData.csv usage log file from XMGR")

    question_parser = subparsers.add_parser("question", help="extract questions from usage logs")
    subparsers = question_parser.add_subparsers(description="extract questions from usage logs")
    # Extract questions from usage logs.
    question_sample = subparsers.add_parser("sample", parents=[questions_shared_arguments],
                                            help="samples questions from usage logs")
    question_sample.add_argument("--sample-size", metavar="SAMPLE-SIZE", type=int,
                                 help="number of unique questions to sample, default is to use them all")
    question_sample.set_defaults(func=extract_handler)
    # Extract question frequencies from usage logs.
    question_frequencies = subparsers.add_parser("frequency", parents=[questions_shared_arguments],
                                                 help="get question frequencies from usage logs")
    question_frequencies.set_defaults(func=frequencies_handler)


def extract_handler(args):
    questions = create_question_set_from_usage_logs(args.usage_log, args.sample_size)
    print_csv(questions)


def frequencies_handler(args):
    frequency = question_frequency(args.usage_log)
    print_csv(frequency)
