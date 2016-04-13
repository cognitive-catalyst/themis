"""
Commands to download information from XMGR, extract questions from the usage logs, answer questions using various Q&A
systems, annotate the answers and analyze the results.
"""
import argparse

from themis import configure_logger
from themis.cli.answer import answer_command
from themis.cli.download import download_command
from themis.cli.fixup import fixup_command
from themis.cli.judge import judge_command
from themis.cli.plot import plot_command
from themis.cli.question import question_command


def run():
    parser = argparse.ArgumentParser(description="Themis analysis toolkit")
    parser.add_argument("--log", default="INFO", help="logging level")

    subparsers = parser.add_subparsers(title="Q&A System analysis", description=__doc__)
    # Download information from xmgr.
    download_command(subparsers)
    # Do system-specific fixups of downloaded files.
    fixup_command(subparsers)
    # Extract questions from usage logs.
    question_command(subparsers)
    # Ask questions to a Q&A system.
    answer_command(subparsers)
    # Judge answers using Annotation Assist.
    judge_command(subparsers)
    # Generate ROC and precision curves from judged answers.
    plot_command(parser, subparsers)

    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


if __name__ == "__main__":
    run()
