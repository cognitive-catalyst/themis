"""
Command line interface for Themis.
"""
import argparse

from themis import configure_logger
from themis.cli.curves import curves_command
from themis.cli.judge import judge_command
from themis.cli.qa import qa_command
from themis.cli.xmgr import xmgr_command


def run():
    parser = argparse.ArgumentParser(description="Themis analysis toolkit")
    parser.add_argument("--log", default="INFO", help="logging level")

    subparsers = parser.add_subparsers(title="Q&A System analysis",
                                       description="Commands to download information from XMGR, answer questions " +
                                                   "using various Q&A systems, annotate the answers and analyze " +
                                                   "the results",
                                       help="command to run")

    # TODO Rename commands to download, question, answer, judge, curve. Split xmgr functionality between download and question.
    # Download information from xmgr.
    xmgr_command(subparsers)
    # Ask questions to a Q&A system.
    qa_command(subparsers)
    # Judge answers using Annotation Assist.
    judge_command(subparsers)
    # Generate ROC and precision curves from judged answers.
    curves_command(parser, subparsers)

    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


if __name__ == "__main__":
    run()
