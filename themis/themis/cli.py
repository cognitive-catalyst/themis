import argparse

from themis import configure_logger, CsvFileType, QUESTION, ANSWER, print_csv, CONFIDENCE
from wea import QUESTION_TEXT, TOP_ANSWER_TEXT, USER_EXPERIENCE, TOP_ANSWER_CONFIDENCE
from wea import wea_test, create_test_set_from_wea_logs
from xmgr import download_from_xmgr


def run():
    parser = argparse.ArgumentParser(description="Themis analysis toolkit")
    parser.add_argument('--log', default='INFO', help='logging level')
    subparsers = parser.add_subparsers(dest="command", help="command to run")

    xmgr_parser = subparsers.add_parser("xmgr", help="download information from XMGR")
    xmgr_parser.add_argument("url", type=str, help="XMGR url")
    xmgr_parser.add_argument("username", type=str, help="XMGR username")
    xmgr_parser.add_argument("password", type=str, help="XMGR password")
    xmgr_parser.add_argument("output_directory", type=str, help="output directory")
    xmgr_parser.add_argument("--max-docs", type=int, help="maximum number of corpus documents to download")

    test_set_parser = subparsers.add_parser("test-set", help="create test set from XMGR logs")
    test_set_parser.add_argument("logs",
                                 type=CsvFileType([QUESTION_TEXT, TOP_ANSWER_TEXT, USER_EXPERIENCE],
                                                  {QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER}),
                                 help="QuestionsData.csv log file from XMGR")
    test_set_parser.add_argument("--n", type=int, help="sample size")

    wea_parser = subparsers.add_parser("wea", help="answer questions with WEA logs")
    wea_parser.add_argument("test_set", type=CsvFileType(), help="test set")
    wea_parser.add_argument("logs",
                            type=CsvFileType(
                                [QUESTION_TEXT, TOP_ANSWER_TEXT, TOP_ANSWER_CONFIDENCE, USER_EXPERIENCE],
                                {QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER,
                                 TOP_ANSWER_CONFIDENCE: CONFIDENCE}),
                            help="QuestionsData.csv log file from XMGR")

    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")

    if args.command == "xmgr":
        download_from_xmgr(args.url, args.username, args.password, args.output_directory, args.max_docs)
    elif args.command == "test-set":
        test_set = create_test_set_from_wea_logs(args.logs, args.n)
        print_csv(test_set)
    elif args.command == "wea":
        results = wea_test(args.test_set, args.logs)
        print_csv(results)


if __name__ == "__main__":
    run()
