import argparse

from themis import configure_logger
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
    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")

    if args.command == "xmgr":
        download_from_xmgr(args.url, args.username, args.password, args.output_directory, args.max_docs)


if __name__ == "__main__":
    run()
