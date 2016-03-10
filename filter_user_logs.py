#!/usr/bin/env python
"""
Filter the user interaction logs
"""
import argparse
import logging
import re

import pandas

logger = logging.getLogger(__name__)

DATE_TIME = "DateTime"

WEA_DATE_FORMAT = re.compile(
    r"(?P<month>\d\d)(?P<day>\d\d)(?P<year>\d\d\d\d):(?P<hour>\d\d)(?P<min>\d\d)(?P<sec>\d\d):UTC")


def filter_logs(logs_file, before, after):
    logs = pandas.read_csv(logs_file, encoding="utf-8")
    logger.info("%d total interactions" % len(logs))
    date_time = pandas.to_datetime(logs[DATE_TIME].apply(standard_date_format))
    if after is not None:
        logs = logs[date_time >= after]
    if before is not None:
        logs = logs[date_time <= before]
    logger.info("%d after filtering interactions" % len(logs))
    return logs.set_index("QuestionId")


def standard_date_format(s):
    """
    Convert from WEA's idiosyncratic string date format to the ISO standard
    :param s: WEA date
    :return: standard date
    """
    m = WEA_DATE_FORMAT.match(s).groupdict()
    return "%s-%s-%sT%s:%s:%sZ" % (m['year'], m['month'], m['day'], m['hour'], m['min'], m['sec'])


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=argparse.FileType(), help="user logs in QuestionsData.csv")
    parser.add_argument("--before", type=pandas.to_datetime, help="keep interactions before the specified date")
    parser.add_argument("--after", type=pandas.to_datetime, help="keep interactions after the specified date")
    parser.add_argument('--log', type=str, default="ERROR", help="logging level")
    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(message)s")
    filtered = filter_logs(args.logs, args.before, args.after)
    print(filtered.to_csv(encoding="utf-8"))
