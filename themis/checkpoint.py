"""
Checkpointing provides a framework for writing intermediary results of long-running operations to disk so that they can
resume where they left off if they fail in the middle.
"""
import time

import pandas

from themis import logger, percent_complete_message


def get_items(item_type, names, checkpoint, get_item, write_frequency):
    recovered = checkpoint.recovered
    if recovered:
        logger.info("Recovered %d %s from previous run" % (len(recovered), item_type))
    total = len(names)
    start = 1 + len(recovered)
    try:
        names_to_get = sorted(set(names) - recovered)
        for i, name in enumerate(names_to_get, start):
            if i == start or i == total or i % write_frequency == 0:
                logger.info("Get " + percent_complete_message(item_type, i, total))
            item = get_item(name)
            checkpoint.write(name, item)
    finally:
        checkpoint.close()
    return checkpoint


class DataFrameCheckpoint(object):
    def __init__(self, output_filename, columns, interval=None):
        try:
            recovered = pandas.read_csv(open(output_filename), usecols=[0], encoding="utf-8")
            self.recovered = set(recovered[recovered.columns[0]])
            self.need_header = False
            logger.debug("Recovered %d items from disk" % len(self.recovered))
        except IOError:
            self.recovered = set()
            self.need_header = True
        except ValueError:
            raise Exception("Cannot recover data from %s" % output_filename)
        self.output_file = open(output_filename, "a")
        self.columns = columns
        self.buffer = pandas.DataFrame(columns=self.columns)
        self.interval = interval

    def write(self, *values):
        self.buffer = self.buffer.append(dict(zip(self.buffer.columns, values)), ignore_index=True)
        if self.interval is not None and len(self.buffer) % self.interval is 0:
            self.flush()

    def close(self):
        self.flush()
        self.output_file.close()

    def flush(self):
        logger.debug("Flush %d items to %s" % (len(self.buffer), self.output_file.name))
        self.buffer.to_csv(self.output_file, header=self.need_header, index=False, encoding="utf-8")
        self.output_file.flush()
        self.buffer = pandas.DataFrame(columns=self.columns)
        self.need_header = False


def retry(function, times):
    """
    Retry a function call that may fail a specified number of times.

    This attempts to call the function a specified number of times. If the function throws an exception, sleep for a
    minute and try again until we have made the specified number of attempts.

    If None is passed for the number of times, just try once and throw any exception that occurs.

    :param function: a function to be called
    :type function: function
    :param times: the number of times to call the function before giving up
    :type times: int
    """
    if times is None:
        function()
    else:
        assert times > 0
        try:
            function()
        except Exception as e:
            logger.info("Error %s" % e)
            times -= 1
            if times:
                logger.info("Retry %d more times" % times)
                time.sleep(60)
                retry(function, times)
            else:
                logger.info("Done retrying")
