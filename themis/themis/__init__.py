import logging

import pandas

logger = logging.getLogger(__name__)

QUESTION = "Question"
ANSWER = "Answer"
ANSWER_ID = "Answer Id"
CONFIDENCE = "Confidence"
FREQUENCY = "Frequency"


def to_csv(filename, dataframe, **kwargs):
    dataframe.to_csv(filename, encoding="utf-8", **kwargs)


def print_csv(dataframe, **kwargs):
    print(dataframe.to_csv(encoding="utf-8", **kwargs))


class CsvFileType(object):
    """Pandas CSV file type used with argparse

    This allows you to specify the columns you wish to use and optionally rename them.
    """

    def __init__(self, columns=None, rename=None):
        self.columns = columns
        self.rename = rename

    def __call__(self, filename):
        csv = pandas.read_csv(filename, usecols=self.columns, encoding="utf-8")
        if self.rename is not None:
            csv = csv.rename(columns=self.rename)
        return csv


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
        logger.debug("Flush %d items to disk" % len(self.buffer))
        self.buffer.to_csv(self.output_file, header=self.need_header, index=False, encoding="utf-8")
        self.output_file.flush()
        self.buffer = pandas.DataFrame(columns=self.columns)
        self.need_header = False


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)
