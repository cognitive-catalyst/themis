#!/usr/bin/env python

"""Download all PAUs from an XMGR instance and print them as a CSV document to a specified output file.
"""

import argparse
import itertools
import json
import logging
import os

import pandas
import requests

logger = logging.getLogger(__name__)


def get_paus_from_xmgr(xmgr, output, limit=None, interval=None):
    # Get the IDs of the documents in the corpus.
    if limit is not None:
        # Requesting items=0-0 returns all the documents. This appears to be a bug in XMGR.
        if limit > 1:
            limit -= 1
        headers = {"Range": "items=0-%d" % limit}
    else:
        headers = None
    document_ids = set([document["id"] for document in xmgr.get('xmgr/corpus/document', headers=headers)])
    logger.info("%d documents" % len(document_ids))
    # Download the individual PAUs in the documents. Checkpoint periodically so that we don't
    # lose everything if the connection drops in the middle of the run.
    checkpoint_file = "%s.temp.json" % output
    checkpoint = IndexedJsonCheckpoint(checkpoint_file, interval, "document_id")
    # Don't redownload the PAUs in a document that is already in our checkpoint file.
    saved_document_ids = set(checkpoint.saved_indexes())
    skip_document_ids = document_ids & saved_document_ids
    if skip_document_ids:
        logger.info("Skipping %d documents already downloaded" % len(skip_document_ids))
    document_ids = sorted(document_ids - saved_document_ids)
    for document_id in document_ids:
        logger.info("Document %d of %d" % (document_ids.index(document_id) + 1, len(document_ids)))
        paus = get_paus_from_document(xmgr, document_id)
        checkpoint.write({"document_id": document_id, "paus": paus})
    checkpoint.flush()
    # Write a CSV file
    documents = checkpoint.load()
    paus = flatten(document_paus["paus"] for document_paus in documents)
    logger.info("%d paus, %d documents" % (len(paus), len(documents)))
    paus = pandas.DataFrame(paus)
    headers = paus.columns.tolist()
    # Ensure that the id column is first.
    headers.remove("id")
    headers.insert(0, "id")
    paus = paus[headers]
    os.remove(checkpoint_file)
    return paus


def get_paus_from_document(xmgr, document_id):
    trec_document = xmgr.get("xmgr/corpus/wea/trec", {"srcDocId": document_id})
    pau_ids = [item["DOCNO"] for item in trec_document["items"]]
    logger.info("Document %s, %d PAUs" % (document_id, len(pau_ids)))
    if not len(pau_ids) == len(set(pau_ids)):
        logger.warning("Document %s contains duplicate PAUs" % document_id)
    return flatten(xmgr.get(os.path.join('wcea/api/GroundTruth/paus/', pau_id))["hits"] for pau_id in pau_ids)


class Xmgr(object):
    def __init__(self, project_url, username, password):
        self.project_url = project_url
        self.username = username
        self.password = password

    def __repr__(self):
        return "XMGR: %s" % self.project_url

    def get(self, path, params=None, headers=None):
        url = os.path.join(self.project_url + "/", path)
        r = requests.get(url, auth=(self.username, self.password), params=params, headers=headers)
        logger.debug("GET %s\t%d" % (url, r.status_code))
        return r.json()


class IndexedJsonCheckpoint(object):
    """A indexed list of JSON objects in a file

    This object manages a list of JSON objects indexed by a specified property. At
    periodic intervals, the buffer is flushed to a file. If the file already exists,
    the buffer contents are appended to it.
    """

    def __init__(self, filename, interval, index):
        self.filename = filename
        self.interval = interval
        self.index = index
        self.buffer = {}

    def write(self, item):
        index = item[self.index]
        if index in self.buffer:
            raise Exception("Duplicated item %s" % index)
        self.buffer[index] = item
        if self.interval is not None and len(self.buffer) % self.interval == 0:
            self.flush()

    def flush(self):
        logger.debug("Flush %d items in buffer" % len(self.buffer))
        if self.buffer:
            current = self.load()
            current_indexes = set([item[self.index] for item in current])
            duplicate_indexes = set(self.buffer.keys()) & current_indexes
            if duplicate_indexes:
                raise Exception("Duplicated items %s" % duplicate_indexes)
            if os.path.exists(self.filename):
                os.remove(self.filename)
            new_items = self.buffer.values()
            current += new_items
            with open(self.filename, "w") as f:
                json.dump(current, f)
            logger.debug("Wrote %d new items to %s" % (len(new_items), self.filename))
        self.buffer = {}

    def load(self):
        stored = []
        if os.path.exists(self.filename):
            with open(self.filename) as f:
                stored += json.load(f)
            logger.debug("Loaded %d items from %s" % (len(stored), self.filename))
        return stored

    def saved_indexes(self):
        return [item[self.index] for item in self.load()]


def flatten(items):
    return list(itertools.chain.from_iterable(items))


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="""
This script periodically saves downloaded PAUs to a temporary file. If the download fails in the
middle, you can rerun the same command and it will pick up where it left off. The temporary
file will be created with the name of your output file followed by .temp.json. After the entire
corpus has been downloaded the temporary file is deleted.""")
    parser.add_argument("xmgr", type=str, help="XMGR url")
    parser.add_argument("username", type=str, help="XMGR username")
    parser.add_argument("password", type=str, help="XMGR password")
    parser.add_argument("output", type=str, help="output CSV file")
    parser.add_argument("--interval", type=int, default=50,
                        help="PAU checkpoint interval, no checkpointing if less than 1")
    parser.add_argument("--limit", type=int, help="limit number of documents downloaded")
    parser.add_argument('--log', type=str, default="ERROR", help="logging level")
    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(message)s")
    xmgr = Xmgr(args.xmgr, args.username, args.password)
    if args.interval < 1:
        args.interval = None
    paus = get_paus_from_xmgr(xmgr, args.output, args.limit, args.interval)
    paus.to_csv(args.output, encoding="utf-8", index=False)
