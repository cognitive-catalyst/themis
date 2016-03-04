#!/usr/bin/env python

"""Convert the corpus file to the Annotation Assist format."""

import argparse
import json

import pandas


def convert_corpus(corpus_file, n):
    corpus = pandas.read_csv(corpus_file, encoding="utf-8", nrows=n)
    corpus.rename(columns={"id": "pauId", "responseMarkup": "text", "sourceName": "fileName"}, inplace=True)
    corpus["splitPauTitle"] = corpus["title"].apply(lambda title: title.split(":"))
    corpus.drop(["answeredByPau", "normalizedScore", "sourceDocUrl", "score", "title"], axis="columns", inplace=True)
    return corpus.to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=argparse.FileType(), help="corpus file to convert")
    parser.add_argument("--n", type=int, help="rows to load")
    args = parser.parse_args()

    corpus = convert_corpus(args.corpus, args.n)
    print(json.dumps(json.loads(corpus, encoding="utf-8"), None, indent=2))
