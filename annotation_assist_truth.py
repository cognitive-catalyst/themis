#!/usr/bin/env python

"""Convert the ground truth file to the Annotation Assist format."""
import argparse

import pandas

QUESTION = "Question"
RESPONSE_MARKUP = "responseMarkup"
ID = "id"
ANSWER_ID = "AnswerId"


def convert_ground_truth(corpus_file, truth_file, n):
    corpus = pandas.read_csv(corpus_file, encoding="utf-8")
    truth = pandas.read_csv(truth_file, encoding="utf-8", nrows=n)
    corpus = corpus[[RESPONSE_MARKUP, ID]]
    corpus.rename(columns={RESPONSE_MARKUP: "ANS_LONG", ID: ANSWER_ID}, inplace=True)
    truth = pandas.merge(truth, corpus, on=ANSWER_ID)
    truth.rename(columns={QUESTION: "QUESTION"}, inplace=True)
    truth["ANS_SHORT"] = None
    truth["IS_ON_TOPIC"] = True
    truth = truth.drop([ANSWER_ID], axis="columns")
    truth.index.rename("QUESTION_ID", inplace=True)
    return truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=argparse.FileType(), help="corpus file to convert")
    parser.add_argument("ground_truth", type=argparse.FileType(), help="ground truth file to convert")
    parser.add_argument("--n", type=int, help="rows to load")
    args = parser.parse_args()

    print(convert_ground_truth(args.corpus, args.ground_truth, args.n).to_csv(encoding="utf-8"))
