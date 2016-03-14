#!/usr/bin/env python

"""Collate results from the Annotation Assist tool"""
import argparse

import pandas

QUESTION = "Question"
FREQUENCY = "Frequency"
ANSWER_ID = "AnswerId"
CONFIDENCE = "Confidence"
IN_PURVIEW = "InPurview"
JUDGEMENT = "Judgement"
ANSWER = "Answer"
# Corpus column names
RESPONSE_MARKUP = "responseMarkup"
ID = "id"
# Annotation Assist column names
QUESTION_TEXT = "Question_Text"
IS_IN_PURVIEW = "Is_In_Purview"
SYSTEM_ANSWER = "System_Answer"
ANNOTATION_SCORE = "Annotation_Score"


def collate_judgements(questions, corpus, answers, judgements):
    # Convert answer text in judgements to answer ID.
    corpus[ANSWER] = corpus[ANSWER].str.replace("\n", "")
    judgements = pandas.merge(judgements, corpus, on=ANSWER).drop(ANSWER, axis="columns")
    assert sum(judgements[ANSWER_ID].isnull()) == 0, "Some answers missing from the corpus"
    # Merge on (question, answer id) pairs.
    judgements = pandas.merge(answers, judgements, on=(QUESTION, ANSWER_ID))
    # Add question frequency information
    judgements = pandas.merge(judgements, questions, on=QUESTION)
    return judgements[[QUESTION, FREQUENCY, ANSWER_ID, CONFIDENCE, IN_PURVIEW, JUDGEMENT]]


class CsvFileType(object):
    def __init__(self, columns=None, rename=None):
        self.columns = columns
        self.rename = rename

    def __call__(self, filename):
        csv = pandas.read_csv(filename, usecols=self.columns, encoding="utf-8")
        if self.rename is not None:
            csv = csv.rename(columns=self.rename)
        return csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=CsvFileType([QUESTION, FREQUENCY]),
                        help="test set, e.g. as created by test_set_from_wea_logs.py")
    parser.add_argument("corpus", type=CsvFileType([ID, RESPONSE_MARKUP], {ID: ANSWER_ID, RESPONSE_MARKUP: ANSWER}),
                        help="corpus")
    parser.add_argument("answers", type=CsvFileType([QUESTION, ANSWER_ID, CONFIDENCE]),
                        help="answers file created by answer_questions.py")
    parser.add_argument("judgements",
                        type=CsvFileType([QUESTION_TEXT, IS_IN_PURVIEW, SYSTEM_ANSWER, ANNOTATION_SCORE],
                                         {QUESTION_TEXT: QUESTION, SYSTEM_ANSWER: ANSWER, IS_IN_PURVIEW: IN_PURVIEW,
                                          ANNOTATION_SCORE: JUDGEMENT}),
                        help="annotations returned by Annotation Assist")
    args = parser.parse_args()

    j = collate_judgements(args.questions, args.corpus, args.answers, args.judgements)
    print(j.to_csv(encoding="utf-8", index=False))
