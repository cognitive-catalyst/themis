import json
import os

import pandas

from themis import ensure_directory_exists, ANSWER, ANSWER_ID, TITLE, FILENAME, QUESTION, logger, CONFIDENCE, to_csv

QUESTION_TEXT = "Question_Text"
IS_IN_PURVIEW = "Is_In_Purview"
SYSTEM_ANSWER = "System_Answer"
ANNOTATION_SCORE = "Annotation_Score"
TOP_ANSWER_TEXT = "TopAnswerText"
TOP_ANSWER_CONFIDENCE = "TopAnswerConfidence"
DATE_TIME = "DateTime"


def create_annotation_assist_files(corpus, truth, answers, output_directory):
    annotation_assist_corpus = convert_corpus(corpus)
    annotation_assist_truth = convert_ground_truth(corpus, truth)
    annotation_assist_answers = convert_answers(corpus, answers)
    ensure_directory_exists(output_directory)
    with open(os.path.join(output_directory, "annotation_assist_corpus.json"), "w") as f:
        json.dump(annotation_assist_corpus, f, indent=2)
    to_csv(os.path.join(output_directory, "annotation_assist_truth.csv"), annotation_assist_truth)
    to_csv(os.path.join(output_directory, "annotation_assist_answers.csv"), annotation_assist_answers)


def convert_corpus(corpus):
    corpus = corpus.rename(columns={ANSWER: "text", ANSWER_ID: "pauId", TITLE: "title", FILENAME: "fileName"})
    corpus["splitPauTitle"] = corpus[TITLE].apply(lambda title: title.split(":"))
    return json.loads(corpus.to_json(orient="records"), encoding="utf-8")


def convert_ground_truth(corpus, truth):
    corpus = corpus[[ANSWER, ANSWER_ID]]
    corpus = corpus.rename(columns={ANSWER: "ANS_LONG"})
    truth = pandas.merge(truth, corpus, on=ANSWER_ID)
    truth = truth.rename(columns={QUESTION: "QUESTION"})
    truth["ANS_SHORT"] = None
    truth["IS_ON_TOPIC"] = True
    truth = truth.drop([ANSWER_ID], axis="columns")
    truth = truth.index.rename("QUESTION_ID")
    return truth


def convert_answers(corpus, systems):
    # Get a mapping of answer Ids to answers from the corpus.
    corpus = corpus[[ANSWER, ANSWER_ID]]
    systems = [pandas.merge(system, corpus, on=ANSWER_ID) for system in systems]
    # noinspection PyUnresolvedReferences
    systems = pandas.concat(systems).drop_duplicates([QUESTION, ANSWER_ID])
    # Add a dummy date time because the Annotation Assist README says that this column is required.
    systems[DATE_TIME] = "06052015:061049:UTC"
    logger.info("%d total Q&A pairs" % len(systems))
    systems = systems.rename(
        columns={QUESTION: QUESTION_TEXT, ANSWER: TOP_ANSWER_TEXT, CONFIDENCE: TOP_ANSWER_CONFIDENCE})
    return systems[[DATE_TIME, QUESTION_TEXT, TOP_ANSWER_TEXT, TOP_ANSWER_CONFIDENCE]]
