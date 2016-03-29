import json
import os

import pandas

from themis import ensure_directory_exists, ANSWER, ANSWER_ID, TITLE, FILENAME, QUESTION, logger, CONFIDENCE, to_csv, \
    CsvFileType, IN_PURVIEW, CORRECT

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


class AnnotationAssistFileType(CsvFileType):
    """
    Read the file produced by the `Annotation Assist <https://github.com/cognitive-catalyst/annotation-assist>` tool.
    """

    def __init__(self):
        super(self.__class__, self).__init__([QUESTION_TEXT, IS_IN_PURVIEW, SYSTEM_ANSWER, ANNOTATION_SCORE],
                                             {QUESTION_TEXT: QUESTION, IS_IN_PURVIEW: IN_PURVIEW,
                                              SYSTEM_ANSWER: ANSWER})

    def __call__(self, filename):
        annotation_assist = super(self.__class__, self).__call__(filename)
        annotation_assist[IN_PURVIEW] = annotation_assist[IN_PURVIEW].astype("bool")
        return annotation_assist[[QUESTION, ANSWER, IN_PURVIEW, ANNOTATION_SCORE]]


def add_judgements_and_frequencies_to_qa_pairs(qa_pairs, judgements, question_frequencies):
    # The Annotation Assist tool strips newlines, so remove them from the answer text in the system output as well.
    qa_pairs[ANSWER] = qa_pairs[ANSWER].str.replace("\n", "")
    qa_pairs = pandas.merge(qa_pairs, judgements, on=(QUESTION, ANSWER)).dropna()
    return pandas.merge(qa_pairs, question_frequencies, on=QUESTION)


def mark_annotation_assist_correct(annotation_assist, judgement_threshold):
    """
    Convert the annotation score column to a boolean correct column by applying a threshold.

    :param annotation_assist: Annotation Assist file
    :param judgement_threshold: threshold above which an answer is deemed correct
    :return: dataframe with a boolean Correct column
    """
    annotation_assist[CORRECT] = annotation_assist[ANNOTATION_SCORE] >= judgement_threshold
    return annotation_assist.drop(ANNOTATION_SCORE, axis="columns")
