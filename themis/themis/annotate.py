import json
import os

import pandas

from themis import ensure_directory_exists, ANSWER, ANSWER_ID, TITLE, FILENAME, QUESTION, logger, CONFIDENCE, to_csv, \
    CsvFileType, IN_PURVIEW, CORRECT

QUESTION_TEXT = "QuestionText"
IS_IN_PURVIEW = "Is_In_Purview"
SYSTEM_ANSWER = "System_Answer"
ANNOTATION_SCORE = "Annotation_Score"
TOP_ANSWER_TEXT = "TopAnswerText"
TOP_ANSWER_CONFIDENCE = "TopAnswerConfidence"
ANS_LONG = "ANS_LONG"
ANS_SHORT = "ANS_SHORT"
IS_ON_TOPIC = "IS_ON_TOPIC"


def create_annotation_assist_files(corpus, answers, output_directory):
    annotation_assist_corpus = convert_corpus(corpus)
    annotation_assist_answers = convert_answers(answers)
    ensure_directory_exists(output_directory)
    with open(os.path.join(output_directory, "annotation_assist_corpus.json"), "w") as f:
        json.dump(annotation_assist_corpus, f, indent=2)
    to_csv(os.path.join(output_directory, "annotation_assist_answers.csv"), annotation_assist_answers, index=False)


def convert_corpus(corpus):
    corpus["splitPauTitle"] = corpus[TITLE].apply(lambda title: title.split(":"))
    corpus = corpus.rename(columns={ANSWER: "text", ANSWER_ID: "pauId", TITLE: "title", FILENAME: "fileName"})
    return json.loads(corpus.to_json(orient="records"), encoding="utf-8")


def convert_answers(systems):
    systems = pandas.concat(systems).drop_duplicates([QUESTION, ANSWER])
    # noinspection PyTypeChecker
    logger.info("%d total Q&A pairs" % len(systems))
    systems = systems.rename(
        columns={QUESTION: QUESTION_TEXT, ANSWER: TOP_ANSWER_TEXT, CONFIDENCE: TOP_ANSWER_CONFIDENCE})
    systems = systems[[QUESTION_TEXT, TOP_ANSWER_TEXT, TOP_ANSWER_CONFIDENCE]]
    return systems.sort_values([QUESTION_TEXT, TOP_ANSWER_CONFIDENCE])


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
    qa_pairs = pandas.merge(qa_pairs, judgements, on=(QUESTION, ANSWER), how="left")
    missing = sum(qa_pairs[ANNOTATION_SCORE].isnull())
    if missing:
        logger.warn("%d unannotated Q&A pairs out of %d" % (missing, len(qa_pairs)))
    qa_pairs = qa_pairs.dropna()
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
