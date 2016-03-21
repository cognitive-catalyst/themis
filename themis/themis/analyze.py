import pandas

from themis import QUESTION, CORRECT, CsvFileType, IN_PURVIEW, ANSWER

# Annotation Assist column names
QUESTION_TEXT = "Question_Text"
IS_IN_PURVIEW = "Is_In_Purview"
SYSTEM_ANSWER = "System_Answer"
ANNOTATION_SCORE = "Annotation_Score"


def analyze(test_set, systems, judgements):
    """
    Analyze judged answers, collating results for different systems.

    :param test_set: questions and their frequencies
    :param systems: dictionary of system name to answers file
    :param judgements: human judgements of answer correctness, dataframe with (Answer, Correct) columns
    :return: collated dataframes of judged answers for all the systems
    """
    # questions = pandas.merge(judgements, test_set, on=QUESTION)
    judgements = judgements.set_index([QUESTION, ANSWER])
    fs = []
    for name in systems:
        s = systems[name]
        # The Annotation Assist tool strips newlines, so remove them from the answer text in the system output as well.
        s[ANSWER] = s[ANSWER].str.replace("\n", "")
        s = s.set_index([QUESTION, ANSWER])
        s = s.join(judgements)
        s = s.dropna()
        s.columns = pandas.MultiIndex.from_tuples([(name, c) for c in s.columns])
        fs.append(s)
    f = reduce(lambda m, f: m.join(f), fs)
    # Add frequency information from the test set.
    f = f.join(test_set.set_index(QUESTION))
    return f


class AnnotationAssistFileType(CsvFileType):
    def __init__(self):
        super(self.__class__, self).__init__([QUESTION_TEXT, IS_IN_PURVIEW, SYSTEM_ANSWER, ANNOTATION_SCORE],
                                             {QUESTION_TEXT: QUESTION, IS_IN_PURVIEW: IN_PURVIEW,
                                              SYSTEM_ANSWER: ANSWER})


def from_annotation_assist(annotation_assist_judgements, judgement_threshold):
    """
    Convert from the file format produced by
    `Annotation Assist <https://github.com/cognitive-catalyst/annotation-assist>`.

    :param annotation_assist_judgements: Annotation Assist file
    :param judgement_threshold: threshold above which an answer is deemed correct
    :return: dataframe with (Answer, Correct) columns
    """
    annotation_assist_judgements[IN_PURVIEW] = annotation_assist_judgements[IN_PURVIEW].astype("bool")
    annotation_assist_judgements[CORRECT] = annotation_assist_judgements[ANNOTATION_SCORE] >= judgement_threshold
    annotation_assist_judgements = annotation_assist_judgements.drop(ANNOTATION_SCORE, axis="columns")
    return annotation_assist_judgements[[QUESTION, ANSWER, IN_PURVIEW, CORRECT]]
