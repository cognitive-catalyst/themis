import functools
import itertools
import json
import math
import os
import os.path
import tempfile
import textwrap

import numpy as np
import pandas
from bs4 import BeautifulSoup
from nltk import FreqDist, word_tokenize
from watson_developer_cloud import \
    NaturalLanguageClassifierV1 as NaturalLanguageClassifier

from themis import (ANSWER, ANSWER_ID, CONFIDENCE, CORRECT, FREQUENCY,
                    IN_PURVIEW, QUESTION, CsvFileType, ensure_directory_exists,
                    logger, percent_complete_message, pretty_print_json,
                    to_csv)
from themis.checkpoint import DataFrameCheckpoint
from themis.metrics import (__standardize_confidence, confidence_thresholds,
                            precision, questions_attempted)
from themis.nlc import NLC, classifier_status

SYSTEM = "System"
ANSWERING_SYSTEM = "Answering System"
NLC_ROUTER_FOLDS = 8


def corpus_statistics(corpus):
    """
    Generate statistics for the corpus.

    :param corpus: corpus generated by 'xmgr corpus' command
    :type corpus: pandas.DataFrame
    :return: answers in corpus, tokens in the corpus, histogram of answer length in tokens
    :rtype: (int, int, dict(int, int))
    """
    answers = len(corpus)
    token_frequency = FreqDist([len(word_tokenize(BeautifulSoup(answer, "lxml").text)) for answer in corpus[ANSWER]])
    histogram = {}
    for frequency, count in token_frequency.items():
        histogram[frequency] = histogram.get(frequency, 0) + count
    tokens = sum(token_frequency.keys())
    n = sum(corpus.duplicated(ANSWER_ID))
    if n:
        logger.warning("%d duplicated answer IDs (%0.3f%%)" % (n, 100.0 * n / answers))
    return answers, tokens, histogram


def truth_statistics(truth):
    """
    Generate statistics for the truth.

    :param truth: question to answer mapping used in training
    :type truth: pandas.DataFrame
    :return: number of training pairs, number of unique questions, number of unique answers, histogram of number of
            questions per answer
    :rtype: (int, int, int, pandas.DataFrame)
    """
    pairs = len(truth)
    questions = len(truth[QUESTION].unique())
    answers = len(truth[ANSWER_ID].unique())
    question_histogram = truth[[ANSWER_ID, QUESTION]].groupby(ANSWER_ID).count()
    return pairs, questions, answers, question_histogram


def system_similarity(systems_data):
    """
    For each system pair, return the number of questions they answered the same.

    :param systems_data: collated results for all systems
    :type systems_data: pandas.DataFrame
    :return: table of pairs of systems and their similarity statistics
    :rtype: pandas.DataFrame
    """
    systems_data = drop_missing(systems_data)
    systems = systems_data[SYSTEM].drop_duplicates().sort_values()
    columns = ["System 1", "System 2", "Same Answer", "Same Answer %"]
    results = pandas.DataFrame(columns=columns)
    for x, y in itertools.combinations(systems, 2):
        data_x = systems_data[systems_data[SYSTEM] == x]
        data_y = systems_data[systems_data[SYSTEM] == y]
        m = pandas.merge(data_x, data_y, on=QUESTION)
        n = len(m)
        logger.info("%d question/answer pairs in common for %s and %s" % (n, x, y))
        same_answer = sum(m["%s_x" % ANSWER] == m["%s_y" % ANSWER])
        same_answer_pct = 100.0 * same_answer / n
        results = results.append(
            pandas.DataFrame([[x, y, same_answer, same_answer_pct]], columns=columns))
    results["Same Answer"] = results["Same Answer"].astype("int64")
    return results.set_index(["System 1", "System 2"])


def compare_systems(systems_data, x, y, comparison_type):
    """
    On which questions did system x do better or worse than system y?

    System x did better than system y if it correctly answered a question when system y did not, and vice versa.

    :param systems_data: collated results for all systems
    :type systems_data: pandas.DataFrame
    :param x: system name
    :type x: str
    :param y: system name
    :type y: str
    :param comparison_type: "better" or "worse"
    :type comparison_type: str
    :return: all question/answer pairs from system x that were either better or worse than system y
    :rtype: pandas.DataFrame
    """

    def col_name(type, system):
        return type + " " + system

    systems_data = drop_missing(systems_data)
    systems_data = systems_data[systems_data[IN_PURVIEW]]
    data_x = systems_data[systems_data[SYSTEM] == x]
    data_y = systems_data[systems_data[SYSTEM] == y][[QUESTION, ANSWER, CONFIDENCE, CORRECT]]
    questions = pandas.merge(data_x, data_y, on=QUESTION, how="left", suffixes=(" " + x, " " + y)).dropna()
    n = len(questions)
    logger.info("%d shared question/answer pairs between %s and %s" % (n, x, y))
    x_correct = col_name(CORRECT, x)
    y_correct = col_name(CORRECT, y)
    if comparison_type == "better":
        a = questions[x_correct] == True
        b = questions[y_correct] == False
    elif comparison_type == "worse":
        a = questions[x_correct] == False
        b = questions[y_correct] == True
    else:
        raise ValueError("Invalid comparison type %s" % comparison_type)
    d = questions[a & b]
    m = len(d)
    logger.info("%d %s (%0.3f%%)" % (m, comparison_type, 100.0 * m / n))
    d = d[[QUESTION, FREQUENCY,
           col_name(ANSWER, x), col_name(CONFIDENCE, x), col_name(ANSWER, y), col_name(CONFIDENCE, y)]]
    d = d.sort_values([col_name(CONFIDENCE, x), FREQUENCY, QUESTION], ascending=(False, False, True))
    return d.set_index(QUESTION)


def analyze_answers(systems_data, freq_le, freq_gr):
    """
    Statistics about all the answered questions in a test set broken down by system.

    :param systems_data: collated results for all systems
    :type systems_data: pandas.DataFrame
    :param freq_gr: optionally only consider questions with frequency greater than this
    :type freq_gr: int
    :param freq_le: optionally only consider questions with frequency less than or equal to this
    :type freq_le: int
    :return: answer summary statistics
    :rtype: pandas.DataFrame
    """
    total = "Total"
    in_purview_percent = IN_PURVIEW + " %"
    correct_percent = CORRECT + " %"
    unique = "Unique"
    systems_data = pandas.concat(systems_data).dropna()
    if freq_le is not None:
        systems_data = systems_data[systems_data[FREQUENCY] <= freq_le]
    if freq_gr is not None:
        systems_data = systems_data[systems_data[FREQUENCY] > freq_gr]
    systems = systems_data.groupby(SYSTEM)
    summary = systems[[IN_PURVIEW, CORRECT]].sum()
    summary[[IN_PURVIEW, CORRECT]] = summary[[IN_PURVIEW, CORRECT]].astype("int")
    summary[total] = systems.count()[QUESTION]
    summary[unique] = systems[ANSWER].nunique()
    summary[in_purview_percent] = summary[IN_PURVIEW] / summary[total] * 100.0
    summary[correct_percent] = summary[CORRECT] / summary[IN_PURVIEW] * 100.0
    return summary.sort_values(correct_percent, ascending=False)[
        [total, unique, IN_PURVIEW, in_purview_percent, CORRECT, correct_percent]]


def truth_coverage(corpus, truth, systems_data):
    """
    Statistics about which answers came from the truth set broken down by system.

    :param corpus: corpus generated by 'xmgr corpus' command
    :type corpus: pandas.DataFrame
    :param truth: question to answer mapping used in training
    :type truth: pandas.DataFrame
    :param systems_data: collated results for all systems
    :type systems_data: pandas.DataFrame
    :return: truth coverage summary statistics
    :rtype: pandas.DataFrame
    """
    truth_answers = pandas.merge(corpus, truth, on=ANSWER_ID)[ANSWER].drop_duplicates()
    n = len(corpus)
    m = len(truth_answers)
    logger.info("%d answers out of %d possible answers in truth (%0.3f%%)" % (m, n, 100.0 * m / n))
    systems_data = pandas.concat(systems_data).dropna()
    answers = systems_data.groupby(SYSTEM)[[CORRECT]].count()
    answers_in_truth = systems_data[systems_data[ANSWER].isin(truth_answers)].groupby(SYSTEM)[[ANSWER]]
    summary = answers_in_truth.count()
    summary["Answers"] = answers
    summary = summary.rename(columns={ANSWER: "Answers in Truth"})
    summary["Answers in Truth %"] = 100 * summary["Answers in Truth"] / summary["Answers"]
    correct_answers = systems_data[systems_data[CORRECT]]
    correct_answers_in_truth = correct_answers[correct_answers[ANSWER].isin(truth_answers)]
    summary["Correct Answers"] = correct_answers.groupby(SYSTEM)[CORRECT].count()
    summary["Correct Answers in Truth"] = correct_answers_in_truth.groupby(SYSTEM)[CORRECT].count()
    summary["Correct Answers in Truth %"] = 100 * summary["Correct Answers in Truth"] / summary["Correct Answers"]
    return summary[
        ["Answers", "Correct Answers",
         "Answers in Truth", "Answers in Truth %",
         "Correct Answers in Truth", "Correct Answers in Truth %"]].sort_values("Correct Answers", ascending=False)


# noinspection PyTypeChecker
def long_tail_fat_head(frequency_cutoff, systems_data):
    """
    Accuracy statistics broken down by question "fat head" and "long tail".

    The fat head is defined to be all questions with a frequency above a cutoff value. The long tail is defined to be
    all questions with a frequency below that value.

    :param frequency_cutoff: question frequency dividing fat head from long tail
    :type frequency_cutoff: int
    :param systems_data: collated results for all systems
    :type systems_data: pandas.DataFrame
    :return: truth coverage summary statistics for the fat head and long tail
    :rtype: (pandas.DataFrame, pandas.DataFrame)
    """
    fat_head = analyze_answers(systems_data, None, frequency_cutoff)
    long_tail = analyze_answers(systems_data, frequency_cutoff, None)
    return fat_head, long_tail


def in_purview_disagreement(systems_data):
    """
    Return collated data where in-purview judgments are not unanimous for a question.

    These questions' purview should be rejudged to make them consistent.

    :param systems_data: collated results for all systems
    :type systems_data: pandas.DataFrame
    :return: subset of collated data where the purview judgments are not unanimous for a question
    :rtype: pandas.DataFrame
    """
    question_groups = systems_data[[QUESTION, IN_PURVIEW]].groupby(QUESTION)
    index = question_groups.filter(lambda qg: len(qg[IN_PURVIEW].unique()) > 1).index
    purview_disagreement = systems_data.loc[index]
    m = len(purview_disagreement[QUESTION].drop_duplicates())
    if m:
        n = len(systems_data[QUESTION].drop_duplicates())
        logger.warning("%d out of %d questions have non-unanimous in-purview judgments (%0.3f%%)"
                       % (m, n, 100.0 * m / n))
    return purview_disagreement


def _get_in_purview_judgment(question):
    judgment = raw_input(textwrap.dedent("""
    ******** JUDGE THE PURVIEW OF THE FOLLOWING QUESTION ********
    QUESTION:  {0}
    (1) IN PURVIEW
    (2) OUT OF PURVIEW

    YOUR JUDGMENT: """).format(question))

    if judgment == '1':
        return True
    elif judgment == '2':
        return False
    else:
        return _get_in_purview_judgment(question)


def _judge_answer(row):
    judgment = raw_input(textwrap.dedent("""
    ******** JUDGE THE ANSWER TO THE FOLLOWING QUESTION ********
    QUESTION:  {0}
    ANSWER:    {1}
    (1) CORRECT
    (2) INCORRECT

    YOUR JUDGMENT: """).format(row[1][QUESTION], row[1][ANSWER]))

    if judgment == '1':
        return True
    elif judgment == '2':
        return False
    else:
        return _judge_answer(row)


def in_purview_disagreement_evaluate(systems_data, output_file):

    purview_disagreement = in_purview_disagreement(systems_data)
    questions_to_judge = purview_disagreement[QUESTION].unique()
    for question in questions_to_judge:

        purview_judgment = _get_in_purview_judgment(question)

        current_question_rows = systems_data[systems_data[QUESTION] == question]
        for row in current_question_rows.iterrows():
            index = row[0]
            original_judgment = row[1]["In Purview"]

            if purview_judgment != original_judgment:
                if purview_judgment:
                    systems_data.ix[index, IN_PURVIEW] = True
                    systems_data.ix[index, CORRECT] = _judge_answer(row)
                else:
                    systems_data.ix[index, IN_PURVIEW] = False
                    systems_data.ix[index, CORRECT] = False
            to_csv(output_file, systems_data, index=False)
        # break
    # print systems_data[systems_data[QUESTION] == question
    return systems_data


def oracle_combination(systems_data, system_names, oracle_name):
    """
    Combine results from multiple systems into a single oracle system. The oracle system gets a question correct if any
    of its component systems did. If the answer is correct use the highest confidence. If it is incorrect, use the
    lowest confidence.

    (A question is in purview if judgments from all the systems say it is in purview. These judgments should be
    unanimous. The 'themis analyze purview' command finds when this is not the case.)

    :param systems_data: collated results for all systems
    :type systems_data: pandas.DataFrame
    :param system_names: names of systems to combine
    :type system_names: list of str
    :param oracle_name: the name of the combined system
    :type oracle_name: str
    :return: oracle results in collated format
    :rtype: pandas.DataFrame
    """
    def log_correct(system_data, name):
        n = len(system_data)
        m = sum(system_data[CORRECT])
        logger.info("%d of %d correct in %s (%0.3f%%)" % (m, n, name, 100.0 * m / n))

    percentile = "Percentile"
    systems_data = drop_missing(systems_data)
    # Extract the systems of interest and map confidences to percentile rank.
    systems = []
    for system_name in system_names:
        system = systems_data[systems_data[SYSTEM] == system_name].set_index(QUESTION)
        system[percentile] = __standardize_confidence(system)
        log_correct(system, system_name)
        systems.append(system)
    # Get the questions asked to all the systems.
    questions = functools.reduce(lambda m, i: m.intersection(i), (system.index for system in systems))
    # Start the oracle with a copy of one of the systems.
    oracle = systems[0].loc[questions].copy()
    oracle = oracle.drop([ANSWER, percentile], axis="columns")
    oracle[SYSTEM] = oracle_name
    # An oracle question is in purview if all systems mark it as in purview. There should be consensus on this.
    systems_in_purview = [system.loc[questions][[IN_PURVIEW]] for system in systems]
    oracle[[IN_PURVIEW]] = functools.reduce(lambda m, x: m & x, systems_in_purview)
    # An oracle question is correct if any system gets it right.
    systems_correct = [system.loc[questions][[CORRECT]] for system in systems]
    oracle[[CORRECT]] = functools.reduce(lambda m, x: m | x, systems_correct)
    # If the oracle answer is correct, use the highest confidence.
    confidences = [system[[percentile]].rename(columns={percentile: system[SYSTEM][0]}) for system in systems]
    system_confidences = functools.reduce(lambda m, x: pandas.merge(m, x, left_index=True, right_index=True),
                                          confidences)
    correct = oracle[CORRECT].astype("bool")
    oracle.loc[correct, CONFIDENCE] = system_confidences[correct].max(axis=1)
    oracle.loc[correct, ANSWERING_SYSTEM] = system_confidences[correct].idxmax(axis=1)
    # If the question is out of purview or the answer is incorrect, use the lowest confidence.
    oracle.loc[~correct, CONFIDENCE] = system_confidences[~correct].min(axis=1)
    oracle.loc[~correct, ANSWERING_SYSTEM] = system_confidences[~correct].idxmin(axis=1)
    # Use the answer produced by the system incorporated into the oracle.
    oracle = oracle.reset_index()
    oracle[ANSWER] = pandas.merge(systems_data, oracle,
                                  left_on=[QUESTION, SYSTEM], right_on=[QUESTION, ANSWERING_SYSTEM])[ANSWER]
    log_correct(oracle, oracle_name)
    return oracle


def _create_combined_fallback_system_at_threshold(default_systems_data, secondary_system_data, threshold):
    """
    Combine results from two systems into a single fallback system. The default system will answer the question if
    the confidence is above the given threshold.

    :param default_systems_data: collated results for the default system (if confidence > t)
    :type default_systems_data: pandas.DataFrame
    :param secondary_system_data: collated results for the secondary system (if default_confidence < t)
    :type default_system: pandas.DataFrame
    :param threshold: (t) the confidence threshold to determine what system answers the question
    :type secondary_system: float
    :return: Fallback results in collated format
    :rtype: pandas.DataFrame
    """
    percentile = 'Percentile'
    default_systems_data[percentile] = __standardize_confidence(default_systems_data)
    secondary_system_data[percentile] = __standardize_confidence(secondary_system_data)

    combined_from_default = default_systems_data[default_systems_data[CONFIDENCE] >= threshold]
    combined_from_secondary = secondary_system_data[~secondary_system_data[QUESTION].isin(combined_from_default[QUESTION])]

    combined = pandas.concat([combined_from_default, combined_from_secondary])
    combined[CONFIDENCE] = combined[percentile]
    combined.drop([percentile], axis="columns")
    return combined


def fallback_combination(systems_data, default_system, secondary_system):
    """
    Combine results from two systems into a single fallback system. The default system will answer the question if
    the confidence is above a certain threshold. This method will find the optimal confidence threshold.

    :param systems_data: collated results for the input systems
    :type systems_data: pandas.DataFrame
    :param default_system: the name of the default system (if confidence > t)
    :type default_system: str
    :param secondary_system: the name of the fallback system (if default_confidence < t)
    :type secondary_system: str
    :return: Fallback results in collated format
    :rtype: pandas.DataFrame
    """
    default_system_data = systems_data[systems_data[SYSTEM] == default_system]
    secondary_system_data = systems_data[systems_data[SYSTEM] == secondary_system]

    intersecting_questions = set(default_system_data[QUESTION]).intersection(set(secondary_system_data[QUESTION]))

    logger.warn("{0} questions in default system".format(len(default_system_data)))
    logger.warn("{0} questions in secondary system".format(len(secondary_system_data)))
    logger.warn("{0} questions in overlapping set".format(len(intersecting_questions)))

    default_system_data = default_system_data[default_system_data[QUESTION].isin(intersecting_questions)]
    secondary_system_data = secondary_system_data[secondary_system_data[QUESTION].isin(intersecting_questions)]

    unique_confidences = default_system_data[CONFIDENCE].unique()

    best_threshold, best_precision = 0, 0
    for threshold in unique_confidences:
        combined_system = _create_combined_fallback_system_at_threshold(default_system_data, secondary_system_data, threshold)

        system_precision = precision(combined_system, 0)
        if system_precision > best_precision:
            best_precision = system_precision
            best_threshold = threshold

    logger.info("Default system accuracy:   {0}%".format(str(precision(default_system_data, 0) * 100)[:4]))
    logger.info("Secondary system accuracy: {0}%".format(str(precision(secondary_system_data, 0) * 100)[:4]))
    logger.info("Combined system accuracy:  {0}%".format(str(best_precision * 100)[:4]))

    logger.info("Combined system best threshold: {0}".format(best_threshold))

    best_system = _create_combined_fallback_system_at_threshold(default_system_data, secondary_system_data, best_threshold)
    best_system[ANSWERING_SYSTEM] = best_system[SYSTEM]
    best_system[SYSTEM] = "{0}_FALLBACK_{1}_AT_{2}".format(default_system, secondary_system, str(best_threshold)[:4])

    logger.info("Questions answered by {0}: {1}%".format(default_system, str(100 * float(len(best_system[best_system[ANSWERING_SYSTEM] == default_system])) / len(best_system))[:4]))

    best_system[CONFIDENCE] = __standardize_confidence(best_system)
    return best_system


def voting_router(systems_data, system_names, voting_name):
    """
    Combine results from multiple systems into a single that uses voting to decide which system should answer.

    :param systems_data: collated results for all systems.
    :type systems_data: pandas.DataFrame
    :param system_names: names of systems to combine
    :type system_names: list of str
    :param voting_name: the name of the combined system
    :type voting_name: str
    :return: voting system results in collated format
    :rtype: pandas.DataFrame
    """
    def log_correct(system_data, name):
        n = len(system_data)
        m = sum(system_data[CORRECT])
        logger.info("%d of %d correct in %s (%0.3f%%)" % (m, n, name, 100.0 * m / n))

    systems_data = drop_missing(systems_data)
    systems = []
    for system_name in system_names:
        system = systems_data[systems_data[SYSTEM] == system_name].set_index(QUESTION)
        log_correct(system, system_name)
        systems.append(system)

        # Calculate the precision and question_attempted for each threshold, use it for standardized confidences
        ts = confidence_thresholds(system, False)
        ps = [precision(system, t) for t in ts]
        qas = [questions_attempted(system, t) for t in ts]
        system['pgc'] = __standardize_confidence(system, method='precision')
        # system.apply(lambda x: precision_grounded_confidence(ts, ps, qas, x[CONFIDENCE],
        #                                                                 method='precision_only'), axis=1)

    # Get the questions asked to all the systems.
    questions = functools.reduce(lambda m, i: m.intersection(i), (system.index for system in systems))
    # Start the voting results with a copy of one of the systems using only the intersecting questions
    voting = systems[0].loc[questions].copy()
    voting = voting.drop([ANSWER, CONFIDENCE, "pgc", CORRECT], axis="columns")
    voting[SYSTEM] = voting_name

    # Find the best precision grounded confidences to find the top system.
    pgcs = [system[['pgc']].rename(columns={'pgc': system[SYSTEM][0]}) for system in systems]
    system_pgcs = functools.reduce(lambda m, x: pandas.merge(m, x, left_index=True, right_index=True), pgcs)
    rows = [True for x in range(0, len(voting))]
    voting.loc[rows, ANSWERING_SYSTEM] = system_pgcs[rows].idxmax(axis=1)
    voting.loc[rows, CONFIDENCE] = system_pgcs[rows].max(axis=1)
    #voting.loc[rows, 'PGC'] = system_pgcs[rows].max(axis=1)

    voting = voting.reset_index()
    voting[ANSWER] = pandas.merge(systems_data, voting,
                                  left_on=[QUESTION, SYSTEM], right_on=[QUESTION, ANSWERING_SYSTEM])[ANSWER]
    voting[CORRECT] = pandas.merge(systems_data, voting,
                                   left_on=[QUESTION, SYSTEM], right_on=[QUESTION, ANSWERING_SYSTEM])[CORRECT]
    log_correct(voting, voting_name)
    return voting


def filter_judged_answers(systems_data, correct, system_names):
    """
    Filter out just the correct or incorrect in-purview answers.

    :param systems_data: questions, answers, and judgments across systems
    :type systems_data: list of pandas.DataFrame
    :param correct: filter correct or incorrect answers?
    :type correct: bool
    :param system_names: systems to filter to, if None show all systems
    :type system_names: list of str
    :return: set of in-purview questions with answers judged either correct or incorrect
    :rtype: pandas.DataFrame
    """
    systems_data = pandas.concat(systems_data).dropna()
    if system_names is not None:
        systems_data = systems_data[systems_data[SYSTEM].isin(system_names)]
    filtered = systems_data[(systems_data[IN_PURVIEW] == True) & (systems_data[CORRECT] == correct)]
    n = len(systems_data)
    m = len(filtered)
    logger.info("%d in-purview %s answers out of %d (%0.3f%%)" %
                (m, {True: "correct", False: "incorrect"}[correct], n, 100 * m / n))
    return filtered


def add_judgments_and_frequencies_to_qa_pairs(qa_pairs, judgments, question_frequencies, remove_newlines):
    """
    Collate system answer confidences and annotator judgments by question/answer pair.
    Add to each pair the question frequency. Collated system files are used as input to subsequent cross-system
    analyses.

    Though you expect the set of question/answer pairs in the system answers and judgments to not be disjoint, it may
    be the case that neither is a subset of the other. If annotation is incomplete, there may be Q/A pairs in the
    system answers that haven't been annotated yet. If multiple systems are being judged, there may be Q/A pairs in the
    judgements that don't appear in the system answers.

    Some versions of Annotation Assist strip newlines from the answers they return in the judgement files, so
    optionally take this into account when joining on question/answer pairs.

    :param qa_pairs: question, answer, and confidence provided by a Q&A system
    :type qa_pairs: pandas.DataFrame
    :param judgments: question, answer, in purview, and judgement provided by annotators
    :type judgments: pandas.DataFrame
    :param question_frequencies: question and question frequency in the test set
    :type question_frequencies: pandas.DataFrame
    :param remove_newlines: join judgments on answers with newlines removed
    :type remove_newlines: bool
    :return: question and answer pairs with confidence, in purview, judgement and question frequency
    :rtype: pandas.DataFrame
    """
    qa_pairs = pandas.merge(qa_pairs, question_frequencies, on=QUESTION, how="left")
    if remove_newlines:
        qa_pairs["Temp"] = qa_pairs[ANSWER].str.replace("\n", "")
        qa_pairs = qa_pairs.rename(columns={"Temp": ANSWER, ANSWER: "Temp"})

        judgments[ANSWER] = judgments[ANSWER].str.replace("\n", "")
    qa_pairs = pandas.merge(qa_pairs, judgments, on=(QUESTION, ANSWER), how="left")
    qa_pairs = qa_pairs.drop_duplicates([QUESTION, ANSWER])

    if remove_newlines:
        del qa_pairs[ANSWER]
        qa_pairs = qa_pairs.rename(columns={"Temp": ANSWER})
    return qa_pairs


def drop_missing(systems_data):
    if any(systems_data.isnull()):
        n = len(systems_data)
        systems_data = systems_data.dropna()
        m = n - len(systems_data)
        if m:
            logger.warning("Dropping %d of %d question/answer pairs missing information (%0.3f%%)" %
                           (m, n, 100.0 * m / n))
    return systems_data

def kfold_split(df, outdir, _folds=5, _training_header=False):
    """
    
    Split the data-set into equal training and testing sets. Put training and testing set into local directory
    as csv files.

    :param df: data frame to be splited
    :param outdir: output directory path
    :param _folds: number of folds to be performed
    :param _training_header: header og the training file
    :return: list of directory for training set and teting set
    """
    # Randomize the order of the input dataframe
    df = df.iloc[np.random.permutation(len(df))]
    df = df.reset_index(drop=True)
    foldSize = int(math.ceil(len(df) / float(_folds)))
    logger.info("Total records: " + str(len(df)))
    logger.info("Fold size: " + str(foldSize))
    logger.info("Results written to output folder " + outdir)

    for x in range(0, _folds):
        fold_low = x * foldSize
        fold_high = (x + 1) * foldSize

        if fold_high >= len(df):
            fold_high = len(df)

        test_df = df.iloc[fold_low:fold_high]
        train_df = df.drop(df.index[fold_low:fold_high])

        test_df.to_csv(os.path.join(outdir, 'Test' + str(x) + '.csv'), encoding='utf-8', index=False)
        train_df.to_csv(os.path.join(outdir, 'Train' + str(x) + '.csv'), header=_training_header, encoding='utf-8', index=False)

        logger.info("--- Train_Fold_" + str(x) + ' size = ' + str(len(train_df)))
        logger.info("--- Test_Fold_" + str(x) + ' size = ' + str(len(test_df)))

# NLC as router functions


# k-folding and training
def nlc_router_train(url, username, password, oracle_out, path, all_correct):

    """
    NLC Training on the oracle experiment output to determine which system(NLC or Solr) should
    answer particular question.

    1. Splitting up the oracle experiment output data into 8 equal training records and testing records. This is to
    ensure 8-fold cross validation of the data-set. All training and Testing files will be stored
    at the "path"

     2. Perform NLC training on the all 8 training set simultaneously and returns list of classifier
     ids as json file in the working directory

    :param url: URL of NLC instance
    :param username: NLC Username
    :param password: NLC password
    :param oracle_out: file created by oracle experiment
    :param path: directory path to save intermediate results
    :param all_correct: optional boolean parameter to train with only correct QA pairs
    :return: list of classifier ids by NLC training
    """
    ensure_directory_exists(path)

    sys_name = oracle_out[SYSTEM][0]
    oracle_out[QUESTION] = oracle_out[QUESTION].str.replace("\n", " ")
    kfold_split(oracle_out, path, NLC_ROUTER_FOLDS, True)
    classifier_list = []
    list = []

    for x in range(0, NLC_ROUTER_FOLDS):
        train = pandas.read_csv(os.path.join(path, "Train{0}.csv".format(str(x))))
        if all_correct:
            logger.info("Training only on CORRECT examples.")
            # Ignore records from training which are not correct
            train = train[train[CORRECT]]
            train = train[train[IN_PURVIEW]]
        train = train[[QUESTION, ANSWERING_SYSTEM]]
        logger.info("Training set size = {0}".format(str(len(train))))
        with tempfile.TemporaryFile() as training_file:
            to_csv(training_file, train[[QUESTION, ANSWERING_SYSTEM]], header=False, index=False)
            training_file.seek(0)
            nlc = NaturalLanguageClassifier(url=url, username=username, password=password)
            classifier_id = nlc.create(training_data=training_file, name="{0}_fold_{1}".format(str(sys_name), str(x)))
            classifier_list.append(classifier_id["classifier_id"].encode("utf-8"))
            list.append({classifier_id["name"].encode("utf-8"): classifier_id["classifier_id"].encode("utf-8")})
            logger.info(pretty_print_json(classifier_id))
            pretty_print_json(classifier_id)

    with open(os.path.join(path, 'classifier.json'), 'wb') as f:
        json.dump(list, f)
    return classifier_list

# training status checking
def nlc_router_status(url, username, password, path):
    """
    Determine the status of NLC training instance and returns whether the instance is finished training or not.

    :param url: URL of NLC instance
    :param username: NLC Username
    :param password: NLC password
    :param path: directory path to save intermediate results
    :return: status of instance on stdout
    """
    # import list of classifier from file
    classifier_list = []
    with open(os.path.join(path, 'classifier.json'), 'r') as f:
        data = json.load(f)
    for x in range(0, NLC_ROUTER_FOLDS):
        classifier_list.append(data[x]['NLC+Solr Oracle_fold_{0}'.format(str(x))].encode("utf-8"))
    classifier_status(url, username, password, classifier_list)

# testing and merging
def nlc_router_test(url, username, password, collate_file, path):
    """
    Querying NLC for testing set to determine the system(NLC or Solr) and then lookup related
    fields from collated file (used as an input to the oracle experiment)

    :param url: URL of NLC instance
    :param username: NLC Username
    :param password: NLC password
    :param oracle_out: file created by oracle experiment
    :param collate_file: collated file created for oracle experiment as input
    :param path: directory path to save intermediate results

    :return: output file with best system NLC or Solr and relevant fields
    """
    def log_correct(system_data, name):
        n = len(system_data)
        m = sum(system_data[CORRECT])
        logger.info("%d of %d correct in %s (%0.3f%%)" % (m, n, name, 100.0 * m / n))

    # import list of classifier from file
    classifier_list = []
    with open(os.path.join(path, 'classifier.json'), 'r') as f:
        data = json.load(f)
    for x in range(0, NLC_ROUTER_FOLDS):
        classifier_list.append(data[x]['NLC+Solr Oracle_fold_{0}'.format(str(x))].encode("utf-8"))

    for x in range(0, NLC_ROUTER_FOLDS):
        test = pandas.read_csv(os.path.join(path, "Test{0}.csv".format(str(x))))
        test = test[[QUESTION]]
        test[QUESTION] = test[QUESTION].str.replace("\n", " ")
        classifier_id = classifier_list[x]
        n = NLC(url, username, password, classifier_id, test)
        out_file = os.path.join(path, "Out{0}.csv".format(str(x)))
        logger.info("Testing on fold {0} using NLC classifier {1}".format(str(x), str(classifier_list[x])))
        answer_router_questions(n, set(test[QUESTION]), out_file)

    # Concatenate multiple trained output into single csv file
    dfList = []
    columns = [QUESTION, SYSTEM]
    for x in range(0, NLC_ROUTER_FOLDS):
        df = pandas.read_csv(os.path.join(path, "Out{0}.csv".format(str(x))), header=0)
        dfList.append(df)

    concateDf = pandas.concat(dfList, axis=0)
    concateDf.columns = columns
    concateDf.to_csv(os.path.join(path, "Interim-Result.csv"), encoding='utf-8', index=None)

    # Join operation to get fields from oracle collated file
    result = pandas.merge(concateDf, collate_file, on=[QUESTION, SYSTEM])
    result = result.rename(columns={SYSTEM: ANSWERING_SYSTEM})
    result[SYSTEM] = 'NLC-as-router'
    result[CONFIDENCE] = __standardize_confidence(result)
    log_correct(result, 'NLC-as-router')
    return result


def answer_router_questions(system, questions, output):

    """
    Get Answer from given system to the question asked and store it in the output file
    :param system: System NLC or Solr
    :param questions: Question set
    :param output: Output file
    :return:
    """
    logger.info("Get answers to %d questions from %s" % (len(questions), system))
    answers = DataFrameCheckpoint(output, [QUESTION, ANSWERING_SYSTEM])
    try:
        if answers.recovered:
            logger.info("Recovered %d answers from %s" % (len(answers.recovered), output))
        questions = sorted(questions - answers.recovered)
        n = len(answers.recovered) + len(questions)
        for i, question in enumerate(questions, len(answers.recovered) + 1):
            if i is 1 or i == n or i % 25 == 0:
                logger.info(percent_complete_message("Question", i, n))
            answer = system.query(question.replace("\n", " "))
            #logger.debug("%s\t%s" % (question, answer))
            answers.write(question, answer)
    finally:
        answers.close()


class CollatedFileType(CsvFileType):
    columns = [QUESTION, SYSTEM, ANSWER, CONFIDENCE, IN_PURVIEW, CORRECT, FREQUENCY]

    def __init__(self):
        super(self.__class__, self).__init__(self.__class__.columns)

    def __call__(self, filename):
        if os.path.isfile(filename):
            collated = super(self.__class__, self).__call__(filename)
            m = sum(collated[collated[IN_PURVIEW] == False][CORRECT])
            if m:
                n = len(collated)
                logger.warning(
                    "%d out of %d question/answer pairs in %s are marked as out of purview but correct (%0.3f%%)"
                    % (m, n, filename, 100.0 * m / n))
            return collated
        else:
            logger.info("{0} does not exist".format(filename))
            return None

    @classmethod
    def output_format(cls, collated):
        collated = collated[cls.columns]
        collated = collated.sort_values([QUESTION, SYSTEM])
        return collated.set_index([QUESTION, SYSTEM, ANSWER])


class OracleFileType(CollatedFileType):
    columns = CollatedFileType.columns[:2] + [ANSWERING_SYSTEM] + CollatedFileType.columns[2:]
