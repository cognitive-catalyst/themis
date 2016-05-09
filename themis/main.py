"""
Command line interface to download information from XMGR, extract questions from the usage logs, answer questions using
various Q&A systems, annotate the answers and analyze the results.
"""

import argparse
import os

import pandas

from themis import configure_logger, CsvFileType, to_csv, QUESTION, ANSWER_ID, pretty_print_json, logger, print_csv, \
    __version__, FREQUENCY, ANSWER, IN_PURVIEW, CORRECT, DOCUMENT_ID
from themis.analyze import SYSTEM, CollatedFileType, add_judgments_and_frequencies_to_qa_pairs, system_similarity, \
    compare_systems, oracle_combination, filter_judged_answers, corpus_statistics, truth_statistics
from themis.answer import answer_questions, Solr, get_answers_from_usage_log, AnswersFileType
from themis.checkpoint import retry
from themis.fixup import filter_usage_log_by_date, filter_usage_log_by_user_experience, deakin, filter_corpus
from themis.judge import AnnotationAssistFileType, annotation_assist_qa_input, create_annotation_assist_corpus, \
    interpret_annotation_assist, JudgmentFileType, augment_usage_log
from themis.nlc import train_nlc, NLC, classifier_list, classifier_status, remove_classifiers
from themis.plot import generate_curves, plot_curves
from themis.question import QAPairFileType, UsageLogFileType, extract_question_answer_pairs_from_usage_logs, \
    QuestionFrequencyFileType
from themis.trec import corpus_from_trec
from themis.xmgr import CorpusFileType, XmgrProject, DownloadCorpusFromXmgrClosure, download_truth_from_xmgr, \
    validate_truth_with_corpus, TruthFileType, examine_truth, validate_answers_with_corpus


def main():
    parser = argparse.ArgumentParser(description="Themis analysis toolkit, version %s" % __version__)
    parser.add_argument("--log", default="INFO", help="logging level")

    subparsers = parser.add_subparsers(title="Q&A System analysis", description=__doc__)
    # Download information from xmgr.
    xmgr_command(subparsers)
    # Extract questions from usage logs.
    question_command(subparsers)
    # Ask questions to a Q&A system.
    answer_command(subparsers)
    # Judge answers using Annotation Assist.
    judge_command(subparsers)
    # Analyze results.
    analyze_command(parser, subparsers)
    # Various utilities.
    util_command(subparsers)
    # Print the version number.
    version_command(subparsers)

    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


def xmgr_command(subparsers):
    xmgr_shared_arguments = argparse.ArgumentParser(add_help=False)
    xmgr_shared_arguments.add_argument("url", help="XMGR url")
    xmgr_shared_arguments.add_argument("username", help="XMGR username")
    xmgr_shared_arguments.add_argument("password", help="XMGR password")

    verify_arguments = argparse.ArgumentParser(add_help=False)
    verify_arguments.add_argument("corpus", type=CorpusFileType(),
                                  help="corpus file created by the 'download corpus' command")
    verify_arguments.add_argument("truth", type=TruthFileType(), help="truth file created by the 'xmgr truth' command")

    output_directory = argparse.ArgumentParser(add_help=False)
    output_directory.add_argument("--output-directory", metavar="OUTPUT-DIRECTORY", type=str, default=".",
                                  help="output directory")

    xmgr_parser = subparsers.add_parser("xmgr", help="download information from XMGR")
    subparsers = xmgr_parser.add_subparsers(description="download information from XMGR")
    # Download corpus from XMGR.
    xmgr_download = subparsers.add_parser("download-corpus",
                                          parents=[xmgr_shared_arguments, output_directory], help="download corpus")
    xmgr_download.add_argument("--max-docs", metavar="MAX-DOCS", type=int,
                               help="maximum number of corpus documents to download")
    xmgr_download.add_argument("--checkpoint-frequency", metavar="CHECKPOINT-FREQUENCY", type=int, default=10,
                               help="flush corpus to checkpoint file after downloading this many documents")
    xmgr_download.add_argument("--retries", type=int, help="number of times to retry downloading after an error")
    xmgr_download.set_defaults(func=download_handler)
    # Get corpus from TREC documents directory.
    xmgr_trec = subparsers.add_parser("trec-corpus", parents=[output_directory], help="extract corpus from TREC files")
    xmgr_trec.add_argument("directory", help="directory containing XML TREC files")
    xmgr_trec.add_argument("--max-docs", metavar="MAX-DOCS", type=int,
                           help="maximum number of TREC documents to examine")
    xmgr_trec.add_argument("--checkpoint-frequency", metavar="CHECKPOINT-FREQUENCY", type=int, default=1000,
                           help="flush corpus to checkpoint file after parsing this many TREC files")
    xmgr_trec.set_defaults(func=trec_handler)
    # Download truth from XMGR.
    xmgr_truth = subparsers.add_parser("truth", parents=[xmgr_shared_arguments, output_directory],
                                       help="download truth file")
    xmgr_truth.set_defaults(func=truth_handler)
    # Download PAU ids corresponding to a document.
    xmgr_pau = subparsers.add_parser("pau-ids", parents=[xmgr_shared_arguments],
                                     help="list PAU ids corresponding to a document")
    xmgr_pau.add_argument("document", help="document id")
    xmgr_pau.set_defaults(func=document_handler)
    # Download individual PAU.
    xmgr_pau = subparsers.add_parser("pau", parents=[xmgr_shared_arguments], help="download an individual PAU")
    xmgr_pau.add_argument("pau", help="PAU id")
    xmgr_pau.set_defaults(func=pau_handler)
    # Filter corpus.
    xmgr_filter = subparsers.add_parser("filter", help="fix up corpus")
    xmgr_filter.add_argument("corpus", type=CorpusFileType(), help="file downloaded by 'xmgr corpus'")
    xmgr_filter.add_argument("--max-size", metavar="MAX-SIZE", type=int,
                             help="maximum size of answer text in characters")
    xmgr_filter.set_defaults(func=filter_corpus_handler)
    # Verify that truth answer Ids are in the corpus.
    xmgr_validate_truth = subparsers.add_parser("validate-truth", parents=[verify_arguments, output_directory],
                                                help="ensure that all truth answer Ids are in the corpus")
    xmgr_validate_truth.set_defaults(func=validate_truth_handler)
    # Verify that usage log answers are in the corpus.
    xmgr_validate_answers = subparsers.add_parser("validate-answers", parents=[output_directory],
                                                  help="ensure that all WEA answers are in the corpus")
    xmgr_validate_answers.add_argument("corpus", type=CorpusFileType(),
                                       help="corpus file created by the 'download corpus' command")
    xmgr_validate_answers.add_argument("qa_pairs", metavar="qa-pairs", type=QAPairFileType(),
                                       help="Q&A pair file generated by `question extract`")
    xmgr_validate_answers.set_defaults(func=validate_answers_handler)
    # Write questions and answers in truth to an HTML file.
    xmgr_examine = subparsers.add_parser("examine-truth", parents=[verify_arguments],
                                         help="create human-readable truth file with answers and their " +
                                              "associated questions")
    xmgr_examine.set_defaults(func=examine_handler)


def download_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    closure = DownloadCorpusFromXmgrClosure(xmgr, args.output_directory, args.checkpoint_frequency, args.max_docs)
    retry(closure, args.retries)


def trec_handler(args):
    checkpoint_filename = os.path.join(args.output_directory, "corpus.trec.temp.csv")
    corpus = corpus_from_trec(checkpoint_filename, args.directory, args.checkpoint_frequency, args.max_docs)
    to_csv(os.path.join(args.output_directory, "corpus.csv"), CorpusFileType.output_format(corpus))
    logger.info("%d documents and %d PAUs in corpus" % (len(corpus[DOCUMENT_ID].drop_duplicates()), len(corpus)))
    os.remove(checkpoint_filename)


def truth_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    download_truth_from_xmgr(xmgr, args.output_directory)


def pau_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    print(pretty_print_json(xmgr.get_paus(args.pau)))


def document_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    print(", ".join(xmgr.get_pau_ids_in_document(args.document)))


def filter_corpus_handler(args):
    corpus = filter_corpus(args.corpus, args.max_size)
    print_csv(corpus)


def validate_truth_handler(args):
    validate_truth_with_corpus(args.corpus, args.truth, args.output_directory)


def validate_answers_handler(args):
    validate_answers_with_corpus(args.corpus, args.qa_pairs, args.output_directory)


def examine_handler(args):
    examine_truth(args.corpus, args.truth)


def question_command(subparsers):
    question_parser = subparsers.add_parser("question", help="get questions to ask a Q&A system")
    subparsers = question_parser.add_subparsers(description="get questions to ask a Q&A system")
    # Extract questions from usage logs.
    question_extract = subparsers.add_parser("extract", help="extract question/answer pairs from usage logs")
    question_extract.add_argument("usage_log", metavar="usage-log", nargs="+", type=UsageLogFileType(),
                                  help="QuestionsData.csv usage log file from XMGR")
    question_extract.add_argument("--before", metavar="DATE", type=pandas.to_datetime,
                                  help="keep interactions before the specified date")
    question_extract.add_argument("--after", metavar="DATE", type=pandas.to_datetime,
                                  help="keep interactions after the specified date")
    question_extract.add_argument("--user-experience", nargs="+", default=set(),
                                  help="disallowed User Experience values (DIALOG is always disallowed)")
    question_extract.add_argument("--deakin", action="store_true", help="fixups specific to the Deakin system")
    question_extract.set_defaults(func=extract_handler)
    # Sample questions by frequency.
    question_sample = subparsers.add_parser("sample", help="sample questions")
    question_sample.add_argument("questions", type=QAPairFileType(),
                                 help="question/answer pairs extracted from usage log by the 'question extract' command")
    question_sample.add_argument("sample_size", metavar="sample-size", type=int,
                                 help="number of unique questions to sample")
    question_sample.set_defaults(func=sample_handler)


# noinspection PyTypeChecker
def extract_handler(args):
    # Do custom fixup of usage logs.
    usage_log = pandas.concat(args.usage_log)
    n = len(usage_log)
    if args.before or args.after:
        usage_log = filter_usage_log_by_date(usage_log, args.before, args.after)
    user_experience = set(args.user_experience) | {"DIALOG"}  # DIALOG is always disallowed
    usage_log = filter_usage_log_by_user_experience(usage_log, user_experience)
    if args.deakin:
        usage_log = deakin(usage_log)
    m = n - len(usage_log)
    if n:
        logger.info("Removed %d of %d questions (%0.3f%%)" % (m, n, 100.0 * m / n))
    # Extract Q&A pairs from fixed up usage logs.
    qa_pairs = extract_question_answer_pairs_from_usage_logs(usage_log)
    print_csv(QAPairFileType.output_format(qa_pairs))


def sample_handler(args):
    # Sample questions by frequency.
    questions = args.questions[[QUESTION, FREQUENCY]].drop_duplicates(QUESTION)
    sample = questions.sample(args.sample_size, weights=FREQUENCY)
    print_csv(QuestionFrequencyFileType.output_format(sample))


def answer_command(subparsers):
    """
    Get answers to questions from various Q&A systems.

    WEA
        Extract answer from usage logs.

    Solr
        Lookup answers from a Solr database using questions as queries.

    NLC
        Train an NLC model to answer questions using the truth file downloaded from XMGR.
        1. train
        2. use
        3. list
        4. status
        5. delete
    """
    qa_shared_arguments = argparse.ArgumentParser(add_help=False)
    qa_shared_arguments.add_argument("questions", type=QuestionSetFileType(),
                                     help="question set generated by either the 'question extract' " +
                                          "or 'question sample' command")
    qa_shared_arguments.add_argument("output", type=str, help="output filename")

    checkpoint_argument = argparse.ArgumentParser(add_help=False)
    checkpoint_argument.add_argument("--checkpoint-frequency", metavar="CHECKPOINT-FREQUENCY", type=int, default=100,
                                     help="how often to flush to a checkpoint file")

    answer_parser = subparsers.add_parser("answer", help="answer questions with Q&A systems")
    subparsers = answer_parser.add_subparsers(description="answer questions with Q&A systems", help="Q&A systems")

    # Extract answers from the usage log.
    answer_wea = subparsers.add_parser("wea", parents=[qa_shared_arguments], help="extract WEA answers from usage log")
    answer_wea.add_argument("qa_pairs", metavar="qa-pairs", type=QAPairFileType(),
                            help="question/answer pairs produced by the 'question extract' command")
    answer_wea.set_defaults(func=wea_handler)

    # Query answers from a Solr database.
    answer_solr = subparsers.add_parser("solr", parents=[qa_shared_arguments, checkpoint_argument],
                                        help="query answers from a Solr database")
    answer_solr.add_argument("url", type=str, help="solr URL")
    answer_solr.set_defaults(func=solr_handler)

    # Manage an NLC model.
    nlc_shared_arguments = argparse.ArgumentParser(add_help=False)
    nlc_shared_arguments.add_argument("url", help="NLC url")
    nlc_shared_arguments.add_argument("username", help="NLC username")
    nlc_shared_arguments.add_argument("password", help="NLC password")

    nlc_parser = subparsers.add_parser("nlc", help="answer questions with NLC")
    nlc_subparsers = nlc_parser.add_subparsers(title="Natural Language Classifier",
                                               description="train, use, and manage NLC models", help="NLC actions")
    # Train NLC model.
    nlc_train = nlc_subparsers.add_parser("train", parents=[nlc_shared_arguments], help="train an NLC model")
    nlc_train.add_argument("truth", type=TruthFileType(), help="truth file created by the 'xmgr truth' command")
    nlc_train.add_argument("name", help="classifier name")
    nlc_train.set_defaults(func=nlc_train_handler)
    # Use an NLC model.
    nlc_use = nlc_subparsers.add_parser("use", parents=[nlc_shared_arguments, qa_shared_arguments, checkpoint_argument],
                                        help="use NLC model")
    nlc_use.add_argument("classifier", help="classifier id")
    nlc_use.add_argument("corpus", type=CorpusFileType(), help="corpus file created by the 'download corpus' command")
    nlc_use.set_defaults(func=nlc_use_handler)
    # List all NLC models.
    nlc_list = nlc_subparsers.add_parser("list", parents=[nlc_shared_arguments], help="list NLC models")
    nlc_list.set_defaults(func=nlc_list_handler)
    # Get status of NLC models.
    nlc_status = nlc_subparsers.add_parser("status", parents=[nlc_shared_arguments], help="status of NLC model")
    nlc_status.add_argument("classifiers", nargs="+", help="classifier ids")
    nlc_status.set_defaults(func=nlc_status_handler)
    # Delete NLC models.
    nlc_delete = nlc_subparsers.add_parser("delete", parents=[nlc_shared_arguments], help="delete an NLC model")
    nlc_delete.add_argument("classifiers", nargs="+", help="classifier ids")
    nlc_delete.set_defaults(func=nlc_delete_handler)


def wea_handler(args):
    wea_answers = get_answers_from_usage_log(args.questions, args.qa_pairs)
    to_csv(args.output, wea_answers)


def solr_handler(args):
    answer_questions(Solr(args.url), set(args.questions[QUESTION]), args.output, args.checkpoint_frequency)


def nlc_train_handler(args):
    print(train_nlc(args.url, args.username, args.password, args.truth, args.name))


def nlc_use_handler(args):
    corpus = args.corpus.set_index(ANSWER_ID)
    n = NLC(args.url, args.username, args.password, args.classifier, corpus)
    answer_questions(n, set(args.questions[QUESTION]), args.output, args.checkpoint_frequency)


def nlc_list_handler(args):
    print(pretty_print_json(classifier_list(args.url, args.username, args.password)))


def nlc_status_handler(args):
    classifier_status(args.url, args.username, args.password, args.classifiers)


def nlc_delete_handler(args):
    remove_classifiers(args.url, args.username, args.password, args.classifiers)


class QuestionSetFileType(CsvFileType):
    def __init__(self):
        super(self.__class__, self).__init__([QUESTION])

    def __call__(self, filename):
        questions = super(self.__class__, self).__call__(filename)
        return questions.drop_duplicates()


def judge_command(subparsers):
    """
    Judge answers using Annotate Assist

    pairs
        Generate question and answer pairs for Annotation Assist to judge. This takes question list filter, system answers,
        optional previous annotations.

    corpus
        Create the corpus file used by Annotation Assist.

    interpret
        Apply judgement threshold to file retrieved from Annotation Assist.
    """
    judge_parser = subparsers.add_parser("judge", help="judge answers provided by Q&A systems")
    subparsers = judge_parser.add_subparsers(description="create and interpret files used by Annotation Assist")
    # Annotation Assistant Q&A pairs.
    judge_pairs = subparsers.add_parser("pairs",
                                        help="generate question and answer pairs for judgment by Annotation Assistant")
    judge_pairs.add_argument("answers", type=CsvFileType(), nargs="+",
                             help="answers generated by one of the 'answer' commands")
    judge_pairs.add_argument("--questions", type=CsvFileType([QUESTION]),
                             help="limit Q&A pairs to just these questions")
    judge_pairs.add_argument("--judgments", type=JudgmentFileType(), nargs="+",
                             help="Q&A pair judgments generated by the 'judge interpret' command")
    judge_pairs.set_defaults(func=annotation_pairs_handler)
    # Annotation Assistant corpus.
    judge_corpus = subparsers.add_parser("corpus", help="generate corpus file for Annotation Assistant")
    judge_corpus.add_argument("corpus", type=CorpusFileType(),
                              help="corpus file created by the 'download corpus' command")
    judge_corpus.set_defaults(func=annotation_corpus_handler)
    # Interpret Annotation Assistant judgments.
    judge_interpret = subparsers.add_parser("interpret", help="interpret Annotation Assistant judgments")
    judge_interpret.add_argument("judgments", type=AnnotationAssistFileType(),
                                 help="judgments file downloaded from Annotation Assistant")
    judge_interpret.add_argument("--judgment-threshold", metavar="JUDGMENT-THRESHOLD", type=float, default=50,
                                 help="cutoff value for a correct score, default 50")
    judge_interpret.set_defaults(func=annotation_interpret_handler)
    # Create sample of already judged questions.
    judge_sample = subparsers.add_parser("sample", help="create sample of already judged questions")
    judge_sample.add_argument("frequency", type=QuestionFrequencyFileType(),
                              help="question frequency file " +
                                   "generated by the 'question extract' or 'question sample' commands")
    judge_sample.add_argument("judgments", nargs="+", type=JudgmentFileType(),
                              help="Q&A pair judgments generated by the 'judge interpret' command")
    judge_sample.set_defaults(func=judge_sample_handler)
    # Augment usage logs with judgments.
    judge_augment = subparsers.add_parser("augment", help="augment usage logs with judgments")
    judge_augment.add_argument("usage_log", metavar="usage-log", nargs="+", type=CsvFileType(),
                               help="QuestionsData.csv usage log file from XMGR")
    judge_augment.add_argument("judgments", type=JudgmentFileType(),
                               help="judgments file created by 'judge interpret' command")
    judge_augment.set_defaults(func=augment_handler)


def annotation_pairs_handler(args):
    qa_pairs = annotation_assist_qa_input(args.answers, args.questions, args.judgments)
    print_csv(qa_pairs, index=False)


def annotation_corpus_handler(args):
    print(create_annotation_assist_corpus(args.corpus))


def annotation_interpret_handler(args):
    judgments = interpret_annotation_assist(args.judgments, args.judgment_threshold)
    print_csv(JudgmentFileType.output_format(judgments))


def judge_sample_handler(args):
    questions = pandas.concat(args.judgments)[[QUESTION]].drop_duplicates()
    sample = pandas.merge(questions, args.frequency, on=QUESTION, how="left")
    n = len(sample)
    logger.info("%d judged questions" % n)
    m = sum(sample[FREQUENCY].isnull())
    if m:
        logger.warning("Missing frequencies for %d questions (%0.3f%%)" % (m, 100.0 * m / n))
    print_csv(QuestionFrequencyFileType.output_format(sample))


def augment_handler(args):
    usage_log = pandas.concat(args.usage_log)
    # noinspection PyTypeChecker
    augment_usage_log(usage_log, args.judments)


def analyze_command(parser, subparsers):
    filter_arguments = argparse.ArgumentParser(add_help=False)
    filter_arguments.add_argument("collated", nargs="+", type=CollatedFileType(),
                                  help="combined system answers and judgments created by 'analyze collate'")
    filter_arguments.add_argument("--system-names", metavar="system", nargs="+",
                                  help="name of systems to view, by default view them all")

    analyze_parser = subparsers.add_parser("analyze", help="analyze results")

    subparsers = analyze_parser.add_subparsers(description="analyze results")
    # Collate results.
    collate = subparsers.add_parser("collate", help="combine Q&A pairs and judgments across systems")
    collate.add_argument("frequency", type=QuestionFrequencyFileType(),
                         help="question frequency file " +
                              "generated by the 'question extract' or 'question sample' commands")
    collate.add_argument("answers", type=AnswersFileType(), nargs="+",
                         help="answers generated by one of the 'answer' commands")
    collate.add_argument("--labels", nargs="+", help="names of the Q&A systems")
    collate.add_argument("--judgments", required=True, nargs="+", type=JudgmentFileType(),
                         help="Q&A pair judgments generated by the 'judge interpret' command")
    collate.add_argument("--remove-newlines", action="store_true", help="join on answers with newlines removed")
    collate.set_defaults(func=HandlerClosure(collate_handler, parser))
    # Plot collated results.
    plot_parser = subparsers.add_parser("plot", help="generate performance plots from judged answers")
    plot_parser.add_argument("type", choices=["roc", "precision"], help="type of plot to create")
    plot_parser.add_argument("collated", nargs="+", type=CollatedFileType(),
                             help="combined system answers and judgments created by 'analyze collate'")
    plot_parser.add_argument("--output", default=".", help="output directory")
    plot_parser.add_argument("--draw", action="store_true", help="draw plots")
    plot_parser.set_defaults(func=plot_handler)
    # Print in-purview correct answers.
    correct_parser = subparsers.add_parser("correct", parents=[filter_arguments], help="in-purview correct answers")
    correct_parser.set_defaults(func=correct_handler)
    # Print in-purview incorrect answers.
    incorrect_parser = subparsers.add_parser("incorrect", parents=[filter_arguments],
                                             help="in-purview incorrect answers")
    incorrect_parser.set_defaults(func=incorrect_handler)
    # Similarity of system answers.
    similarity_parser = subparsers.add_parser("similarity", help="measure similarity of different systems' answers")
    similarity_parser.add_argument("collated", type=CollatedFileType(),
                                   help="combined system answers and judgments created by 'analyze collate'")
    similarity_parser.set_defaults(func=similarity_handler)
    # Comparison of system pairs.
    comparison_parser = subparsers.add_parser("compare", help="compare two systems' performance")
    comparison_parser.add_argument("type", choices=["better", "worse"],
                                   help="relative performance of first to second system")
    comparison_parser.add_argument("system_1", metavar="system-1", help="first system")
    comparison_parser.add_argument("system_2", metavar="system-2", help="second system")
    comparison_parser.add_argument("collated", type=CollatedFileType(),
                                   help="combined system answers and judgments created by 'analyze collate'")
    comparison_parser.set_defaults(func=comparison_handler)
    # Create multi-system oracle.
    oracle_parser = subparsers.add_parser("oracle",
                                          help="combine multiple systems into a single oracle system " +
                                               "that is correct when any one of them is correct")
    oracle_parser.add_argument("collated", type=CollatedFileType(),
                               help="combined system answers and judgments created by 'analyze collate'")
    oracle_parser.add_argument("system_names", metavar="system", nargs="+", help="name of systems to combine")
    oracle_parser.set_defaults(func=oracle_handler)
    # Corpus statistics.
    corpus_parser = subparsers.add_parser("corpus", help="corpus statistics")
    corpus_parser.add_argument("corpus", type=CorpusFileType(),
                               help="corpus file created by the 'download corpus' command")
    corpus_parser.add_argument("--histogram", help="token frequency per answer histogram")
    corpus_parser.set_defaults(func=analyze_corpus_handler)
    # Truth statistics.
    truth_parser = subparsers.add_parser("truth", help="truth statistics")
    truth_parser.add_argument("truth", type=TruthFileType(), help="truth file created by the 'xmgr truth' command")
    truth_parser.add_argument("--histogram", help="answers per question histogram")
    truth_parser.add_argument("--corpus", type=CorpusFileType(),
                              help="corpus file created by the 'download corpus' command, " +
                                   "used to add answer text to the histogram")
    truth_parser.set_defaults(func=HandlerClosure(analyze_truth_handler, parser))


# noinspection PyTypeChecker
def collate_handler(parser, args):
    labeled_qa_pairs = answer_labels(parser, args)
    judgments = pandas.concat(args.judgments)
    all_systems = []
    for label, qa_pairs in labeled_qa_pairs:
        # Only consider the questions listed in the frequency file.
        qa_pairs = qa_pairs[qa_pairs[QUESTION].isin(args.frequency[QUESTION])]
        collated = add_judgments_and_frequencies_to_qa_pairs(qa_pairs, judgments, args.frequency, args.remove_newlines)
        collated[SYSTEM] = label
        all_systems.append(collated)
    collated = pandas.concat(all_systems)
    logger.info("%d question/answer pairs" % len(collated))
    n = len(collated)
    for column, s in [(ANSWER, "answers"), (IN_PURVIEW, "in purview judgments"), (CORRECT, "correctness judgments")]:
        m = sum(collated[column].isnull())
        if m:
            logger.warning("%d question/answer pairs out of %d missing %s (%0.3f%%)" % (m, n, s, 100.0 * m / n))
    print_csv(CollatedFileType.output_format(collated))


def answer_labels(parser, args):
    if args.labels is None:
        args.labels = [answers.filename for answers in args.answers]
    elif not len(args.answers) == len(args.labels):
        parser.print_usage()
        parser.error("There must be a name for each plot.")
    return zip(args.labels, args.answers)


def correct_handler(args):
    correct = filter_judged_answers(args.collated, True, args.system_names)
    print_csv(CollatedFileType.output_format(correct))


def incorrect_handler(args):
    incorrect = filter_judged_answers(args.collated, False, args.system_names)
    print_csv(CollatedFileType.output_format(incorrect))


def plot_handler(args):
    curves = generate_curves(args.type, args.collated)
    # Write curves data.
    for label, curve in curves.items():
        filename = os.path.join(args.output, "%s.%s.csv" % (args.type, label))
        to_csv(filename, curve)
    # Optionally draw plot.
    if args.draw:
        plot_curves(curves, args.type)


def similarity_handler(args):
    similarity = system_similarity(args.collated)
    print_csv(similarity)


def comparison_handler(args):
    comparison = compare_systems(args.collated, args.system_1, args.system_2, args.type)
    print_csv(comparison)


def oracle_handler(args):
    oracle_name = "%s Oracle" % "+".join(args.system_names)
    oracle = oracle_combination(args.collated, args.system_names, oracle_name)
    print_csv(CollatedFileType.output_format(oracle))


def analyze_corpus_handler(args):
    answers, tokens, histogram = corpus_statistics(args.corpus)
    print("%d answers, %d tokens, average %0.3f tokens per answer" % (answers, tokens, tokens / float(answers)))
    if args.histogram:
        r = pandas.DataFrame(list(histogram.items()), columns=("Tokens", "Count")).set_index("Tokens").sort_index()
        to_csv(args.histogram, r)


def analyze_truth_handler(parser, args):
    if args.histogram is None and args.corpus is not None:
        parser.print_usage()
        parser.error("The corpus is only used when drawing a histogram.")
    pairs, questions, answers, question_histogram = truth_statistics(args.truth)
    print("%d training pairs, %d unique questions, %d unique answers, average %0.3f questions per answer" %
          (pairs, questions, answers, questions / float(answers)))
    if args.histogram:
        if args.corpus is not None:
            question_histogram = pandas.merge(question_histogram, args.corpus[[ANSWER_ID, ANSWER]].set_index(ANSWER_ID),
                                              left_index=True, right_index=True)
            question_histogram = question_histogram[[ANSWER, QUESTION]]
        to_csv(args.histogram,
               question_histogram.sort_values(QUESTION, ascending=False).rename(columns={QUESTION: "Questions"}))


def util_command(subparsers):
    util_parser = subparsers.add_parser("util", help="various utilities")
    subparsers = util_parser.add_subparsers(description="various utilities")
    rows = subparsers.add_parser("rows", help="get the number of rows in a CSV file")
    rows.add_argument("file", type=CsvFileType(), help="CSV file")
    rows.set_defaults(func=rows_handler)
    drop_null = subparsers.add_parser("drop-null", help="drop rows that contain null values from a CSV file")
    drop_null.add_argument("file", type=CsvFileType(), help="CSV file")
    drop_null.set_defaults(func=drop_null_handler)


def rows_handler(args):
    n = len(args.file)
    m = sum(args.file.count(axis="columns") < len(args.file.columns))
    print("%d rows, %d with null values (%0.3f%%)" % (n, m, 100.0 * m / n))


def drop_null_handler(args):
    n = len(args.file)
    non_null = args.file.dropna()
    m = n - len(non_null)
    logger.info("Dropped %d rows with null values from %d rows (%0.3f%%)" % (m, n, 100.0 * m / n))
    print_csv(non_null, index=False)


def version_command(subparsers):
    version_parser = subparsers.add_parser("version", help="print version number")
    version_parser.set_defaults(func=version_handler)


def version_handler(_):
    print("Themis version %s" % __version__)


class HandlerClosure(object):
    def __init__(self, func, parser):
        self.func = func
        self.parser = parser

    def __call__(self, args):
        self.func(self.parser, args)


if __name__ == "__main__":
    main()
