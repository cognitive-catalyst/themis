"""
Command line interface to download information from XMGR, extract questions from the usage logs, answer questions using
various Q&A systems, annotate the answers and analyze the results.
"""

import argparse
import os

import pandas

from themis import configure_logger, CsvFileType, to_csv, QUESTION, ANSWER_ID, pretty_print_json, logger, print_csv, \
    retry, __version__
from themis.answer import answer_questions, Solr, get_answers_from_usage_log
from themis.fixup import filter_usage_log_by_date, filter_usage_log_by_user_experience, deakin, filter_corpus
from themis.judge import AnnotationAssistFileType, annotation_assist_qa_input, create_annotation_assist_corpus, \
    interpret_annotation_assist, JudgmentFileType, augment_usage_log
from themis.nlc import train_nlc, NLC, classifier_list, classifier_status, remove_classifiers
from themis.plot import generate_curves, plot_curves
from themis.question import QAPairFileType, UsageLogFileType, extract_question_answer_pairs_from_usage_logs, \
    sample_questions
from themis.xmgr import CorpusFileType, XmgrProject, DownloadCorpusFromXmgrClosure, download_truth_from_xmgr, \
    validate_truth_with_corpus, TruthFileType, examine_truth, validate_answers_with_corpus


def main():
    parser = argparse.ArgumentParser(description="Themis analysis toolkit")
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
    # Generate ROC and precision curves from judged answers.
    plot_command(parser, subparsers)
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
    xmgr_corpus = subparsers.add_parser("corpus", parents=[xmgr_shared_arguments, output_directory],
                                        help="download corpus")
    xmgr_corpus.add_argument("--checkpoint-frequency", metavar="CHECKPOINT-FREQUENCY", type=int, default=10,
                             help="flush corpus to checkpoint file after downloading this many documents")
    xmgr_corpus.add_argument("--max-docs", metavar="MAX-DOCS", type=int,
                             help="maximum number of corpus documents to download")
    xmgr_corpus.add_argument("--retries", type=int, help="number of times to retry downloading after an error")
    xmgr_corpus.set_defaults(func=corpus_handler)
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


def corpus_handler(args):
    xmgr = XmgrProject(args.url, args.username, args.password)
    c = DownloadCorpusFromXmgrClosure(xmgr, args.output_directory, args.checkpoint_frequency, args.max_docs)
    retry(c, args.retries)


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
    sample = sample_questions(args.questions, args.sample_size)
    print_csv(sample)


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


def augment_handler(args):
    usage_log = pandas.concat(args.usage_log)
    # noinspection PyTypeChecker
    augment_usage_log(usage_log, args.judments)


def plot_command(parser, subparsers):
    """
    Generate and optionally draw precision and ROC curves.
    """
    plot_parser = subparsers.add_parser("plot", help="generate performance plots from judged answers")
    plot_parser.add_argument("type", choices=["roc", "precision"], help="type of plot to create")
    plot_parser.add_argument("qa_pairs", metavar="qa-pairs", type=QAPairFileType(),
                             help="question/answer pairs generated by the 'question extract' command")
    plot_parser.add_argument("answers", type=CsvFileType(), nargs="+",
                             help="answers generated by one of the 'answer' commands")
    plot_parser.add_argument("--labels", nargs="+", help="names of the Q&A systems")
    plot_parser.add_argument("--judgments", required=True, nargs="+", type=JudgmentFileType(),
                             help="Q&A pair judgments generated by the 'judge interpret' command")
    plot_parser.add_argument("--output", default=".", help="output directory")
    plot_parser.add_argument("--draw", action="store_true", help="draw plots")
    plot_parser.set_defaults(func=CurvesHandlerClosure(parser))


def curves_handler(parser, args):
    if args.labels is None:
        args.labels = [answers.filename for answers in args.answers]
    elif not len(args.answers) == len(args.labels):
        parser.print_usage()
        parser.error("There must be a name for each plot.")
    labeled_qa_pairs = zip(args.labels, args.answers)
    judgments = pandas.concat(args.judgments)
    # noinspection PyTypeChecker
    curves = generate_curves(args.type, labeled_qa_pairs, judgments, args.qa_pairs)
    # Write curves data.
    for label, data in curves.items():
        filename = os.path.join(args.output, "%s.%s.csv" % (args.type, label))
        to_csv(filename, data)
    # Optionally draw plot.
    if args.draw:
        plot_curves(curves)


class CurvesHandlerClosure(object):
    def __init__(self, parser):
        self.parser = parser

    def __call__(self, args):
        curves_handler(self.parser, args)


def version_command(subparsers):
    version_parser = subparsers.add_parser("version", help="print version number")
    version_parser.set_defaults(func=version_handler)


def version_handler(_):
    print("Themis version %s" % __version__)


if __name__ == "__main__":
    main()
