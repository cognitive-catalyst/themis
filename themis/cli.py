import argparse
import json

import pandas

from annotate import create_annotation_assist_files, AnnotationAssistFileType, \
    add_judgements_and_frequencies_to_qa_pairs, mark_annotation_assist_correct
from curves import roc_curve, precision_curve, plot_curves
from nlc import classifier_list, NLC, train_nlc, remove_classifiers, classifier_status
from test import answer_questions, Solr
from themis import configure_logger, CsvFileType, QUESTION, ANSWER, print_csv, CONFIDENCE, FREQUENCY, logger, CORRECT, \
    ANSWER_ID, retry
from wea import QUESTION_TEXT, TOP_ANSWER_TEXT, augment_system_logs, \
    filter_corpus, WeaLogFileType
from wea import wea_test, create_test_set_from_wea_logs
from xmgr import DownloadFromXmgrClosure


def run():
    global parser  # global so that it can be used by the command line handlers
    parser = argparse.ArgumentParser(description="Themis analysis toolkit")
    parser.add_argument("--log", default="INFO", help="logging level")
    subparsers = parser.add_subparsers(title="Q&A System analysis",
                                       description="Commands to download information from XMGR, answer questions " +
                                                   "using various Q&A systems, annotate the answers and analyze " +
                                                   "the results",
                                       help="command to run")

    xmgr_parser = subparsers.add_parser("xmgr", help="download information from XMGR")
    xmgr_parser.add_argument("url", type=str, help="XMGR url")
    xmgr_parser.add_argument("username", type=str, help="XMGR username")
    xmgr_parser.add_argument("password", type=str, help="XMGR password")
    xmgr_parser.add_argument("--output-directory", type=str, default=".", help="output directory")
    xmgr_parser.add_argument("--checkpoint-frequency", type=int, default=100,
                             help="how often to flush to a checkpoint file")
    xmgr_parser.add_argument("--max-docs", type=int, help="maximum number of corpus documents to download")
    xmgr_parser.add_argument("--retries", type=int, help="number of times to retry downloading after an error")
    xmgr_parser.set_defaults(func=xmgr_handler)

    filter_corpus_parser = subparsers.add_parser("filter", help="filter the corpus downloaded from XMGR")
    filter_corpus_parser.add_argument("corpus", type=CsvFileType(), help="corpus file created by the xmgr command")
    filter_corpus_parser.add_argument("--max-size", type=int, help="maximum size of answer text")
    filter_corpus_parser.set_defaults(func=filter_handler)

    test_set_parser = subparsers.add_parser("test-set",
                                            help="create test set of questions and their frequencies from XMGR logs")
    test_set_parser.add_argument("logs", type=WeaLogFileType(), help="QuestionsData.csv log file from XMGR")
    test_set_parser.add_argument("--before", type=pandas.to_datetime,
                                 help="keep interactions before the specified date")
    test_set_parser.add_argument("--after", type=pandas.to_datetime, help="keep interactions after the specified date")
    test_set_parser.add_argument("--n", type=int, help="sample size")
    test_set_parser.set_defaults(func=test_set_handler)

    wea_parser = subparsers.add_parser("wea", help="answer questions with WEA logs")
    wea_parser.add_argument("test_set", metavar="test-set", type=CsvFileType(),
                            help="questions and frequencies created by the test-set command")
    wea_parser.add_argument("logs", type=WeaLogFileType(), help="QuestionsData.csv log file from XMGR")
    wea_parser.set_defaults(func=wea_handler)

    solr_parser = subparsers.add_parser("solr", help="answer questions with solr")
    solr_parser.add_argument("url", type=str, help="solr URL")
    solr_parser.add_argument("test_set", type=CsvFileType(),
                             help="questions and frequencies created by the test-set command")
    solr_parser.add_argument("output", type=str, help="output filename")
    solr_parser.add_argument("--checkpoint-frequency", type=int, default=100,
                             help="how often to flush to a checkpoint file")
    solr_parser.set_defaults(func=solr_handler)

    nlc_arguments = argparse.ArgumentParser(add_help=False)
    nlc_arguments.add_argument("url", help="NLC url")
    nlc_arguments.add_argument("username", help="NLC username")
    nlc_arguments.add_argument("password", help="NLC password")

    nlc_parser = subparsers.add_parser("nlc", help="answer questions with NLC")
    nlc_subparsers = nlc_parser.add_subparsers(title="Natural Language Classifier",
                                               description="train, use, and manage NLC models", help="NLC actions")
    nlc_train = nlc_subparsers.add_parser("train", parents=[nlc_arguments], help="train an NLC model")
    nlc_train.add_argument("truth", type=CsvFileType(), help="truth file created by the xmgr command")
    nlc_train.add_argument("name", help="classifier name")
    nlc_train.set_defaults(func=nlc_train_handler)
    nlc_use = nlc_subparsers.add_parser("use", parents=[nlc_arguments], help="use NLC model")
    nlc_use.add_argument("classifier", help="classifier id")
    nlc_use.add_argument("test_set", metavar="test-set", type=CsvFileType(),
                         help="questions and frequencies created by the test-set command")
    nlc_use.add_argument("output", type=str, help="output filename")
    nlc_use.add_argument("corpus", type=CsvFileType([ANSWER, ANSWER_ID]),
                         help="corpus file created by the xmgr command")
    nlc_use.add_argument("--checkpoint-frequency", type=int, default=100,
                         help="how often to flush to a checkpoint file")
    nlc_use.set_defaults(func=nlc_use_handler)
    nlc_list = nlc_subparsers.add_parser("list", parents=[nlc_arguments], help="list NLC models")
    nlc_list.set_defaults(func=nlc_list_handler)
    nlc_status = nlc_subparsers.add_parser("status", parents=[nlc_arguments], help="status of NLC model")
    nlc_status.add_argument("classifiers", nargs="+", help="classifier ids")
    nlc_status.set_defaults(func=nlc_status_handler)
    nlc_delete = nlc_subparsers.add_parser("delete", parents=[nlc_arguments], help="delete an NLC model")
    nlc_delete.add_argument("classifiers", nargs="+", help="classifier ids")
    nlc_delete.set_defaults(func=nlc_delete_handler)

    annotate_parser = subparsers.add_parser("annotate", help="work with annotation assist")
    annotate_parser.add_argument("corpus", type=CsvFileType(), help="corpus file created by the xmgr command")
    annotate_parser.add_argument("answers", type=CsvFileType(), nargs="+", help="answered questions file")
    annotate_parser.add_argument("--output", default=".", help="output directory")
    annotate_parser.add_argument("--sample", type=int, help="number of unique questions to sample")
    annotate_parser.add_argument("--frequency", type=CsvFileType(),
                                 help="frequency file generated by the test-set command")
    annotate_parser.set_defaults(func=annotate_handler)

    curves_parser = subparsers.add_parser("curves", help="plot curves")
    curves_parser.add_argument("type", choices=["roc", "precision"], help="type of curve to create")
    curves_parser.add_argument("test_set", metavar="test-set", type=CsvFileType(),
                               help="questions and frequencies created by the test-set command")
    curves_parser.add_argument("judgements", type=AnnotationAssistFileType(), help="Annotation Assist judgements")
    curves_parser.add_argument("--judgement-threshold", type=float, default=50,
                               help="cutoff value for a correct score, default 50")
    curves_parser.add_argument("answers", type=CsvFileType(), help="answers returned by a system")
    curves_parser.set_defaults(func=curves_handler)

    draw_parser = subparsers.add_parser("draw", help="draw curves")
    draw_parser.add_argument("curves", type=CsvFileType(), nargs="+", help="Curve data generated by the curves option")
    draw_parser.add_argument("--labels", nargs="+", help="curve labels, by default use the file names")
    draw_parser.set_defaults(func=draw_handler)

    collate_parser = subparsers.add_parser("collate", help="collate answers and judgements")
    collate_parser.add_argument("test_set", metavar="test-set", type=CsvFileType(),
                                help="questions and frequencies created by the test-set command")
    collate_parser.add_argument("judgements", type=AnnotationAssistFileType(), help="Annotation Assist judgements")
    collate_parser.add_argument("--judgement-threshold", type=float, default=50,
                                help="cutoff value for a correct score, default 50")
    collate_parser.add_argument("answers", type=CsvFileType(), help="answers returned by a system")
    collate_parser.set_defaults(func=collate_handler)

    augment_parser = subparsers.add_parser("augment", help="augment system logs with annotation")
    augment_parser.add_argument("logs",
                                type=CsvFileType(rename={QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER}),
                                help="QuestionsData.csv log file from XMGR")
    augment_parser.add_argument("judgements", type=AnnotationAssistFileType(), help="Annotation Assist judgements")
    augment_parser.set_defaults(func=augment_handler)

    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


def xmgr_handler(args):
    c = DownloadFromXmgrClosure(args.url, args.username, args.password, args.output_directory,
                                args.checkpoint_frequency,
                                args.max_docs)
    retry(c, args.retries)


def filter_handler(args):
    corpus = filter_corpus(args.corpus, args.max_size)
    print_csv(corpus)


def test_set_handler(args):
    test_set = create_test_set_from_wea_logs(args.logs, args.before, args.after, args.n)
    print_csv(test_set)


def wea_handler(args):
    results = wea_test(args.test_set, args.logs)
    print_csv(results)


def solr_handler(args):
    answer_questions(Solr(args.url), args.test_set, args.output, args.checkpoint_frequency)


def nlc_train_handler(args):
    print(train_nlc(args.url, args.username, args.password, args.truth, args.name))


def nlc_use_handler(args):
    corpus = args.corpus.set_index(ANSWER_ID)
    n = NLC(args.url, args.username, args.password, args.classifier, corpus)
    answer_questions(n, args.test_set, args.output, args.checkpoint_frequency)


def nlc_list_handler(args):
    print(json.dumps(classifier_list(args.url, args.username, args.password), indent=4))


def nlc_status_handler(args):
    classifier_status(args.url, args.username, args.password, args.classifiers)


def nlc_delete_handler(args):
    remove_classifiers(args.url, args.username, args.password, args.classifiers)


def annotate_handler(args):
    if args.sample is not None and args.frequency is None:
        parser.print_usage()
        parser.error("You must specify a frequency file if you want to take a sample")
    create_annotation_assist_files(args.corpus, args.answers, args.sample, args.frequency, args.output)


def curves_handler(args):
    data = add_judgements_and_frequencies_to_qa_pairs(args.answers, args.judgements, args.test_set)
    data = mark_annotation_assist_correct(data, args.judgement_threshold)
    if args.type == "roc":
        curve = roc_curve(data)
    elif args.type == "precision":
        curve = precision_curve(data)
    else:
        raise Exception("Invalid curve type %s" % args.type)
    print_csv(curve)


def draw_handler(args):
    if args.labels is None:
        args.labels = [curve.filename for curve in args.curves]
    elif not len(args.curves) == len(args.labels):
        parser.print_usage()
        parser.error("There must be a name for each plot.")
    plot_curves(args.curves, args.labels)


def collate_handler(args):
    data = add_judgements_and_frequencies_to_qa_pairs(args.answers, args.judgements, args.test_set).set_index(
        [QUESTION, ANSWER]).sort_values(by=[CONFIDENCE, FREQUENCY])
    print_csv(data)
    logger.info("Confidence range %0.3f - %0.3f" % (data[CONFIDENCE].min(), data[CONFIDENCE].max()))
    correct_confidence = data[data[CORRECT]][CONFIDENCE]
    logger.info(
        "Correct answer confidence range %0.3f - %0.3f" % (correct_confidence.min(), correct_confidence.max()))


def augment_handler(args):
    augmented_logs = augment_system_logs(args.logs, args.judgements)
    print_csv(augmented_logs)


if __name__ == "__main__":
    run()
