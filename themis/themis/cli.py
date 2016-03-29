import argparse
import json

from analyze import AnnotationAssistFileType, mark_annotation_assist_correct, \
    add_judgements_and_frequencies_to_qa_pairs, \
    roc_curve, precision_curve, plot_curve
from annotate import create_annotation_assist_files
from nlc import classifier_list, NLC, train_nlc
from test import answer_questions, Solr
from themis import configure_logger, CsvFileType, QUESTION, ANSWER, print_csv, CONFIDENCE, FREQUENCY, logger, CORRECT
from wea import QUESTION_TEXT, TOP_ANSWER_TEXT, USER_EXPERIENCE, TOP_ANSWER_CONFIDENCE, augment_system_logs, \
    filter_corpus
from wea import wea_test, create_test_set_from_wea_logs
from xmgr import download_from_xmgr


def run():
    parser = argparse.ArgumentParser(description="Themis analysis toolkit")
    parser.add_argument('--log', default='INFO', help='logging level')
    subparsers = parser.add_subparsers(help="command to run")

    xmgr_parser = subparsers.add_parser("xmgr", help="download information from XMGR")
    xmgr_parser.add_argument("url", type=str, help="XMGR url")
    xmgr_parser.add_argument("username", type=str, help="XMGR username")
    xmgr_parser.add_argument("password", type=str, help="XMGR password")
    xmgr_parser.add_argument("output_directory", type=str, help="output directory")
    xmgr_parser.add_argument("--checkpoint-frequency", type=int, default=100,
                             help="how often to flush to a checkpoint file")
    xmgr_parser.add_argument("--max-docs", type=int, help="maximum number of corpus documents to download")
    xmgr_parser.set_defaults(func=xmgr_handler)

    filter_corpus_parser = subparsers.add_parser("filter", help="filter the corpus downloaded from XMGR")
    filter_corpus_parser.add_argument("corpus", type=CsvFileType(), help="corpus file")
    filter_corpus_parser.add_argument("--max-size", type=int, help="maximum size of answer text")
    filter_corpus_parser.set_defaults(func=filter_handler)

    test_set_parser = subparsers.add_parser("test-set", help="create test set from XMGR logs")
    test_set_parser.add_argument("logs",
                                 type=CsvFileType([QUESTION_TEXT, TOP_ANSWER_TEXT, USER_EXPERIENCE],
                                                  {QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER}),
                                 help="QuestionsData.csv log file from XMGR")
    test_set_parser.add_argument("--n", type=int, help="sample size")
    test_set_parser.set_defaults(func=test_set_handler)

    wea_parser = subparsers.add_parser("wea", help="answer questions with WEA logs")
    wea_parser.add_argument("test_set", type=CsvFileType(), help="test set")
    wea_parser.add_argument("logs",
                            type=CsvFileType(
                                [QUESTION_TEXT, TOP_ANSWER_TEXT, TOP_ANSWER_CONFIDENCE, USER_EXPERIENCE],
                                {QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER,
                                 TOP_ANSWER_CONFIDENCE: CONFIDENCE}),
                            help="QuestionsData.csv log file from XMGR")
    wea_parser.set_defaults(func=wea_handler)

    solr_parser = subparsers.add_parser("solr", help="answer questions with solr")
    solr_parser.add_argument("url", type=str, help="solr URL")
    solr_parser.add_argument("test_set", type=CsvFileType(), help="test set")
    solr_parser.add_argument("output", type=str, help="output filename")
    solr_parser.add_argument("--checkpoint-frequency", type=int, default=100,
                             help="how often to flush to a checkpoint file")
    solr_parser.set_defaults(func=solr_handler)

    nlc_arguments = argparse.ArgumentParser(add_help=False)
    nlc_arguments.add_argument("url", help="NLC url")
    nlc_arguments.add_argument("username", help="NLC username")
    nlc_arguments.add_argument("password", help="NLC password")

    nlc_parser = subparsers.add_parser("nlc", help="answer questions with NLC")
    nlc_subparsers = nlc_parser.add_subparsers(help="NLC actions")
    nlc_train = nlc_subparsers.add_parser("train", parents=[nlc_arguments], help="train an NLC model")
    nlc_train.add_argument("truth", type=CsvFileType(), help="ground truth")
    nlc_train.add_argument("name", help="classifier name")
    nlc_train.set_defaults(func=nlc_train_handler)
    nlc_use = nlc_subparsers.add_parser("use", parents=[nlc_arguments], help="use NLC model")
    nlc_use.add_argument("classifier", help="classifier id")
    nlc_use.add_argument("test_set", metavar="test-set", type=CsvFileType(), help="test set")
    nlc_use.add_argument("output", type=str, help="output filename")
    nlc_use.add_argument("--checkpoint-frequency", type=int, default=100,
                         help="how often to flush to a checkpoint file")
    nlc_use.set_defaults(func=nlc_use_handler)
    nlc_list = nlc_subparsers.add_parser("list", parents=[nlc_arguments], help="list NLC models")
    nlc_list.set_defaults(func=nlc_list_handler)

    annotate_parser = subparsers.add_parser("annotate", help="work with annotation assist")
    annotate_parser.add_argument("corpus", type=CsvFileType(), help="corpus file")
    annotate_parser.add_argument("truth", type=CsvFileType(), help="truth file")
    annotate_parser.add_argument("answers", type=argparse.FileType(), nargs="+", help="answered questions file")
    annotate_parser.add_argument("--output", default=".", help="output directory")
    annotate_parser.set_defaults(func=annotate_handler)

    curves_parser = subparsers.add_parser("curves", help="plot curves")
    curves_parser.add_argument("type", choices=["roc", "precision"], help="type of curve to create")
    curves_parser.add_argument("test_set", metavar="test-set", type=CsvFileType(), help="test set")
    curves_parser.add_argument("judgements", type=AnnotationAssistFileType(), help="Annotation Assist judgements")
    curves_parser.add_argument("--judgement-threshold", type=float, default=50,
                               help="cutoff value for a correct score, default 50")
    curves_parser.add_argument("answers", type=CsvFileType(), help="answers returned by a system")
    curves_parser.set_defaults(func=curves_handler)

    draw_parser = subparsers.add_parser("draw", help="draw curves")
    draw_parser.add_argument("curve", type=CsvFileType(), help="Curve data generated by the curves option")
    draw_parser.set_defaults(func=draw_handler)

    collate_parser = subparsers.add_parser("collate", help="collate answers and judgements")
    collate_parser.add_argument("test_set", metavar="test-set", type=CsvFileType(), help="test set")
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
    download_from_xmgr(args.url, args.username, args.password, args.output_directory, args.checkpoint_frequency,
                       args.max_docs)


def filter_handler(args):
    corpus = filter_corpus(args.corpus, args.max_size)
    print_csv(corpus)


def test_set_handler(args):
    test_set = create_test_set_from_wea_logs(args.logs, args.n)
    print_csv(test_set)


def wea_handler(args):
    results = wea_test(args.test_set, args.logs)
    print_csv(results)


def solr_handler(args):
    answer_questions(Solr(args.url), args.test_set, args.output, args.checkpoint_frequency)


def nlc_train_handler(args):
    print(train_nlc(args.url, args.username, args.password, args.truth, args.name))


def nlc_use_handler(args):
    n = NLC(args.url, args.username, args.password, args.classifier)
    answer_questions(n, args.test_set, args.output, args.checkpoint_frequency)


def nlc_list_handler(args):
    print json.dumps(classifier_list(args.url, args.username, args.password), indent=4)


def annotate_handler(args):
    create_annotation_assist_files(args.corpus, args.truth, args.answers, args.output)


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
    x_label = args.curve.columns[1]
    y_label = args.curve.columns[2]
    plot_curve(args.curve[x_label], args.curve[y_label], x_label, y_label)


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
