import argparse

from annotate import AnnotationAssistFileType, interpret_annotation_assist
from curves import roc_curve, precision_curve
from themis import configure_logger, CsvFileType, QUESTION, ANSWER, print_csv, CONFIDENCE, FREQUENCY, logger, CORRECT
from themis.curves import add_judgments_and_frequencies_to_qa_pairs
from wea import QUESTION_TEXT, TOP_ANSWER_TEXT, augment_system_logs


def run():
    global parser  # global so that it can be used by the command line handlers
    parser = argparse.ArgumentParser(description="Themis analysis toolkit")
    parser.add_argument("--log", default="INFO", help="logging level")
    subparsers = parser.add_subparsers(title="Q&A System analysis",
                                       description="Commands to download information from XMGR, answer questions " +
                                                   "using various Q&A systems, annotate the answers and analyze " +
                                                   "the results",
                                       help="command to run")

    curves_parser = subparsers.add_parser("curves", help="plot curves")
    curves_parser.add_argument("type", choices=["roc", "precision"], help="type of curve to create")
    curves_parser.add_argument("test_set", metavar="test-set", type=CsvFileType(),
                               help="questions and frequencies created by the test-set command")
    curves_parser.add_argument("judgments", type=AnnotationAssistFileType(), help="Annotation Assist judgments")
    curves_parser.add_argument("--judgment-threshold", type=float, default=50,
                               help="cutoff value for a correct score, default 50")
    curves_parser.add_argument("answers", type=CsvFileType(), help="answers returned by a system")
    curves_parser.set_defaults(func=curves_handler)

    collate_parser = subparsers.add_parser("collate", help="collate answers and judgments")
    collate_parser.add_argument("test_set", metavar="test-set", type=CsvFileType(),
                                help="questions and frequencies created by the test-set command")
    collate_parser.add_argument("judgments", type=AnnotationAssistFileType(), help="Annotation Assist judgments")
    collate_parser.add_argument("--judgment-threshold", type=float, default=50,
                                help="cutoff value for a correct score, default 50")
    collate_parser.add_argument("answers", type=CsvFileType(), help="answers returned by a system")
    collate_parser.set_defaults(func=collate_handler)

    augment_parser = subparsers.add_parser("augment", help="augment system logs with annotation")
    augment_parser.add_argument("logs",
                                type=CsvFileType(rename={QUESTION_TEXT: QUESTION, TOP_ANSWER_TEXT: ANSWER}),
                                help="QuestionsData.csv log file from XMGR")
    augment_parser.add_argument("judgments", type=AnnotationAssistFileType(), help="Annotation Assist judgments")
    augment_parser.set_defaults(func=augment_handler)

    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


def curves_handler(args):
    data = add_judgments_and_frequencies_to_qa_pairs(args.answers, args.judgments, args.test_set)
    data = interpret_annotation_assist(data, args.judgment_threshold)
    if args.type == "roc":
        curve = roc_curve(data)
    elif args.type == "precision":
        curve = precision_curve(data)
    else:
        raise Exception("Invalid curve type %s" % args.type)
    print_csv(curve)


def collate_handler(args):
    data = add_judgments_and_frequencies_to_qa_pairs(args.answers, args.judgments, args.test_set).set_index(
        [QUESTION, ANSWER]).sort_values(by=[CONFIDENCE, FREQUENCY])
    print_csv(data)
    logger.info("Confidence range %0.3f - %0.3f" % (data[CONFIDENCE].min(), data[CONFIDENCE].max()))
    correct_confidence = data[data[CORRECT]][CONFIDENCE]
    logger.info(
        "Correct answer confidence range %0.3f - %0.3f" % (correct_confidence.min(), correct_confidence.max()))


def augment_handler(args):
    augmented_logs = augment_system_logs(args.logs, args.judgments)
    print_csv(augmented_logs)


if __name__ == "__main__":
    run()
