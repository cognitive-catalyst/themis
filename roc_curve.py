"""Create a ROC curve from judged question/anwer pairs"""

import argparse

import pandas


def roc_curve(judgements, correct):
    # Get binary correct/incorrect
    # Get confidence
    # plot with sklearn.metrics.roc_curve
    pass


def csv_file(filename):
    return pandas.read_csv(filename, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judgements", type=csv_file, help="judgements file created by annotation_assist_results.py")
    parser.add_argument("--correct", type=int, default=50, help="cutoff value for a correct score, default 50")
    parser.add_argument("--plot", action="store_true", help="draw the ROC curve")
    args = parser.parse_args()

    roc_curve(args.judgements, args.correct)
