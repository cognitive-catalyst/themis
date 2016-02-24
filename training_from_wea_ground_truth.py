#!/usr/bin/env python

"""Extract training data from the the WEA ground truth file"""
import argparse

import pandas

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("ground_truth_file", type=argparse.FileType(), help="ground truth csv file from WEA")
args = parser.parse_args()

ground_truth = pandas.DataFrame.from_csv(args.ground_truth_file, index_col=False)
print(ground_truth.to_csv(columns=["QUESTION", "PAU_ID"], encoding="utf-8", index=False))
