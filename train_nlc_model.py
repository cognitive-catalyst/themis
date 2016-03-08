#!/usr/bin/env python

"""Train an NLC model"""
import argparse
import json

from watson_developer_cloud import NaturalLanguageClassifierV1 as NaturalLanguageClassifier


def train_nlc_model(nlc, training_file, name):
    r = nlc.create(training_data=training_file, name=name)
    print((json.dumps(r, indent=2)))
    return r["classifier_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", type=str, help="NLC username")
    parser.add_argument("password", type=str, help="NLC password")
    parser.add_argument("truth", type=argparse.FileType("rb"), help="truth file")
    parser.add_argument("name", type=str, help="model name")
    parser.add_argument("--url", type=str,
                        default="https://gateway-s.watsonplatform.net/natural-language-classifier/api",
                        help="NLC url")
    args = parser.parse_args()

    nlc = NaturalLanguageClassifier(url=args.url, username=args.username, password=args.password)
    train_nlc_model(nlc, args.truth, args.name)
