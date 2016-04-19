# Themis Data Analysis

This package provides a suite of tools for evaluating question-answering systems: the 
[Watson Engagement Advisor (WEA)](http://www.ibm.com/smarterplanet/us/en/ibmwatson/engagement_advisor.html),
[Solr](http://lucene.apache.org/solr),
and the [Natural Language Classifier (NLC)](http://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/doc/nl-classifier/)

## Installation

You can install directly from github using the following command

    pip install git+https://github.com/cognitive-catalyst/themis.git

You can also install from source by running

    python setup.py install

## Example Usage

Here is a step-by-step example of a typical Themis experiment.
See `themis --help` for more details.

### Get Corpus and Truth

Download the corpus and the truth files from the Watson Experience Manager (XMGR) interface to WEA.

The corpus maps answer IDs to answer text.
To download it, run the following command.

    themis xmgr corpus XMGR-URL USERNAME PASSWORD

The `XMGR-URL` points to an XMGR project instance, e.g. `https://watson.ihost.com/instance/283/predeploy/$150dd167e4e`.
(Note that these may contain "$" characters in that have to be escaped on the command line.)
This will create a `corpus.csv` file.
The command may take multiple hours to run.
It saves intermediate state, so if it drops in the middle you can run it again and it will pick up where it left off.
Optionally you may specify a `--retries` parameter which automatically restarts a specified number of times.

The truth maps answer IDs to questions they are known to answer.
This is the information used to train the WEA instance and will be used to train the NLC model.
To download the truth file, run the following command.

    themis xmgr truth XMGR-URL USERNAME PASSWORD

This creates `truth.json` and `truth.csv` files.
The json file is a verbose archive of truth information, while the csv file is used in subsequent Themis commands.
Subsequent actions assume that the answer Ids referenced in the truth are all present in the corpus.
Sometimes this is not the case.
See `themis xmgr verify` for how to rectify this.


### Get Test Set from Usage Logs

Download a usage report from XMGR.
The `QuestionsData.csv` in the zip file contains records of the questions that were asked of Watson and the answers it
provided.
Use this to extract a set of questions that were asked to Watson along with the answers it gave.

    themis question extract QuestionsData.csv > qa-pairs.csv

This file also records the number of times each question was asked.

### Ask Questions to Various Systems

Now we ask the questions in the test set to various Q&A systems and compare the answers they return.

First we "ask" questions to Watson by reading the answers out of the usage logs.

    themis answer wea qa-pairs.csv answers.wea.csv qa-pairs.csv

The answers are written to `answers.wea.csv`.
The `qa-pairs.csv` file created in the previous step appears twice on the command line above because it is used as a
source of both questions and answers.

To ask questions to a Solr instance, import the corpus file using the example `solr/schema.xml`.

    themis answer solr qa-pairs.csv answers.solr.csv http://localhost:8983/solr/test

The answers are written to `answers.solr.csv`.

To ask questions of the NLC we must first train a model using the truth file downloaded from XMGR as training data.

    themis answer nlc train NLC-URL USERNAME PASSWORD truth.csv model-name
    
The `NLC-URL` points to an NLC API endpoint,
e.g. `https://gateway-s.watsonplatform.net/natural-language-classifier/api`.
This will return a model ID that can be used in subsequent commands.
Note that it may take some time to train the model before it is ready to be presented questions.
The Themis toolkit has utilities to manage NLC models.
See the help for more details.

After the model has been trained, you can submit questions to it.

    themis answer nlc use NLC-URL USERNAME PASSWORD qa-pairs.csv answers.nlc.csv MODEL-ID corpus.csv

If the command to ask questions to either Solr or NLC fails you can rerun it and it will pick up where it left off.

### Submit Answers to Annotation Assist

A human annotator needs to judge whether the answers to the questions returned by the various systems are correct.
The [Annotation Assist](https://github.com/cognitive-catalyst/annotation-assist) tool provides a visual interface for
doing this.

Annotation Assist takes corpus and question/answer pairs files as input.
To generate the corpus file run

    themis judge corpus corpus.csv > annotation-assist.corpus.json

To generate the question/answer pairs file run

    themis judge pairs answers.wea.csv answers.solr.csv answers.nlc.csv > annotation-assist.pairs.json

Annotation is a time consuming task, and it may be prohibitively difficult to annotate all system answers.
You may optionally take a sample of questions with the following command

    themis question sample qa-pairs.csv 1000 > sample.1000.csv

This will sample 1000 unique questions from the set of all questions in `qa-pairs.csv`.
Questions are sampled from a distribution determined by the frequency with which they were asked in the usage logs.
The following command will generate annotation assist question/answer input for just these 1000 questions.

    themis judge --questions sample.1000.csv pairs answers.wea.csv answers.solr.csv answers.nlc.csv > annotation-assist.pairs.json

It is also possible to incorporate previous judgments.
See the command help for details.

After annotating the answers you can download the results in a file: call it `annotation-assist.judgments.csv`.
This can be converted into the format used by Themis with the following command.

    themis judge interpret annotation-assist.judgments.csv > judgments.csv

### Plot Curves

The correctness judgments along with the question frequencies can then be used to plot precision and
(receiver operating characteristic) ROC curves.

The following command generates precision curve data for the WEA, Solr, and NLC systems using the judgments obtained
from Annotation Assist.

    themis plot precision qa-pairs.csv answers.wea.csv answers.solr.csv answers.nlc.csv --labels WEA Solr NLC --judgments judgments.csv

This will create `WEA.precision.csv`, `Solr.precision.csv`, and `NLC.precision.csv` files containing threshold values
and X and Y scatter plot values.
ROC curves can be generated with the `roc` option in the place of `precision`. 
If you specify the `--draw` option, the curves will be drawn.

## License

See [License.txt](License.txt).
