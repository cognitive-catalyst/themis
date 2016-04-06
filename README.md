# Themis Data Analysis

This package provides a suite of tools for evaluating question-answering systems.

## Installation

You can install directly from github using the following command

    pip install git+https://github.ibm.com/watson-prototypes/themis.git

## Example Usage

Here is a step-by-step example of a typical Themis experiment.

### Get Corpus and Truth

Download the corpus and the truth files from the Watson Experience Manager (XMGR).
The corpus maps answer IDs to answer text while the truth maps answer IDs to questions they are known to answer.

    themis xmgr https://watson.ihost.com/instance/283/predeploy/\$150dd167e4e xmgr_administrator PASSWORD

This creates corpus.csv and truth.csv files.
The process may take several minutes.
If it stops in the middle for any reason you can run the same command line and it will pick up where it left off.
You may also specify an optional number of retries.
If the download fails, the script will sleep for one minute then try again the specified number of times.

### Get Test Set from Usage Logs

Download a usage report from XMGR.
The `QuestionsData.csv` in the zip file contains records of the questions that were asked of Watson and the answers it
provided.
Use this to extract a test set of questions.

    themis test-set QuestionsData.csv > test-set.csv

This will contain a list of all the questions and the number of times they were asked.

### Ask Questions to Various Systems

Now we ask the questions in the test set to various Q&A systems and compare the answers they return.

First we "ask" questions to Watson by reading the answers out of the usage logs.

    themis wea test-set.csv QuestionsData.csv > wea.answers.csv

To ask questions to a [Solr](http://lucene.apache.org/solr) system, import the corpus file using the example `solr/schema.xml`.

    themis solr http://localhost:8983/solr/test test-set.csv solr.answers.csv

Unlike many other Themis commands, this writes to an output file instead of standard out.
If this command stops in the middle you can run the same command line and it will pick up where it left off.

To ask a questions of the
[Natural Language Classifier](http://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/doc/nl-classifier/) we
must first train a model using the truth file downloaded from XMGR as training data.

    themis nlc train https://gateway-s.watsonplatform.net/natural-language-classifier/api USERNAME PASSWORD truth.csv
    
This will return a model ID that can be used in subsequent commands.
Note that it may take some time to train the model before it is ready to be presented questions.
The Themis toolkit has utilities to maintain various NLC models.
See the help for more details.

After the model has been trained, you can submit questions to it.

    themis nlc use https://gateway-s.watsonplatform.net/natural-language-classifier/api USERNAME PASSWORD MODEL-ID test-set.csv nlc.answers.csv corpus.csv

### Submit Answers to Annotation Assist

Run the `annotate` command to generate files that can be used by the
[Annotation Assist](https://github.com/cognitive-catalyst/annotation-assist) tool.

    themis annotate corpus.csv wea.answers.csv solr.answers.csv nlc.answers.csv

This produces `annotation_assist_corpus.json` and `annotation_assist_answers.csv` files.
After annotating the answers you can download the results in a file: call it `judgments.csv`.
 
### Plot Curves

Use the `curves` command to generate data for precision and ROC curves.
For a given set of answers this generates a files containing threshold values, and `x` and `y` coordinates.

    themis curves roc test-set.csv judgments.csv wea.answers.csv > wea.roc.csv
    themis curves precision test-set.csv judgments.csv nlc.answers.csv > nlc.precision.csv
    etc.

You can use these files as inputs into your preferred plotting program.
You can also plot them with the `draw` command.

    themis draw wea.precision.csv solr.precision.csv nlc.precision.csv --labels WEA Solr NLC
