import csv


with open('all_systems.results.csv', 'r') as f:
    results = [r for r in csv.reader(f)]

with open('deakin.corpus.csv', 'r') as f:
    corpus = [r for r in csv.reader(f)]

with open('sample-test-set.1000.csv', 'r') as f:
    sample = [r for r in csv.reader(f)]

questions = [q[0] for q in sample]

#  Storing ID and PAU text
corpus_dict = {}
for c in corpus:
    if len(c) > 0:
        corpus_dict[c[0]] = c[3]

annotation_list = [['DateTime', 'QuestionText', 'TopAnswerText', 'TopAnswerConfidence']]

for result in results:
    if len(result) > 0 and result[0].lower() != 'question':
        if result[0] in questions:
            # Storing ID and confidence
            temp_dict={}
            temp_dict[result[1]] = result[2]
            if result[3] not in temp_dict:
                temp_dict[result[3]] = result[4]
            if result[5] not in temp_dict:
                temp_dict[result[5]] = result[6]

            for key, value in temp_dict.items():
                if key in corpus_dict:
                    annotation_list.append(['01/01/2016', result[0], corpus_dict[key], value])

answer_writer = csv.writer(open('AnnotationAssist.csv', 'w'))
for row in annotation_list:
    answer_writer.writerow(row)