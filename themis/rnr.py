from watson_developer_cloud import RetrieveAndRankV1 as RetriveandRank
from themis import logger, to_csv, pretty_print_json
import time
import json
import csv
import pandas
import requests, os
import subprocess
import shlex
import urllib

def create_cluster(url, username, password,cluster_name):
    if not cluster_name:
        cluster_name = "solr_cluster"
    rnr = RetriveandRank(url=url, username=username, password=password)
    cluster = rnr.create_solr_cluster(cluster_name=cluster_name)
    logger.info('Creating solr cluster....')

    # waiting for cluster to be ready
    end = time.time() + 600
    try:
        while time.time() < end:
            if(cluster_status(url,username,password,cluster) == 'READY'):
                logger.info('Cluster created successfully and ready to use. Cluster Id: %s' % cluster['solr_cluster_id'])
                break
            time.sleep(10)
    except:
        logger.info('Error in cluster creation')

def cluster_status(url,username,password,cluster):
    rnr = RetriveandRank(url=url, username=username, password=password)
    return rnr.get_solr_cluster_status(cluster['solr_cluster_id'])['solr_cluster_status']


def create_config(url, username, password,c_id,path,schema_file,corpus_file,config_name,collection_name):

    # create config
    if not config_name:
        config_name = "solr_configuration"
    try:
        zip_file = open(os.path.join(path,schema_file),'rb')
    except:
        logger.info("Error in uploading zip file from:  %s" % os.path.join(path,schema_file))
        exit()

    rnr = RetriveandRank(url=url, username=username, password=password)
    config = rnr.create_config(c_id,config_name,zip_file)
    logger.info(pretty_print_json(config['message']))

    # create collection
    if not collection_name:
        collection_name = "solr_collection"
    collection = rnr.create_collection(c_id,collection_name,config_name)
    if collection['success']:
        logger.info('Collection successfully created as: %s'%collection_name)
    else:
        exit()

    # convert corpus to json format
    convert_corpus_to_json(os.path.join(path,corpus_file))
    logger.info('Corpus file successfully converted to json as: corpus.json')

    # add documents to collection
    logger.info('Adding documents to collection...')
    status = upload_corpus(url, username, password, c_id, os.path.join(path,'corpus.json'),collection_name)
    if status.__contains__('Error Code'):
        logger.info('Error in uploading documents to collection')
        exit()
    logger.info('Documents added to the collection successfully')



def create_ranker(url, username, password,c_id,path,truth,ranker_name,collection_name):

    if not collection_name:
        collection_name = "solr_collection"

    # modify truth file to add relevance
    logger.info('Adding relevance to ground truth....')
    create_truth(os.path.join(path,truth),path)
    logger.info('New file generated with relevance: rnr_relevance.csv')

    # convert relevance file into rnr format file
    logger.info('Converting file....')
    ranker_training_file(url, username, password, c_id, collection_name, os.path.join(path,'rnr_relevance.csv'),path)
    logger.info('Conversion completed successfully. New file generated : training.txt')


    # ranker creation and training
    '''
    if not ranker_name:
        ranker_name = "rnr_ranker"
    rnr = RetriveandRank(url=url, username=username, password=password)
    ranker = rnr.create_ranker(os.path.join(path,'training.txt'),ranker_name)
    logger.info('Ranker instance is created successfully')
    logger.info(ranker['status_description'])
    logger.info(pretty_print_json(ranker))


    '''

    # Train the ranker with the training data that was generate above from the query/relevance input
    logger.info('Creating and training ranker...')
    cred = username + ":" + password
    if not ranker_name:
        ranker_name = "rnr_ranker"

    ranker_curl_cmd = 'curl -k -X POST -u %s -F training_data=@%s -F training_metadata="{\\"name\\":\\"%s\\"}" %s' % (
     cred, os.path.join(path,'training.txt'), ranker_name, url+'/v1/rankers')

    process = subprocess.Popen(shlex.split(ranker_curl_cmd), stdout=subprocess.PIPE)
    response = process.communicate()[0]
    #print response[14:34]

    # waiting for ranker to be ready
    end = time.time() + 900
    try:
        while time.time() < end:
            if (ranker_status(url, username, password, response) == 'Available'):
                logger.info('Ranker is created successfully and trained with Ranker Id: %s' % response[14:34])
                break
            time.sleep(10)
    except:
        logger.info('Error in ranker creation')

# Ranker status
def ranker_status(url, username, password, ranker):
    rnr = RetriveandRank(url=url, username=username, password=password)
    return rnr.get_ranker_status(ranker[14:34])['status']

# modify truth file to add relevance
def create_truth(truth,path):
    df = pandas.read_csv(truth)
    df = df[['Question', 'Answer Id']]
    df['Question'] = df['Question'].str.replace(":", "")
    df['Relevance'] = 4
    df.to_csv(os.path.join(path,'rnr_relevance.csv'), index = False, header = False)

# convert truth file in rnr format from relevance file
def ranker_training_file(url,username,password,c_id,collection_name,relevance_file,path):
    url = url+'/v1/'+'solr_clusters/'+c_id+'/solr/'+collection_name+'/fcselect/'
    number_row = '10'
    cred = username+":"+password
    with open(relevance_file, 'rb') as csvfile:
        add_header = 'true'
        question_relevance = csv.reader(csvfile)
        with open(os.path.join(path,'training.txt'), "a") as training_file:
            print ('Generating training data...')
            for row in question_relevance:
                question = urllib.quote(row[0])
                relevance = ','.join(row[1:])
                curl_cmd = 'curl -k -s %s -u %s -d "q=%s&gt=%s&generateHeader=%s&rows=%s&returnRSInput=true&wt=json" "%s"' % (
                '-v', cred, question, relevance, add_header, number_row, url)

                process = subprocess.Popen(shlex.split(curl_cmd),stdout=subprocess.PIPE)
                output = process.communicate()[0]

                try:
                    parsed_json = json.loads(output)
                    if 'RSInput' in parsed_json:
                        training_file.write(parsed_json['RSInput'])
                    else:
                        continue
                except:
                    print ('Command:')
                    print (curl_cmd)
                    print ('Response:')
                    print (output)
                    print (question)
                    raise
                add_header = 'false'
    print ('Generating training data complete.')

# query ranker change
def query_ranker(url, username, password,c_id, ranker_id, query,collection_name):
    if not collection_name:
        collection_name = "solr_collection"
    print query
    cred = (username, password)
    resp = requests.get(url+'/v1/solr_clusters/'+c_id+'/solr/'+collection_name+'/fcselect?ranker_id='+ranker_id+'&q='+query+'&wt=json', auth=cred)
    return resp.text

# query untrained ranker change
def query_untrained_ranker(url, username, password,c_id, query,collection_name):
    cred = (username, password)
    resp = requests.get(url+'/v1/solr_clusters/'+c_id+'/solr/'+collection_name+'/fcselect?q='+query+'&wt=json', auth=cred)
    return resp.text

# query trained rnr
def query_trained_rnr(url, username, password,c_id, ranker_id, question):
    answers = []
    with open(question, 'r') as f:
        input_reader = csv.DictReader( f, delimiter=',' )
        rows = [r for r in input_reader]
    print "number of sample questions: " ,len(rows)
    for row in rows:
        query = row['Question'].replace("#", "").replace(":","")
        resp = query_ranker(url, username, password,c_id, ranker_id, query)
        try:
            res = json.loads(resp)
        except:
            print resp.text
            answers.append([query,0,"Query Error"])
            continue

        if res['response']['docs']:
            answers.append([query,res['response']['docs'][0]['score'],res['response']['docs'][0]['Answer'][0]])
        else:
            answers.append([query, 0, "No docs returned from RnR"])

    with open('answers.trained.rnr.csv', 'w') as f:
        output_writer = csv.writer(f)
        output_writer.writerow(['Question', 'Confidence', 'Answer'])
        for r in answers:
            output_writer.writerow((r))

# query untrained rnr
def query_untrained_rnr(url, username, password,c_id, question,collection_name):
    if not collection_name:
        collection_name = "solr_collection"
    answers = []
    with open(question, 'r') as f:
        input_reader = csv.DictReader( f, delimiter=',' )
        rows = [r for r in input_reader]
    print "number of sample questions: " ,len(rows)
    for row in rows:
        query = row['Question'].replace("#", "")
        resp = query_untrained_ranker(url, username, password,c_id, query,collection_name)
        try:
            res = json.loads(resp)
        except:
            print resp.text
            answers.append([query,0,"Query Error"])
            continue

        if res['response']['docs']:
            answers.append([query,res['response']['docs'][0]['score'],res['response']['docs'][0]['Answer'][0]])
        else:
            answers.append([query, 0, "No docs returned from RnR"])

    with open('answers.untrained.rnr.csv', 'w') as f:
        output_writer = csv.writer(f)
        output_writer.writerow(['Question', 'Confidence', 'Answer'])
        for r in answers:
            output_writer.writerow((r))

# convert csv to json
def convert_corpus_to_json(corpus):
    df = pandas.read_csv(corpus)
    df = df[['Answer Id', 'Answer']]
    f = open('corpus_temp.json', 'w')
    df.to_json(f, orient='records')

    with open('corpus_temp.json', 'r') as f:
        data = json.load(f)
    a = []
    for row in data:
        temp = {"doc": row}
        a.append(("add", temp))

    out = '{%s' % ',\n'.join(['"{}": {}'.format(action, json.dumps(dictionary)) for action, dictionary in a])
    out = out + ',"commit" : { }}'

    # can be commented as not required
    with open('corpus.json', 'w') as f:
        f.write(out)

# upload documents to collection
def upload_corpus(url, username, password,c_id,corpus,collection_name):
    cred = (username, password)
    headers = {
    'Content-Type': 'application/json',
    }
    data = open(corpus)
    resp = requests.post(url+'/v1/solr_clusters/'+c_id+'/solr/'+collection_name+'/update', headers=headers, data=data, auth = cred)
    return resp.text

# upload test corpus
def upload_test_corpus(url, username, password,c_id,collection_name):
    cred = (username, password)
    resp = requests.get(url+'/v1/solr_clusters/'+c_id+'/solr/'+collection_name+'/select?q=*:*&fl=*&df=Answer', auth = cred)
    return resp.text









