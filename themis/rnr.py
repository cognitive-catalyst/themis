from watson_developer_cloud import RetrieveAndRankV1 as RetriveandRank
from themis import logger, to_csv, pretty_print_json
import time
import json
import csv
import pandas
import requests, os

def create_cluster(url, username, password,cluster_name):
    if not cluster_name:
        cluster_name = "solr_cluster"
    rnr = RetriveandRank(url=url, username=username, password=password)
    cluster = rnr.create_solr_cluster(cluster_name=cluster_name)
    logger.info('Cluster creation starting....')
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
        logger.info("Error opening zip file from:  %s" % os.path.join(path,schema_file))
        exit()

    rnr = RetriveandRank(url=url, username=username, password=password)
    config = rnr.create_config(c_id,config_name,zip_file)
    logger.info(pretty_print_json(config['message']))

    # create collection
    if not collection_name:
        collection_name = "solr_collection"
    collection = rnr.create_collection(c_id,collection_name,config_name)
    if collection['success']:
        logger.info('Collection successfully created with name: %s'%collection_name)
    else:
        exit()

    # convert corpus to json format
    convert_corpus_to_json(os.path.join(path,corpus_file))
    logger.info('corpus file successfully converted to json as: corpus.json')

    # add documents to collection
    logger.info('Adding documents to collection...')
    status = upload_corpus(url, username, password, c_id, os.path.join(path,'corpus.json'),collection_name)
    #logger.info(pretty_print_json(status))
    if status.__contains__('Error Code'):
        logger.info('Error in uploading documents to collection')
        exit()
    logger.info('Documents added to the collection successfully')

def create_ranker(url, username, password,path,truth,ranker_name):
    """
    # upload test corpus
    upload_test_corpus(url, username, password, c_id)
    logger.info('Test corpus uploaded')
    """
    # modify truth file to add relevance
    logger.info('Converting ground truth file for rnr....')
    create_truth(os.path.join(path,truth))
    logger.info('Conversion completed')

    # ranker
    if not ranker_name:
        ranker_name = "rnr_ranker"
    rnr = RetriveandRank(url=url, username=username, password=password)
    ranker = rnr.create_ranker(os.path.join(path,'rnr_truthincorpus.csv'),ranker_name)
    logger.info('Ranker instance is created successfully')
    logger.info(ranker['status_description'])
    logger.info(pretty_print_json(ranker))
    # waiting for ranker to be ready
    end = time.time() + 900
    try:
        while time.time() < end:
            if(ranker_status(url,username,password,ranker) == 'Available'):
                logger.info('Ranker is created successfully and trained with Ranker Id: %s'%ranker['ranker_id'])
                break
            time.sleep(10)
    except:
        logger.info('Error in ranker creation')

# Ranker status
def ranker_status(url, username, password, ranker):
    rnr = RetriveandRank(url=url, username=username, password=password)
    return rnr.get_ranker_status(ranker['ranker_id'])['status']

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

# modify truth file to add relevance
def create_truth(truth):
    df = pandas.read_csv(truth)
    df = df[['Question', 'Answer Id']]
    df['Question'] = df['Question'].str.replace(":", "")
    df['Relevance'] = 4
    df.to_csv('rnr_truthincorpus.csv', index = False, header = False)







