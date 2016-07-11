from watson_developer_cloud import RetrieveAndRankV1 as RetriveandRank
from themis import logger, to_csv, pretty_print_json

def create_cluster(url, username, password):
    name = "solr_cluster"
    rnr = RetriveandRank(url=url, username=username, password=password)
    cluster = rnr.create_solr_cluster(cluster_name=name)
    logger.info(pretty_print_json(cluster))
    #status = rnr.get_solr_cluster_status(cluster["solr_cluster_id"])
    return cluster["solr_cluster_id"]

def create_config(url, username, password,c_id,zip_path):

    # create config
    config_name = "solr_configuration"
    try:
        zip_file = open(zip_path,'rb')
    except:
        print("Error opening zip file from:  %s" % zip_path)

    rnr = RetriveandRank(url=url, username=username, password=password)
    config = rnr.create_config(c_id,config_name,zip_file)
    logger.info(pretty_print_json(config["message"]))

    #list config
    list = rnr.list_configs(c_id)
    logger.info(pretty_print_json(list["solr_configs"]))

    # status of config
    get = rnr.get_config(c_id,config_name)
    logger.info(pretty_print_json(get))

    # create collection
    collection_name = "solr_collection"
    collection = rnr.create_collection(c_id,collection_name,config_name)
    logger.info(pretty_print_json(collection))

