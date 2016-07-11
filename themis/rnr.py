from watson_developer_cloud import RetrieveAndRankV1 as RetriveandRank
from themis import logger, to_csv, pretty_print_json

def create_cluster(url, username, password):
    name = "Solr Cluser"
    rnr = RetriveandRank(url=url, username=username, password=password)
    cluster = rnr.create_solr_cluster(cluster_name=name)
    logger.info(pretty_print_json(cluster))
    return cluster["solr_cluster_id"]

def create_config(url, username, password,c_id):
    pass