
"""
using elasticsearch dsl
https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html
"""

from datetime import datetime
import argparse
from tqdm import tqdm

# from elasticsearch import Elasticsearch
from elasticsearch_dsl import Document, Text, connections


########################################################


parser = argparse.ArgumentParser()
parser.add_argument('--data',
                    type=str,
                    help='location of the notes to be added to the search index')
parser.add_argument('--indexname',
                    type=str,
                    default='ehr-data-index',
                    help='name of search-index')
args = parser.parse_args()

ALIAS = args.indexname
PATTERN = ALIAS + "-*"


########################################################

# specify custom bm25 similarity measure in index class of EHR_note class
class EHRNote(Document):
    body = Text(similarity='my_bm25')

    class Index:
        # we will use an alias instead of the index
        name = ALIAS
        # analyzers
        settings = {
        'index': {
            'similarity': {
                'my_bm25': {
                    'type': 'BM25',
                    'b': 0.75,
                    'k':1.2
                    }
                }
            }
        }


def migrate(move_data=True, update_alias=True):
    """
    Upgrade function that creates a new index for the data. Optionally it also can
    (and by default will) reindex previous copy of the data into the new index
    (specify ``move_data=False`` to skip this step) and update the alias to
    point to the latest index (set ``update_alias=False`` to skip).
    Note that while this function is running the application can still perform
    any and all searches without any loss of functionality. It should, however,
    not perform any writes at this time as those might be lost.
    """
    # construct a new index name by appending current timestamp
    next_index = PATTERN.replace("*", datetime.now().strftime("%Y%m%d%H%M%S%f"))

    # get the low level connection
    #es = es
    es = connections.create_connection()

    # create new index, it will use the settings from the template
    es.indices.create(index=next_index)

    if move_data:
        # move data from current alias to the new index
        es.reindex(
            body={"source": {"index": ALIAS}, "dest": {"index": next_index}},
            request_timeout=3600,
        )
        # refresh the index to make the changes visible
        es.indices.refresh(index=next_index)

    if update_alias:
        # repoint the alias to point to the newly created index
        es.indices.update_aliases(
            body={
                "actions": [
                    {"remove": {"alias": ALIAS, "index": PATTERN}},
                    {"add": {"alias": ALIAS, "index": next_index}},
                ]
            }
        )


def setup():
    # create an index template
    index_template = EHRNote._index.as_template(ALIAS, PATTERN)

    # upload the template into elasticsearch potentially overriding the one already there
    index_template.save()

    # create the first index if it doesn't exist
    if not EHRNote._index.exists():
        migrate(move_data=False)




if __name__ == '__main__':

    #initiate connection to elasticsearch
    connections.create_connection()

    #SETUP
    setup()

    ##ADD NOTES TO INDEX
    #import data
    with open(args.data,'r') as fin:
        training_notes = fin.readlines()
        training_notes = [l.strip() for l in training_notes if l.strip()]

    for note in tqdm(training_notes):
        ehr = EHRNote(body = note)
        ehr.save(refresh=True)

    migrate()
