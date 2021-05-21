
from elasticsearch_dsl import connections, Search
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--queries',
                    type=str,
                    default='output/pytorch_lstm/synthetic_data/test_queries.txt',
                    help='location of the text queries, e.g. a set of synthetic notes')
parser.add_argument('--outf',
                    type=str,
                    default='output/pytorch_lstm/evaluation/elastic_query_matches.txt',
                    help='filepath of output file with match results'
                    )
parser.add_argument('--n_hits',
                    type=int,
                    default=1,
                    help='top n hits per query will be written to results')

args = parser.parse_args()

#SYNTH_PATH = 'output/pytorch_lstm/synthetic_data/test_queries.txt'
SYNTH_PATH = args.queries
#OUT_PATH = 'output/pytorch_lstm/evaluation/elastic_query_matches_TEST.txt'
OUT_PATH = args.outf

#########################


def query(client,querytext,n_hits):
    file = open(OUT_PATH, 'a')
    file.write('querytext: \n{}\n\n'.format(querytext))
    s = Search().using(es).query("match", body=querytext) #content = querytext
    s.execute()
    for hit in s[:n_hits]:
        score = hit.meta.score
        file.write('\nscore: {}'.format(score))
        text = hit.body #hit.content
        file.write('\n{}\n\n'.format(text))
    file.write('\n***************************\n\n')
    file.close()




if __name__ == '__main__':
    es = connections.create_connection()

    #querytext = 'Mevr. heeft vannacht om half 12 diarree gehad , vanmiddag is met een collega naar buiten gelopen . We zijn bij mevr. op de bank gaan zitten en wisselligging .'
    #query(es,querytext,2)

    with open(SYNTH_PATH,'r') as fin:
        queries = fin.readlines()
        queries = [l.strip() for l in queries if l.strip()]

    for text in queries:
        query(es,text,args.n_hits)


"""
In Kibana:

GET /_search
{
  "query": {
    "match": {
      "content": “YOUR TEXT QUERY”
    }
  }
}

"""
