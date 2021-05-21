
import json
import os
import re
import zipfile
from pathlib import Path

from pprint import pprint
import sys
import argparse

#import flair
import pandas as pd
from tqdm import tqdm

#import jsonlines
from deidentify.base import Document
from deidentify.taggers import FlairTagger
from deidentify.tokenizer import TokenizerFactory
from deidentify.util import surrogate_annotations



###########################################################################

DATA_PATH = Path(__file__).parent.parent.parent / 'data'
SECTORMAP_PATH = DATA_PATH / 'external' / 'sectors.csv'
INCLUDED_CUSTOMERS_PATH = DATA_PATH / 'interim' / 'included_customers.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--dump_path',
                    type = bool,
                    default=False,
                    help='If True, script takes Nedap data dump from environment variable as input.')
parser.add_argument('--dummy',
                    type=str,
                    default = DATA_PATH / 'external' / 'dummy_data.csv',
                    help='Define custom path for input csv, default is dummy dataset.')


###########################################################################

# zipped jsonl 2 pandas
def iter_jsonl(fp, include_header=False):
    header = json.loads(fp.readline())
    if include_header:
        yield header
    for line in fp:
        yield json.loads(line)


def iter_documents_zip(customer_dump, include_header=False):
    customer_dump = Path(customer_dump)
    with zipfile.ZipFile(customer_dump, 'r') as archive:
        with archive.open(customer_dump.name.replace(".zip", ".jsonl")) as corpus:
            yield from iter_jsonl(corpus, include_header=include_header)


# keep track of processed files
def logit(log_path, customer_code):
    with open(log_path, "a") as log:
        log.write(customer_code)
        log.write("\n")


# load log and update to-do list
def update2do(log_path, customer_codes):
    with open(log_path, "r") as f:
        logs = f.read().split('\n')

    to_do_list = [c for c in customer_codes if not c in logs]
    return to_do_list


def annotate_in_text(doc):
    text = ''
    current_end = 0
    for annotation in doc.annotations:
        text += doc.text[current_end:annotation.start]
        
        text += f' <{annotation.tag}START> {annotation.text} <{annotation.tag}END> '
        current_end = annotation.end
        
    text += doc.text[current_end:]
    text = text.replace('  ',' ')
    return text


def add_par_tok(text):
    text = text.replace('\n', ' <PAR> ')
    if text.endswith(" <PAR> "):
        text = text[:-7]
    return text


###########################################################################

def produce_data(df, customer_code, customer_domain, out_path):

    # load flair tagger
    tagger = FlairTagger(
        model='model_bilstmcrf_ons_fast-v0.1.0',
        tokenizer=TokenizerFactory().tokenizer(corpus='ons', disable=("tagger", "ner")),
        verbose=True,
        mini_batch_size=256,
    )

    # prepare df
    df['text'] = df.text.apply(lambda x: str(x).replace('\t', ' '))
    # add report id: 'customercode-rowindex'
    df['report_id'] = df.index.astype(str) + '-' + customer_code
    df['customer_domain'] = customer_domain

    # Wrap text in documents for deidentify & apply deidentify pipeline with surrogate replacement
    deidentify_docs = [Document(name='', text=doc, annotations=[]) for doc in df['text']]
    annotated_docs = tagger.annotate(deidentify_docs)
    iter_docs = surrogate_annotations(docs=annotated_docs, seed=1, errors='ignore')
    surrogate_docs = list(iter_docs)
    df['annotated_doc'] = surrogate_docs
    
    # Use custom function to embed annotations as string tokens in text + exchange newlines with <PAR>
    df['annotated_text'] = df['annotated_doc'].apply(lambda doc: add_par_tok(annotate_in_text(doc)))

    # Done. write to files:
    new_df = df[['report_id', 'annotated_text', 'type',
                 'client_age', 'client_gender', 'customer_domain']]
    
    with open(out_path, 'a') as f:
        json_string = new_df.to_json(lines=True, orient='records')
        f.write(json_string + '\n')


###########################################################################


def main():

    # original Nedap data dump
    if args.dump_path:
        dump_path = Path(os.environ['CUSTOMER_DUMPS'])
        
        log_path = DATA_PATH / 'interim' / 'log.txt'
        # write to these files in the /data directory:
        out_path = DATA_PATH / 'interim' / 'annotated_ehr.jsonl'

        # load customer domain mappings
        mapping_df = pd.read_csv(SECTORMAP_PATH)
        customer_domain_mapping = mapping_df.set_index('Customer Code')['Sector'].to_dict()
        
        # get customer codes to be included (& update to-do)
        with open(INCLUDED_CUSTOMERS_PATH) as fin:
            customer_codes = [code.strip() for code in fin.readlines() if code.strip()]
        to_do_list = update2do(log_path, customer_codes)
        print('Restart from customer: '+str(to_do_list[0])+'...')
        
        # iterate through to-do list of customer codes
        for customer_code in tqdm(to_do_list):
            print(customer_code)

            # some customer codes do not appear in the map, so add NA values to those:
            customer_domain = customer_domain_mapping.get(customer_code, '')

            # get data
            dump_zip = dump_path / f'{customer_code}.zip'
            docs = iter_documents_zip(dump_zip)
            df = pd.DataFrame(docs, columns=['text', 'type', 'client_gender', 'client_age'])

            #process and log
            produce_data(df, customer_code, customer_domain, out_path)
            logit(log_path, customer_code)

    # use dummy (or different) dataset
    else:
        data_path = Path(args.dummy) if type(args.dummy)==str else args.dummy
        print('Using data from: {}'.format(data_path))
        out_path = DATA_PATH / 'interim' / 'annotated_dummy.jsonl'
        df = pd.read_csv(data_path)
        df = df[:50]
        produce_data(df,'dummy','dummy_domain',out_path)
        


if __name__ == "__main__":
    args = parser.parse_args()
    main()
