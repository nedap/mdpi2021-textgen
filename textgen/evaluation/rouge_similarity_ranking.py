# iterates through synth corpus and finds most similar training doc based on Rouge-n recall.
# Results are saved in pandas df -> csv

import argparse
import os.path
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rouge_score import rouge_scorer
from tqdm import tqdm

from textgen.evaluation.util import tqdm_joblib

parser = argparse.ArgumentParser()
parser.add_argument('--realpath',
                    help='path to real data')
parser.add_argument('--synthpath',
                    help='path to synthetic data')
parser.add_argument('--outdir',
                    help='path to directory where output of ranking should be saved')
parser.add_argument('--n_jobs',
                    help='Number of synthetic documents in process in parallel',
                    type=int,
                    default=1)
parser.add_argument('--batch_size',
                    help='Number of syntehtic documents to process before saving results',
                    type=int,
                    default=32)


########################################################################


RougeMatch = namedtuple('RougeMatch', ['query', 'result', 'precision', 'recall', 'fmeasure'])


def get_best_match_rouge(trainingset, synth_ehr):
    # get all N-ROUGE scores (n-gram = 5)
    scorer = rouge_scorer.RougeScorer(['rouge5'], use_stemmer=False)

    highscore = 0.0
    best_match = None
    # iterate through all real notes & compare synth with each
    for real_ehr in trainingset:
        score = scorer.score(synth_ehr, real_ehr)  # synth vs. real1/real2/real3...
        precision, recall, fmeasure = score['rouge5']
        # rank using recall, we care less about precision (i.e. amount of non-overlap doesn't matter)
        if recall > highscore:
            highscore = recall
            best_match = RougeMatch(synth_ehr, real_ehr, precision, recall, fmeasure)

    # in the unlikely case that no similarity was found at all:
    if best_match is None:
        best_match = RougeMatch(synth_ehr, None, 0, 0, 0,)

    return best_match


def write_batch_results(matches, index_start, out_file):
    df = pd.DataFrame(matches)
    # Instead of starting from 0, we start from counting from `index_start`.
    df.index = np.arange(index_start, index_start + len(df))

    if os.path.exists(out_file):
        df.to_csv(out_file, mode='a', header=False)
    else:
        df.to_csv(out_file)


def match_all_parallel(training_docs, synth_docs, out_file, n_jobs=1, batch_size=32):
    total_batches = int(len(synth_docs) / batch_size)

    try:
        df = pd.read_csv(out_file)
        # Start where we left off.
        start_index = len(df)
    except FileNotFoundError:
        # Start from scratch.
        start_index = 0

    with Parallel(n_jobs=n_jobs) as parallel:
        for i in range(start_index, len(synth_docs), batch_size):
            batch = synth_docs[i:i + batch_size]

            current_batch_index = int(i / batch_size)
            progress = f"batch = {current_batch_index}/{total_batches}"

            with tqdm_joblib(tqdm(desc=progress, total=len(batch))):
                matches = parallel(
                    delayed(get_best_match_rouge)(training_docs, doc) for doc in batch
                )

                # if we were to write this sequentially, this is how it would look like:
                # matches = []
                # for doc in batch:
                #     matches.append(get_best_match_rouge(training_docs doc))

                write_batch_results(matches, index_start=i, out_file=out_file)


########################################################################

def main(args):
    # get trainingdata
    with open(args.realpath) as f:
        training_docs = [line.strip() for line in f.readlines() if line.strip() != '']

    synthfilepath = Path(args.synthpath)
    with open(synthfilepath) as f:
        synth_docs = [line.strip() for line in f.readlines() if line.strip() != '']

    outfilepath = Path(args.outdir) / 'rougescores_5gram_{}.csv'.format(synthfilepath.stem)

    match_all_parallel(training_docs, synth_docs, outfilepath, args.n_jobs, args.batch_size)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
