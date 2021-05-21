
"""
Distribution across PHI tags in input files:

usage:
python -m evaluation.phi_tag_distribution --datadir 'path-to-preprocessed-dataset' --synthdir 'path-to-synth-dir'

"""

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict
import pandas as pd
import argparse


#args how many words to process:
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', 
                    type = str,
                    help='Location of the folder with your preprocessed data.')
parser.add_argument('--synthdir',
                    type = str,
                    help='Location of the folder with synthetic data.')
args = parser.parse_args()

########################################################################


def getfile(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines


def phi_freqs(data) -> Dict[str, int]:
    pattern = r'<(\w*)START>'
    matches = re.finditer(pattern, ' '.join(data))
    phi_freq = Counter(match.group(1) for match in matches)
    return phi_freq

########################################################################


def main():
    #original_data = Path(__file__).parent.parent.parent / 'data' / 'preprocessed'
    original_data = Path(args.datadir)

    files = [
        original_data / 'train.txt',
        original_data / 'valid.txt',
        original_data / 'test.txt'
    ]

    #synth_path = Path(sys.argv[1])
    synth_path = Path(args.synthdir)
    synth_files = list(synth_path.glob('*.phi_cleaned.txt'))
    print(synth_files)
    files = files + synth_files

    colnames = []
    phi_per_file = []
    for file in files:
        colnames.append(file.stem)
        phi_freq = phi_freqs(getfile(file))
        phi_per_file.append(phi_freq)

    df = pd.DataFrame(phi_per_file, index=colnames).T.fillna(0).astype(int)
    for col in colnames:
        df[f'{col}_relative'] = df[col] / df[col].sum()

    model_output_path = synth_path.parent
    output_path = model_output_path / 'evaluation' / 'phi-eval.csv'
    df.to_csv(output_path, sep='\t')


if __name__ == "__main__":
    main()
