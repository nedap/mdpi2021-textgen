"""
e.g.:
python -m textgen.evaluation.postprocessing.prepdata4deidentify_batched --input_file output/dummy/pytorch_lstm_40ep_tied/synthetic_data/synth10000w.phi_cleaned.txt --output_dir output/dummy/pytorch_lstm_40ep_tied/deidentify_data/synthetic --n_batch 5

save synthetic data brat format IN BATCHES
"""
import argparse
import re
from pathlib import Path
import sys

from deidentify.base import Annotation
from deidentify.dataset.brat import write_brat_document
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file',
                    help='Input file of in-text annotated EHR notes (synth must have been phi-tag-checked)')
parser.add_argument('--input_dir',
                    help='Input directory of phi-checked synth notes. Will only take files ending in phi_checked.txt')
parser.add_argument('--output_dir',
                    help='Output directory for batched brat annotations and text files for training deidentify')
parser.add_argument('--max_batch',
                    type=int,
                    help='Number of batches (i.e. output files) that reports are grouped into.')
parser.add_argument('--lengthfilter', 
                    action="store_true",
                    help='If applied, notes that have too many or too few tokens are filtered out.')

def getfile(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines

def len_clean(report):
    no_phi = re.sub('<([A-Za-z\_]+)((START)|(END))>','',report)
    toks = [tok for tok in no_phi.split(' ') if tok!=' ']
    return len(toks)

def clean_up(text):
    text = text.replace('<PAR> ','\n').replace('<eos>','')
    text = re.sub('<([A-Za-z\_]+)START>( +?)<[A-Za-z\_]+END>', '', text)
    text = text.replace('  ',' ')
    text = text.strip()
    text = text.replace('<PAR>','\n')
    text = text+' '
    return text

def batch(list_of_reports, n_batch):
    # takes list of reports (str) and creates smaller list of batched rerports (str)
    # batched reports have a delimiter ===delim===
    num_per_batch = len(list_of_reports) // n_batch
    if num_per_batch == 0:
        num_per_batch = 1
        print('More batches than notes.')

    print('Max. number of notes per batch is {}'.format(num_per_batch))

    batches = []
    batch = []
    for r in list_of_reports:
        if len(batch) <= num_per_batch:
            batch.append(r)
        else:
            batches.append(batch)
            batch = []

    batches_concatted = ['\n=== Report: 42 ===\n'.join(b) for b in batches]
    return batches_concatted


def reformat_annotations(text):
    # Regex with two capture groups: 1) tag of annotation, 2) annotated text itself
    pattern = r'<([A-Za-z\_]+)START> (.*?) <[A-Za-z\_]+END>'
    matched_annotations = re.finditer(pattern, text, re.DOTALL)

    current_end = 0
    text_rewritten = ''
    annotations = []

    for ann_id, match in enumerate(matched_annotations):
        text_rewritten += text[current_end:match.start()]
        
        if match.group(2).strip():
            annotations.append(Annotation(
                text=match.group(2),
                start=len(text_rewritten),
                end=len(text_rewritten) + len(match.group(2)),
                tag=match.group(1),
                doc_id='',
                ann_id='T'+str(ann_id)
            ))

        text_rewritten += match.group(2)
        current_end = match.end()

    text_rewritten += text[current_end:]
    return text_rewritten, annotations


########################################################################


def main():
    
    # get input data
    if args.input_file:
        input_path = Path(args.input_file)
        print('\nThis program assumes that the input, if synthetically generated, has well formed in-text annotations. If badly formed annotations are encountered, an error may occur. Make sure you ran annotation_check.py on your synthetic data first.')
        print()
        reports = getfile(input_path)
        
    elif args.input_dir:
        reports = []
        input_path = Path(args.input_dir)
        file_paths = list(input_path.glob('*.phi_cleaned.txt'))
        for p in file_paths:
            reports += getfile(p)
            
    else:
        print('Please specify an input file or directory with EITHER --input_file OR --input_dir.')
        sys.exit(2)
    
    
    # replace <PAR> tokens with \n char:
    reports = [clean_up(text) for text in reports]
    print('Removed <eos> tokens and replaced <PAR> tokens with newline.')
    
    
    # if specified, apply length filter
    if args.lengthfilter:
        originalnum = len(reports)
        print('Only keeping notes between 50-1000 tokens (PHI tags are not counted).')
        reports = [text for text in reports if 50<=len_clean(text)<=1000]
        print('Keeping {} of {} notes.'.format(len(reports),originalnum))
        
    
    # Make batches.
    # One batch is just a string of concatted reports (with delimiter)
    reports_batched = batch([r for r in reports if r != '\n'],
                            args.max_batch)
    print('Created {} batches.'.format(len(reports_batched)))

    # For each batch, reformat annotations and save
    output_path = Path(args.output_dir)
    batch_id = 1
    for b in tqdm(reports_batched):
        text, annotations = reformat_annotations(b)
        # save annotations
        write_brat_document(output_path, f'document_{batch_id}', text, annotations)
        batch_id += 1


if __name__ == "__main__":
    args = parser.parse_args()
    main()
