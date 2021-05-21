# Usage: python -m textgen.evaluation.postprocessing.annotation_check <inputfile>

import sys
import re
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('inputfile', 
                    help='Location of synthetic data you want to check and clean.')


########################################################################

def add_safety_spaces(text):
    # some tags are attached to the previous or next word
    # this function makes sure that there is a space so they are tokenized correctly
    spaced = text.replace('<',' <').replace('>','> ')
    nodouble = spaced.replace('  ',' ')
    return nodouble.strip()


def has_annotations(text):
    pattern = r'<[A-Za-z\_]+((START)|(END))>'
    m = re.search(pattern, text)
    return m is not None

    
def matches(tok_start,tok_end):
    # checks if start & end tag match (e.g. Name == Name).
    # some safety guards if tag token contains part of non-tag.
    # If add_safety_spaces used, the splits are redundant.
    tok_start = (tok_start.split('START')[0]).split('<')[-1]
    tok_end = (tok_end.split('END')[0]).split('<')[-1]
    #if tok_start.replace('START','') == tok_end.replace('END',''):
    return tok_start == tok_end


def remove_unbalanced(synth_ehr):
    # returns the text without unbalanced or empty tags
    # also returns array with tag correctness scores
    
    tokens = synth_ehr.split(' ')
    
    # eval
    matched_count = 0
    empty_count = 0
    unmatched_count = 0
    # sequentially check tokens using stack to keep track of tags
    cleaned_up = [] # only annotation independent text or correct annotations  
    stack = [] # for start tokens
    in_observation = [] # sequence to be observed once start tag found
    
    pattern_start = r'.*<[A-Za-z\_]*(START)>.*'
    pattern_end = r'.*<[A-Za-z\_]*(END)>.*'
    
    for i, t in enumerate(tokens):
        
        # if start token, add to stack and start observing
        if re.match(pattern_start,t):
            cleaned_up += in_observation
            in_observation = []
            stack.append(t)

        # if end token, check if tag can be closed
        elif re.match(pattern_end,t):
            if (len(stack)>0) and matches(stack[-1],t):
                #get rid of newlines within annotations
                in_observation = [t for t in in_observation if t.strip()!='<PAR>']
                if not (in_observation == []):
                    # pair matched, add all to cleaned_up
                    cleaned_up.append(stack[-1])
                    cleaned_up += in_observation
                    cleaned_up.append(t)
                    # empty memory, log event
                    in_observation = []
                    stack.pop()
                    matched_count+=1
                else:
                    # pair matched, but no tag content
                    # empty memory, log event
                    stack.pop()
                    empty_count+=1
            
            # if recent start tag doesn't match end tag      
            else:
                # just keep content without tag markers, log event
                cleaned_up += in_observation
                in_observation = []
                unmatched_count +=1
        
        # if non-tag token
        else:
            if len(stack)>0:
                in_observation.append(t)
            else:
                cleaned_up.append(t)
    
    cleaned_up += in_observation
    text_output = (' '.join(cleaned_up)).strip()
    text_output = re.sub(r' +',' ', text_output)
    scores = [matched_count, empty_count, unmatched_count]
    
    return text_output, scores


########################################################################

def main():

    # filepaths input, output, log
    input_file= Path(args.inputfile)
    output_file = input_file.with_suffix('.phi_cleaned.txt')
    log_file = input_file.with_suffix('.log')
    print('\nCleaned up file will be saved to: {}'.format(output_file))
    print('Evaluation of annotations will be saved to: {}\n'.format(log_file))
    
    # get synthetic data to process
    with open(input_file) as fin:
        text = fin.readlines()
        
    # record output & scores
    cleaned_notes = []
    matchscore = 0
    emptyscore = 0
    unmatchedscore = 0
    
    # set up log
    with open(log_file, 'w') as logf:
        logf.write('Evaluating annotations from file: {}\n'.format(input_file))
        logf.write('='*89)
        logf.write('\n\n')
        
        # iterate through input data
        for t in text:
            t = add_safety_spaces(t)
            if has_annotations(t): # if tags in text
                cleaned, scores = remove_unbalanced(t)
                cleaned_notes.append(cleaned)
                matchscore += scores[0]
                emptyscore += scores[1]
                unmatchedscore += scores[2]
                logf.write(t)
                logf.write('\nMatched: {}\n'.format(scores[0]))
                logf.write('Empty: {}\n'.format(scores[1]))
                logf.write('Unmatched: {}\n'.format(scores[2]))
                logf.write('\n')
            else: # if no tags, just add to output
                cleaned_notes.append(t)

        # log score summary
        logf.write('=' * 89)
        logf.write('\nTotal matched: {}'.format(matchscore))
        logf.write('\nTotal empty: {}'.format(emptyscore))
        logf.write('\nTotal unmatched: {}'.format(unmatchedscore))
        print('PHI evaluation summary:\nTotal matched: {}\nTotal empty: {}\nTotal unmatched: {}\n'.format(matchscore, emptyscore, unmatchedscore))
        print('Total number of checked documents: {}'.format(len(cleaned_notes)))
    
    # write output to file
    with open(output_file,'w') as fout:
        for n in cleaned_notes:
            fout.write(n)
            fout.write('\n')
    

if __name__== "__main__" :
    args = parser.parse_args()
    main()
    
