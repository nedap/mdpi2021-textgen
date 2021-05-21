
import argparse
import torch

from textgen.word_language_model import data



#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--model',
                    help='path to model (.pt)')
parser.add_argument('--samples', 
                    action='store',
                    help='path to sample of notes to reconstruct (.txt)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', default=111,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--outf', type=str, default='./output/pytorch_lstm/evaluation/reconstruction_results.txt',
                    help='location of the output file with reconstruction results')
args = parser.parse_args()

# Set the random seed manually for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# set device cpu or cuda
device = torch.device("cuda" if args.cuda else "cpu")

##############################################################################
        
def get_notes(path):
    with open(path,'r') as f:
        notes = f.readlines()
    notes = [n for n in notes if n.strip]
    return notes

def tokenize(text):
    # like in tokenize function from data.Corpus
    toks = text.split() + ['<eos>']
    return toks

def get_reconstruction(corpus,model,original_sequence):
    
    # we treat the whole original sequence as the 'final' trigger
    # the first trigger is the first word, the second is the first two etc.
    trigger = tokenize(original_sequence)
    w2i = corpus.dictionary.word2idx
    i2w = corpus.dictionary.idx2word
    
    # count correct word_predictions given trigger at each iteration:
    compare_desired_actual = []
    
    #################################################################
    
    ####### prepare first model input: first trigger word
    hidden = model.init_hidden(1)
    first_word = trigger[0]
    input = torch.ones(1,1).mul(w2i[first_word]).long().to(device)

    ####### if there is a trigger sequence, update model trigger by trigger
    while trigger:
        with torch.no_grad():
            for t in trigger:
                
                #feed model and compute weights given first input
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                
                # IF there is a next word...
                if len(trigger)>1:
                    
                    # save correct vs. desired output pair
                    correct_next_word = trigger[1]                    
                    predicted_next_word = i2w[torch.multinomial(word_weights, 1)[0]]
                    compare_desired_actual.append((correct_next_word,predicted_next_word))
                    
                    # update model: pretend predicted == correct
                    idx_next= torch.tensor(w2i[correct_next_word])
                    input.fill_(idx_next)
                        
                # remove first trigger word from trigger (reduce to-do stack)
                trigger = trigger[1:]

    return compare_desired_actual


def main():
    
    # load model 
    with open(args.model, 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()
    
    #get corpus dictionary for vocabulary to build with
    corpus = data.Corpus(args.data)
    
    #get samples that we want to try reconstructing (or entire training corpus?)
    input_samples = get_notes(args.samples)
    
    #attempt reconstruction and collect results
    file = open(args.outf,'w')
    for text in input_samples:
        reconstruction = get_reconstruction(corpus,model,text)
        print(reconstruction)
        success = sum([1 for (actual, desired) in reconstruction if actual==desired])
        success_rate = success/len(reconstruction)
        print('success rate = {} / {} = {}'.format(success,len(reconstruction),success_rate))        
        print()
        
        file.write(str(reconstruction))
        file.write('\n')
        file.write(f'success rate = {success} / {len(reconstruction)} = {success_rate}')
        file.write('\n***************************\n')
    
    file.close()
            
if __name__ == '__main__':
    main()
