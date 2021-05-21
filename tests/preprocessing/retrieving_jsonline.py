#from textgen.data.preprocessing import read_document

from pathlib import Path
import jsonlines

def read_document(DATA_PATH):
    print('reading data')
    data = []
    f = jsonlines.open(DATA_PATH, mode='r')
    for line in f.iter(skip_empty=True, skip_invalid=True):
        data.append(line['annotated_text'])
    return data


if __name__=="__main__":
    DATA_PATH = Path(__file__).parent.parent.parent / 'data' /'interim'/'annotated_ehr.jsonl'
    data = read_document(DATA_PATH)
    print(len(data))
    print(data[0])