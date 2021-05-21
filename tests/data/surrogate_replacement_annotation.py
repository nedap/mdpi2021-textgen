import re

#import jsonlines
from deidentify.base import Document,Annotation
from deidentify.taggers import FlairTagger
from deidentify.tokenizer import TokenizerFactory
from deidentify.util import surrogate_annotations

###########################################################################

def annotate_in_text(doc):
    # get original text
    text = doc.text
    
    # get list of start and end indices & split original text on each of those
    indices = [0]+[idx for idx_pair in [[a.start,a.end] for a in doc.annotations] for idx in idx_pair]
    print(indices)
    
    parts = [text[i:j] for i,j in zip(indices, indices[1:]+[None])]
    
    # get list of annotations and replace each text chunk that corresponds to an annotation with an in-text-annotated chunk
    # concat both normal and annotated chunks back into text
    annotations=doc.annotations
    new = ''
    for p in parts:
        print(p)
        if annotations and p==annotations[0].text:
                new_annotation_token = ' <' + annotations[0].tag + 'START> ' + p + ' <' + annotations[0].tag + 'END> '
                new += new_annotation_token
                annotations = annotations[1:]
        else:
            new += p
    
    # clean up new text: remove double whitespaces
    new = new.replace('  ',' ')
    
    return new

def main():
    # load flair tagger
    tagger = FlairTagger(
        model='model_bilstmcrf_ons_fast-v0.1.0',
        tokenizer=TokenizerFactory().tokenizer(corpus='ons', disable=("tagger", "ner")),
        verbose=True,
        mini_batch_size=256,
    )
    
    t = 'Ilja, journaliste en schrijfster, is net getrouwd met Petros, een succesvol voetbaltrainer. Zij is nieuw in het voetbalwereldje en voelde zich er in Zoeterwoude totaal niet in thuis. Nog maar net terug in Nederland, waar ze een frisse start wel ziet zitten, krijgt Petros een contract als journalist op Cyprus. Daar gaan ze weer… Gelukkig is het tropische Cyprus héél anders dan Zoeterwoude en maakt Veroni daar wel snel vriendinnen onder de andere voetbalvrouwen. Maar dan komt ze er langzaamaan achter dat er een hoop ellende schuilt achter alle glitter en glamour, en dat de mannen (en de vrouwen…) het niet zo nauw nemen met huwelijkse trouw. Zij en Petros zijn anders, zij zijn écht gelukkig samen. Toch?  Wat een heerlijk verhaal is dit! De nuchtere en misschien wat naïeve Ilja is een verademing tussen de ‘echte’ voetbalvrouwen vol uiterlijke schijn. Zij gelooft nog echt in liefde en trouw en je hoopt maar dat haar vertrouwen niet beschaamd wordt.  Het verhaal leest heerlijk vlot, is lekker luchtig, heeft humor en een fijn zomers sfeertje. Een ideaal zomerboek dus! (of daarna om een zonnetje in huis te brengen ;) ) Ik heb van begin tot eind zitten smullen van alle intriges in dit smeuïge verhaal! Het leest als een heerlijk dramatische soap :D Genieten!! \'Wat beweegt de jonge zwarte deelpachter Célina van Impelen om huis, vee en akkers te vernietigen en met vrouw en kind naar De Ommekeer te vertrekken?\'- Win Uit de maat voor je hele leesgroep! \'Het is autobiografisch, helemaal waargebeurd maar toch zie je elementen van fictie in de stijl en vooral de opbouw, dat maakt het des te sterker.\' - Win boeken voor je hele leesclub! We gaan Wil van Pleun van Oijen luisteren via de gratis Hebban Luisterboeken-app. Doe je mee?'
    
    doc = Document(name='',text = t, annotations=[])
    annotated_docs = tagger.annotate([doc])
        
    iter_docs = surrogate_annotations(docs=annotated_docs, seed=1, errors='ignore')
    surrogate_docs = list(iter_docs)
    
    for ann in surrogate_docs[0].annotations:
        print(ann)
    
    in_text = annotate_in_text(surrogate_docs[0])
    print(in_text)

if __name__ == "__main__":
    main()
