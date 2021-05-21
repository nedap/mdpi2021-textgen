#import jsonlines
from deidentify.base import Document
from deidentify.taggers import FlairTagger
from deidentify.tokenizer import TokenizerFactory
from deidentify.util import surrogate_annotations

###########################################################################

def main():
    # load flair tagger
    tagger = FlairTagger(
        model='model_bilstmcrf_ons_fast-v0.1.0',
        tokenizer=TokenizerFactory().tokenizer(corpus='ons', disable=("tagger", "ner")),
        verbose=True,
        mini_batch_size=256,
    )
    
    #t = '#DeMayonaiseMoorden (2012)  (Ludo)Enckels en (Jos)Dewit, de één schreef jeugdverhalen en romans, de ander schreef een debuutverhalenbundel, dat werd bekroond met de prijs voor beste literaire debuut.  Samen schreven ze de thriller met de bijzondere titel De Mayonaise Moorden.  Een boek waarin we kennis maken met inspecteur Kareem Zeiz. Een zwaar toegetakeld lijk, waar het niet bij blijft. Kareem bijt zich vast in de zaak en worstelt ondertussen met zijn privé problemen, zijn vrouw Cathy die hem onlangs verlaten heeft, herinneringen aan vroeger en nu betreffende zijn ouders en familie en een nieuw ontluikende liefde?  Dit allemaal prachtig en met diepgang en overtuiging geschreven. IJzersterke karakters, spanning en een dader uit onverwachte hoek.  Mij hebben deze heren volledig overtuigd van hun schrijfkunsten en ik verheug me erop binnenkort het vervolg te gaan lezen. Het boek wat de titel Operatie Monstrans (2013) draagt. Als deze net zo goed is, dan hoop ik dat het duo momenteel aan boek 3 schrijft en wellicht nog later dit jaar uitkomt?  Een aanrader, ook voor de liefhebbers van EllenG en Hjorth Rosenfeldt'
    t = '  (Ludo)Enckels en (Jos)Dewit'
    doc = Document(name='',text = t, annotations=[])
    annotated_docs = tagger.annotate([doc])
    
    for ann in annotated_docs[0].annotations:
        print(ann)
    
    print(annotated_docs[0].text)
    
    iter_docs = surrogate_annotations(docs=annotated_docs, seed=1, errors='ignore')
    surrogate_docs = list(iter_docs)


if __name__ == "__main__":
    main()
