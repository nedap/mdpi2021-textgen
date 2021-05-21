from textgen.data.make_dataset import annotate_in_text
from deidentify.base import Annotation, Document

def test_annotate_in_text():
    text = "De patient J. Jansen (e: j.jnsen@email.com, t: 06-12345678)"
    annotations = [
        Annotation(text='J. Jansen', start=11, end=20, tag='Name', doc_id='', ann_id='T0'),
        Annotation(text='j.jnsen@email.com', start=25, end=42, tag='Email', doc_id='', ann_id='T1'),
        Annotation(text='06-12345678', start=47, end=58, tag='Phone_fax', doc_id='', ann_id='T2')
    ]

    doc = Document(name='test_doc', text=text, annotations=annotations)

    text_rewritten = annotate_in_text(doc)
    print(text_rewritten)
        
    assert text_rewritten == 'De patient <NameSTART> J. Jansen <NameEND> (e: <EmailSTART> j.jnsen@email.com <EmailEND> , t: <Phone_faxSTART> 06-12345678 <Phone_faxEND> )'


test_annotate_in_text()