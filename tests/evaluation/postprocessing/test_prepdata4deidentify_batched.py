from deidentify.base import Annotation

from textgen.evaluation.postprocessing.prepdata4deidentify_batched import \
    reformat_annotations


def test_reformat_annotations():
    text = 'De patient <NameSTART> J. Jansen <NameEND> (e: <EmailSTART> j.jnsen@email.com <EmailEND> , t: <Phone_faxSTART> 06-12345678 <Phone_faxEND> )'

    text, annotations = reformat_annotations(text)

    assert text == 'De patient J. Jansen (e: j.jnsen@email.com , t: 06-12345678 )'
    assert annotations == [
        Annotation(text='J. Jansen', start=11, end=20, tag='Name', doc_id='', ann_id='T1'),
        Annotation(text='j.jnsen@email.com', start=25, end=42, tag='Email', doc_id='', ann_id='T2'),
        Annotation(text='06-12345678', start=48, end=59, tag='Phone_fax', doc_id='', ann_id='T3')
    ]
