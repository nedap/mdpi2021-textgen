from evaluation.phi_tag_distribution import phi_freqs
import pandas as pd

def test_phi_freqs():
    data = [
        "Dit is stukje tekst met daarin de naam <NameSTART> J. Jansen <NameEND>.",
        "De patient <NameSTART> Jan Jansen <NameEND> (e: <EmailSTART> j.jnsen@email.com <EmailEND>"
    ]

    print(phi_freqs(data))
    assert phi_freqs(data) == {
        'Name': 2,
        'Email': 1,
    }
