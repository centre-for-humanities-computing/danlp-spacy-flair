"""the entire package"""
import ssl

from danlp.models import load_flair_ner_model, load_flair_pos_model
from flair.data import Sentence, Token
from spacy.language import Language
from spacy.tokens import Doc


@Language.component("danlp_flair_component")
def my_component(nlp):
    return FlairComponent(nlp=nlp)


class FlairComponent:
    def __init__(self, nlp):
        # self.name = name
        self.nlp = nlp
        # setup certificate to download flair
        ssl._create_default_https_context = ssl._create_unverified_context

        self.tagger_ner = load_flair_ner_model()
        self.tagger_pos = load_flair_pos_model()

    def __call__(self, doc):
        sent = Sentence()
        [sent.add_token(Token(t.text)) for t in self.nlp(doc.text)]
        self.tagger_ner.predict([sent], verbose=True)
        self.tagger_pos.predict([sent], verbose=True)

        text, iob, upos, ws = zip(
            *[
                (
                    tok.text,
                    tok.get_tag("ner").value,
                    tok.get_tag("upos").value,
                    tok.whitespace_after,
                )
                for tok in sent
            ]
        )
        doc = Doc(self.nlp.vocab, words=text, spaces=ws, tags=upos, ents=iob)


if __name__ == "__main__":
    import spacy
    nlp = spacy.blank("da")
    nlp.add_pipe("danlp_flair_component", name="flair", last=True)
    doc = nlp("Jeg hedder Anders og bor i Odense.")
    print(doc.ents)