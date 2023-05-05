# danlp-spacy-flair
A spacy wrapper for DaNLPs flair model. It is a simple wrapper that adds the flair model as a spacy pipeline. It is not
optimized for speed.

We don't recommend using this for other than testing as better models are available. We use it for validation using DaCy.


## Installation
```bash
pip install https://github.com/centre-for-humanities-computing/danlp-spacy-flair
```

## Usage
```python
import spacy
from danlp_spacy_flair import DanlpFlairComponent  # just to register the component

nlp = spacy.blank("da")
nlp.add_pipe("danlp_flair", last=True)
doc = nlp("Jeg hedder Anders og bor i Odense.")
print(doc.ents)
# (Anders, Odense)
```