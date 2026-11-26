"""
Part 3 - Text preprocessing (spaCy).

Input : data/raw_data.csv            (150 synthetic case narratives)
Output: data/clean_text.csv          (one row per sentence)

Design decisions
* Sentence splitting uses spaCy's dependency parser (more reliable than a
  regex around abbreviations such as "H. pylori").
* The ORIGINAL sentence text is kept in `text_raw` because dictionary NER and
  dependency-based relation extraction (Parts 4-5) must run on unmodified
  text - lemmatised / stop-word-stripped text would break phrase matching
  ("lactic acidosis" -> "lactic acidosis" is fine, but "was attributed to"
  -> "attribute" loses the very wording the relation rules look for).
* `text_norm`  : lower-cased, whitespace-collapsed sentence.
* `lemmas`     : lemmatised content words (stop words, punctuation, numbers
  and 1-letter tokens removed) - used for TF-IDF in Part 6.
* Negation words ("no", "not", "without", ...) are normally on spaCy's
  stop-word list; they are RESCUED here because dropping "no new symptoms"
  -> "new symptom" would flip the meaning of a sentence.
* The abstract is processed WITHOUT the title: titles name the event even in
  narratives that go on to say the event was unrelated to the drug.
"""
import re
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner"])   # NER is replaced by our own in Part 4

NEGATIONS = {"no", "not", "n't", "never", "none", "nor", "without", "cannot", "neither", "nobody", "nothing"}
rescued = sorted(NEGATIONS & nlp.Defaults.stop_words)
for w in rescued:
    nlp.vocab[w].is_stop = False
print(f"Stop-word list: {len(nlp.Defaults.stop_words)} words; negations rescued from it: {rescued}")


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_lemmas(sent) -> list:
    out = []
    for t in sent:
        if t.is_stop or t.is_punct or t.is_space or t.like_num or not t.is_alpha or len(t) < 2:
            continue
        out.append(t.lemma_.lower())
    return out


raw = pd.read_csv("data/raw_data.csv")
rows = []
def repaired_sentences(doc):
    """spaCy's parser sometimes cuts after a fronted time phrase ("Several months
    after the dose change," | "the patient presented ..."), separating a drug
    mention from its event. A sentence that does not end in . ! ? is never
    complete, so it is merged with the sentence that follows it."""
    spans, start = [], None
    for sent in doc.sents:
        start = sent.start if start is None else start
        if sent.text.strip()[-1:] in ".!?":
            spans.append(doc[start:sent.end])
            start = None
    if start is not None:                       # trailing fragment (no final punctuation)
        spans.append(doc[start:])
    return spans


n_repaired = 0
for doc_id, doc in zip(raw.doc_id, nlp.pipe(raw.abstract.tolist(), batch_size=32)):
    n_repaired += len(list(doc.sents)) - len(repaired_sentences(doc))
    for k, sent in enumerate(repaired_sentences(doc), start=1):
        text = sent.text.strip()
        rows.append(dict(
            doc_id=doc_id, sent_id=f"{doc_id}_S{k}", sent_no=k,
            text_raw=text, text_norm=normalise(text),
            lemmas=" ".join(content_lemmas(sent)),
            n_tokens_raw=len([t for t in sent if not t.is_space]),
            n_lemmas=len(content_lemmas(sent)),
        ))
print(f"False sentence boundaries repaired: {n_repaired}")

clean = pd.DataFrame(rows)
clean.to_csv("data/clean_text.csv", index=False)
print(f"Wrote {len(clean)} sentences from {clean.doc_id.nunique()} documents -> data/clean_text.csv")
