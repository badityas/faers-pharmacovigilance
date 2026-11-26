"""Hand-written relation stress test (outside template wording). Tests the relation RULES only:
entities are supplied by hand so NER errors do not mix in. Expected labels written before running."""
import re, importlib.util, pandas as pd, spacy
spec = importlib.util.spec_from_file_location("rel", "scripts/05_relations.py"); rel = importlib.util.module_from_spec(spec); spec.loader.exec_module(rel)
nlp = rel.nlp if hasattr(rel, "nlp") else spacy.load("en_core_web_sm")
# (sentences, drug, adr, expected relation_type)
T = [
 (["A woman on metformin developed lactic acidosis.", "The event was attributed to the drug."], "metformin", "lactic acidosis", "causal-strong"),
 (["The patient took omeprazole for reflux.", "She later had hypomagnesaemia, a relationship with the drug could not be excluded."], "omeprazole", "hypomagnesaemia", "causal-weak"),
 (["Olanzapine was started in March.", "Neuroleptic malignant syndrome occurred and resolved after the drug was stopped."], "olanzapine", "neuroleptic malignant syndrome", "causal-strong"),
 (["He had myositis.", "This was unrelated to atorvastatin and was due to a viral illness."], "atorvastatin", "myositis", "unrelated"),
 (["Atorvastatin was prescribed.", "Myositis was not considered related to the medication."], "atorvastatin", "myositis", "unrelated"),
 (["Clopidogrel was given after stenting.", "Bleeding was noted at the access site."], "clopidogrel", "bleeding", "co-occurrence"),
 (["The doctor suspected that paracetamol overdose caused hepatic failure."], "paracetamol", "hepatic failure", "causal-strong"),
 (["Hepatic failure was possibly linked to acetaminophen."], "acetaminophen", "hepatic failure", "causal-weak"),
 (["Metformin was continued.", "No lactic acidosis was observed."], "metformin", "lactic acidosis", "none"),
 (["Following omeprazole, the patient experienced nausea.", "Nausea abated on dechallenge."], "omeprazole", "nausea", "causal-strong"),
 (["Olanzapine and metformin were co-prescribed.", "Lactic acidosis was attributed to metformin."], "metformin", "lactic acidosis", "causal-strong"),
]
rows = []
for i, (sents, drug, adr, exp) in enumerate(T):
    did = f"S{i}"; ss = [dict(sent_id=f"{did}_S{k+1}", sent_no=k+1, text=s) for k, s in enumerate(sents)]
    parsed = {s["sent_id"]: nlp(s["text"]) for s in ss}; ents = []
    for s in ss:
        for lab, term, role in (("DRUG", drug, ""), ("MEDCOND", adr, "ADR")):
            for m in re.finditer(re.escape(term), s["text"], flags=re.I):
                ents.append(dict(doc_id=did, sent_id=s["sent_id"], sent_no=s["sent_no"], ent_text=m.group(0), label=lab, role=role,
                                 canonical=term.lower(), start_sent=m.start(), end_sent=m.end(), start_doc=m.start(), end_doc=m.end(), sentence=s["text"]))
    out = rel.extract_doc(did, ss, pd.DataFrame(ents), parsed)
    got = out[0]["relation_type"] if out else "none"
    rows.append(dict(id=did, drug=drug, adr=adr, expected=exp, predicted=got, ok=got == exp, conf=out[0]["confidence"] if out else None,
                     cue=out[0]["cue_phrase"] if out else None, text=" ".join(sents)))
d = pd.DataFrame(rows); d.to_csv("results/relation_stress_test.csv", index=False)
print(d[["id", "expected", "predicted", "ok", "conf", "cue"]].to_string()); print(f"\n{d.ok.sum()}/{len(d)} correct")
