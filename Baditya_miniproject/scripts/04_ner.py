"""
Part 4 - Named Entity Recognition (dictionary NER + context roles).

Why not BioBERT / scispaCy?  Model weights are hosted on huggingface.co / S3,
which are unreachable from this environment (same restriction that blocked
PubMed, PubChem and ChEMBL earlier).  We therefore build a transparent,
gazetteer-based NER with spaCy's PhraseMatcher, seeded with the REAL
vocabularies from the FAERS signal table produced in Assignments 1-2:
    * 2,365 drug names   -> DRUG
    * 5,145 MedDRA PTs   -> MEDCOND (any clinical term)
Because FAERS "reaction" terms also contain indications and pre-existing
conditions, a second step assigns each MEDCOND a ROLE from sentence context
(ADR / INDICATION / HISTORY / CAUSE / MANAGEMENT / NON_EVENT).  Only role=ADR
entities are carried into relation extraction (Part 5).

Two gazetteer versions are scored against the gold spans recorded by the
narrative generator:
    baseline : FAERS strings exactly as they are
    improved : baseline + US spelling variants of every UK-spelled term
"""
import json
import re
from collections import Counter, defaultdict

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

nlp = spacy.blank("en")                      # tokenizer only - same tokenisation for patterns and text
STOP = spacy.load("en_core_web_sm").Defaults.stop_words

# --------------------------------------------------------------------------- gazetteers
sig = pd.read_csv("data/FAERS_signals_with_EBGM.csv", usecols=["drug", "adr"])

drug_terms = set()
for name in sig.drug.dropna().unique():
    for part in re.split(r"[\\/;,]", name):             # FAERS combination products use "\"
        part = re.sub(r"\s+", " ", part).strip().lower()
        if len(part) >= 4 and not re.search(r"\d", part) and part not in STOP:
            drug_terms.add(part)

adr_terms = {re.sub(r"\s+", " ", a).strip().lower() for a in sig.adr.dropna().unique()}

# A few FAERS "reaction" terms are procedures/treatments, not events.
NON_EVENT = {"infusion", "transfusion", "transplant", "immunosuppression", "tension"}
adr_terms -= NON_EVENT
overlap = drug_terms & adr_terms
drug_terms -= overlap                                    # a string that is both: treat as condition, not drug

UK2US = [("aem", "em"), ("oea", "ea"), ("oe", "e"), ("aes", "es"), ("faec", "fec"), ("sulph", "sulf")]


def us_variant(term):
    t = term
    for uk, us in UK2US:
        t = t.replace(uk, us)
    return t


variant_to_pt = {}
for t in sorted(adr_terms):
    v = us_variant(t)
    if v != t and v not in adr_terms:
        variant_to_pt[v] = t
print(f"Gazetteers: {len(drug_terms)} drug strings | {len(adr_terms)} condition strings (PTs) "
      f"| {len(variant_to_pt)} US-spelling variants | {len(overlap)} strings removed from drug list (also PTs)")

pd.DataFrame({"term": sorted(drug_terms)}).to_csv("data/gazetteer_drugs.csv", index=False)
pd.DataFrame([{"term": t, "source": "FAERS PT"} for t in sorted(adr_terms)] +
             [{"term": v, "source": f"US spelling of '{p}'"} for v, p in sorted(variant_to_pt.items())]).to_csv("data/gazetteer_conditions.csv", index=False)


def build_matcher(conditions):
    m = PhraseMatcher(nlp.vocab, attr="LOWER")
    m.add("DRUG", [nlp.make_doc(t) for t in sorted(drug_terms)])
    m.add("MEDCOND", [nlp.make_doc(t) for t in sorted(conditions)])
    return m


matchers = {"baseline": build_matcher(adr_terms), "improved": build_matcher(adr_terms | set(variant_to_pt))}

# --------------------------------------------------------------------------- role rules
CAUSE_CUE = re.compile(r"(attributed to|pointed to|due to|caused by|secondary to)\s+(?:[\w.\-]+\s+){0,3}$")
FOR_CUE = re.compile(r"\bfor\s+(?:an?\s+|the\s+)?$")
MGMT_CUE = re.compile(r"\b(withdrawn|stopped|discontinued|withheld|held|interrupted|commenced|administered|started|added|given|"
                      r"transfused|transferred|treated|managed|prescribed|reduced|settled|resolved|continued|applied|"
                      r"required|needed|used|received|normalised|improved|switching)\b")


def assign_role(sentence, start, end, sent_no):
    """Rule-based context role for a MEDCOND mention."""
    before = sentence[:start].lower()
    if FOR_CUE.search(before) and re.search(r"\b(taking|receiving|treated)\b", before):
        return "INDICATION"
    if CAUSE_CUE.search(before):
        return "CAUSE"
    if sent_no == 1:
        return "HISTORY"                     # pre-existing condition in the patient-introduction sentence
    if MGMT_CUE.search(sentence.lower()):
        return "MANAGEMENT"
    return "ADR"


# --------------------------------------------------------------------------- run NER
clean = pd.read_csv("data/clean_text.csv")
clean["doc_off"] = clean.groupby("doc_id").text_raw.transform(lambda s: (s.str.len() + 1).cumsum().shift(fill_value=0))


def run_ner(version):
    matcher, rows = matchers[version], []
    for r in clean.itertuples():
        doc = nlp.make_doc(r.text_raw)
        spans = [doc[s:e] for _, s, e in matcher(doc)]
        labels = {(sp.start, sp.end): nlp.vocab.strings[mid] for mid, s, e in matcher(doc) for sp in [doc[s:e]]}
        spans = sorted(spans, key=lambda sp: labels[(sp.start, sp.end)] != "DRUG")   # DRUG wins exact ties
        for sp in filter_spans(spans):                                                 # longest match wins
            label = labels[(sp.start, sp.end)]
            low = sp.text.lower()
            role = assign_role(r.text_raw, sp.start_char, sp.end_char, r.sent_no) if label == "MEDCOND" else ""
            rows.append(dict(doc_id=r.doc_id, sent_id=r.sent_id, sent_no=r.sent_no, ent_text=sp.text, label=label, role=role,
                             canonical=variant_to_pt.get(low, low) if label == "MEDCOND" else low,
                             start_sent=sp.start_char, end_sent=sp.end_char,
                             start_doc=r.doc_off + sp.start_char, end_doc=r.doc_off + sp.end_char, sentence=r.text_raw))
    return pd.DataFrame(rows)


ents = {v: run_ner(v) for v in matchers}
ents["improved"].to_csv("data/entities.csv", index=False)
for v, e in ents.items():
    print(f"[{v:8s}] {len(e)} entities -> " + ", ".join(f"{k}: {n}" for k, n in e.label.value_counts().items()) +
          " | roles: " + ", ".join(f"{k}: {n}" for k, n in e[e.label == 'MEDCOND'].role.value_counts().items()))

# --------------------------------------------------------------------------- evaluation vs. gold spans
raw = pd.read_csv("data/raw_data.csv")
gold = {r.doc_id: json.loads(r.gt_spans) for r in raw.itertuples()}
meta = raw.set_index("doc_id")
COARSE = {"EVENT": "ADR", "SYMPTOM": "ADR", "INDICATION": "NOT_ADR", "COMORBIDITY": "NOT_ADR", "ALT_CAUSE": "NOT_ADR",
          "MANAGEMENT": "NOT_ADR", "FINDING": "AMBIGUOUS"}


def form_category(pt, form):
    f, p = form.lower(), pt.lower()
    if f == p:
        return "1 exact PT"
    if us_variant(p) == f:
        return "2 US spelling of PT"
    if p in f:
        return "3 PT + modifier"
    return "4 synonym / lay term / abbreviation"


def evaluate(e):
    out = []
    byd = {d: g for d, g in e.groupby("doc_id")}
    # --- (a) DRUG recall on gold drug spans
    tally = defaultdict(lambda: [0, 0])
    for d, spans in gold.items():
        ee = byd.get(d, e.iloc[0:0])
        conc = {x.lower() for x in str(meta.at[d, "gt_other_drugs"]).split("|") if x and x != "nan"}
        for sp in spans:
            if sp["role"] != "DRUG":
                continue
            low = sp["text"].lower()
            cat = ("concomitant drug" if low in conc else
                   "suspect drug - synonym (paracetamol)" if low == "paracetamol" else
                   "suspect drug - generic" if low == meta.at[d, "gt_drug"].lower() else "suspect drug - brand")
            hit = ((ee.label == "DRUG") & (ee.start_doc == sp["start"]) & (ee.end_doc == sp["end"])).any()
            tally[cat][0] += hit
            tally[cat][1] += 1
    for cat, (h, n) in sorted(tally.items()):
        out.append(dict(task="DRUG recall (exact span)", group=cat, hits=h, n=n, score=h / n))
    # --- (b) target-event recall by surface-form category
    tally = defaultdict(lambda: [0, 0, 0, 0])
    for d, spans in gold.items():
        ee = byd.get(d, e.iloc[0:0])
        pt = meta.at[d, "gt_adr"]
        for sp in spans:
            if sp["role"] != "EVENT":
                continue
            cat = form_category(pt, sp["text"])
            inside = ee[(ee.label == "MEDCOND") & (ee.start_doc < sp["end"]) & (ee.end_doc > sp["start"])]
            strict = ((ee.label == "MEDCOND") & (ee.role == "ADR") & (ee.start_doc == sp["start"]) & (ee.end_doc == sp["end"])).any()
            ptfound = ((inside.role == "ADR") & (inside.canonical == pt.lower())).any()
            lenient = (inside.role == "ADR").any()
            t = tally[cat]; t[0] += strict; t[1] += ptfound; t[2] += lenient; t[3] += 1
    for cat, (s, p, l, n) in sorted(tally.items()):
        out.append(dict(task="EVENT recall: exact span", group=cat, hits=s, n=n, score=s / n))
        out.append(dict(task="EVENT recall: target PT recovered", group=cat, hits=p, n=n, score=p / n))
        out.append(dict(task="EVENT recall: any ADR entity overlaps", group=cat, hits=l, n=n, score=l / n))
    s = [sum(v[i] for v in tally.values()) for i in range(4)]
    out += [dict(task="EVENT recall: exact span", group="ALL", hits=s[0], n=s[3], score=s[0] / s[3]),
            dict(task="EVENT recall: target PT recovered", group="ALL", hits=s[1], n=s[3], score=s[1] / s[3]),
            dict(task="EVENT recall: any ADR entity overlaps", group="ALL", hits=s[2], n=s[3], score=s[2] / s[3])]
    # --- (c) role classification: each MEDCOND entity inside a gold span
    recs = []
    for r in e[e.label == "MEDCOND"].itertuples():
        cont = [g for g in gold[r.doc_id] if g["start"] <= r.start_doc and r.end_doc <= g["end"] and g["role"] != "DRUG"]
        g = min(cont, key=lambda g: g["end"] - g["start"]) if cont else None
        recs.append(dict(pred=("ADR" if r.role == "ADR" else "NOT_ADR"), pred_fine=r.role,
                         gold=(COARSE[g["role"]] if g else "UNANNOTATED"), gold_fine=(g["role"] if g else "UNANNOTATED")))
    rec = pd.DataFrame(recs)
    scored = rec[rec.gold.isin(["ADR", "NOT_ADR"])]
    tp = ((scored.pred == "ADR") & (scored.gold == "ADR")).sum()
    fp = ((scored.pred == "ADR") & (scored.gold == "NOT_ADR")).sum()
    fn = ((scored.pred == "NOT_ADR") & (scored.gold == "ADR")).sum()
    unann = ((rec.pred == "ADR") & (rec.gold == "UNANNOTATED")).sum()
    out += [dict(task="ROLE: ADR precision (vs gold slots)", group="entities in annotated slots", hits=tp, n=tp + fp, score=tp / (tp + fp)),
            dict(task="ROLE: ADR recall (vs gold slots)", group="entities in annotated slots", hits=tp, n=tp + fn, score=tp / (tp + fn)),
            dict(task="ROLE: coarse accuracy ADR vs not-ADR", group="entities in annotated slots", hits=(scored.pred == scored.gold).sum(), n=len(scored),
                 score=(scored.pred == scored.gold).mean()),
            dict(task="ROLE: ADR entities in no annotated slot", group="(template wording / dictionary noise)", hits=unann, n=(rec.pred == 'ADR').sum(),
                 score=unann / (rec.pred == 'ADR').sum())]
    return pd.DataFrame(out), rec


res = {}
for v in ents:
    res[v], rec = evaluate(ents[v])
    if v == "improved":
        rec_improved = rec
cmp = res["baseline"].merge(res["improved"], on=["task", "group"], suffixes=("_base", "_impr"), how="outer")
cmp = cmp[["task", "group", "hits_base", "n_base", "score_base", "hits_impr", "n_impr", "score_impr"]]
cmp.to_csv("results/ner_evaluation.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 60)
show = cmp.copy()
for c in ("score_base", "score_impr"):
    show[c] = show[c].map(lambda x: f"{x:.1%}")
print()
print(show.to_string(index=False))

print("\nFine-grained role confusion (improved gazetteer; rows = gold slot, cols = predicted role):")
print(pd.crosstab(rec_improved.gold_fine, rec_improved.pred_fine))

# --------------------------------------------------------------------------- distinct-string audit of extracted DRUG entities
TEMPLATE_DRUGS = ({x.lower() for v in json.load(open("data/drug_surface_forms.json")).values() for x in v} |
                  {"lisinopril", "aspirin", "furosemide", "levothyroxine", "bisoprolol", "insulin glargine", "warfarin", "amlodipine",
                   "ramipril", "pantoprazole", "paracetamol", "acetylcysteine", "dantrolene", "bromocriptine", "prednisolone",
                   "methotrexate", "heparin", "immunoglobulin", "bicarbonate", "magnesium sulfate", "magnesium sulphate"})
dd = ents["improved"][ents["improved"].label == "DRUG"].ent_text.str.lower().value_counts()
aud = pd.DataFrame({"string": dd.index, "n_mentions": dd.values, "known_to_be_inserted_drug": [s in TEMPLATE_DRUGS for s in dd.index]})
aud.to_csv("results/ner_drug_string_audit.csv", index=False)
print(f"\nDRUG string audit: {len(aud)} distinct strings; {(~aud.known_to_be_inserted_drug).sum()} not in the list of drugs the generator inserted:")
print(aud[~aud.known_to_be_inserted_drug].to_string(index=False))

# --------------------------------------------------------------------------- entities in no annotated slot (dictionary noise)
noise = rec_improved.copy()
e2 = ents["improved"][ents["improved"].label == "MEDCOND"].reset_index(drop=True)
noise["ent"] = e2.ent_text.str.lower(); noise["sentence"] = e2.sentence
nz = noise[(noise.pred == "ADR") & (noise.gold == "UNANNOTATED")]
print("\nADR-role entities that fall in no annotated slot (top strings):")
print(nz.ent.value_counts().head(15).to_string())

# --------------------------------------------------------------------------- manual-validation sheet for the student
sample = clean.sample(40, random_state=7)
e = ents["improved"]
lines = []
for r in sample.itertuples():
    es = e[e.sent_id == r.sent_id]
    lines.append(dict(sent_id=r.sent_id, sentence=r.text_raw,
                      extracted="; ".join(f"{x.ent_text} [{x.label if x.label == 'DRUG' else x.role}]" for x in es.itertuples()) or "(none)",
                      student_check_correct="", student_notes=""))
pd.DataFrame(lines).to_csv("results/ner_manual_check.csv", index=False)
print("\nWrote data/entities.csv, results/ner_evaluation.csv, results/ner_drug_string_audit.csv, results/ner_manual_check.csv")


# --------------------------------------------------------------------------- stress test on hand-written sentences
# The synthetic narratives and the role rules were written by the same author, so the near-perfect
# scores above are an internal-consistency check, NOT a performance estimate.  These 14 sentences
# are written in ordinary clinical phrasing outside the template vocabulary.
STRESS = [
    ("Metformin-associated lactic acidosis is a rare complication in patients with renal impairment.", {"lactic acidosis": "ADR", "renal impairment": "NOT_ADR"}),
    ("Atorvastatin rarely causes rhabdomyolysis in patients receiving high doses.", {"rhabdomyolysis": "ADR"}),
    ("The patient has a history of hypertension and type 2 diabetes mellitus.", {"hypertension": "NOT_ADR", "diabetes": "NOT_ADR"}),
    ("Omeprazole is indicated for gastro-oesophageal reflux disease.", {"reflux": "NOT_ADR"}),
    ("Olanzapine therapy was complicated by neuroleptic malignant syndrome.", {"neuroleptic malignant syndrome": "ADR"}),
    ("Liver enzymes normalised after stopping acetaminophen, supporting a role in the hepatotoxicity.", {"hepatotoxicity": "ADR"}),
    ("She developed severe hypomagnesemia after two years of proton pump inhibitor use.", {"hypomagnesemia": "ADR"}),
    ("Patients with schizophrenia treated with olanzapine are at risk of weight gain.", {"schizophrenia": "NOT_ADR", "weight gain": "ADR"}),
    ("Myalgia resolved after atorvastatin was discontinued.", {"myalgia": "ADR"}),
    ("Sertraline was started for major depression.", {"depression": "NOT_ADR"}),
    ("Clopidogrel has been linked to haemorrhage after percutaneous coronary intervention.", {"haemorrhage": "ADR"}),
    ("Dapagliflozin treatment was complicated by euglycaemic diabetic ketoacidosis.", {"ketoacidosis": "ADR"}),
    ("The elevated lactate was attributed to septic shock rather than the drug.", {"shock": "NOT_ADR"}),
    ("The patient presented with nausea, vomiting and dizziness two days after starting the tablets.", {"nausea": "ADR", "vomiting": "ADR", "dizziness": "ADR"}),
]
mt = matchers["improved"]
rows = []
for text, expected in STRESS:
    doc = nlp.make_doc(text)
    labmap = {(doc[s:e].start_char, doc[s:e].end_char): nlp.vocab.strings[m] for m, s, e in mt(doc)}
    ents_here = []
    for sp in filter_spans([doc[s:e] for _, s, e in mt(doc)]):
        lab = labmap[(sp.start_char, sp.end_char)]
        role = assign_role(text, sp.start_char, sp.end_char, 3) if lab == "MEDCOND" else "DRUG"
        ents_here.append((sp.text.lower(), role))
    for key, exp in expected.items():
        hit = [(t, r) for t, r in ents_here if key in t]
        if not hit:
            rows.append(dict(sentence=text, expected_entity=key, expected_role=exp, extracted="(missed)", predicted_role="", status="MISSED"))
        else:
            t, r = hit[0]
            pred = "ADR" if r == "ADR" else "NOT_ADR"
            rows.append(dict(sentence=text, expected_entity=key, expected_role=exp, extracted=t, predicted_role=r,
                             status="correct" if pred == exp else "WRONG ROLE"))
st = pd.DataFrame(rows)
st.to_csv("results/ner_stress_test.csv", index=False)
print("\nSTRESS TEST (hand-written sentences, outside template vocabulary):")
print(st[["expected_entity", "expected_role", "extracted", "predicted_role", "status"]].to_string(index=False))
print(st.status.value_counts().to_string())
