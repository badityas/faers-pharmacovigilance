"""
Part 5 - Relation extraction (drug -> ADR).

Input : data/entities.csv (Part 4), data/clean_text.csv (Part 3), data/raw_data.csv (gold labels)
Output: data/relations.csv         one row per (document, drug, ADR) pair with a relation type + confidence
        data/drug_adr_pairs.csv    pairs aggregated over documents (input to Part 6)
        results/relation_evaluation.csv, results/relation_stress_test.csv

Why a document-level approach?  In case narratives the drug is introduced in the first sentence and the
event appears 1-3 sentences later, so SENTENCE co-occurrence alone misses most pairs.  We therefore
    1. identify the SUSPECT drug of each narrative (drug the patient "had been taking"),
    2. pair it with every ADR-role entity of the narrative,
    3. read the causality wording (WHO-UMC style: probable / possible / unrelated) to type the relation,
    4. score each pair 0-1 from transparent evidence (cue strength, event vs symptom, proximity, dechallenge).
Three methods are compared against the gold labels: sentence co-occurrence, abstract co-occurrence, and
this rule-based method.  A brand/synonym alias table stands in for RxNorm (unreachable).
"""
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import roc_auc_score
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

nlp = spacy.load("en_core_web_sm", disable=["ner"])        # parser is needed for dependency distance

ALIASES = {"glucophage": "metformin", "tylenol": "acetaminophen", "paracetamol": "acetaminophen",
           "zyprexa": "olanzapine", "lipitor": "atorvastatin", "prilosec": "omeprazole", "plavix": "clopidogrel"}


def norm_drug(s):
    s = s.lower()
    return ALIASES.get(s, s)


# ------------------------------------------------------------------------------------------ cue rules
REL = re.compile(r"related|relation\w*|association|associated|caus\w+|due to|attributed|drug-related|link(?:ed)?|"
                 r"contribut\w*|responsible|induc\w+|trigger\w+|led to|resulted? (?:from|in)")
NEG = re.compile(r"\b(?:unrelated|not related|not (?:caused|attributed|due)|no (?:causal )?(?:relationship|association|link|connection)|"
                 r"not (?:considered|thought|felt|believed) (?:to be )?(?:related|due|caused)|unlikely|ruled out|rather than|(?:did|does|do) not (?:cause|contribute|induce|trigger))\b")
WEAK = re.compile(r"\b(?:possibly|possible|may have|might have|may be|might be|could have|suspected|uncertain)\b")
WEAK_EXPLICIT = re.compile(r"\b(?:could not|cannot|can not) be (?:excluded|ruled out|dismissed)\b")
STRONG = re.compile(r"\b(?:probabl[ye]|likely|certain(?:ly)?|definite(?:ly)?|clearly)\b")
ASSERT = re.compile(r"\b(?:caused by|was caused|induced by|triggered by|led to|resulted from|attributed to)\b")
DECHALLENGE = re.compile(r"\b(?:discontinued|stopped|withdrawn|withheld|held|interrupted|withdrew|ceased)\b")
EVENT_CTX = re.compile(r"(?:diagnosed with|developed|diagnosis of)\s+(?:[\w\-]+\s+){0,2}$")
SUSPECT_CTX = re.compile(r"\b(?:taking|treated with|started on|prescribed|on)\s+(?:\w+\s+){0,1}$")


GENERIC_DRUG = re.compile(r"\bthe (?:drug|medication|medicine|treatment|therapy)\b")


def causality_cue(sentence, has_drug):
    """WHO-UMC-style reading of one sentence -> ('NEG'|'WEAK'|'STRONG'|None, matched phrase)."""
    t = sentence.lower()
    weak_explicit = WEAK_EXPLICIT.search(t)
    t2 = WEAK_EXPLICIT.sub(" ", t)
    m = NEG.search(t2)
    if m:
        return "NEG", m.group(0)
    has_rel = REL.search(t2)
    if weak_explicit:
        return "WEAK", weak_explicit.group(0)
    m = WEAK.search(t2)
    if m and has_rel:
        return "WEAK", m.group(0)
    m = STRONG.search(t2)
    if m and has_rel:
        return "STRONG", m.group(0)
    has_drug = has_drug or bool(GENERIC_DRUG.search(t2))   # anaphora: "the drug", "the medication"
    m = ASSERT.search(t2)
    if m and has_drug:
        return "STRONG", m.group(0)
    return None, ""


def dep_distance(a, b):
    """Shortest path length between two tokens of one parsed sentence."""
    anc_a, anc_b = [a] + list(a.ancestors), [b] + list(b.ancestors)
    pos_b = {t.i: k for k, t in enumerate(anc_b)}
    for ka, t in enumerate(anc_a):
        if t.i in pos_b:
            return ka + pos_b[t.i]
    return None


CUE_BASE = {"causal-strong": 0.70, "causal-weak": 0.45, "co-occurrence": 0.20}
REL_TYPE = {"STRONG": "causal-strong", "WEAK": "causal-weak", "NEG": "unrelated", None: "co-occurrence"}


def extract_doc(doc_id, sents, ents, parsed):
    """sents: list of dict(sent_id, sent_no, text); ents: DataFrame of entities of this doc;
    parsed: sent_id -> spaCy Doc.  Returns list of relation rows."""
    drugs = ents[ents.label == "DRUG"]
    adrs = ents[(ents.label == "MEDCOND") & (ents.role == "ADR")]
    if drugs.empty:
        return []
    # 1. suspect drug = drug the patient "had been taking" (fallback: first drug that is not a concomitant)
    suspect = None
    for r in drugs.sort_values("start_doc").itertuples():
        before = r.sentence[:r.start_sent].lower()
        if "also receiving" in before or before.rstrip().endswith("after starting"):
            continue
        if SUSPECT_CTX.search(before):
            suspect = r
            break
        suspect = suspect or r
    if suspect is None:
        return []
    sus = norm_drug(suspect.ent_text)

    # 2. causality cues per drug (cue applies to the drugs named in its sentence, else to the suspect)
    cue, dech = {}, False
    for s in sents:
        d_here = drugs[drugs.sent_id == s["sent_id"]]
        d_here = d_here[~d_here.apply(lambda r: "also receiving" in r.sentence[:r.start_sent].lower(), axis=1)] if len(d_here) else d_here
        label, phrase = causality_cue(s["text"], has_drug=len(d_here) > 0)
        targets = {norm_drug(x) for x in d_here.ent_text} or {sus}
        if label:
            for t in targets:
                prev = cue.get(t)
                order = {"NEG": 3, "WEAK": 2, "STRONG": 1}
                if prev is None or order[label] > order[prev[0]]:
                    cue[t] = (label, phrase, s["sent_id"])
        if DECHALLENGE.search(s["text"].lower()) and (not len(d_here) or sus in targets):
            dech = True
    candidates = {sus} | set(cue)

    # 3. pair every candidate drug with every ADR-role entity of the narrative
    rows = []
    if adrs.empty:
        return rows
    for d in candidates:
        d_mentions = drugs[drugs.ent_text.map(norm_drug) == d]
        label, phrase, cue_sid = cue.get(d, (None, "", ""))
        rtype = REL_TYPE[label]
        for pt, g in adrs.groupby("canonical"):
            same, dist, gap = False, None, 99
            for a in g.itertuples():
                for m in d_mentions.itertuples():
                    gap = min(gap, abs(a.sent_no - m.sent_no))
                    if a.sent_id == m.sent_id:
                        same = True
                        sd = parsed[a.sent_id]
                        sa, sm = sd.char_span(a.start_sent, a.end_sent, alignment_mode="expand"), sd.char_span(m.start_sent, m.end_sent, alignment_mode="expand")
                        if sa is not None and sm is not None:
                            dd = dep_distance(sa.root, sm.root)
                            dist = dd if dist is None else min(dist, dd)
            event = any(EVENT_CTX.search(a.sentence[:a.start_sent].lower()) or (a.start_sent == 0 and ASSERT.search(a.sentence.lower()))
                        for a in g.itertuples())
            if rtype == "unrelated":
                conf = 0.05
            else:
                prox = (0.10 if (dist is not None and dist <= 4) else 0.07) if same else (0.05 if gap <= 1 else 0.0)
                conf = min(1.0, CUE_BASE[rtype] + (0.15 if event else 0.0) + prox + (0.05 if dech else 0.0))
            first = g.sort_values("start_doc").iloc[0]
            rows.append(dict(doc_id=doc_id, drug=suspect.ent_text if d == sus else d, drug_generic=d, is_suspect=(d == sus),
                             adr_pt=pt, adr_surface=first.ent_text, relation_type=rtype, confidence=round(conf, 3),
                             cue_phrase=phrase, cue_sentence_id=cue_sid, same_sentence=same, dep_distance=dist,
                             sentence_gap=gap if gap != 99 else None, event_type=bool(event), dechallenge=dech,
                             n_mentions=len(g), context=first.sentence))
    return rows


if __name__ == "__main__":
    clean = pd.read_csv("data/clean_text.csv")
    clean["doc_off"] = clean.groupby("doc_id").text_raw.transform(lambda s: (s.str.len() + 1).cumsum().shift(fill_value=0))
    ents = pd.read_csv("data/entities.csv").fillna({"role": ""})
    raw = pd.read_csv("data/raw_data.csv").set_index("doc_id")

    # alias gazetteer: "paracetamol" is a synonym that FAERS does not use (Part 4 missed it)
    extra = []
    for r in clean.itertuples():
        for m in re.finditer(r"\bparacetamol\b", r.text_raw, flags=re.I):
            extra.append(dict(doc_id=r.doc_id, sent_id=r.sent_id, sent_no=r.sent_no, ent_text=m.group(0), label="DRUG", role="",
                              canonical="paracetamol", start_sent=m.start(), end_sent=m.end(),
                              start_doc=r.doc_off + m.start(), end_doc=r.doc_off + m.end(), sentence=r.text_raw))
    ents = pd.concat([ents, pd.DataFrame(extra)], ignore_index=True)
    print(f"Alias gazetteer added {len(extra)} 'paracetamol' DRUG mentions to the {len(ents) - len(extra)} entities from Part 4")

    print("Parsing sentences for dependency distances ...")
    parsed = dict(zip(clean.sent_id, nlp.pipe(clean.text_raw.tolist(), batch_size=64)))

    rel_rows, doc_ents = [], {d: g for d, g in ents.groupby("doc_id")}
    for doc_id, g in clean.groupby("doc_id"):
        sents = [dict(sent_id=r.sent_id, sent_no=r.sent_no, text=r.text_raw) for r in g.sort_values("sent_no").itertuples()]
        rel_rows += extract_doc(doc_id, sents, doc_ents.get(doc_id, ents.iloc[0:0]), parsed)
    rel = pd.DataFrame(rel_rows).sort_values(["doc_id", "drug", "adr_pt", "cue_sentence_id"], kind="stable").reset_index(drop=True)   # deterministic order
    rel.to_csv("data/relations.csv", index=False)
    print(f"\n{len(rel)} (document, drug, ADR) relations from {rel.doc_id.nunique()} documents")
    print(rel.relation_type.value_counts().to_string())
    print(f"same-sentence pairs: {rel.same_sentence.sum()} ({rel.same_sentence.mean():.0%}); dependency distance available for {rel.dep_distance.notna().sum()}")

    # ------------------------------------------------------------------------------ evaluation (document level)
    docs = raw.index.tolist()
    gold_pos = raw.gt_relation.astype(int)
    drug_e = ents[ents.label == "DRUG"]
    adr_e = ents[(ents.label == "MEDCOND") & (ents.role == "ADR")]
    b1 = {d: bool(set(drug_e[drug_e.doc_id == d].sent_id) & set(adr_e[adr_e.doc_id == d].sent_id)) for d in docs}
    b2 = {d: bool((drug_e.doc_id == d).any() and (adr_e.doc_id == d).any()) for d in docs}
    pos_types = {"causal-strong", "causal-weak"}
    p3 = {d: bool(set(rel[(rel.doc_id == d) & rel.is_suspect].relation_type) & pos_types) for d in docs}


    def prf(pred):
        y, p = gold_pos.values, np.array([pred[d] for d in docs])
        tp, fp, fn = ((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum()
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return dict(TP=tp, FP=fp, FN=fn, precision=prec, recall=rec, F1=2 * prec * rec / max(prec + rec, 1e-9), accuracy=(p == y).mean())


    ev = pd.DataFrame({"1 sentence co-occurrence": prf(b1), "2 abstract co-occurrence": prf(b2), "3 rule-based (this work)": prf(p3)}).T
    print("\nDOCUMENT-LEVEL: does the narrative assert a drug-caused ADR?  (gold: 120 causal narratives vs 30 unrelated / no-event)")
    print(ev.round(3).to_string())

    # relation strength at document level
    top = {}
    for d in docs:
        r = rel[(rel.doc_id == d) & rel.is_suspect]
        top[d] = ("no-pair" if r.empty else r.relation_type.iloc[0])
    pred_strength = pd.Series(top).map({"causal-strong": "strong", "causal-weak": "weak", "unrelated": "none", "co-occurrence": "none", "no-pair": "none"})
    conf_tab = pd.crosstab(raw.gt_strength.rename("gold"), pd.Series(top).rename("predicted relation type"))
    print("\nGold strength vs predicted relation type (documents):")
    print(conf_tab.to_string())
    print(f"3-class strength accuracy (strong / weak / none): {(pred_strength == raw.gt_strength).mean():.1%}")

    # confidence quality: does the score rank related narratives above the rest?
    maxconf = {d: (rel[(rel.doc_id == d) & rel.is_suspect].confidence.max() if (rel.doc_id == d).any() else 0.0) for d in docs}
    maxconf = {d: (0.0 if pd.isna(v) else v) for d, v in maxconf.items()}
    auc = roc_auc_score(gold_pos.values, [maxconf[d] for d in docs])
    m_strong = np.mean([maxconf[d] for d in docs if raw.at[d, "gt_strength"] == "strong"])
    m_weak = np.mean([maxconf[d] for d in docs if raw.at[d, "gt_strength"] == "weak"])
    m_none = np.mean([maxconf[d] for d in docs if raw.at[d, "gt_strength"] == "none"])
    print(f"Confidence: ROC-AUC (related vs not) = {auc:.3f}; mean max-confidence strong {m_strong:.2f} | weak {m_weak:.2f} | none {m_none:.2f}")

    # target-pair recall: is the gold (drug, ADR-PT) pair extracted as a positive relation?
    def form_category(pt, form):
        f, p = form.lower(), pt.lower()
        if f == p:
            return "1 exact PT"
        if p.replace("aem", "em").replace("oea", "ea") == f:
            return "2 US spelling of PT"
        if p in f:
            return "3 PT + modifier"
        return "4 synonym / lay term / abbreviation"
    tgt = defaultdict(lambda: [0, 0])
    for d in docs:
        if raw.at[d, "gt_relation"] != 1:
            continue
        hit = ((rel.doc_id == d) & (rel.drug_generic == raw.at[d, "gt_drug"].lower()) & (rel.adr_pt == raw.at[d, "gt_adr"].lower())
               & rel.relation_type.isin(pos_types)).any()
        cat = form_category(raw.at[d, "gt_adr"], raw.at[d, "gt_adr_surface"])
        tgt[cat][0] += hit
        tgt[cat][1] += 1
        tgt["ALL"][0] += hit
        tgt["ALL"][1] += 1
    print("\nTarget (drug, ADR-PT) pair recovered as a positive relation, by how the event was worded:")
    for k, (h, n) in sorted(tgt.items()):
        print(f"  {k:40s} {h:3d}/{n:3d} = {h / n:.1%}")
    pos_rows = rel[rel.relation_type.isin(pos_types) & rel.is_suspect]
    tgt_rows = pos_rows.merge(raw[["gt_drug", "gt_adr"]].reset_index(), on="doc_id")
    tgt_rows["is_target"] = (tgt_rows.drug_generic == tgt_rows.gt_drug.str.lower()) & (tgt_rows.adr_pt == tgt_rows.gt_adr.str.lower())
    print(f"Positive pair rows: {len(pos_rows)}; target event pairs {tgt_rows.is_target.sum()}, symptom/other ADR pairs {(~tgt_rows.is_target).sum()}; "
          f"event-type mentions among rows: {pos_rows.event_type.sum()}")

    out = [dict(section="document-level", method=m, **row.to_dict()) for m, row in ev.iterrows()]
    out += [dict(section="target pair recall", method=k, TP=h, FN=n - h, recall=h / n) for k, (h, n) in sorted(tgt.items())]
    out += [dict(section="confidence", method="ROC-AUC related vs not", F1=auc),
            dict(section="strength", method="3-class accuracy", accuracy=(pred_strength == raw.gt_strength).mean())]
    pd.DataFrame(out).to_csv("results/relation_evaluation.csv", index=False)

    # ------------------------------------------------------------------------------ aggregate pairs for Part 6
    r = rel.copy()
    r["pos"] = r.relation_type.isin(pos_types)
    agg = r.groupby(["drug_generic", "adr_pt"]).apply(lambda g: pd.Series(dict(
        n_docs=g.doc_id.nunique(), n_positive=int(g.pos.sum()), n_strong=int((g.relation_type == "causal-strong").sum()),
        n_weak=int((g.relation_type == "causal-weak").sum()), n_unrelated=int((g.relation_type == "unrelated").sum()),
        n_cooccurrence=int((g.relation_type == "co-occurrence").sum()), n_event_mentions=int(g.event_type.sum()),
        mean_conf_positive=g[g.pos].confidence.mean() if g.pos.any() else 0.0, max_conf=g.confidence.max(),
        sum_conf_positive=g[g.pos].confidence.sum())), include_groups=False).reset_index()
    agg = agg.sort_values(["sum_conf_positive", "n_positive"], ascending=False)
    agg.to_csv("data/drug_adr_pairs.csv", index=False)
    print(f"\nAggregated {len(agg)} unique drug-ADR pairs -> data/drug_adr_pairs.csv ; top 12 by summed confidence:")
    print(agg.head(12).round(2).to_string(index=False))
