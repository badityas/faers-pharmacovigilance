"""
Part 2 - Data acquisition: synthetic case narratives.

PubMed (NCBI Entrez) is unreachable from this environment, so we generate
clinical case-report style narratives instead (explicitly allowed by the
mini-project brief). Narratives are grounded in the Assignment-1/2 FAERS
signal table: 6 high-EBGM "signal" pairs + 4 low-EBGM "control" pairs, all
taken from the Assignment-2 20-drug dataset.

Each narrative is template-generated with randomised demographics, dose,
onset, findings, management, outcome and causality wording, so surface text
varies. ADR mentions use several surface forms (PT, spelling variants, lay /
synonym phrasing) so that downstream dictionary NER has to cope with
imperfect vocabulary coverage. Ground-truth columns are stored so that every
later step (NER, relation extraction, ranking) can be validated.
"""
import json
import random
import pandas as pd

random.seed(42)
N_CAUSAL, N_UNRELATED, N_DRUG_ONLY = 12, 2, 1   # per pair -> 15 x 10 pairs = 150

# --------------------------------------------------------------------------
# Pair definitions. drug / adr_pt match the Assignment-2 table exactly.
# --------------------------------------------------------------------------
PAIRS = [
    dict(drug="METFORMIN", adr_pt="Lactic acidosis", ptype="signal",
         drug_forms=["metformin", "metformin", "metformin", "Glucophage"],
         adr_forms=["lactic acidosis", "lactic acidosis", "severe lactic acidosis",
                    "metformin-associated lactic acidosis", "lactic acid accumulation"],
         indication="type 2 diabetes mellitus", doses=["500 mg twice daily", "1000 mg twice daily", "850 mg three times daily"],
         comorbid=["chronic kidney disease stage 4", "congestive heart failure", "acute kidney injury", "hepatic impairment"],
         symptoms=["profound malaise, abdominal pain and rapid deep breathing", "vomiting, hyperventilation and altered mental status", "severe myalgia, hypotension and drowsiness"],
         findings=["arterial pH of {ph} with serum lactate of {lac} mmol/L", "an anion-gap metabolic acidosis with lactate {lac} mmol/L and pH {ph}"],
         management=["Metformin was withdrawn and the patient underwent haemodialysis.", "Metformin was stopped and bicarbonate infusion with renal replacement therapy was started.", "The drug was discontinued and the patient was admitted to intensive care for supportive management."],
         alt_causes=["septic shock", "mesenteric ischaemia", "a prolonged seizure"], severe=True),
    dict(drug="ACETAMINOPHEN", adr_pt="Acute hepatic failure", ptype="signal",
         drug_forms=["acetaminophen", "paracetamol", "acetaminophen", "Tylenol"],
         adr_forms=["acute hepatic failure", "acute liver failure", "fulminant hepatic failure", "acute hepatic failure"],
         indication="chronic back pain", doses=["4 g daily", "1 g every six hours", "6 g over 24 hours", "500 mg tablets, up to 10 tablets daily"],
         comorbid=["chronic alcohol use", "malnutrition", "hepatitis C infection", "no known liver disease"],
         symptoms=["jaundice, nausea and right upper quadrant pain", "confusion, vomiting and coagulopathy", "progressive fatigue, dark urine and encephalopathy"],
         findings=["ALT of {alt} U/L and INR of {inr}", "markedly raised transaminases (ALT {alt} U/L) with INR {inr}"],
         management=["Acetaminophen was stopped and intravenous N-acetylcysteine was commenced.", "The drug was discontinued, N-acetylcysteine was given and the patient was listed for transplant assessment.", "Paracetamol was withheld and the patient was transferred to a liver unit."],
         alt_causes=["viral hepatitis", "ischaemic hepatitis", "autoimmune hepatitis"], severe=True),
    dict(drug="OLANZAPINE", adr_pt="Neuroleptic malignant syndrome", ptype="signal",
         drug_forms=["olanzapine", "olanzapine", "Zyprexa"],
         adr_forms=["neuroleptic malignant syndrome", "neuroleptic malignant syndrome", "NMS", "malignant neuroleptic syndrome"],
         indication="schizophrenia", doses=["10 mg daily", "20 mg daily", "15 mg at night", "dose increased from 10 mg to 25 mg"],
         comorbid=["dehydration", "recent intramuscular antipsychotic injection", "bipolar disorder", "no significant comorbidity"],
         symptoms=["high fever, generalised muscle rigidity and altered consciousness", "hyperthermia, tachycardia and marked rigidity", "fever, diaphoresis, tremor and autonomic instability"],
         findings=["a temperature of {temp} C and creatine kinase of {ck} U/L", "creatine kinase rising to {ck} U/L with leucocytosis and temperature {temp} C"],
         management=["Olanzapine was discontinued and the patient received dantrolene and aggressive cooling.", "The antipsychotic was stopped and supportive care with bromocriptine was started.", "Olanzapine was withdrawn and the patient was managed in the intensive care unit."],
         alt_causes=["meningitis", "serotonin syndrome", "heat stroke"], severe=True),
    dict(drug="ATORVASTATIN", adr_pt="Immune-mediated myositis", ptype="signal",
         drug_forms=["atorvastatin", "atorvastatin", "Lipitor"],
         adr_forms=["immune-mediated myositis", "immune-mediated necrotizing myopathy", "statin-associated autoimmune myopathy", "immune-mediated myositis"],
         indication="hypercholesterolaemia", doses=["40 mg daily", "80 mg daily", "20 mg daily"],
         comorbid=["hypothyroidism", "type 2 diabetes", "prior coronary stenting", "no other muscle disease"],
         symptoms=["progressive proximal muscle weakness and myalgia", "difficulty climbing stairs and persistent muscle pain", "symmetrical proximal weakness persisting after statin withdrawal"],
         findings=["creatine kinase of {ck} U/L and anti-HMGCR antibodies", "markedly elevated creatine kinase ({ck} U/L) with positive anti-HMGCR antibody"],
         management=["Atorvastatin was stopped and the patient was treated with prednisolone and intravenous immunoglobulin.", "The statin was discontinued and immunosuppression with methotrexate was added.", "Atorvastatin was withdrawn; weakness improved slowly on corticosteroids."],
         alt_causes=["polymyositis of unknown cause", "hypothyroid myopathy", "inclusion body myositis"], severe=False),
    dict(drug="OMEPRAZOLE", adr_pt="Hypomagnesaemia", ptype="signal",
         drug_forms=["omeprazole", "omeprazole", "Prilosec"],
         adr_forms=["hypomagnesaemia", "hypomagnesemia", "low serum magnesium", "severe hypomagnesaemia"],
         indication="gastro-oesophageal reflux disease", doses=["20 mg daily", "40 mg daily", "40 mg twice daily"],
         comorbid=["long-term diuretic use", "chronic diarrhoea", "alcohol use disorder", "no other electrolyte disorder"],
         symptoms=["muscle cramps, tremor and palpitations", "tetany, paraesthesia and a generalised seizure", "fatigue, muscle spasms and cardiac arrhythmia"],
         findings=["serum magnesium of {mg} mmol/L with secondary hypocalcaemia", "a serum magnesium level of {mg} mmol/L"],
         management=["Omeprazole was discontinued and intravenous magnesium sulphate was administered.", "The proton pump inhibitor was stopped and oral magnesium supplementation was started.", "Omeprazole was withdrawn and magnesium levels normalised over two weeks."],
         alt_causes=["chronic diuretic therapy", "alcohol-related malabsorption", "diabetic nephropathy"], severe=False),
    dict(drug="CLOPIDOGREL", adr_pt="Procedural haemorrhage", ptype="signal",
         drug_forms=["clopidogrel", "clopidogrel", "Plavix"],
         adr_forms=["procedural haemorrhage", "post-procedural bleeding", "peri-procedural haemorrhage", "procedural haemorrhage"],
         indication="acute coronary syndrome with stent placement", doses=["75 mg daily", "75 mg daily with aspirin 100 mg", "300 mg loading then 75 mg daily"],
         comorbid=["peripheral arterial disease", "chronic kidney disease", "anaemia", "prior gastrointestinal bleed"],
         symptoms=["persistent bleeding from the access site", "a large haematoma following the procedure", "uncontrolled oozing after the dental extraction"],
         findings=["a haemoglobin drop from {hb0} to {hb1} g/dL", "haemoglobin falling to {hb1} g/dL requiring transfusion"],
         management=["Clopidogrel was interrupted and two units of red cells were transfused.", "The antiplatelet was held and local haemostatic measures were applied.", "Clopidogrel was discontinued and platelet transfusion was given."],
         alt_causes=["a surgical vessel injury", "inadequate local haemostasis", "heparin overdose"], severe=False),
    # ---- low-EBGM controls: common, non-specific events --------------------
    dict(drug="METFORMIN", adr_pt="Diarrhoea", ptype="control",
         drug_forms=["metformin", "metformin", "Glucophage"],
         adr_forms=["diarrhoea", "diarrhea", "loose stools"],
         indication="type 2 diabetes mellitus", doses=["500 mg twice daily", "1000 mg daily"],
         comorbid=["obesity", "hypertension", "no gastrointestinal history"],
         symptoms=["mild watery stools and abdominal discomfort", "loose stools after meals"],
         findings=["normal renal function and unremarkable stool studies"],
         management=["The dose was reduced and symptoms settled.", "Symptoms resolved after switching to the extended-release formulation."],
         alt_causes=["gastroenteritis", "dietary change"], severe=False),
    dict(drug="ATORVASTATIN", adr_pt="Headache", ptype="control",
         drug_forms=["atorvastatin", "atorvastatin", "Lipitor"],
         adr_forms=["headache", "mild headache", "headache"],
         indication="hypercholesterolaemia", doses=["10 mg daily", "20 mg daily"],
         comorbid=["hypertension", "migraine history", "no neurological history"],
         symptoms=["a mild bilateral headache", "intermittent tension-type headaches"],
         findings=["a normal neurological examination"],
         management=["No action was required and the headaches settled spontaneously.", "Paracetamol was used occasionally and atorvastatin was continued."],
         alt_causes=["tension headache", "poor sleep"], severe=False),
    dict(drug="OLANZAPINE", adr_pt="Vomiting", ptype="control",
         drug_forms=["olanzapine", "olanzapine", "Zyprexa"],
         adr_forms=["vomiting", "vomiting", "emesis"],
         indication="bipolar disorder", doses=["5 mg daily", "10 mg at night"],
         comorbid=["anxiety", "no gastrointestinal history"],
         symptoms=["occasional vomiting in the first week of treatment", "episodes of vomiting and mild nausea"],
         findings=["normal electrolytes and liver tests"],
         management=["Symptoms resolved without change in therapy.", "An anti-emetic was prescribed and olanzapine was continued."],
         alt_causes=["viral gastroenteritis", "anxiety"], severe=False),
    dict(drug="OMEPRAZOLE", adr_pt="Nausea", ptype="control",
         drug_forms=["omeprazole", "omeprazole", "Prilosec"],
         adr_forms=["nausea", "mild nausea", "nausea"],
         indication="dyspepsia", doses=["20 mg daily", "20 mg twice daily"],
         comorbid=["Helicobacter pylori infection", "no other comorbidity"],
         symptoms=["transient nausea after the morning dose", "mild nausea with occasional bloating"],
         findings=["normal liver function tests"],
         management=["Taking the tablet with food resolved the symptom.", "No change in therapy was needed."],
         alt_causes=["H. pylori gastritis", "a viral illness"], severe=False),
]

CONCOMITANT = ["lisinopril", "aspirin", "furosemide", "levothyroxine", "bisoprolol",
               "insulin glargine", "warfarin", "amlodipine", "ramipril", "pantoprazole"]
SEX = [("man", "he"), ("woman", "she")]
STRONG = ["The event was considered probably related to {drug}.",
          "Causality assessment indicated a probable association with {drug}.",
          "{ADR} was attributed to {drug}.",
          "The reaction was judged to be drug-related, with {drug} the most likely cause."]
WEAK = ["A contribution from {drug} could not be excluded.",
        "The event was considered possibly related to {drug}.",
        "{drug} was regarded as a possible cause."]
OUTCOMES_SEVERE = [("The patient recovered fully.", .45), ("The patient recovered with sequelae.", .15),
                   ("The patient required prolonged intensive care but survived.", .25), ("The patient died despite treatment.", .15)]
OUTCOMES_MILD = [("The patient recovered fully.", .8), ("The patient was recovering at last follow-up.", .2)]


def pick(opts):
    r, c = random.random(), 0
    for txt, w in opts:
        c += w
        if r <= c:
            return txt
    return opts[-1][0]


def fill_findings(t):
    return t.format(ph=round(random.uniform(6.8, 7.1), 2), lac=round(random.uniform(9, 24), 1),
                    alt=random.randint(1200, 9800), inr=round(random.uniform(2.1, 6.5), 1),
                    temp=round(random.uniform(38.8, 41.2), 1), ck=random.randint(2500, 28000),
                    mg=round(random.uniform(0.25, 0.5), 2), hb0=round(random.uniform(11, 14), 1),
                    hb1=round(random.uniform(6.2, 8.4), 1))


def locate(sentence, offset, items):
    """Gold-standard spans. items = [(role, prefix, text), ...] in the order they occur in
    `sentence`; `prefix` anchors the slot so repeated strings are never confused.
    Returns spans with absolute character offsets inside the full abstract."""
    out, cursor = [], 0
    for role, prefix, text in items:
        i = sentence.index(prefix + text, cursor)
        start = offset + i + len(prefix)
        out.append(dict(role=role, text=text, start=start, end=start + len(text)))
        cursor = i
    return out


def build(pair, kind, i):
    drug = random.choice(pair["drug_forms"])
    adr = random.choice(pair["adr_forms"])
    age, (sex, pron) = random.randint(34, 89), random.choice(SEX)
    others = random.sample([c for c in CONCOMITANT if c != pair["drug"].lower()], random.choice([0, 1, 2]))
    dur = random.choice(["3 weeks", "2 months", "6 months", "3 years", "10 days", "18 months"])
    onset = random.choice(["two days", "one week", "three weeks", "about a month", "several months"])
    comorb = random.choice(pair["comorbid"])
    dose = pair["doses"][0] if kind == "drug_only" else random.choice(pair["doses"])   # keep routine-review doses therapeutic
    s1 = f"A {age}-year-old {sex} with {comorb} had been taking {drug} ({dose}) for {pair['indication']} for {dur}."
    if others:
        s1 += f" The patient was also receiving {' and '.join(others)}."
    s1_items = [("COMORBIDITY", "with ", comorb), ("DRUG", "taking ", drug), ("INDICATION", "for ", pair["indication"])]
    spans = locate(s1, 0, s1_items)
    if others:                                   # concomitant drugs, in order of appearance
        spans += locate(s1, 0, [("DRUG", "also receiving " if k == 0 else " and ", o) for k, o in enumerate(others)])
    sym = random.choice(pair["symptoms"])
    if kind == "causal":
        if random.random() < 0.3 and others:
            s2 = f"After starting {others[0]}, {onset} later the patient presented with {sym} and was diagnosed with {adr} while on {drug}."
        else:
            s2 = f"{onset.capitalize()} after the dose change, the patient presented with {sym} and was diagnosed with {adr}."
        finding = fill_findings(random.choice(pair["findings"]))
        s3 = f"Investigations showed {finding}."
        s4 = random.choice(pair["management"])
        pool = STRONG if pair["ptype"] == "signal" else WEAK
        s5 = random.choice(pool).format(drug=drug, ADR=adr[0].upper() + adr[1:])
        s5 = s5[0].upper() + s5[1:]          # sentence-initial capital (e.g. '{drug} was regarded ...')
        s6 = pick(OUTCOMES_SEVERE if pair["severe"] else OUTCOMES_MILD)
        text = " ".join([s1, s2, s3, s4, s5, s6])
        o2, o3, o4, o5 = (len(" ".join([s1, s2, s3, s4, s5, s6][:k])) + 1 for k in (1, 2, 3, 4))
        if "After starting" in s2:
            spans += locate(s2, o2, [("DRUG", "After starting ", others[0]), ("SYMPTOM", "presented with ", sym), ("EVENT", "diagnosed with ", adr)])
        else:
            spans += locate(s2, o2, [("SYMPTOM", "presented with ", sym), ("EVENT", "diagnosed with ", adr)])
        spans += locate(s3, o3, [("FINDING", "Investigations showed ", finding)])
        spans += [dict(role="MANAGEMENT", text=s4, start=o4, end=o4 + len(s4))]
        adr_cap = adr[0].upper() + adr[1:]
        if s5.startswith(adr_cap):
            spans += [dict(role="EVENT", text=adr_cap, start=o5, end=o5 + len(adr_cap))]
        rel, strength = 1, ("strong" if pair["ptype"] == "signal" else "weak")
        gt_adr = pair["adr_pt"]
    elif kind == "unrelated":
        alt = random.choice(pair["alt_causes"])
        s2 = f"{onset.capitalize()} later the patient developed {adr} with {sym}."
        s3 = f"Investigations pointed to {alt}."
        s4 = f"The event was judged unrelated to {drug} and was attributed to {alt}."
        s5 = "No recurrence was seen once the underlying cause had been treated."
        text = " ".join([s1, s2, s3, s4, s5])
        o2, o3, o4 = (len(" ".join([s1, s2, s3, s4, s5][:k])) + 1 for k in (1, 2, 3))
        spans += locate(s2, o2, [("EVENT", "developed ", adr), ("SYMPTOM", " with ", sym)])
        spans += locate(s3, o3, [("ALT_CAUSE", "pointed to ", alt)])
        spans += locate(s4, o4, [("ALT_CAUSE", "attributed to ", alt)])
        rel, strength, gt_adr = 0, "none", pair["adr_pt"]
    else:  # drug_only: no adverse event described
        s2 = f"At routine review, the patient reported no new symptoms on {drug}."
        s3 = random.choice(["Blood pressure and weight were stable.", "Routine blood tests were unremarkable.", "Adherence was good and no dose change was needed."])
        s4 = f"The patient will be reviewed again in {random.choice(['three', 'six'])} months."
        text = " ".join([s1, s2, s3, s4])
        rel, strength, gt_adr = 0, "none", ""
        adr = ""
    for sp in spans:                                # every gold span must point at exactly its text
        assert text[sp["start"]:sp["end"]] == sp["text"], (sp, text)
    title_adr = adr if adr else "routine follow-up"
    title = (f"Case report: {title_adr} in a patient treated with {drug}" if kind != "drug_only"
             else f"Case report: routine follow-up of a patient treated with {drug}")
    return dict(title=title, abstract=text, source="synthetic_case_narrative",
                gt_drug=pair["drug"], gt_drug_surface=drug, gt_adr=gt_adr, gt_adr_surface=adr,
                gt_relation=rel, gt_strength=strength, narrative_type=kind, pair_type=pair["ptype"],
                gt_other_drugs="|".join(others), gt_spans=json.dumps(spans))


rows, seen = [], set()
for pair in PAIRS:
    for kind, n in [("causal", N_CAUSAL), ("unrelated", N_UNRELATED), ("drug_only", N_DRUG_ONLY)]:
        made = 0
        while made < n:
            r = build(pair, kind, made)
            if r["abstract"] in seen:          # keep every narrative unique
                continue
            seen.add(r["abstract"])
            rows.append(r)
            made += 1

random.shuffle(rows)
df = pd.DataFrame(rows)
df.insert(0, "doc_id", [f"DOC{str(i + 1).zfill(3)}" for i in range(len(df))])
df.to_csv("data/raw_data.csv", index=False)

# save the ADR surface-form map so Part 4 can evaluate vocabulary coverage honestly
json.dump({p["adr_pt"]: sorted(set(p["adr_forms"])) for p in PAIRS}, open("data/adr_surface_forms.json", "w"), indent=2)
json.dump({p["drug"]: sorted(set(p["drug_forms"])) for p in PAIRS}, open("data/drug_surface_forms.json", "w"), indent=2)
print(f"Wrote {len(df)} narratives -> data/raw_data.csv")
