"""Part 6 - signal prioritisation: literature evidence (Part 5) + FAERS disproportionality (Assignment 1/2).
Transparent composite score (no trained model: 48 pairs is too few to fit weights without overfitting).
  LIT   = percentile rank of  evidence = sum_conf_positive * (1 - unrelated_share) * (1 if event-type mentions else 0.5)
  FAERS = percentile rank of  log10(EBGM) ; EB05 and PRR(Haldane) reported alongside
  COMPOSITE = w*LIT + (1-w)*FAERS , w = 0.5 ; sensitivity to w in 0.3..0.7 reported."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
p = pd.read_csv("data/drug_adr_pairs.csv")
f = pd.read_csv("data/FAERS_signals_with_EBGM.csv")
f["drug_l"], f["adr_l"] = f.drug.str.lower().str.strip(), f.adr.str.lower().str.strip()

def faers_row(drug, adr):
    c = f[(f.adr_l == adr) & (f.drug_l.str.contains(rf"\b{drug}\b", regex=True))]
    return c.sort_values("a", ascending=False).iloc[0] if len(c) else None

rows = []
for r in p.itertuples():
    x = faers_row(r.drug_generic, r.adr_pt)
    d = r._asdict(); d.pop("Index")
    if x is None:
        d.update(faers_a=0, PRR=np.nan, ROR=np.nan, EBGM=np.nan, EB05=np.nan, faers_signal=False, in_faers=False)
    else:
        a, b, c, dd = x.a, x.b, x.c, x.d
        prr = ((a + .5) / (a + b + 1)) / ((c + .5) / (c + dd + 1))          # Haldane-Anscombe: finite when c = 0
        d.update(faers_a=a, PRR=round(prr, 2), ROR=round(((a+.5)*(dd+.5))/((b+.5)*(c+.5)), 2), EBGM=x.EBGM, EB05=x.EB05,
                 faers_signal=bool(x.signal_flag), in_faers=True)
    rows.append(d)
d = pd.DataFrame(rows)
d["unrelated_share"] = d.n_unrelated / d.n_docs
d["evidence"] = d.sum_conf_positive * (1 - d.unrelated_share) * np.where(d.n_event_mentions > 0, 1.0, 0.5)
d["lit_score"] = d.evidence.rank(pct=True)
d["faers_score"] = np.log10(d.EBGM.fillna(0).clip(lower=1)).rank(pct=True).where(d.in_faers, 0.0)
def comp(w): return w * d.lit_score + (1 - w) * d.faers_score
d["composite"] = comp(0.5).round(3)
d["rank"] = d.composite.rank(ascending=False, method="min").astype(int)
def tier(r):
    lit = r.n_strong >= 3 and r.n_event_mentions > 0
    if lit and r.faers_signal: return "A: confirmed (literature + FAERS)"
    if lit: return "B: literature-only (not a FAERS signal)"
    if r.faers_signal: return "C: FAERS-only / weak literature"
    return "D: low priority"
d["tier"] = d.apply(tier, axis=1)
d = d.sort_values(["rank", "evidence"], ascending=[True, False])
cols = ["rank", "drug_generic", "adr_pt", "tier", "composite", "lit_score", "faers_score", "n_docs", "n_strong", "n_weak", "n_unrelated",
        "n_event_mentions", "mean_conf_positive", "evidence", "faers_a", "PRR", "ROR", "EBGM", "EB05", "faers_signal", "in_faers"]
d[cols].round(3).to_csv("results/ranked_drug_adr_signals.csv", index=False)
print(d[cols[:8] + ["faers_a", "EBGM"]].head(20).round(2).to_string(index=False))
print("\ntiers:\n", d.tier.value_counts().to_string(), f"\nin FAERS: {d.in_faers.sum()}/{len(d)}  FAERS signal flag: {d.faers_signal.sum()}")
base = d["composite"]
for w in (0.3, 0.7, 1.0, 0.0):
    print(f"w={w}: Spearman vs w=0.5 = {spearmanr(base, comp(w))[0]:.2f}")
print("Spearman(lit_score, faers_score) =", round(spearmanr(d.lit_score, d.faers_score)[0], 2))
