"""Part 7a - static figures and Streamlit dashboard entry point."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, networkx as nx
r = pd.read_csv("results/ranked_drug_adr_signals.csv")
TIER_COL = {"A": "#1b6ca8", "B": "#e69f00", "C": "#7a7a7a", "D": "#c9c9c9"}   # colour-blind-safe
r["t"] = r.tier.str[0]; r["pair"] = r.drug_generic + " - " + r.adr_pt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: top-15 ranked pairs, stacked literature + FAERS contributions
top = r.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.barh(top.pair, 0.5 * top.lit_score, color="#56b4e9", label="literature (50%)")
ax.barh(top.pair, 0.5 * top.faers_score, left=0.5 * top.lit_score, color="#0b3d6b", label="FAERS EBGM (50%)")
for y, (c, t) in enumerate(zip(top.composite, top.t)): ax.text(c + .01, y, f"{c:.2f} [{t}]", va="center", fontsize=8)
ax.set_xlim(0, 1.12); ax.set_xlabel("composite priority score"); ax.legend(loc="upper center", bbox_to_anchor=(.45, -.14), ncol=2, frameon=False)
ax.set_title("Top 15 drug-ADR signals (tier in brackets)")
fig.tight_layout(); fig.savefig("figures/fig1_top_signals.png", dpi=200); plt.close(fig)

# Fig 2: literature evidence vs FAERS signal strength
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for t, g in r.groupby("t"):
    ax.scatter(g.lit_score, g.faers_score, s=25 + 12 * g.n_docs, c=TIER_COL[t], alpha=.85, edgecolor="white", label=g.tier.iloc[0].split(" (")[0])
for x in r.head(6).itertuples(): ax.annotate(f"{x.drug_generic[:5]}-{x.adr_pt[:12]}", (x.lit_score, x.faers_score), fontsize=7, xytext=(4, -9), textcoords="offset points")
ax.set_xlabel("literature evidence (percentile)"); ax.set_ylabel("FAERS log10 EBGM (percentile)"); ax.legend(frameon=False, fontsize=7, loc="lower left")
ax.set_title("Literature vs FAERS evidence (size = documents)")
fig.tight_layout(); fig.savefig("figures/fig2_lit_vs_faers.png", dpi=200); plt.close(fig)

# Fig 3: drug-ADR network (tiers A-C), edge width = composite
G = nx.Graph(); sub = r[r.t.isin(list("ABC"))]
for x in sub.itertuples(): G.add_edge(x.drug_generic, x.adr_pt, w=x.composite, t=x.t)
drugs = sorted(set(sub.drug_generic)); pos = nx.spring_layout(G, seed=7, k=1.3)
fig, ax = plt.subplots(figsize=(7.5, 5.6))
nx.draw_networkx_edges(G, pos, ax=ax, width=[1 + 5 * G[u][v]["w"] ** 2 for u, v in G.edges], edge_color=[TIER_COL[G[u][v]["t"]] for u, v in G.edges], alpha=.8)
nx.draw_networkx_nodes(G, pos, nodelist=drugs, node_color="#d55e00", node_size=2300, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G if n not in drugs], node_color="#f2f2f2", edgecolors="#555", node_size=140, ax=ax)
nx.draw_networkx_labels(G, pos, labels={d: d for d in drugs}, font_size=6.5, font_color="white", ax=ax)
nx.draw_networkx_labels(G, pos, labels={n: n for n in G if n not in drugs}, font_size=6, ax=ax)
ax.set_title("Drug-ADR signal network (edge colour = tier A blue / B orange / C grey, width = score)"); ax.axis("off")
fig.tight_layout(); fig.savefig("figures/fig3_network.png", dpi=200); plt.close(fig)

# Generate static figures. The interactive dashboard is results/dashboard.py.
print("wrote figures/fig1-3; run the Streamlit dashboard with: streamlit run results/dashboard.py")
