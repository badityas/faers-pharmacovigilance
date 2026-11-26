from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent / "ranked_drug_adr_signals.csv"

TIER_COLORS = {
    "A": "#1b6ca8",
    "B": "#e69f00",
    "C": "#7a7a7a",
    "D": "#c9c9c9",
}

st.set_page_config(
    page_title="ADR Signal Dashboard",
    page_icon="💊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["tier_group"] = data["tier"].str[0]
    data["pair"] = data["drug_generic"] + " - " + data["adr_pt"]
    return data


data = load_data()

st.title("AI-driven ADR Signal Dashboard")
st.caption("Synthetic literature narratives + FAERS 2025Q3")

# Sidebar filters
st.sidebar.header("Filters")
tier_options = [tier for tier in ("A", "B", "C", "D") if tier in set(data["tier_group"])]
selected_tiers = st.sidebar.multiselect(
    "Priority tier",
    tier_options,
    default=tier_options,
)

if selected_tiers:
    filtered = data[data["tier_group"].isin(selected_tiers)].copy()
else:
    filtered = data.iloc[0:0].copy()

search = st.sidebar.text_input("Search drug or ADR")
if search:
    query = search.casefold()
    filtered = filtered[
        filtered["pair"].str.casefold().str.contains(query, na=False)
    ]

minimum_docs = st.sidebar.slider(
    "Minimum literature documents",
    0,
    int(data["n_docs"].fillna(0).max()),
    0,
)
filtered = filtered[filtered["n_docs"].fillna(0) >= minimum_docs]

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Drug–ADR pairs", len(filtered))
col2.metric("Tier A signals", int((filtered["tier_group"] == "A").sum()))
col3.metric("FAERS signals", int(filtered["faers_signal"].fillna(False).sum()))
col4.metric("Highest composite score", f"{filtered['composite'].max():.2f}" if not filtered.empty else "—")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Ranked signals")
    top_n = st.slider("Number of signals", 5, min(50, len(data)), 20)
    top = filtered.head(top_n).iloc[::-1]

    if top.empty:
        st.info("No signals match the selected filters.")
    else:
        ranked_fig = go.Figure()
        ranked_fig.add_trace(go.Bar(
            y=top["pair"],
            x=top["composite"],
            orientation="h",
            marker_color=[TIER_COLORS[tier] for tier in top["tier_group"]],
            customdata=top[["tier", "lit_score", "faers_score"]],
            hovertemplate=(
                "%{y}<br>Composite: %{x:.3f}"
                "<br>Tier: %{customdata[0]}"
                "<br>Literature: %{customdata[1]:.3f}"
                "<br>FAERS: %{customdata[2]:.3f}<extra></extra>"
            ),
        ))
        ranked_fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Composite priority score",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(ranked_fig, use_container_width=True)

with right:
    st.subheader("Literature vs FAERS evidence")

    if filtered.empty:
        st.info("No signals match the selected filters.")
    else:
        evidence_fig = go.Figure()
        for tier in sorted(filtered["tier_group"].unique(), key=lambda value: "ABCD".index(value)):
            group = filtered[filtered["tier_group"] == tier]
            evidence_fig.add_trace(go.Scatter(
                x=group["lit_score"],
                y=group["faers_score"],
                mode="markers",
                name=f"Tier {tier}",
                text=group["pair"],
                marker=dict(
                    size=8 + group["n_docs"].fillna(0).clip(upper=12),
                    color=TIER_COLORS[tier],
                    line=dict(width=1, color="white"),
                ),
                customdata=group[["n_strong", "n_weak", "n_unrelated", "EBGM", "PRR"]],
                hovertemplate=(
                    "%{text}"
                    "<br>Literature percentile: %{x:.3f}"
                    "<br>FAERS percentile: %{y:.3f}"
                    "<br>Strong: %{customdata[0]}, weak: %{customdata[1]}, unrelated: %{customdata[2]}"
                    "<br>EBGM: %{customdata[3]:.1f}, PRR: %{customdata[4]:.1f}<extra></extra>"
                ),
            ))

        evidence_fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Literature evidence percentile",
            yaxis_title="FAERS log10 EBGM percentile",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(evidence_fig, use_container_width=True)

st.divider()
st.subheader("Signal details")

if filtered.empty:
    st.warning("No rows to display.")
else:
    display_columns = [
        "rank", "drug_generic", "adr_pt", "tier", "composite",
        "lit_score", "faers_score", "n_docs", "n_strong", "n_weak",
        "n_unrelated", "EBGM", "PRR", "faers_signal", "in_faers",
    ]
    available_columns = [column for column in display_columns if column in filtered.columns]
    st.dataframe(
        filtered[available_columns].sort_values("composite", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered results",
    data=csv,
    file_name="filtered_drug_adr_signals.csv",
    mime="text/csv",
)
