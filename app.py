from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st


CORPUS_DIR = Path("Lexus_plain_text_only")


st.set_page_config(
    page_title="Lexus Corpus Overview",
    layout="wide"
)

st.title("Lexus Corpus Overview")
st.write("Simple overview of document distribution across the corpus.")


@st.cache_data
def load_metadata():
    rows = []

    for file_path in CORPUS_DIR.rglob("*.txt"):
        subcorpus = file_path.parent.name
        filename = file_path.name

        year_match = re.search(r"(?:19|20)\d{2}", filename)
        year = int(year_match.group(0)) if year_match else None

        rows.append({
            "subcorpus": subcorpus,
            "year": year,
            "filename": filename,
            "path": str(file_path)
        })

    return pd.DataFrame(rows)


df = load_metadata()

if df.empty:
    st.error("No .txt files found. Check that Lexus_plain_text_only is in the same folder as app.py.")
    st.stop()


st.sidebar.header("Filters")

subcorpora = sorted(df["subcorpus"].unique())
selected_subcorpora = st.sidebar.multiselect(
    "Subcorpus",
    subcorpora,
    default=subcorpora
)

filtered = df[df["subcorpus"].isin(selected_subcorpora)]

min_year = int(df["year"].min())
max_year = int(df["year"].max())

selected_years = st.sidebar.slider(
    "Year range",
    min_year,
    max_year,
    (min_year, max_year)
)

filtered = filtered[
    (filtered["year"] >= selected_years[0]) &
    (filtered["year"] <= selected_years[1])
]


col1, col2, col3 = st.columns(3)

col1.metric("Documents", len(filtered))
col2.metric("Subcorpora", filtered["subcorpus"].nunique())
col3.metric("Years", filtered["year"].nunique())


st.subheader("Documents per year — whole selected corpus")

docs_per_year = (
    filtered
    .groupby("year")
    .size()
    .reset_index(name="documents")
    .sort_values("year")
)

fig_year = px.bar(
    docs_per_year,
    x="year",
    y="documents",
    labels={"year": "Year", "documents": "Number of documents"},
)

st.plotly_chart(fig_year, use_container_width=True)


st.subheader("Documents per year by subcorpus")

docs_by_subcorpus_year = (
    filtered
    .groupby(["year", "subcorpus"])
    .size()
    .reset_index(name="documents")
    .sort_values("year")
)

fig_sub = px.line(
    docs_by_subcorpus_year,
    x="year",
    y="documents",
    color="subcorpus",
    markers=True,
    labels={
        "year": "Year",
        "documents": "Number of documents",
        "subcorpus": "Subcorpus"
    },
)

st.plotly_chart(fig_sub, use_container_width=True)


st.subheader("Documents by subcorpus")

docs_by_subcorpus = (
    filtered
    .groupby("subcorpus")
    .size()
    .reset_index(name="documents")
    .sort_values("documents", ascending=False)
)

fig_bar = px.bar(
    docs_by_subcorpus,
    x="subcorpus",
    y="documents",
    labels={"subcorpus": "Subcorpus", "documents": "Number of documents"},
)

st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Term frequency over time")

st.write(
    """
    Search for a word or phrase of your choice and see how often it appears over time.
    
    Use * as wildcard (to see all expressions compatible with your search expression, ex. whistle* will also yield results on 
    whistle-blowing.
    
    Examples:
    - whistle*
    - corruption
    - "public disclosures act"
    """
)

search_term = st.text_input("Search term")


@st.cache_data
def count_matches(filtered_df, query):
    rows = []

    # wildcard -> regex
    query = query.lower().strip()

    # Fjern eventuelle anførselstegn rundt fraser
    query = query.strip('"').strip("'")

    # Escape alt først, så gjør * om til wildcard
    regex_pattern = re.escape(query).replace(r"\*", r"\w*")

    # La mellomrom i søket matche ett eller flere mellomrom/linjeskift i teksten
    regex_pattern = regex_pattern.replace(r"\ ", r"\s+")

    for _, row in filtered_df.iterrows():
        path = Path(row["path"])

        try:
            text = path.read_text(encoding="utf-8").lower()
        except:
            continue

        matches = re.findall(regex_pattern, text)

        rows.append({
            "year": row["year"],
            "subcorpus": row["subcorpus"],
            "matches": len(matches)
        })

    return pd.DataFrame(rows)


if search_term.strip():
    match_df = count_matches(filtered, search_term)

    yearly_matches = (
        match_df
        .groupby("year")["matches"]
        .sum()
        .reset_index()
    )

    fig_matches = px.line(
        yearly_matches,
        x="year",
        y="matches",
        markers=True,
        labels={
            "year": "Year",
            "matches": "Occurrences"
        },
        title=f"Occurrences of '{search_term}' over time"
    )

    st.plotly_chart(fig_matches, use_container_width=True)

    subcorpus_matches = (
        match_df
        .groupby(["year", "subcorpus"])["matches"]
        .sum()
        .reset_index()
    )

    fig_sub_matches = px.line(
        subcorpus_matches,
        x="year",
        y="matches",
        color="subcorpus",
        markers=True,
        labels={
            "year": "Year",
            "matches": "Occurrences",
            "subcorpus": "Subcorpus"
        },
        title=f"Occurrences of '{search_term}' by subcorpus"
    )

    st.plotly_chart(fig_sub_matches, use_container_width=True)


st.subheader("File list")

st.dataframe(
    filtered.sort_values(["subcorpus", "year", "filename"]),
    use_container_width=True,
    hide_index=True
)