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


# ---------------------------------------------------------
# CASE RELEVANCE SCREENING
# ---------------------------------------------------------

st.subheader("Case relevance screening")

st.write(
    """
    Cases are ranked according to the occurrence of a small set of
    strongly relevant terms. The relevance score combines:

    - **Breadth**: how many different relevant term groups occur
    - **Absolute frequency**: total number of relevant occurrences
    - **Relative frequency**: occurrences per 1,000 words

    The terms currently used are: **whistle\\***, **protected disclosure\\***,
    **PDA**, **retaliation**, and **unfair dismissal**.
    """
)


@st.cache_data
def calculate_case_relevance(case_df):

    # Regex patterns for the five strong term groups
    patterns = {
        # Matches e.g. whistleblower, whistleblowing,
        # whistle-blower, whistle blower, whistle
        "whistle*": re.compile(
            r"\bwhistle(?:[\s\-]?blow\w*)?\b",
            re.IGNORECASE
        ),

        # Matches protected disclosure / protected disclosures
        "protected disclosure*": re.compile(
            r"\bprotected\s+disclosures?\b",
            re.IGNORECASE
        ),

        # Exact acronym
        "PDA": re.compile(
            r"\bPDA\b",
            re.IGNORECASE
        ),

        "retaliation": re.compile(
            r"\bretaliation\b",
            re.IGNORECASE
        ),

        # Allows multiple spaces / line breaks
        "unfair dismissal": re.compile(
            r"\bunfair\s+dismissal\b",
            re.IGNORECASE
        )
    }

    rows = []

    for _, row in case_df.iterrows():

        path = Path(row["path"])

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Simple word count
        word_count = len(re.findall(r"\b\w+\b", text))

        counts = {}

        for term, pattern in patterns.items():
            counts[term] = len(pattern.findall(text))

        total_hits = sum(counts.values())

        # Number of different term groups represented
        breadth = sum(count > 0 for count in counts.values())

        # Relative frequency
        hits_per_1000 = (
            (total_hits / word_count) * 1000
            if word_count > 0 else 0
        )

        rows.append({
            "year": row["year"],
            "filename": row["filename"],
            "path": row["path"],
            "words": word_count,
            "breadth": breadth,
            "total_hits": total_hits,
            "hits_per_1000": hits_per_1000,
            **counts
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    # -----------------------------------------
    # NORMALISE COMPONENTS FOR RELEVANCE SCORE
    # -----------------------------------------

    # Breadth is naturally 0–5
    result["breadth_score"] = result["breadth"] / 5

    # log1p prevents a few documents with extremely many hits
    # from dominating the entire scale
    abs_log = result["total_hits"].apply(lambda x: __import__("math").log1p(x))
    rel_log = result["hits_per_1000"].apply(lambda x: __import__("math").log1p(x))

    if abs_log.max() > abs_log.min():
        result["absolute_score"] = (
            (abs_log - abs_log.min()) /
            (abs_log.max() - abs_log.min())
        )
    else:
        result["absolute_score"] = 0

    if rel_log.max() > rel_log.min():
        result["relative_score"] = (
            (rel_log - rel_log.min()) /
            (rel_log.max() - rel_log.min())
        )
    else:
        result["relative_score"] = 0

    # Combined relevance score
    result["relevance_score"] = (
        0.45 * result["breadth_score"]
        + 0.20 * result["absolute_score"]
        + 0.35 * result["relative_score"]
    )

    result["relevance_score"] = result["relevance_score"].round(3)
    result["hits_per_1000"] = result["hits_per_1000"].round(2)

    return result.sort_values(
        "relevance_score",
        ascending=False
    ).reset_index(drop=True)


# Only analyse the Cases subcorpus
case_df = df[df["subcorpus"] == "Cases"].copy()

case_relevance = calculate_case_relevance(case_df)


if not case_relevance.empty:

    # -----------------------------------------
    # CONTROLS
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        minimum_score = st.slider(
            "Minimum relevance score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01
        )

    with col2:
        high_relevance_threshold = st.slider(
            "Threshold for 'highly relevant' cases",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.01
        )


    ranked_cases = case_relevance[
        case_relevance["relevance_score"] >= minimum_score
    ].copy()


    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cases above selected score",
        len(ranked_cases)
    )

    col2.metric(
        "Highly relevant cases",
        int(
            (
                case_relevance["relevance_score"]
                >= high_relevance_threshold
            ).sum()
        )
    )

    col3.metric(
        "Total cases analysed",
        len(case_relevance)
    )


    # -----------------------------------------
    # RANKED TABLE
    # -----------------------------------------

    st.markdown("#### Cases ranked by relevance")

    display_columns = [
        "year",
        "filename",
        "relevance_score",
        "breadth",
        "total_hits",
        "hits_per_1000",
        "whistle*",
        "protected disclosure*",
        "PDA",
        "retaliation",
        "unfair dismissal",
        "words"
    ]

    st.dataframe(
        ranked_cases[display_columns],
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------
    # RELEVANCE OVER TIME
    # -----------------------------------------

    st.markdown("#### Relevance over time")

    time_df = case_relevance.dropna(subset=["year"]).copy()

    time_df["highly_relevant"] = (
        time_df["relevance_score"]
        >= high_relevance_threshold
    )


    # 1. Average relevance score per year
    average_by_year = (
        time_df
        .groupby("year")["relevance_score"]
        .mean()
        .reset_index()
    )

    fig_average = px.line(
        average_by_year,
        x="year",
        y="relevance_score",
        markers=True,
        labels={
            "year": "Year",
            "relevance_score": "Average relevance score"
        },
        title="Average case relevance by year"
    )

    st.plotly_chart(
        fig_average,
        use_container_width=True
    )


    # 2. Number of highly relevant cases per year
    relevant_by_year = (
        time_df
        .groupby("year")["highly_relevant"]
        .sum()
        .reset_index(name="highly_relevant_cases")
    )

    fig_number = px.bar(
        relevant_by_year,
        x="year",
        y="highly_relevant_cases",
        labels={
            "year": "Year",
            "highly_relevant_cases": "Highly relevant cases"
        },
        title="Number of highly relevant cases by year"
    )

    st.plotly_chart(
        fig_number,
        use_container_width=True
    )


    # 3. Share of cases that are highly relevant
    share_by_year = (
        time_df
        .groupby("year")
        .agg(
            total_cases=("filename", "count"),
            highly_relevant_cases=("highly_relevant", "sum")
        )
        .reset_index()
    )

    share_by_year["share_relevant"] = (
        share_by_year["highly_relevant_cases"]
        / share_by_year["total_cases"]
        * 100
    )

    fig_share = px.line(
        share_by_year,
        x="year",
        y="share_relevant",
        markers=True,
        labels={
            "year": "Year",
            "share_relevant": "Highly relevant cases (%)"
        },
        title="Share of highly relevant cases by year"
    )

    st.plotly_chart(
        fig_share,
        use_container_width=True
    )