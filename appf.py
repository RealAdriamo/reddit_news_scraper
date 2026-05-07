import re
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Reddit r/news Articles",
    layout="wide"
)

HEADERS = {
    "User-Agent": "streamlit:reddit-news-app:v1.0 (by /u/example)"
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at", "by",
    "with", "from", "after", "over", "under", "is", "are", "was", "were", "be",
    "as", "it", "that", "this", "its", "into", "about", "has", "have", "had",
    "will", "would", "can", "could", "new", "says", "say", "amid", "more",
    "than", "not", "up", "out", "off", "their", "his", "her", "they", "them",
    "you", "your", "our", "we", "but", "who", "what", "when", "why", "how",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "man", "woman", "person", "people", "company", "companies", "group", "groups",
    "news", "article", "articles", "story", "stories", "report", "reports",
    "found", "he", "she", "it", "they", "us", "them", "their", "my", "our",
    "we", "said"
}

TIME_MAP = {
    "past hour": "hour",
    "day": "day",
    "week": "week",
    "month": "month",
    "year": "year",
    "all time": "all",
}

POSITIVE_WORDS = {
    "win", "wins", "won", "victory", "success", "successful",
    "growth", "improve", "improves", "improved", "help",
    "helps", "saved", "safe", "peace", "deal", "breakthrough",
    "record", "rescue", "rescued", "approved", "benefit",
    "benefits", "recover", "recovery"
}

NEGATIVE_WORDS = {
    "war", "killed", "dead", "death", "dies", "crash",
    "attack", "attacks", "shooting", "murder", "crime",
    "fraud", "scandal", "crisis", "threat", "threatens",
    "collapse", "fire", "fired", "lawsuit", "abuse",
    "ban", "banned", "violence", "violent", "corruption",
    "guilty", "charges", "terror", "terrorist"
}

LEFT_TERMS = {
    "climate", "union", "labor", "workers", "minimum wage",
    "abortion", "lgbt", "transgender", "immigration reform",
    "medicaid", "welfare", "student debt", "gun control",
    "healthcare", "civil rights", "equity", "renewable",
    "racism", "police reform"
}

RIGHT_TERMS = {
    "border", "illegal immigration", "tax cuts",
    "gun rights", "second amendment", "religious freedom",
    "crime crackdown", "fossil fuel", "school choice",
    "pro-life", "election fraud", "small government",
    "deportation", "national security", "woke",
    "parental rights"
}

CENTER_TERMS = {
    "bipartisan", "compromise", "moderate", "centrist",
    "independent", "poll", "election", "economy",
    "inflation", "budget", "congress", "court",
    "senate", "house", "president", "policy"
}


def build_start_url(feed_type, timeframe):
    base = "https://www.reddit.com/r/news"

    if feed_type == "hot":
        return f"{base}/hot.json?limit=25"

    if feed_type == "new":
        return f"{base}/new.json?limit=25"

    if feed_type == "rising":
        return f"{base}/rising.json?limit=25"

    if feed_type == "top":
        t = TIME_MAP.get(timeframe, "day")
        return f"{base}/top.json?t={t}&limit=25"

    if feed_type == "controversial":
        t = TIME_MAP.get(timeframe, "day")
        return f"{base}/controversial.json?t={t}&limit=25"

    return f"{base}/hot.json?limit=25"


def scrape_page(url, page_num):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    if response.status_code != 200:
        st.error(f"Reddit returned status code {response.status_code}")
        return [], None

    data = response.json()

    records = []

    children = data.get("data", {}).get("children", [])

    for post in children:
        p = post.get("data", {})

        permalink = p.get("permalink", "")

        records.append(
            {
                "Title": p.get("title"),
                "Score": p.get("score"),
                "Comments": p.get("num_comments"),
                "Post URL": f"https://reddit.com{permalink}",
                "Comments URL": f"https://reddit.com{permalink}",
                "Posted": None,
                "Posted Datetime": pd.to_datetime(
                    p.get("created_utc"),
                    unit="s",
                    errors="coerce"
                ),
                "Page": page_num,
            }
        )

    after = data.get("data", {}).get("after")

    next_url = None

    if after:
        if "&after=" in url:
            next_url = re.sub(
                r"after=[^&]+",
                f"after={after}",
                url
            )
        else:
            next_url = f"{url}&after={after}"

    return records, next_url


@st.cache_data(ttl=900, show_spinner=False)
def scrape_news(num_pages=5, feed_type="hot", timeframe=None):
    url = build_start_url(feed_type, timeframe)

    all_records = []

    for page_num in range(1, num_pages + 1):

        try:
            records, next_url = scrape_page(url, page_num)

        except Exception as e:
            st.error(f"Scraping failed: {e}")
            break

        all_records.extend(records)

        if not next_url:
            break

        url = next_url

    df = pd.DataFrame(all_records)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "Title",
                "Score",
                "Comments",
                "Post URL",
                "Comments URL",
                "Posted",
                "Posted Datetime",
                "Page"
            ]
        )

    df["Score"] = (
        pd.to_numeric(df["Score"], errors="coerce")
        .astype("Int64")
    )

    df["Comments"] = (
        pd.to_numeric(df["Comments"], errors="coerce")
        .astype("Int64")
    )

    df["Page"] = (
        pd.to_numeric(df["Page"], errors="coerce")
        .astype("Int64")
    )

    df["Posted Datetime"] = pd.to_datetime(
        df["Posted Datetime"],
        errors="coerce"
    )

    df = (
        df.drop_duplicates(subset=["Title", "Post URL"])
        .reset_index(drop=True)
    )

    return df


def filter_dataframe(df, keyword, min_score, min_comments):
    filtered = df.copy()

    if keyword.strip():
        filtered = filtered[
            filtered["Title"]
            .fillna("")
            .str.contains(
                keyword,
                case=False,
                regex=True,
                na=False
            )
        ]

    filtered = filtered[
        filtered["Score"].isna()
        | (filtered["Score"] >= min_score)
    ]

    filtered = filtered[
        filtered["Comments"].isna()
        | (filtered["Comments"] >= min_comments)
    ]

    return filtered.reset_index(drop=True)


def top_words(series, n=10):
    text = " ".join(series.dropna().astype(str))

    words = re.findall(
        r"\b[a-z][a-z'-]{2,}\b",
        text.lower()
    )

    words = [
        w for w in words
        if w not in STOPWORDS
    ]

    if not words:
        return pd.DataFrame(columns=["Word", "Count"])

    values, counts = np.unique(words, return_counts=True)

    out = pd.DataFrame(
        {
            "Word": values,
            "Count": counts
        }
    )

    return (
        out.sort_values("Count", ascending=False)
        .head(n)
    )


def simple_sentiment(title):
    text = str(title).lower()

    words = set(
        re.findall(
            r"\b[a-z][a-z'-]{2,}\b",
            text
        )
    )

    positive_score = sum(
        1 for word in words
        if word in POSITIVE_WORDS
    )

    negative_score = sum(
        1 for word in words
        if word in NEGATIVE_WORDS
    )

    if positive_score > negative_score:
        return "Positive"

    if negative_score > positive_score:
        return "Negative"

    return "Neutral"


def classify_leaning(title):
    text = str(title).lower()

    left_score = sum(
        1 for term in LEFT_TERMS
        if term in text
    )

    right_score = sum(
        1 for term in RIGHT_TERMS
        if term in text
    )

    center_score = sum(
        1 for term in CENTER_TERMS
        if term in text
    )

    scores = {
        "Left": left_score,
        "Right": right_score,
        "Center": center_score,
    }

    max_score = max(scores.values())

    if max_score == 0:
        return "Unclear"

    top_labels = [
        label
        for label, score in scores.items()
        if score == max_score
    ]

    if len(top_labels) > 1:
        return "Mixed"

    return top_labels[0]


def add_analysis_columns(df):
    analyzed = df.copy()

    analyzed["Sentiment"] = (
        analyzed["Title"]
        .apply(simple_sentiment)
    )

    analyzed["Political Leaning"] = (
        analyzed["Title"]
        .apply(classify_leaning)
    )

    return analyzed


def train_comment_prediction_model(df):
    model_df = df.dropna(
        subset=[
            "Score",
            "Comments",
            "Page"
        ]
    ).copy()

    if len(model_df) < 10:
        return None, None, None

    features = pd.get_dummies(
        model_df[
            [
                "Score",
                "Page",
                "Sentiment",
                "Political Leaning"
            ]
        ],
        columns=[
            "Sentiment",
            "Political Leaning"
        ],
        drop_first=True
    )

    target = model_df["Comments"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions)
    }

    prediction_df = pd.DataFrame(
        {
            "Actual Comments": y_test.values,
            "Predicted Comments": predictions
        }
    )

    return model, metrics, prediction_df


st.title("Reddit r/news Articles")

with st.sidebar:
    st.header("Scrape settings")

    refresh = st.button("Refresh data")

    pages = st.slider(
        "Pages to scrape",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    feed_type = st.selectbox(
        "Which posts to scrape",
        [
            "Hot",
            "New",
            "Rising",
            "Controversial",
            "Top"
        ]
    ).lower()

    timeframe = None

    if feed_type in {"controversial", "top"}:
        timeframe = st.selectbox(
            "Timeframe",
            [
                "Past Hour",
                "Day",
                "Week",
                "Month",
                "Year",
                "All Time"
            ]
        ).lower()

if refresh:
    scrape_news.clear()

df = scrape_news(
    pages,
    feed_type,
    timeframe
)

st.write(df.head())

if df.empty:
    st.error("No data was scraped.")
    st.stop()

df = add_analysis_columns(df)

with st.sidebar:
    st.header("Filters")

    keyword = st.text_input(
        "Keyword or regex",
        ""
    )

    max_score = int(
        df["Score"]
        .fillna(0)
        .max()
    )

    max_comments = int(
        df["Comments"]
        .fillna(0)
        .max()
    )

    min_score = st.slider(
        "Minimum Upvotes",
        min_value=0,
        max_value=max(max_score, 1),
        value=0
    )

    min_comments = st.slider(
        "Minimum Comments",
        min_value=0,
        max_value=max(max_comments, 1),
        value=0
    )

    sort_by = st.selectbox(
        "Sort by",
        [
            "Score",
            "Comments",
            "Posted Datetime",
            "Page",
            "Title",
            "Sentiment",
            "Political Leaning"
        ]
    )

    ascending = st.checkbox(
        "Sort ascending",
        value=False
    )

filtered_df = filter_dataframe(
    df,
    keyword,
    min_score,
    min_comments
)

filtered_df = (
    filtered_df
    .sort_values(
        sort_by,
        ascending=ascending,
        na_position="last"
    )
    .reset_index(drop=True)
)

st.subheader("Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Posts scraped",
    len(df)
)

c2.metric(
    "Posts after filters",
    len(filtered_df)
)

c3.metric(
    "Average upvotes",
    (
        f"{filtered_df['Score'].dropna().mean():.1f}"
        if len(filtered_df)
        else "0.0"
    )
)

c4.metric(
    "Average comments",
    (
        f"{filtered_df['Comments'].dropna().mean():.1f}"
        if len(filtered_df)
        else "0.0"
    )
)

st.subheader("Sentiment and leaning overview")

s1, s2, s3, s4 = st.columns(4)

most_common_sentiment = (
    filtered_df["Sentiment"].mode().iloc[0]
    if len(filtered_df)
    else "N/A"
)

most_common_leaning = (
    filtered_df["Political Leaning"].mode().iloc[0]
    if len(filtered_df)
    else "N/A"
)

s1.metric(
    "Most common sentiment",
    most_common_sentiment
)

s2.metric(
    "Most common leaning",
    most_common_leaning
)

s3.metric(
    "Highest upvotes",
    int(filtered_df["Score"].max())
    if len(filtered_df)
    else 0
)

s4.metric(
    "Highest comments",
    int(filtered_df["Comments"].max())
    if len(filtered_df)
    else 0
)

st.subheader("Post data")

show_df = filtered_df.copy()

if "Posted Datetime" in show_df.columns:
    show_df["Posted Datetime"] = (
        show_df["Posted Datetime"]
        .astype(str)
        .replace("NaT", "")
    )

st.dataframe(
    show_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Post URL": st.column_config.LinkColumn(
            "Post URL",
            display_text="Open article",
        ),
        "Comments URL": st.column_config.LinkColumn(
            "Comments URL",
        ),
    },
)

csv_data = (
    filtered_df.to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    "Download filtered data as CSV",
    data=csv_data,
    file_name="reddit_news.csv",
    mime="text/csv",
)

left, right = st.columns(2)

with left:
    st.subheader("Upvote distribution")

    plot_df = filtered_df.dropna(subset=["Score"])

    if len(plot_df):
        fig = px.histogram(
            plot_df,
            x="Score",
            nbins=min(12, max(len(plot_df), 1)),
            labels={"Score": "Upvotes"},
            title="Distribution of upvotes",
        )

        fig.update_layout(
            yaxis_title="Number of posts"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

with right:
    st.subheader("Upvotes vs comments")

    plot_df = filtered_df.dropna(
        subset=["Score", "Comments"]
    )

    if len(plot_df):
        fig = px.scatter(
            plot_df,
            x="Score",
            y="Comments",
            hover_name="Title",
            labels={
                "Score": "Upvotes",
                "Comments": "Comments"
            },
            title="Upvotes compared with comments",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

left2, right2 = st.columns(2)

with left2:
    st.subheader("Most common headline words")

    word_df = top_words(
        filtered_df["Title"],
        10
    )

    if len(word_df):
        fig = px.bar(
            word_df,
            x="Word",
            y="Count",
            labels={
                "Word": "Word",
                "Count": "Count"
            },
            title="Common words in headlines",
        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

with right2:
    st.subheader("Average comments by page")

    page_avg = (
        filtered_df
        .dropna(subset=["Comments"])
        .groupby("Page", dropna=True)["Comments"]
        .mean()
        .sort_index()
    )

    if len(page_avg):
        page_avg_df = page_avg.reset_index()

        page_avg_df["Page"] = (
            page_avg_df["Page"]
            .astype(str)
        )

        fig = px.bar(
            page_avg_df,
            x="Page",
            y="Comments",
            labels={
                "Page": "Page",
                "Comments": "Average comments"
            },
            title="Average comments for posts on each page",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

left3, right3 = st.columns(2)

with left3:
    st.subheader("Headline sentiment distribution")

    sentiment_counts = (
        filtered_df["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    if len(sentiment_counts):
        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Headline sentiment distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

with right3:
    st.subheader("Political leaning distribution")

    leaning_counts = (
        filtered_df["Political Leaning"]
        .value_counts()
        .reset_index()
    )

    leaning_counts.columns = [
        "Political Leaning",
        "Count"
    ]

    if len(leaning_counts):
        fig = px.bar(
            leaning_counts,
            x="Political Leaning",
            y="Count",
            title="Estimated political leaning distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

st.subheader("Comment prediction model")

model, metrics, prediction_df = (
    train_comment_prediction_model(filtered_df)
)

if model is not None:

    p1, p2 = st.columns(2)

    p1.metric(
        "Mean absolute error",
        f"{metrics['MAE']:.1f} comments"
    )

    p2.metric(
        "R² score",
        f"{metrics['R2']:.2f}"
    )

    fig = px.scatter(
        prediction_df,
        x="Actual Comments",
        y="Predicted Comments",
        title="Predicted comments vs actual comments",
        labels={
            "Actual Comments": "Actual comments",
            "Predicted Comments": "Predicted comments",
        },
    )

    fig.add_shape(
        type="line",
        x0=prediction_df["Actual Comments"].min(),
        y0=prediction_df["Actual Comments"].min(),
        x1=prediction_df["Actual Comments"].max(),
        y1=prediction_df["Actual Comments"].max(),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
