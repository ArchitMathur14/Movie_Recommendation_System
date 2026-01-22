import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Movie Recommendation System")
st.subheader("Discover similar movies using content-based filtering (TF-IDF + Cosine Similarity)")

# --------------------------------------------------
# Load & Prepare Data
# --------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ArchitMathur14/MovieRec_Sys/main/movies.csv"
    df = pd.read_csv(url)

    # Sample 10,001 rows
    df = df.sample(n=10001, random_state=42).reset_index(drop=True)

    # Handle missing overview
    df["overview"] = df["overview"].fillna("")

    # Extract year
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    return df


@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim, tfidf


df = load_data()
cosine_sim, tfidf = build_model(df)

# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------
def get_recommendations(title, cosine_sim, df, n=10):
    try:
        idx = df[df["title"] == title].index[0]
        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        seen = set()
        movie_indices = []
        similarity_scores = []

        for i, score in scores:
            movie_title = df.iloc[i]["title"]
            if movie_title != title and movie_title not in seen:
                movie_indices.append(i)
                similarity_scores.append(score)
                seen.add(movie_title)
            if len(movie_indices) == n:
                break

        result = df.iloc[movie_indices].copy()
        result["similarity_score"] = similarity_scores
        return result

    except IndexError:
        return pd.DataFrame()


# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("🎯 Select a Movie")

input_method = st.sidebar.radio("Input Method", ["Search", "Dropdown"])

if input_method == "Search":
    query = st.sidebar.text_input("Search movie title")
    matches = df[df["title"].str.contains(query, case=False, na=False)]["title"].unique()
    selected_movie = st.sidebar.selectbox("Matching movies", matches) if len(matches) > 0 else None
else:
    selected_movie = st.sidebar.selectbox("Choose a movie", sorted(df["title"].unique()))

st.sidebar.markdown("---")

n_recommendations = st.sidebar.slider("Number of recommendations", 5, 20, 10)
min_rating = st.sidebar.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)
popular_only = st.sidebar.checkbox("Only popular movies (Top 50%)")

# --------------------------------------------------
# Main App Logic
# --------------------------------------------------
if selected_movie:
    movie = df[df["title"] == selected_movie].iloc[0]

    st.markdown("### 🎥 Selected Movie")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Title", movie["title"])
    col2.metric("⭐ Rating", f"{movie['vote_average']:.1f}")
    col3.metric("🔥 Popularity", f"{movie['popularity']:.1f}")
    col4.metric("📅 Year", int(movie["year"]) if pd.notna(movie["year"]) else "N/A")

    st.markdown("**Overview**")
    st.write(movie["overview"])

    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Finding similar movies..."):
            recs = get_recommendations(selected_movie, cosine_sim, df, n_recommendations)

            if min_rating > 0:
                recs = recs[recs["vote_average"] >= min_rating]

            if popular_only:
                threshold = df["popularity"].quantile(0.5)
                recs = recs[recs["popularity"] >= threshold]

            if recs.empty:
                st.warning("No movies match your filters.")
            else:
                st.markdown("### 🎬 Recommended Movies")

                fig = px.bar(
                    recs,
                    x="similarity_score",
                    y="title",
                    orientation="h",
                    title="Similarity Scores",
                    labels={"similarity_score": "Similarity"},
                )
                st.plotly_chart(fig, use_container_width=True)

                for _, row in recs.iterrows():
                    st.markdown(f"#### 🎞️ {row['title']}")
                    st.write(row["overview"][:300] + "...")
                    st.write(
                        f"⭐ Rating: {row['vote_average']} | "
                        f"🔥 Popularity: {row['popularity']} | "
                        f"📅 Year: {int(row['year']) if pd.notna(row['year']) else 'N/A'} | "
                        f"🎯 Match: {row['similarity_score']*100:.1f}%"
                    )
                    st.markdown("---")
else:
    st.info("👈 Select a movie from the sidebar to begin")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | TF-IDF & Cosine Similarity")
