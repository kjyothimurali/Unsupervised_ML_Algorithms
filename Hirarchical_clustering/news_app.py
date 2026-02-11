import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chardet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(layout="wide")
st.title("⭕ News Topic Discovery with Hierarchical Clustering")
st.caption("This system groups similar news articles automatically based on textual similarity.")

# -----------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------
st.sidebar.header("⚙ Configuration")

max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 500)
use_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)
ngram_option = st.sidebar.selectbox("N-gram Range", ["Unigrams", "Bigrams", "Uni + Bi"])
linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average"])
dendro_size = st.sidebar.slider("Number of Articles for Dendrogram", 50, 500, 100)
num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

# -----------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------
uploaded_file = st.file_uploader("Upload Financial News Dataset (CSV without header)", type=["csv"])

if uploaded_file:

    # Detect encoding
    rawdata = uploaded_file.read()
    encoding = chardet.detect(rawdata)["encoding"]
    uploaded_file.seek(0)

    # Load WITHOUT header (important)
    df = pd.read_csv(uploaded_file, header=None, encoding=encoding, engine="python")

    # Assign correct column names
    if df.shape[1] == 2:
        df.columns = ["sentiment", "text"]
    elif df.shape[1] == 1:
        df.columns = ["text"]
    else:
        st.error("Dataset format not supported. Expected 1 or 2 columns.")
        st.stop()

    # Clean text column
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].fillna("")

    st.sidebar.success(f"Loaded dataset: {uploaded_file.name}")

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # -----------------------------------------------------
    # TF-IDF SECTION
    # -----------------------------------------------------
    st.header("TF-IDF Vectorization")

    if ngram_option == "Unigrams":
        ngram_range = (1,1)
    elif ngram_option == "Bigrams":
        ngram_range = (2,2)
    else:
        ngram_range = (1,2)

    tfidf = TfidfVectorizer(
        stop_words="english" if use_stopwords else None,
        max_features=max_features,
        ngram_range=ngram_range
    )

    X = tfidf.fit_transform(df["text"])

    st.write(f"TF-IDF Shape: {X.shape}")

    # -----------------------------------------------------
    # DENDROGRAM
    # -----------------------------------------------------
    st.header("Dendrogram")

    subset = X[:min(dendro_size, len(df))].toarray()
    Z = linkage(subset, method=linkage_method)

    fig1 = plt.figure(figsize=(12,5))
    dendrogram(Z)
    plt.xlabel("Article Index")
    plt.ylabel("Distance")
    st.pyplot(fig1)

    st.info("Large vertical gaps suggest natural topic separation.")

    # -----------------------------------------------------
    # CLUSTERING
    # -----------------------------------------------------
    model = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage=linkage_method
    )

    clusters = model.fit_predict(X.toarray())
    df["Cluster"] = clusters

    # -----------------------------------------------------
    # PCA VISUALIZATION
    # -----------------------------------------------------
    st.header("Clustering Output")
    st.subheader("PCA Cluster Visualization")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    fig2 = plt.figure(figsize=(10,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
    plt.title("Cluster Projection")
    st.pyplot(fig2)

    # -----------------------------------------------------
    # CLUSTER SUMMARY TABLE
    # -----------------------------------------------------
    st.subheader("📋 Cluster Summary")

    terms = tfidf.get_feature_names_out()
    summary_data = []

    for i in range(num_clusters):
        cluster_indices = np.where(clusters == i)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_mean = X[cluster_indices].mean(axis=0)
        top_indices = np.argsort(cluster_mean.A1)[-5:]
        top_words = ", ".join([terms[j] for j in top_indices])

        sample_article = df.iloc[cluster_indices[0]]["text"][:100]

        summary_data.append([
            i,
            len(cluster_indices),
            top_words,
            sample_article
        ])

    summary_df = pd.DataFrame(summary_data, columns=[
        "Cluster ID",
        "Number of Articles",
        "Top Keywords",
        "Sample Article"
    ])

    st.dataframe(summary_df)

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------
    st.header("📊 Validation")

    if len(set(clusters)) > 1:
        score = silhouette_score(X, clusters)
        st.metric("Silhouette Score", round(score,4))

        if score < 0.2:
            st.warning("Clusters have some overlap.")
        else:
            st.success("Clusters are reasonably separated.")
    else:
        st.warning("Silhouette score cannot be calculated for 1 cluster.")

    # -----------------------------------------------------
    # EDITORIAL INSIGHTS
    # -----------------------------------------------------
    st.header("Editorial Insights")

    for i in range(len(summary_df)):
        st.write(f"🟣 Cluster {summary_df.iloc[i]['Cluster ID']}: "
                 f"Articles focus on topics related to {summary_df.iloc[i]['Top Keywords']}.")

    st.success("These clusters support automatic tagging, recommendations, and content organization.")

    # -----------------------------------------------------
    # DOWNLOAD CLEAN DATA
    # -----------------------------------------------------
    st.download_button(
        "Download Clustered Dataset",
        df.to_csv(index=False),
        "clustered_news.csv",
        "text/csv"
    )
