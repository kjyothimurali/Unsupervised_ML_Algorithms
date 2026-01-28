import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# ----------------------------------
# App Title & Description
# ----------------------------------
st.title("🟢 Customer Segmentation Dashboard")

st.markdown("""
**This system uses K-Means Clustering to group customers based on their purchasing behavior and similarities.**

👉 Discover hidden customer groups without predefined labels.
""")

# ----------------------------------
# Load Dataset
# ----------------------------------
# Get the directory where this app file exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full path to dataset
DATA_PATH = os.path.join(BASE_DIR, "Wholesale customers data.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)


# Spending-related numeric columns
numeric_cols = [
    'Fresh', 'Milk', 'Grocery',
    'Frozen', 'Detergents_Paper', 'Delicassen'
]

# ----------------------------------
# Sidebar – Input Section
# ----------------------------------
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox("Select Feature 1", numeric_cols)
feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    [col for col in numeric_cols if col != feature_1]
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3
)

random_state = st.sidebar.number_input(
    "Random State (Optional)",
    value=42,
    step=1
)

run_button = st.sidebar.button("🟦 Run Clustering")

# ----------------------------------
# Main Logic
# ----------------------------------
if run_button:

    # Select features
    X = df[[feature_1, feature_2]]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans model
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # Inverse transform centroids for plotting
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # ----------------------------------
    # Visualization Section
    # ----------------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in range(k):
        ax.scatter(
            df[df['Cluster'] == cluster][feature_1],
            df[df['Cluster'] == cluster][feature_2],
            label=f'Cluster {cluster}',
            alpha=0.6
        )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='black',
        s=200,
        marker='X',
        label='Centroids'
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Clusters")
    ax.legend()

    st.pyplot(fig)

    # ----------------------------------
    # Cluster Summary Section
    # ----------------------------------
    st.subheader("📋 Cluster Summary")

    summary = df.groupby('Cluster').agg(
        Count=('Cluster', 'count'),
        Avg_Feature_1=(feature_1, 'mean'),
        Avg_Feature_2=(feature_2, 'mean')
    ).reset_index()

    st.dataframe(summary)

    # ----------------------------------
    # Business Interpretation Section
    # ----------------------------------
    st.subheader("💼 Business Interpretation")

    for _, row in summary.iterrows():
        cluster_id = int(row['Cluster'])

        if row['Avg_Feature_1'] > summary['Avg_Feature_1'].mean():
            spending_type = "high-spending"
        else:
            spending_type = "budget-conscious"

        st.markdown(
            f"🔹 **Cluster {cluster_id}**: {spending_type} customers based on {feature_1} and {feature_2} spending."
        )

    # ----------------------------------
    # User Guidance Box
    # ----------------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.warning("👈 Select features and click **Run Clustering** to begin.")
