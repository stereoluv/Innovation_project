import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from typing import List

# --- CONFIG ---
INPUT_PATH = "basic_data_3.aggregated.jsonl" 
RANDOM_SEED = 42
N_CLUSTERS = 8 # Set K=8 to compare against the 8 known CWE categories
# -----------------------------------

try:
    df = pd.read_json(INPUT_PATH, lines=True)
except FileNotFoundError:
    print(f"Error: Input file not found at '{INPUT_PATH}'.")
    exit()

# Extract feature data and true labels
X_text = df['code_snippet']
y = df['cwe_category']

# 1. Vectorize text (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\w{1,}')
X_tfidf = tfidf_vectorizer.fit_transform(X_text)

# 2. Reduce dimensions for plotting (PCA to 2 components)
pca = PCA(n_components=2, random_state=RANDOM_SEED)
X = pca.fit_transform(X_tfidf.toarray()) # X is now the 2D feature matrix

# 3. Encode true labels for plotting (y is only used for comparison plot)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
target_names = label_encoder.classes_.tolist()

# Create KMeans model and train
model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
model.fit(X) # TODO: fit on feature matrix X (X is now PCA-reduced)

# Predict all data points
all_predictions = model.predict(X) # TODO: predict on X


plt.figure(figsize=(16, 7))

# Plot 1: Clusters colored by TRUE LABEL (Equivalent to Plot original data)
plt.subplot(1, 2, 1)
scatter_true = plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap='Spectral', alpha=0.7)
plt.title('Data Colored by TRUE CWE Category', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(handles=scatter_true.legend_elements()[0], labels=target_names, title="True Label", bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 2: Clusters colored by K-MEANS CLUSTER (Equivalent to Plot clustering results)
plt.subplot(1, 2, 2)
scatter_pred = plt.scatter(X[:, 0], X[:, 1], c=all_predictions, cmap='viridis', alpha=0.7)
# Plot centroids
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='X', s=150, color='red', label='Centroids')
plt.title(f'K-Means Clustering - Predicted Clusters (k={N_CLUSTERS})', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(handles=scatter_pred.legend_elements()[0], labels=[f'Cluster {i}' for i in range(N_CLUSTERS)], title="Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.9, 1])
plot_filename = "kmeans_visualization.png"
plt.savefig(plot_filename)
print(f"[INFO] K-Means visualization saved as '{plot_filename}'")
plt.close()


# --- Predict Single Data Point ---

# Sample vulnerable code snippet (Example: simple SQL injection pattern)
sample_snippet = ["$query = 'SELECT * FROM users WHERE id = ' . $id;"]

# The sample must go through the same TF-IDF and PCA steps as the training data
sample_tfidf = tfidf_vectorizer.transform(sample_snippet)
sample_reduced = pca.transform(sample_tfidf.toarray())

predicted_label = model.predict(sample_reduced)
print(f'\nThe predicted cluster for the sample code snippet is: {predicted_label[0]}')
print("\n--- K-Means Unit Style Analysis Complete ---")
