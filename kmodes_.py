import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from kmodes.kmodes import KModes
import seaborn as sns 
from typing import List
from sklearn.metrics import normalized_mutual_info_score # Keep NMI for completeness

# --- CONFIG ---
INPUT_PATH = "basic_data_3.aggregated.jsonl" 
RANDOM_SEED = 42
N_CLUSTERS = 8 # Set K=8 to compare against the 8 known CWE categories
K_RANGE = range(2, 12) # Range for Elbow Method (K=2 up to K=11)
# -----------------------------------

def plot_kmodes_elbow(X_categorical):
    """
    Calculates and plots the K-Modes Dissimilarity Cost (Elbow Method).
    """
    cost = []
    print("\n--- Running Elbow Method (K-Modes) ---")
    for k in K_RANGE:
        # KModes model is fitted to the categorical data
        kmodes_model = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0, random_state=RANDOM_SEED)
        kmodes_model.fit(X_categorical)
        cost.append(kmodes_model.cost_)
        
    plt.figure(figsize=(8, 6))
    plt.plot(K_RANGE, cost, marker='o', linestyle='-', color='purple')
    plt.title('Elbow Method for Optimal K (K-Modes Dissimilarity Cost)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Dissimilarity Cost (Huang)')
    plt.xticks(K_RANGE)
    plt.grid(True, alpha=0.5)
    
    # Save the plot
    plot_filename = "kmodes_elbow_method.png"
    plt.savefig(plot_filename)
    print(f"[INFO] K-Modes Elbow visualization saved as '{plot_filename}'")
    plt.close()

def plot_composition_heatmap(composition_df):
    """
    Generates a heatmap of the raw cluster counts cross-tabulation table.
    This provides a visual representation of the console table.
    """
    print("\n--- Generating Cluster Composition Heatmap ---")

    plot_df = composition_df.drop(columns=["Total Samples"], index=["Total Samples"], errors='ignore')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap 
    sns.heatmap(
        plot_df, 
        annot=True, 
        fmt='d', # 'd' format ensures integer display
        cmap='rocket_r', 
        linewidths=.5, 
        linecolor='white', 
        cbar_kws={'label': 'Number of Samples'}
    )

   
    ax.set_title(f'Cluster Composition Heatmap (K-Modes Raw Counts, K={N_CLUSTERS})', fontsize=16)
    ax.set_xlabel('True CWE Category', fontsize=12)
    ax.set_ylabel('K-Modes Cluster ID', fontsize=12)
    plt.yticks(rotation=0) # Keep cluster labels horizontal
    
    plt.tight_layout()
    
   
    plot_filename = "kmodes_cluster_composition_heatmap.png"
    plt.savefig(plot_filename)
    print(f"[INFO] Cluster composition heatmap saved as '{plot_filename}'")
    plt.close()


try:
    df = pd.read_json(INPUT_PATH, lines=True)
except FileNotFoundError:
    print(f"Error: Input file not found at '{INPUT_PATH}'.")
    exit()

# Extract categorical feature data (language and cwe_category)
# NOTE: Using only 'language' and 'cwe_category' for clustering based on the original snippet,
# but typically K-Modes would use all categorical features:
# X = df[['language', 'platform', 'os', 'technology']] 
X = df[['language', 'cwe_category']]

# --- K-Modes Analysis ---

# 1. Run and save the Elbow Method plot
plot_kmodes_elbow(X)

# 2. Create KModes model and train
print(f"\n--- Clustering with K={N_CLUSTERS} ---")
# Only fit on the clustering features ('language', 'cwe_category')
model = KModes(n_clusters=N_CLUSTERS, init='Huang', n_init=5, verbose=0, random_state=RANDOM_SEED) 
# The kmodes library expects a numpy array, not a DataFrame
model.fit(X.values) 

# 3. Predict all data points
all_predictions = model.predict(X.values)
df['kmodes_cluster'] = all_predictions

# 4. Create the cross-tabulation table (counts)
composition_table = pd.crosstab(
    df['kmodes_cluster'], 
    df['cwe_category'], 
    margins=True, 
    margins_name="Total Samples"
)

# 5. Generate visualizations
plot_composition_heatmap(composition_table) # Heatmap (Raw Counts)

# --- Console Analysis Output ---

# Print the table to the console
print("\n--- Cluster Composition Crosstab (K-Modes Result) ---")
print("Shows the breakdown of True CWE Categories inside each machine-discovered cluster.")
print(composition_table)
print("--------------------------------------------------")

# Print the cost
print(f'The final Dissimilarity Cost for K={N_CLUSTERS} is: {model.cost_:.2f}')
print("\n--- K-Modes Unit Style Analysis Complete ---")
