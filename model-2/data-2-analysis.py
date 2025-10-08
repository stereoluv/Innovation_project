import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List

# --- CONFIG ---
INPUT_FILE = "model-2/data/basic_data_3.cleaned.jsonl" 
OUTPUT_DIR = "model-2/analysis-2-output" 
# -----------------------------------

# ---------------- Load Data ----------------
def load_jsonl(path: str) -> pd.DataFrame:
    """Loads a JSONL file into a pandas DataFrame."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file not found at '{path}'. Please check the path.")
        return pd.DataFrame()
        
    return pd.DataFrame(rows)

# ---------------- Preprocess ----------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic preprocessing, including filling missing values and creating 
    simple length features for analysis.
    """
    if df.empty:
        return df
        
    for col in ["code_snippet", "exploitation_techniques"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
            
    # Create the code length feature
    df["code_len"] = df["code_snippet"].apply(len)
    return df

# ---------------- Analysis & Visualisation ----------------
def analyze_and_compare(df: pd.DataFrame):
    """
    Generates all required descriptive statistics and comparative visualizations 
    for the dataset, saving all plots to the OUTPUT_DIR.
    """
    if df.empty:
        print("Cannot run analysis: DataFrame is empty.")
        return
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== Initial Data Overview ===")
    print(df.info())
    print("\n=== Data Head ===")
    print(df.head())
    
    # 1. Language counts
    if "language" in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(y=df["language"], order=df["language"].value_counts().index[:10], palette="viridis")
        plt.title("Top 10 Languages in Vulnerability Data")
        plt.xlabel("Count of Samples")
        plt.ylabel("Programming Language")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "language_counts.png"))
        plt.close()

    # 2. Broad CWE Category counts
    if "cwe_category" in df.columns:
        plt.figure(figsize=(10, 7))
        sns.countplot(y=df["cwe_category"], order=df["cwe_category"].value_counts().index, palette="rocket")
        plt.title("Distribution of Broad CWE Categories (Classification Target)")
        plt.xlabel("Count of Samples")
        plt.ylabel("CWE Category (Aggregated)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "cwe_category_distribution.png"))
        plt.close()

    # 3. Distribution of code snippet lengths
    plt.figure(figsize=(8,5))
    sns.histplot(df["code_len"], bins=50, kde=True, color="teal")
    plt.title("Distribution of Code Snippet Lengths")
    plt.xlabel("Code Length (Characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "code_length_distribution.png"))
    plt.close()
    
    # 4. Top 10 Specific CWE-ID counts
    if "cwe_id" in df.columns:
        plt.figure(figsize=(10, 7))
        top_10_cwe = df["cwe_id"].value_counts().nlargest(10).index
        sns.countplot(y=df["cwe_id"], order=top_10_cwe, palette="magma")
        plt.title("Top 10 Specific CWE-IDs")
        plt.xlabel("Count of Samples")
        plt.ylabel("CWE-ID (Specific Vulnerability)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "cwe_id_top10_distribution.png"))
        plt.close()

    # 5. Comparative Analysis: CWE Category vs. Programming Language (Heatmap)
    if 'cwe_category' in df.columns and 'language' in df.columns:
        # Normalize by index (language) to get the percentage composition within each language.
        crosstab_normalized = pd.crosstab(
            df['language'], 
            df['cwe_category'], 
            normalize='index'
        ) * 100 

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            crosstab_normalized, 
            annot=True,          
            fmt=".1f",           
            cmap="YlGnBu",       
            linewidths=.5,      
            cbar_kws={'label': 'Percentage of Vulnerabilities within Language (%)'}
        )
        
        plt.title("Comparative Analysis: Vulnerability Composition by Programming Language", fontsize=16)
        plt.ylabel("Programming Language", fontsize=12)
        plt.xlabel("CWE Category", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        output_path = os.path.join(OUTPUT_DIR, "cwe_language_composition_heatmap.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"\n[INFO] All analysis plots saved to the '{OUTPUT_DIR}' directory.")


# ---------------- Main ----------------
if __name__ == "__main__":
    df = load_jsonl(INPUT_FILE)
    if not df.empty:
        df = preprocess(df)
        analyze_and_compare(df)