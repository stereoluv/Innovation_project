import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os # Import os for path manipulation

# --- CONFIG ---
INPUT_FILE = "basic_data_3.aggregated.jsonl"
OUTPUT_DIR = "analysis_output" # Directory to save plot images
# -----------------------------------

# ---------------- Load Data ----------------
def load_jsonl(path: str) -> pd.DataFrame:
    """
    Loads a JSONL file into a pandas DataFrame.
    """
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file not found at '{path}'. Please run vulnerability_processor.py first.")
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
        
    # Only check for columns that were NOT dropped in the processor script.
    for col in ["code_snippet", "exploitation_techniques"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
            
    # Example numeric features: code snippet length only, as description was dropped.
    df["code_len"] = df["code_snippet"].apply(len)
    return df

# ---------------- Analysis & Visualisation ----------------
def analyze(df: pd.DataFrame):
    """
    Generates basic descriptive statistics and visualizations for the dataset,
    saving all plots to the OUTPUT_DIR.
    """
    if df.empty:
        print("Cannot run analysis: DataFrame is empty.")
        return
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== Basic Info ===")
    print(df.info())
    print("\n=== Head ===")
    print(df.head())
    
    # 1. Language counts
    if "language" in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(y=df["language"], order=df["language"].value_counts().index[:10])
        plt.title("Top 10 Languages")
        plt.savefig(os.path.join(OUTPUT_DIR, "language_counts.png"))
        print(f"[INFO] Saved plot: {OUTPUT_DIR}/language_counts.png")
        plt.close()

    # 2. Broad CWE Category counts (using the aggregated field)
    if "cwe_category" in df.columns:
        plt.figure(figsize=(9, 6))
        sns.countplot(y=df["cwe_category"], order=df["cwe_category"].value_counts().index)
        plt.title("Distribution of Broad CWE Categories")
        plt.savefig(os.path.join(OUTPUT_DIR, "cwe_category_distribution.png"))
        print(f"[INFO] Saved plot: {OUTPUT_DIR}/cwe_category_distribution.png")
        plt.close()

    # 3. Distribution of code snippet lengths
    plt.figure(figsize=(8,5))
    sns.histplot(df["code_len"], bins=50, kde=True)
    plt.title("Code Snippet Lengths")
    plt.savefig(os.path.join(OUTPUT_DIR, "code_length_distribution.png"))
    print(f"[INFO] Saved plot: {OUTPUT_DIR}/code_length_distribution.png")
    plt.close()

# ---------------- Main ----------------
if __name__ == "__main__":
    # Load the new aggregated output file
    df = load_jsonl(INPUT_FILE)
    df = preprocess(df)
    analyze(df)
