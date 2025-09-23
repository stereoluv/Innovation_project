import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ---------------- Load Data ----------------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

# ---------------- Preprocess ----------------
def preprocess(df):
    # Fill missing text fields
    for col in ["description", "code_snippet", "exploitation_techniques", "mitigation"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    # Example numeric features
    df["desc_len"] = df["description"].apply(len)
    df["code_len"] = df["code_snippet"].apply(len)
    return df

# ---------------- Analysis & Visualisation ----------------
def analyze(df):
    print("\n=== Basic Info ===")
    print(df.info())
    print("\n=== Head ===")
    print(df.head())
    
    # Language counts
    if "language" in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(y=df["language"], order=df["language"].value_counts().index[:10])
        plt.title("Top 10 Languages")
        plt.show()

    # Vulnerability type counts
    if "vulnerability_type" in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(y=df["vulnerability_type"], order=df["vulnerability_type"].value_counts().index[:10])
        plt.title("Top 10 Vulnerability Types")
        plt.show()

    # Distribution of description/code lengths
    plt.figure(figsize=(8,5))
    sns.histplot(df["desc_len"], bins=50, kde=True)
    plt.title("Description Lengths")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.histplot(df["code_len"], bins=50, kde=True)
    plt.title("Code Snippet Lengths")
    plt.show()

# ---------------- Main ----------------
if __name__ == "__main__":
    df = load_jsonl("basic_data_3.cleaned.jsonl")
    df = preprocess(df)
    analyze(df)

