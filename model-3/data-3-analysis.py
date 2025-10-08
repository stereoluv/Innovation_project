import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# --- CONFIG ---
# The file path has been updated to reflect common project structure in a modular way.
INPUT_FILE = "model-3/data/cleaned_cve_data.csv"
OUTPUT_DIR = "analysis3_output"                                                                            
# -----------------------------------

# ---------------- Load Data ----------------
def load_data(path: str) -> pd.DataFrame:
    """
    Loads the CSV file into a pandas DataFrame.
    """
    try:
        # Pandas generally handles CSV data from CVEs well
        df = pd.read_csv(path)
        required_cols = ['CVSS-V2', 'SEVERITY', 'DESCRIPTION', 'CWE-ID']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in data. Ensure file contains: {required_cols}")
        return df
    except FileNotFoundError:
        # Adjusted print statement to show the full path from the configuration
        print(f"Error: Input file not found at '{path}'. Please ensure your cleaned data is at this location.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# ---------------- Preprocess ----------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning and creates new analytical features.
    
    Adjustment: Simplified CWE_Cleaned feature generation since 
    'NVD-CWE-Other' and 'NVD-noinfo' rows have been dropped.
    """
    if df.empty:
        return df
        
    # Convert CVSS-V2 to numeric, coercing errors (e.g., 'None' or NaNs)
    df['CVSS-V2'] = pd.to_numeric(df['CVSS-V2'], errors='coerce')
    df = df.dropna(subset=['CVSS-V2'])
    
    # Feature 1: Description Length (important for text analysis)
    df["description_len"] = df["DESCRIPTION"].apply(lambda x: len(str(x)))
    
    # Feature 2: Cleaned CWE-ID (extract just the number for better grouping)
    # The previous filtering ensures 'CWE-ID' should contain only CWE-### format or NaN
    df['CWE_Cleaned'] = df['CWE-ID'].apply(
        # We can directly extract the number
        lambda x: re.sub(r'^CWE-', '', str(x)) if str(x).startswith('CWE-') else 'Unknown'
    )
    
    return df

# ---------------- Analysis & Visualisation ----------------
def analyze(df: pd.DataFrame):
    """
    Generates key visualizations for the CVE dataset, saving all plots to the OUTPUT_DIR.
    No changes are needed here, as the plots remain relevant for the filtered data.
    """
    if df.empty:
        print("Cannot run analysis: DataFrame is empty.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n--- Running CVSS Data Analysis ---")

    # 1. Distribution of the Target Variable (CVSS-V2)
    plt.figure(figsize=(10, 6))
    sns.histplot(df["CVSS-V2"], bins=20, kde=True, color='teal')
    plt.title("Distribution of CVSS-V2 Scores")
    plt.xlabel("CVSS-V2 Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "cvss_v2_distribution.png"))
    print(f"[INFO] Saved plot: {OUTPUT_DIR}/cvss_v2_distribution.png")
    plt.close()

    # 2. Distribution of SEVERITY Categories
    plt.figure(figsize=(8, 5))
    # Using 'order' based on value counts ensures the bars are sorted by frequency
    sns.countplot(y=df["SEVERITY"], order=df["SEVERITY"].value_counts().index, palette='viridis')
    plt.title("Distribution of Severity Levels")
    plt.xlabel("Count")
    plt.ylabel("Severity")
    plt.savefig(os.path.join(OUTPUT_DIR, "severity_counts.png"))
    print(f"[INFO] Saved plot: {OUTPUT_DIR}/severity_counts.png")
    plt.close()
    
    # 3. Top 10 CWE Categories (CWE-ID)
    top_cwe = df['CWE_Cleaned'].value_counts().nlargest(10)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=top_cwe.values, y=top_cwe.index, palette='plasma')
    plt.title("Top 10 CWE Identifiers (Cleaned)")
    plt.xlabel("Count")
    plt.ylabel("CWE ID")
    plt.savefig(os.path.join(OUTPUT_DIR, "top_cwe_counts.png"))
    print(f"[INFO] Saved plot: {OUTPUT_DIR}/top_cwe_counts.png")
    plt.close()

    # 4. Relationship between Description Length and CVSS Score
    plt.figure(figsize=(10, 6))
    # Using a scatter plot with alpha for density visualization
    plt.scatter(df["description_len"], df["CVSS-V2"], alpha=0.1, color='darkred')
    plt.title("CVSS-V2 Score vs. Vulnerability Description Length")
    plt.xlabel("Description Length (Characters)")
    plt.ylabel("CVSS-V2 Score")
    plt.savefig(os.path.join(OUTPUT_DIR, "cvss_vs_description_length.png"))
    print(f"[INFO] Saved plot: {OUTPUT_DIR}/cvss_vs_description_length.png")
    plt.close()


# ---------------- Main ----------------
if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    df = preprocess(df)
    analyze(df)