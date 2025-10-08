import pandas as pd
import io
import os
import re

# Define the expected input file name
INPUT_FILE = "model-3/data/Global_Dataset.csv"


def preprocess_cve_data(csv_content: str) -> pd.DataFrame:
    """
    Cleans and prepares the raw CVE data for both CVSS regression and 
    severity classification modeling, including filtering specific CWE-ID values.
    
    Args:
        csv_content: A string containing the raw CSV data read from the file.
        
    Returns:
        A cleaned Pandas DataFrame.
    """
    # 1. Read the raw string data into a DataFrame
    # Use io.StringIO to treat the string as a file-like object
    df = pd.read_csv(io.StringIO(csv_content))

    print(f"Initial shape: {df.shape}")
    print(f"Initial columns: {list(df.columns)}")

    # 2. Drop unnecessary columns as requested
    COLUMNS_TO_DROP = ['CVE-ID', 'CVSS-V3']
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    # 3. Handle 'None' string values in target and feature columns
    # We replace 'None' string placeholders with actual NaN values
    df = df.replace('None', pd.NA)

    # 4. Convert the CVSS-V2 target column to numeric
    # 'CVSS-V2' is the target for regression, so it must be a float
    df['CVSS-V2'] = pd.to_numeric(df['CVSS-V2'], errors='coerce')
    
    # ------------------ STEP 5 (CWE Filtering) ------------------
    # Filters out NVD-CWE-Other and NVD-noinfo CWE-ID entries as requested.
    CWE_FILTER_VALUES = ['NVD-CWE-Other', 'NVD-noinfo', 'NVD-CWE-noinfo']
    
    # Store initial row count for reporting
    initial_rows_cwe = len(df)
    
    # Filter: keep rows where 'CWE-ID' is NOT in the filter list (case-insensitive).
    df['CWE-ID'] = df['CWE-ID'].astype(str)
    df = df[~df['CWE-ID'].str.lower().isin([v.lower() for v in CWE_FILTER_VALUES])]
    
    rows_dropped_cwe = initial_rows_cwe - len(df)
    # ------------------------------------------------
    

    # 6. Drop rows where the target variable (CVSS-V2) is missing (NaN)
    # This is crucial for supervised learning models (regression)
    initial_rows_cvss = len(df)
    df = df.dropna(subset=['CVSS-V2'])
    rows_dropped_cvss = initial_rows_cvss - len(df)
    
    # 7. Basic text cleaning for the 'DESCRIPTION' feature
    # Ensure all descriptions are strings and strip surrounding whitespace
    df['DESCRIPTION'] = df['DESCRIPTION'].astype(str).str.strip()

    # 8. Reset index after dropping rows
    df = df.reset_index(drop=True)

    print(f"\nPreprocessing complete:")
    print(f"Rows dropped due to NVD-CWE-Other/NVD-noinfo: {rows_dropped_cwe}")
    print(f"Rows dropped due to missing CVSS-V2: {rows_dropped_cvss}")
    print(f"Final shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")
    print(f"CVSS-V2 Dtype: {df['CVSS-V2'].dtype}")
    
    return df

if __name__ == "__main__":
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: The input file '{INPUT_FILE}' was not found.")
        print("Please ensure 'Global_Dataset.csv' is in the same directory as this script.")
    else:
        try:
            # 1. Read the content from the actual file
            print(f"--- Attempting to read data from '{INPUT_FILE}' ---")
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                raw_data_content = f.read()
            
            # 2. Process the data
            cleaned_df = preprocess_cve_data(raw_data_content)
            
            # 3. Save the cleaned dataset
            OUTPUT_FILE = "model-3/data/cleaned_cve_data.csv"
            cleaned_df.to_csv(OUTPUT_FILE, index=False)
            
            print(f"\nCleaned data successfully processed and saved to '{OUTPUT_FILE}'.")
            
        except Exception as e:
            print(f"An error occurred during file reading or processing: {e}")