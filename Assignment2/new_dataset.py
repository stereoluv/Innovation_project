# ---------------- Imports ----------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_parquet("hf://datasets/realvul/LineVul_Test_Dataset/data/test-00000-of-00001.parquet")

# ---------------- Analysis & Visualisation ----------------
def analyze(df):
    print("\n=== Basic Info ===")
    print(df.info())
    print("\n=== Head ===")
    print(df.head())
    
    

# ---------------- Main ----------------
if __name__ == "__main__":
    analyze(df)
