# Machine Learning on Alternative Dataset for Classification (IMPROVED)
# Done by Nathan Rancie

# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Enhanced imports for better models and imbalance handling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report


# --- DATA LOADING AND CLEANING (CORRECTED LOGIC) ---
# NOTE: Using pd.read_csv to avoid parquet file dependency if possible.
# Assuming the user has saved 'train.csv' and 'test.csv' as per the original code.
try:
    # Load data from CSVs
    train_df = pd.read_csv("model-1/data/train.csv")
    test_df = pd.read_csv("model-1/data/test.csv")
except FileNotFoundError:
    print("WARNING: CSV files not found. Attempting to load Parquet files as specified in original script.")
    train_df = pd.read_parquet("model-1/data/train.parquet")
    test_df = pd.read_parquet("model-1/data/test.parquet")


# Clean data
COLUMNS_TO_DROP = ["unique_id", "__index_level_0__"]

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning steps to a single DataFrame."""
    # 1. Drop unnecessary columns
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    # 2. Drop rows with missing 'code' values
    df = df.dropna(subset=["code"])
    # 3. Remove duplicate code entries
    df = df.drop_duplicates(subset=["code"])
    return df

train_df = clean_df(train_df)
test_df = clean_df(test_df)


# Basic data exploration
print("Head of training data (Cleaned):")
print(train_df.head())
print("Head of test data (Cleaned):")
print(test_df.head())


# Assign features and target variable for training set
X = train_df["code"]
y = train_df["target"]


# Text vectorization for increased performance
vectorizer = TfidfVectorizer(max_features=5000) 
X_tfidf = vectorizer.fit_transform(X)


# Split data into training and validation sets (80% train, 20% val)
X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)


# --- MODEL 1: LOGISTIC REGRESSION (WITH IMBALANCE HANDLING) ---
# FIX: class_weight='balanced' is added to improve Recall for the minority class (1).
logreg_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg_balanced.fit(X_train, y_train)

print("\n--- Logistic Regression (Balanced) Validation Report ---")
print(classification_report(y_val, logreg_balanced.predict(X_val)))


# --- MODEL 2: RANDOM FOREST CLASSIFIER (MORE ROBUST ALGORITHM) ---
# FIX: Replaced KNN with Random Forest Classifier (an ensemble model).
# RFC is generally superior for high-dimensional sparse data and handles imbalance better.
rfc = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') 
# Note: class_weight='balanced' is also added here for best results.
rfc.fit(X_train, y_train)

print("\n--- Random Forest Classifier (Balanced) Validation Report ---")
print(classification_report(y_val, rfc.predict(X_val)))


# --- FINAL EVALUATION ON UNSEEN TEST SET ---
X_test = test_df["code"]
y_test = test_df["target"]

# Transform test set using the SAME fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test)

print("\n\n--- FINAL TEST SET EVALUATION ---")
print("1. Logistic Regression (Balanced):")
print(classification_report(y_test, logreg_balanced.predict(X_test_tfidf)))

print("\n2. Random Forest Classifier (Balanced):")
print(classification_report(y_test, rfc.predict(X_test_tfidf)))