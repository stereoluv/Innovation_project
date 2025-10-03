import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from typing import List

# --- CONFIG ---
INPUT_PATH = "basic_data_3.aggregated.jsonl"
RANDOM_SEED = 42
TEST_SIZE = 0.2
# -----------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a JSONL file."""
    try:
        return pd.read_json(filepath, lines=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{filepath}'")
        return pd.DataFrame()

if __name__ == "__main__":
    # 1. Load data
    df = load_data(INPUT_PATH)
    if df.empty:
        exit()
        
    # Define features (X) and labels (Y)
    X = df['code_snippet'] 
    Y = df['cwe_category'] 

    # 2. Encode the target labels (Y)
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    target_names = label_encoder.classes_.tolist() # Store original class names
    
    # 3. Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y_encoded, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=Y_encoded # Maintain class balance
    )
    
    # 4. Define the best-performing pipeline (TF-IDF + LR with optimized settings)
    # Using C=10 (low regularization) and ngram_range=(1, 2) (unigrams and bigrams)
    log_reg_pipeline = Pipeline([
        # Step 1: Feature Extraction (CountVectorizer)
        ('vect', CountVectorizer(token_pattern=r'\w{1,}', ngram_range=(1, 2))), 
        # Step 2: Feature Weighting (TF-IDF)
        ('tfidf', TfidfTransformer()),
        # Step 3: Classifier (Logistic Regression with high C for low regularization)
        ('clf', LogisticRegression(random_state=RANDOM_SEED, solver='liblinear', max_iter=5000, C=10)), 
    ])

    # 5. Train the model
    print("--- Training Logistic Regression (TF-IDF Features) ---")
    log_reg_pipeline.fit(X_train, y_train)

    # 6. Predict on the test data
    y_pred = log_reg_pipeline.predict(X_test)

    # 7. Evaluate the model
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\nLogistic Regression (TF-IDF) Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report (Detailed per-class metrics):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
