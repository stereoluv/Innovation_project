import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier # Using the Random Forest model
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
    df = load_data(INPUT_PATH)
    if df.empty:
        exit()
        
  
    X = df['code_snippet'] 
    Y = df['cwe_category'] 


    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    target_names = label_encoder.classes_.tolist() # Store original class names
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y_encoded, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=Y_encoded # Maintain class balance
    )
    

    rf_pipeline = Pipeline([
        
        ('vect', CountVectorizer(token_pattern=r'\w{1,}', ngram_range=(1, 2))), 
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)), 
    ])

    print("--- Training Random Forest Classifier (TF-IDF Features) ---")
    rf_pipeline.fit(X_train, y_train)

    y_pred = rf_pipeline.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\nRandom Forest Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report (Detailed per-class metrics):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
