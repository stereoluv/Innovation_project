import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
    df = load_data(INPUT_PATH)
    if df.empty:
        exit()
        
    X_full = df['code_snippet'] 
    Y_full = df['cwe_category'] 

    label_encoder = LabelEncoder()
    Y_encoded_full = label_encoder.fit_transform(Y_full)
    target_names = label_encoder.classes_.tolist() 

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, Y_encoded_full, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=Y_encoded_full
    )
    
    print("\n" + "="*80)
    print("--- 4. HYPERPARAMETER TUNING (Grid Search for Logistic Regression) ---")
    print("="*80)
    
    # Define the Pipeline for Grid Search (CountVectorizer -> TF-IDF -> LR)
    lr_pipeline = Pipeline([
        ('vect', CountVectorizer(token_pattern=r'\w{1,}')), 
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(random_state=RANDOM_SEED, solver='liblinear', max_iter=5000)), 
    ])

    # Define the parameter grid to search over
    param_grid = {
        'vect__ngram_range': [(1, 1), (1, 2)], 
        'clf__C': [0.1, 1, 10, 100], 
    }

    # Setup the Grid Search Cross-Validator
    grid_search = GridSearchCV(
        lr_pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1, 
        verbose=1 
    )

    # Run the Grid Search on the full dataset (X_full, Y_encoded_full)
    grid_search.fit(X_full, Y_encoded_full)

    print("\n--- Grid Search Results ---")
    print(f"Best Macro F1 Score (Cross-Validation): {grid_search.best_score_:.4f}")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)

 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\nLogistic Regression (Tuned) Final Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report (Detailed per-class metrics):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
