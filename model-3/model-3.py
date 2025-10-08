import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Swapping back to Ridge Regression for fine-tuning with regularization
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt

# --- Configuration ---
CLEANED_DATA_FILE = "model-3/data/cleaned_cve_data.csv"
MODEL_FILENAME = "cvss_regression_model_fine_tuned.joblib" 

def load_data(file_path):
    """
    Loads the cleaned dataset and checks for required columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned data file not found at: {file_path}. Please run 'preprocessor.py' first.")
    
    try:
        df = pd.read_csv(file_path)
        required_cols = ['CVSS-V2', 'DESCRIPTION', 'CWE-ID']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in data. Found: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading or reading data: {e}")
        return None

def train_regression_model(df: pd.DataFrame):
    """
    Trains a fine-tuned Ridge regression model to predict the CVSS-V2 score.
    
    Args:
        df: The cleaned Pandas DataFrame.
    """
    X = df[['DESCRIPTION', 'CWE-ID']]
    y = df['CVSS-V2']

    print(f"\nTraining data size: {len(X)} samples.")
    print(f"CVSS-V2 target mean: {y.mean():.2f}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. Feature Engineering: Fine-Tuned Preprocessor Pipeline
    
    # Define the preprocessing steps for different feature types
    preprocessor = ColumnTransformer(
        transformers=[
            # FINE-TUNING STEP 1: TF-IDF with increased features and n-grams (1 and 2 words)
            ('text_tfidf', 
             TfidfVectorizer(
                 stop_words='english', 
                 max_features=7500, # Increased max features from 5000 to 7500
                 ngram_range=(1, 2) # Added 1-gram and 2-gram features
             ), 
             'DESCRIPTION'),
             
            # Apply One-Hot Encoding to the CWE-ID (categorical feature)
            ('cwe_ohe', 
             OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             ['CWE-ID'])
        ],
        remainder='drop' # Drop any other columns
    )
    
    # 2. Model Pipeline: Preprocessor + Estimator
    
    # FINE-TUNING STEP 2: Use Ridge Regression (regularization) to handle the increased features
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0, random_state=42)) # Using L2 regularization (alpha=1.0)
    ])
    
    print("\nStarting fine-tuned Ridge model training...")
    
    # Train the pipeline
    model_pipeline.fit(X_train, y_train)
    
    print("Training complete. Evaluating model performance...")
    
    # 3. Model Evaluation
    
    # Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)
    
    # Clip predictions to the valid CVSS range (0.0 to 10.0)
    y_pred = np.clip(y_pred, 0.0, 10.0) 
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Fine-Tuned Ridge Regression Performance (Test Set) ---")
    print(f"Mean Absolute Error (MAE): {mae:.3f} (Lower is better)")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"R-squared (R2) Score: {r2:.3f} (Closer to 1.0 is better)")
    
    # 4. Visualization of Results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='darkgreen', alpha=0.5, label='Predicted Scores (Fine-Tuned)')
    
    # Plot the ideal line (where Actual = Predicted)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')
    
    plt.xlabel('Actual CVSS-V2 Score')
    plt.ylabel('Predicted CVSS-V2 Score')
    plt.title('Fine-Tuned Ridge Regression: Actual vs. Predicted CVSS-V2 Scores')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

    # 5. Save the trained model
    joblib.dump(model_pipeline, MODEL_FILENAME)
    print(f"\n✅ Fine-Tuned Ridge Regression Model saved as '{MODEL_FILENAME}'")
    
    # Note: The entire pipeline includes the preprocessor, so saving the pipeline is sufficient.

if __name__ == "__main__":
    
    try:
        # Load the data processed by preprocessor.py
        data = load_data(CLEANED_DATA_FILE)
        
        if data is not None and not data.empty:
            train_regression_model(data)
        elif data is None:
             print("Data loading failed. Please check the error message above.")
        else:
             print("The loaded dataset is empty after initial processing.")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
