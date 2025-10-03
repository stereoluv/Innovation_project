# Machine Learning on Alternative Dataset for Classification
# Done by Nathan Rancie

# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# Read the parquet file
df = pd.read_parquet("alt_dataset/train.parquet")
df = pd.read_parquet("alt_dataset/test.parquet")

# Save to CSV due to assignment requirement
df.to_csv("alt_dataset/train.csv", index=False)   
df.to_csv("alt_dataset/test.csv", index=False)


# Load data
train_df = pd.read_parquet("alt_dataset/train.parquet")
test_df = pd.read_parquet("alt_dataset/test.parquet")

# Clean data
# Drop unnecessary columns
train_df = df.drop(columns=["unique_id", "__index_level_0__"], errors="ignore", inplace=True)
test_df = df.drop(columns=["unique_id", "__index_level_0__"], errors="ignore", inplace=True)

# Drop rows with missing 'code' values
train_df = df.dropna(subset=["code"])
test_df = df.dropna(subset=["code"])

# Remove duplicate code entries
train_df = df.drop_duplicates(subset=["code"])
test_df = df.drop_duplicates(subset=["code"])


# Basic data exploration
print("Head of training data:")
print(train_df.head())
print("Head of test data:")
print(test_df.head())


# Assign features and target variable for training set
X = train_df["code"]
y = train_df["target"]


# Text vectorization for increased performance
vectorizer = TfidfVectorizer(max_features=5000)  
X_tfidf = vectorizer.fit_transform(X)


# Split data into training and validation sets (80% train, 20% val), random state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)



# Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
# Fit the model
logreg.fit(X_train, y_train)
# Report for training set showing precision, recall, f1-score
print("Logistic Regression:\n", classification_report(y_val, logreg.predict(X_val)))



# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the model
knn.fit(X_train, y_train)
# Report for training set showing precision, recall, f1-score
print("KNN:\n", classification_report(y_val, knn.predict(X_val)))




# Assign features and target variable for test set
X_test = test_df["code"]
y_test = test_df["target"]

# Transform test set using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)


# Final evaluation of models using unseen data in test set
print("Test set evaluation:")
print(classification_report(y_test, logreg.predict(X_test_tfidf)))
print(classification_report(y_test, knn.predict(X_test_tfidf)))

