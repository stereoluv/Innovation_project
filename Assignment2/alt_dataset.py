# ---------------- Imports ----------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

train_df = pd.read_parquet("alt_dataset_train.parquet")
test_df = pd.read_parquet("alt_dataset_test.parquet")

print(train_df.head())
print(test_df.head())

# Features and target
X = train_df.drop(["target", "unique_id", "__index_level_0__"], axis=1)
y = train_df["target"]



vectorizer = TfidfVectorizer(max_features=5000)  # adjust features if needed
X_tfidf = vectorizer.fit_transform(X["code"])

X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)




# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print("Logistic Regression:\n", classification_report(y_val, logreg.predict(X_val)))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN:\n", classification_report(y_val, knn.predict(X_val)))



X_test = test_df.drop(["target", "unique_id", "__index_level_0__"], axis=1)
y_test = test_df["target"]
X_test_tfidf = vectorizer.transform(X_test["code"])

print("Test set evaluation:")
print(classification_report(y_test, logreg.predict(X_test_tfidf)))
print(classification_report(y_test, knn.predict(X_test_tfidf)))