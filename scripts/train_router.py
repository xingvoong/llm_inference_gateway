"""
Train a logistic regression classifier on (prompt, model) pairs.

Input:  data/training_data.json
Output: data/router_model.pkl

The classifier learns which prompts should route to which model.
At inference time, it replaces the zero-shot classifier from Phase 3.

Why logistic regression over DistilBERT fine-tuning:
- Trains in seconds on any hardware
- No GPU required
- Accurate enough for routing decisions
- DistilBERT is the upgrade path when you have more data and better hardware
"""

import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train():
    # Load training data
    with open("data/training_data.json") as f:
        data = json.load(f)

    prompts = [d["prompt"] for d in data]
    labels = [d["model"] for d in data]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        prompts, labels, test_size=0.2, random_state=42
    )

    # TF-IDF converts text to numbers, logistic regression classifies
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("=== Classifier Performance ===")
    print(classification_report(y_test, y_pred))

    # Save
    with open("data/router_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved → data/router_model.pkl")


if __name__ == "__main__":
    train()
