import pandas as pd
import joblib
import re

from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize


def extract_features(review):
    words = word_tokenize(review.lower())

    features = {}

    for word in words:
        if re.match(r"^[a-zA-Z]+$", word):
            features[f"word={word}"] = True

    return features


# Load dataset
data = pd.read_csv("dataset/reviews.csv")

print("Dataset loaded successfully!")
print("Total reviews:", len(data))


# Convert reviews into features
features = []

for _, row in data.iterrows():
    review_features = extract_features(row["review"])
    features.append((review_features, row["label"]))


# Train Naive Bayes model
model = NaiveBayesClassifier.train(features)

print("Model trained successfully!")


# Save model
joblib.dump(model, "model/fake_review_model.pkl")

print("Model saved successfully!")