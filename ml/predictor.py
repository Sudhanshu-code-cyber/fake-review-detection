import joblib
import re
import nltk

from nltk.tokenize import word_tokenize

from config import MODEL_PATH


# Download tokenizer data
nltk.download("punkt_tab")


# Load trained model
model = joblib.load(MODEL_PATH)


def extract_features(review):
    words = word_tokenize(review.lower())

    features = {}

    for word in words:
        if re.match(r"^[a-zA-Z]+$", word):
            features[f"word={word}"] = True

    return features


def predict_review(review):

    features = extract_features(review)

    prediction = model.classify(features)

    probabilities = model.prob_classify(features)

    confidence = probabilities.prob(prediction) * 100

    return prediction, confidence