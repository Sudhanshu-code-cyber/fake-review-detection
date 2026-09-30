from ml.predictor import predict_review


review = input("Enter a review: ")

prediction, confidence = predict_review(review)

print("Prediction:", prediction)
print("Confidence:", round(confidence, 2), "%")