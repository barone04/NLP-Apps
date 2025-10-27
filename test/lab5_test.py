import sys
import os
import pprint
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.models.text_classifier import TextClassifier

def main():
    """
    Main function to test the TextClassifier.
    """
    print("=== TextClassifier Evaluation ===")

    # 1. Define the dataset
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

    print("\nOriginal Data:")
    for i, text in enumerate(texts):
        print(f"  '{text}' -> {labels[i]}")

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print("\nTraining Data:")
    for i, text in enumerate(X_train):
        print(f"  '{text}' -> {y_train[i]}")
    print("\nTest Data:")
    for i, text in enumerate(X_test):
        print(f"  '{text}' -> {y_test[i]}")

    # 3. Instantiate tokenizer and vectorizer
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer=tokenizer)

    # 4. Instantiate and train the classifier
    classifier = TextClassifier(vectorizer=vectorizer)
    classifier.fit(X_train, y_train)
    print("\nClassifier trained.")

    # 5. Make predictions on the test set
    y_pred = classifier.predict(X_test)
    print("\nPredictions on Test Data:")
    for i, text in enumerate(X_test):
        print(f"  '{text}' -> Predicted: {y_pred[i]}, Actual: {y_test[i]}")

    # 6. Evaluate the predictions
    metrics = classifier.evaluate(y_test, y_pred)
    print("\nEvaluation Metrics:")
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()