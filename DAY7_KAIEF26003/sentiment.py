import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_imdb_sample_dataset():
    # Synthetic representative dataset of positive and negative reviews
    reviews = [
        # Positive Reviews
        "This movie was absolutely amazing! Incredible acting and brilliant plot twists.",
        "A masterpiece of cinema. Beautiful cinematography and outstanding directing.",
        "I loved every single minute of this film! Highly recommended for everyone.",
        "Superb performance by the lead actors. One of the best movies of the decade.",
        "An astonishing journey with a great soundtrack and marvelous visuals.",
        "Truly inspiring and heartwarming. Exceeded all my expectations completely.",
        "Fantastic script and brilliant execution. I will definitely watch it again.",
        "A wonderful cinematic experience that keeps you hooked from start to finish.",

        # Negative Reviews
        "Terrible movie. Complete waste of time and money. Do not watch this.",
        "Boring plot, awful character development, and horrible acting throughout.",
        "I fell asleep halfway through. Predictable, slow, and completely dull.",
        "Extremely disappointing performance. The script makes absolutely no sense.",
        "One of the worst films I have ever seen. Frustrating and poorly directed.",
        "Bad direction, weak dialogue, and zero chemistry between the leads.",
        "Unbearably slow and meaningless. Save your time and skip this garbage.",
        "Flat characters and cheap special effects. A total disaster of a movie."
    ]
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 = Positive, 0 = Negative
    return reviews, labels


def inspect_predictive_words(vectorizer, model, top_n=5):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    # Top positive words have largest positive weights
    top_pos_idx = np.argsort(coefficients)[-top_n:][::-1]
    # Top negative words have largest negative weights
    top_neg_idx = np.argsort(coefficients)[:top_n]

    print("\n--- Feature Coefficient Analysis ---")
    print(f"Top {top_n} Positive Indicator Words:")
    for idx in top_pos_idx:
        print(f"  {feature_names[idx]:<15} (Weight: +{coefficients[idx]:.4f})")

    print(f"\nTop {top_n} Negative Indicator Words:")
    for idx in top_neg_idx:
        print(f"  {feature_names[idx]:<15} (Weight: {coefficients[idx]:.4f})")


def main():
    reviews, labels = load_imdb_sample_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.25, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    acc = accuracy_score(y_test, predictions)

    print("--- Sentiment Classifier Evaluation ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, predictions, target_names=["Negative (0)", "Positive (1)"], zero_division=0))

    inspect_predictive_words(vectorizer, model, top_n=5)


if __name__ == "__main__":
    main()