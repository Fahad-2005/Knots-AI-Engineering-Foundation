from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def get_dataset():
    # Sample dataset of spam and ham (legitimate) messages
    texts = [
        # Spam
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "URGENT! You have won a 1 week FREE membership in our $100,000 prize jackpot!",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free!",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt CS to 87575 to cost 150p/day.",
        "You are awarded a free trip to Hawaii. Call 0800293812 now to claim your ticket.",
        "Congratulations! Claim your guaranteed $1000 cash card now by texting WIN to 55555.",
        "Get cheap loans instantly with low interest rates. No credit check required!",
        
        # Ham
        "Hey, are we still meeting up for dinner tonight at 7?",
        "I'll be home late tonight, don't wait up for me.",
        "Can you send me the project report whenever you get a chance?",
        "Great job on the presentation today, everyone was really impressed.",
        "Let's grab a coffee tomorrow morning before our first class starts.",
        "Don't forget to buy milk and bread on your way back home.",
        "What time does the movie start? I need to check train schedules.",
        "Hey Mom, just checking in to see if you received the package I sent."
    ]
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 = Spam, 0 = Ham
    return texts, labels


def train_and_evaluate():
    texts, labels = get_dataset()

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    # Transform raw text into TF-IDF numerical feature vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Classifier (SVM)": SVC(kernel="linear")
    }

    print("--- Spam Classification Performance Metrics ---")

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, predictions)
        print(f"\nModel: {name}")
        print(f"Overall Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, predictions, target_names=["Ham (0)", "Spam (1)"], zero_division=0))


if __name__ == "__main__":
    train_and_evaluate()