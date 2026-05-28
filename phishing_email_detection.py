# phishing_email_detection.py
# Python 3.13 Compatible
# Install required libraries before running:
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------
# STEP 1: Sample Dataset
# ---------------------------------------------------

data = {
    "email": [
        "Congratulations! You won a free iPhone. Click here now!",
        "Your bank account has been suspended. Verify immediately.",
        "Meeting scheduled tomorrow at 10 AM.",
        "Project submission deadline is Friday.",
        "Urgent! Update your password using this link.",
        "Lunch at 1 PM?",
        "Claim your reward now by visiting this website.",
        "Please review the attached project document."
    ],
    "label": [
        "Phishing",
        "Phishing",
        "Safe",
        "Safe",
        "Phishing",
        "Safe",
        "Phishing",
        "Safe"
    ]
}

df = pd.DataFrame(data)

# ---------------------------------------------------
# STEP 2: Feature Cleaning Function
# ---------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df["email"] = df["email"].apply(clean_text)

# ---------------------------------------------------
# STEP 3: Split Dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["email"],
    df["label"],
    test_size=0.25,
    random_state=42
)

# ---------------------------------------------------
# STEP 4: Create ML Pipeline
# ---------------------------------------------------

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# ---------------------------------------------------
# STEP 5: Train Model
# ---------------------------------------------------

model.fit(X_train, y_train)

# ---------------------------------------------------
# STEP 6: Predictions
# ---------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------
# STEP 7: Accuracy & Report
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# STEP 8: Confusion Matrix
# ---------------------------------------------------

cm = confusion_matrix(y_test, y_pred, labels=["Phishing", "Safe"])

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Safe"],
    yticklabels=["Phishing", "Safe"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------------------------------
# STEP 9: Test Custom Email
# ---------------------------------------------------

while True:
    user_email = input("\nEnter an email message (or type 'exit'): ")

    if user_email.lower() == "exit":
        print("Program ended.")
        break

    cleaned = clean_text(user_email)

    prediction = model.predict([cleaned])[0]

    print("Prediction:", prediction)