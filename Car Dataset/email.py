"""Spam / not-spam email checker.

"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("emails.csv").dropna(subset=["text", "label"])
df["label"] = df["label"].str.strip().str.lower()
print(df["label"].value_counts())

X_tr, X_te, y_tr, y_te = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
    ("nb", MultinomialNB()),
])
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print("Accuracy:", round(accuracy_score(y_te, pred), 3))
print(classification_report(y_te, pred))


new_emails = [
    "Congratulations! You won a free prize, click here to claim",
    "Hi, can we move tomorrow's meeting to 3pm?",
]
for mail, label in zip(new_emails, model.predict(new_emails)):
    print(f"{label.upper():9} <- {mail}")