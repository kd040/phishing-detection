import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Better model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Print accuracy in terminal
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Custom prediction with lower threshold
def predict_email(email_text):
    cleaned = clean_text(email_text)
    vector = vectorizer.transform([cleaned])
    probability = model.predict_proba(vector)[0][1]
    boost_words = ["win", "congratulations", "click", "prize", "lottery"]

    for word in boost_words:
        if word in cleaned:
            probability += 0.15
    # LOWER THRESHOLD (Important)
    if probability > 0.35:
        prediction = 1
    else:
        prediction = 0

    return prediction, probability

# Suspicious words for highlighting
suspicious_words = [
    "win", "free", "urgent", "click", "offer",
    "password", "bank", "lottery", "prize", "claim",
    "congratulations"
]

def highlight_words(text):
    words = text.split()
    highlighted = []

    for word in words:
        clean_word = word.lower().strip(".,!?$")
        if clean_word in suspicious_words:
            highlighted.append(f"<mark style='background-color:yellow'>{word}</mark>")
        else:
            highlighted.append(word)

    return " ".join(highlighted)