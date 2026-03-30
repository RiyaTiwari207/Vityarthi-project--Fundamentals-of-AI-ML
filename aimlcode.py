# ==============================
# FAKE REVIEW DETECTION SYSTEM
# ==============================

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------------
# STEP 1: SAMPLE DATA (auto-created)
# ------------------------------
data_dict = {
    "review": [
        "This product is amazing and works perfectly",
        "Very bad quality waste of money",
        "Best product ever!!!",
        "Good",
        "Excellent quality highly recommend",
        "Worst product do not buy",
        "Loved it very useful",
        "Fake product totally useless",
        "Awesome awesome awesome!!!",
        "Not worth it"
    ],
    "label": [0,0,1,1,0,0,0,0,1,0]  # 1 = Fake, 0 = Real
}

data = pd.DataFrame(data_dict)

# ------------------------------
# STEP 2: TEXT PREPROCESSING
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

data["review"] = data["review"].apply(clean_text)

# ------------------------------
# STEP 3: TRAIN MODEL
# ------------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['review'])
y = data['label']

model = LogisticRegression()
model.fit(X, y)

# ------------------------------
# STEP 4: RULE-BASED CHECK
# ------------------------------
def rule_check(review, rating):
    words = review.split()

    if len(words) < 5:
        return True

    if rating == 5 and ("best" in review.lower() or "amazing" in review.lower()):
        return True

    if review.count("!") > 2:
        return True

    return False

# ------------------------------
# STEP 5: PREDICTION FUNCTION
# ------------------------------
def predict_review(review, rating):
    clean = clean_text(review)
    X_input = tfidf.transform([clean])

    ml_pred = model.predict(X_input)[0]
    rule_pred = rule_check(review, rating)

    final = "FAKE ❌" if ml_pred == 1 or rule_pred else "REAL ✅"

    return final

# ------------------------------
# STEP 6: USER INPUT
# ------------------------------
print("=== AI Fake Review Detector ===")

review = input("Enter product review: ")
rating = int(input("Enter rating (1-5): "))

result = predict_review(review, rating)

print("\n🔍 Result:", result)