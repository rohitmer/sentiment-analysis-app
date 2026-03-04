from flask import Flask, render_template, request
import joblib
import re
import os
import nltk

# ✅ Create Flask app FIRST
app = Flask(__name__)

# ✅ Tell NLTK where local data is
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

from nltk.corpus import stopwords

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ===============================
# STOPWORDS SETUP
# ===============================

stop_words = set(stopwords.words("english"))
stop_words.discard("not")  # keep negation

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# ===============================
# ROUTE
# ===============================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            review = request.form.get("review")

            if not review or review.strip() == "":
                error = "Please enter a review before predicting."
            else:
                cleaned = clean_text(review)
                vector = vectorizer.transform([cleaned])

                result = model.predict(vector)[0]
                probabilities = model.predict_proba(vector)[0]

                confidence = round(max(probabilities) * 100, 2)

                prediction = "Positive 😊" if result == 1 else "Negative 😡"

        except Exception:
            error = "Something went wrong. Please try again."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )

# ===============================
# RUN APP
# ===============================

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)

