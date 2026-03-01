from flask import Flask, render_template, request
from model import predict_email, highlight_words

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    highlighted = None

    if request.method == "POST":
        email_text = request.form["email"]
        prediction, probability = predict_email(email_text)
        highlighted = highlight_words(email_text)

    if prediction == 1:
        result = "Phishing Detected"
    else:
        result = "Safe Email"

    return render_template("index.html",
                       result=result,
                       probability=probability,
                       highlighted=highlighted,
                       risk_percentage=round(probability * 100, 2) if probability else 0)

 if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)