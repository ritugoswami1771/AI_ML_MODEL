from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

label_map = {0: "Not Spam 😊", 1: "Spam 🚨"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sms_text = request.form["sms"]
    sms_vec = vectorizer.transform([sms_text])
    prediction = model.predict(sms_vec)[0]
    result = label_map[prediction]

    return render_template("index.html", prediction=result, sms_text=sms_text)

if __name__ == "__main__":
    app.run(debug=True)
