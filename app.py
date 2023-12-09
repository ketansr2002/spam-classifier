from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
clf = pickle.load(open("models/clf.pkl", "rb"))
cv = pickle.load(open("models/cv.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    if request.method == "POST":
        text = request.form.get("content")

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get("content")
    tokenized_mail = cv.transform([email])
    prediction = clf.predict(tokenized_mail)
    prediction = 1 if prediction == 1 else -1

    return render_template("index.html", prediction=prediction, email=email)


if __name__ == "__main__":
    app.run(debug=True)
