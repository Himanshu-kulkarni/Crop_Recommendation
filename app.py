from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv("crop_dataset.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get farmer inputs
    location = request.form["location"]
    season = request.form["season"]

    # Filter dataset based on input
    filtered = dataset[(dataset["location"].str.lower() == location.lower()) & (dataset["season"].str.lower() == season.lower())]

    if not filtered.empty:
        prediction = filtered.iloc[0]["recommended_crop"]
    else:
        prediction = "No data available for this location and season"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
