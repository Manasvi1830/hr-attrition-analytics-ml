from flask import Flask, render_template, request
import joblib
import pandas as pd
from preprocess import preprocess_input
from recommendation import generate_recommendations

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Age": int(request.form["Age"]),
            "MonthlyIncome": int(request.form["MonthlyIncome"]),
            "YearsAtCompany": int(request.form["YearsAtCompany"]),
            "YearsSinceLastPromotion": int(request.form["YearsSinceLastPromotion"]),
            "Overtime": int(request.form["Overtime"])
        }

        processed = preprocess_input(data)
        scaled = scaler.transform(processed)

        probability = model.predict_proba(scaled)[0][1]
        risk_score = round(probability * 100, 2)

        if risk_score < 30:
            category = "Low Risk"
        elif risk_score < 60:
            category = "Medium Risk"
        else:
            category = "High Risk"

        recommendations = generate_recommendations(data, risk_score)

        return render_template("result.html",
                               risk_score=risk_score,
                               category=category,
                               recommendations=recommendations)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
