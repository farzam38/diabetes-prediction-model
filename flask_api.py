from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the model and feature order
model, feature_order = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Encode input values
    def encode(value):
        return 1 if value == "Yes" or value == "Male" else 0

    input_data = [[
        encode(data.get("gender", "")),
        encode(data.get("polyuria", "")),
        encode(data.get("polydipsia", "")),
        encode(data.get("sudden_weight_loss", "")),
        encode(data.get("weakness", "")),
        encode(data.get("polyphagia", "")),
        encode(data.get("genital_thrush", "")),
        encode(data.get("visual_blurring", "")),
        encode(data.get("itching", "")),
        encode(data.get("irritability", "")),
        encode(data.get("delayed_healing", "")),
        encode(data.get("partial_paresis", "")),
        encode(data.get("muscle_stiffness", "")),
        encode(data.get("alopecia", "")),
        encode(data.get("obesity", "")),
        data.get("age", 0)
    ]]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_score = round(probability * 100, 2)

    if risk_score >= 80:
        risk_level = "High Risk"
    elif risk_score >= 50:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    # Health tips
    tips = []
    if encode(data.get("polyuria", "")):
        tips.append("💧 Frequent urination may indicate high blood sugar. Stay hydrated and consult a doctor.")
    if encode(data.get("polydipsia", "")):
        tips.append("🚰 Excessive thirst can be a sign of uncontrolled glucose levels. Monitor water intake and get a checkup.")
    if encode(data.get("sudden_weight_loss", "")):
        tips.append("⚠️ Sudden weight loss without trying is a red flag. Talk to a healthcare provider.")
    if encode(data.get("weakness", "")):
        tips.append("🪫 Feeling weak can result from low energy levels in cells due to insulin resistance.")
    if encode(data.get("visual_blurring", "")):
        tips.append("👓 Blurry vision may be due to glucose affecting your eye lenses. A full eye checkup is recommended.")
    if encode(data.get("delayed_healing", "")):
        tips.append("🩹 Slow wound healing is common with diabetes. Keep your skin clean and monitor wounds.")
    if encode(data.get("partial_paresis", "")):
        tips.append("🧠 Muscle weakness or partial paralysis may require neurological consultation.")
    if encode(data.get("obesity", "")):
        tips.append("⚖️ Obesity increases your risk significantly. Consider lifestyle changes: exercise, diet, and better sleep.")
    if encode(data.get("alopecia", "")):
        tips.append("💇 Hair loss can be a side effect of poor circulation due to diabetes. Manage stress and see a dermatologist.")
    if encode(data.get("itching", "")):
        tips.append("🧴 Itchy skin can occur due to dryness or infections. Use moisturizers and keep skin clean.")

    return jsonify({
        "result": result,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "tips": tips
    })

if __name__ == '__main__':
    app.run() 