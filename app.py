import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import re

# Load the model
model, feature_order = joblib.load('diabetes_model.pkl')

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient information below to assess your diabetes risk:")

# Input fields
age = st.slider("Age", 1, 120, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
polyuria = st.selectbox("Polyuria (Frequent urination)", ["Yes", "No"])
polydipsia = st.selectbox("Polydipsia (Excessive thirst)", ["Yes", "No"])
sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["Yes", "No"])
weakness = st.selectbox("Weakness", ["Yes", "No"])
polyphagia = st.selectbox("Polyphagia (Excessive hunger)", ["Yes", "No"])
genital_thrush = st.selectbox("Genital Thrush", ["Yes", "No"])
visual_blurring = st.selectbox("Visual Blurring", ["Yes", "No"])
itching = st.selectbox("Itching", ["Yes", "No"])
irritability = st.selectbox("Irritability", ["Yes", "No"])
delayed_healing = st.selectbox("Delayed Healing", ["Yes", "No"])
partial_paresis = st.selectbox("Partial Paresis", ["Yes", "No"])
muscle_stiffness = st.selectbox("Muscle Stiffness", ["Yes", "No"])
alopecia = st.selectbox("Alopecia (Hair loss)", ["Yes", "No"])
obesity = st.selectbox("Obesity", ["Yes", "No"])

# Encode input values
def encode(value):
    return 1 if value == "Yes" or value == "Male" else 0

input_data = [[
    encode(gender), encode(polyuria), encode(polydipsia), encode(sudden_weight_loss),
    encode(weakness), encode(polyphagia), encode(genital_thrush), encode(visual_blurring),
    encode(itching), encode(irritability), encode(delayed_healing), encode(partial_paresis),
    encode(muscle_stiffness), encode(alopecia), encode(obesity), age
]]

for key in ["result", "risk_score", "risk_level", "tips"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "tips" else []

patient_name = st.text_input("Patient Name", placeholder="Enter your full name")

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if patient_name:
    st.session_state.patient_name = patient_name

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_score = round(probability * 100, 2)

    if risk_score >= 80:
        risk_level = "🔴 High Risk"
    elif risk_score >= 50:
        risk_level = "🟠 Medium Risk"
    else:
        risk_level = "🟢 Low Risk"

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
        
    st.session_state.result = result
    st.session_state.risk_score = risk_score
    st.session_state.risk_level = risk_level

    st.subheader("🧾 Result Summary")
    st.success(f"Prediction: **{result}**")
    st.info(f"Risk Score: **{risk_score}%**")
    st.warning(f"Risk Level: **{risk_level}**")

feature_order = [
    "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
    "weakness", "Polyphagia", "Genital thrush", "visual blurring",
    "Itching", "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity", "Age"
]

input_df = pd.DataFrame(input_data, columns=feature_order)

# SHAP Explanation - Individual Waterfall Plot
st.subheader("📊 Why this prediction?")
st.markdown("This SHAP waterfall plot explains how each feature contributed to your diabetes risk score:")

# Create SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

# Use SHAP values for class 1 (diabetic)
try:
    shap_value_individual = shap_values[0, :, 1]  # Index: [instance, features, class]
except:
    st.warning("Using fallback class (likely binary).")
    shap_value_individual = shap_values[0]  # Try fallback if slicing fails

# Plot using SHAP's built-in waterfall
fig = plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_value_individual, max_display=16, show=False)
st.pyplot(fig)

st.subheader("💡 Personalized Health Tips")

tips = []

if encode(polyuria):
    tips.append("💧 Frequent urination may indicate high blood sugar. Stay hydrated and consult a doctor.")
if encode(polydipsia):
    tips.append("🚰 Excessive thirst can be a sign of uncontrolled glucose levels. Monitor water intake and get a checkup.")
if encode(sudden_weight_loss):
    tips.append("⚠️ Sudden weight loss without trying is a red flag. Talk to a healthcare provider.")
if encode(weakness):
    tips.append("🪫 Feeling weak can result from low energy levels in cells due to insulin resistance.")
if encode(visual_blurring):
    tips.append("👓 Blurry vision may be due to glucose affecting your eye lenses. A full eye checkup is recommended.")
if encode(delayed_healing):
    tips.append("🩹 Slow wound healing is common with diabetes. Keep your skin clean and monitor wounds.")
if encode(partial_paresis):
    tips.append("🧠 Muscle weakness or partial paralysis may require neurological consultation.")
if encode(obesity):
    tips.append("⚖️ Obesity increases your risk significantly. Consider lifestyle changes: exercise, diet, and better sleep.")
if encode(alopecia):
    tips.append("💇 Hair loss can be a side effect of poor circulation due to diabetes. Manage stress and see a dermatologist.")
if encode(itching):
    tips.append("🧴 Itchy skin can occur due to dryness or infections. Use moisturizers and keep skin clean.")

st.session_state.tips = tips

if tips:
    for tip in tips:
        st.info(tip)
else:
    st.success("✅ Great! Your symptoms don't show major early warning signs. But regular screening is still wise.")

    
def generate_pdf(patient_name, result, risk_score, risk_level, tips):
    def remove_emojis(text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    clean_result = remove_emojis(result)
    clean_risk_level = remove_emojis(risk_level)
    clean_tips = [remove_emojis(tip) for tip in tips]
    clean_name = remove_emojis(patient_name)

    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # Title
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt="Diabetes Risk Report", ln=True, align='C')
    pdf.ln(10)

    # Patient Info
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient Name: {clean_name}", ln=True)
    pdf.cell(0, 10, txt=f"Prediction: {clean_result}", ln=True)
    pdf.cell(0, 10, txt=f"Risk Score: {risk_score}%", ln=True)
    pdf.cell(0, 10, txt=f"Risk Level: {clean_risk_level}", ln=True)
    pdf.ln(10)

    # Health Tips Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, txt="Personalized Health Tips:", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", size=12)
    for tip in clean_tips:
        pdf.multi_cell(0, 10, tip.strip(), align='J')
        pdf.ln(1)

    # Save and encode
    pdf_output = "report.pdf"
    pdf.output(pdf_output)

    with open(pdf_output, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    return base64_pdf

if st.session_state.result is not None and st.session_state.patient_name:
    if st.button("📄 Generate PDF Report"):
        base64_pdf = generate_pdf(
            st.session_state.patient_name,
            st.session_state.result,
            st.session_state.risk_score,
            st.session_state.risk_level,
            st.session_state.tips
        )
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="diabetes_report.pdf">📥 Click here to download your report</a>'
        st.markdown(href, unsafe_allow_html=True)

