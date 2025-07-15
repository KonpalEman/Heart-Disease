import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# Load model
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page Config
st.set_page_config("Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("#### Fill the form to check your risk of heart disease.")

# Input Form
with st.form("patient_form"):
    name = st.text_input("👤 Patient Name")
    age = st.number_input("🎂 Age", 1, 120, 30)
    sex = st.selectbox("⚥ Sex", ["Male", "Female"])
    cp = st.selectbox("💓 Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("🩺 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("🧪 Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("🧠 Resting ECG", ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("🏃‍♂️ Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("😰 Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("📉 ST depression", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("📈 Slope of peak exercise ST", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("🩻 Major Vessels Colored", ["0", "1", "2", "3"])
    thal = st.selectbox("🧬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    submitted = st.form_submit_button("🔍 Predict")

# Encoding
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs_map = {"Yes": 1, "No": 0}
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

if submitted:
    # Default name
    if not name:
        name = "Unknown"

    input_data = np.array([[
        age,
        sex_map[sex],
        cp_map[cp],
        trestbps,
        chol,
        fbs_map[fbs],
        restecg_map[restecg],
        thalach,
        exang_map[exang],
        oldpeak,
        slope_map[slope],
        int(ca),
        thal_map[thal]
    ]])

    prediction = model.predict(input_data)[0]
    risk_proba = model.predict_proba(input_data)[0][1]

    st.subheader("📋 Prediction Result")
    st.success(f"{'❤️ Positive for Heart Disease' if prediction == 1 else '💚 No Heart Disease Detected'}")

    # Risk Bar
    st.markdown("#### 💡 Risk Level")
    st.progress(int(risk_proba * 100))

    # Save full record to CSV
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Sex": [sex],
        "Chest Pain Type": [cp],
        "Resting BP": [trestbps],
        "Cholesterol": [chol],
        "Fasting Sugar": [fbs],
        "Rest ECG": [restecg],
        "Max HR": [thalach],
        "Exercise Angina": [exang],
        "ST Depression": [oldpeak],
        "Slope": [slope],
        "Major Vessels": [ca],
        "Thalassemia": [thal],
        "Prediction": ["Yes" if prediction == 1 else "No"],
        "Risk Level": ["High Risk" if prediction == 1 else "Low Risk"],
        "Risk %": [f"{float(risk_proba)*100:.2f}%" if risk_proba is not None else "0.00%"],
        "Timestamp": [timestamp]
    })

    if not os.path.exists("prediction_history.csv"):
        new_data.to_csv("prediction_history.csv", index=False)
    else:
        new_data.to_csv("prediction_history.csv", mode='a', index=False, header=False)

    st.download_button(
        label="📥 Download Report (CSV)",
        data=new_data.to_csv(index=False).encode('utf-8'),
        file_name=f"{name.replace(' ', '_')}_report.csv",
        mime='text/csv'
    )

# History Sidebar
st.sidebar.title("📜 Prediction History")
if os.path.exists("prediction_history.csv"):
    df_history = pd.read_csv("prediction_history.csv")
    search = st.sidebar.text_input("🔎 Search by Name")
    if search:
        df_history = df_history[df_history['Name'].str.contains(search, case=False, na=False)]

    st.sidebar.dataframe(df_history.tail(10), height=300)

    if st.sidebar.button("🧹 Clear All History"):
        os.remove("prediction_history.csv")
        st.sidebar.success("History Cleared")
else:
    st.sidebar.info("No history available yet.")


# Chart Section
st.markdown("## 📊 Risk Prediction Chart")
if os.path.exists("prediction_history.csv"):
    df_chart = pd.read_csv("prediction_history.csv")

    # Count values
    counts = df_chart['Prediction'].value_counts()
    total = counts.sum()

    fig, ax = plt.subplots()
    sns.countplot(x='Prediction', data=df_chart, palette='Set2', ax=ax)

    # Add percentage labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{(height/total)*100:.1f}%'
        ax.annotate(percentage,
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    st.pyplot(fig)
else:
    st.info("No prediction data available for chart.")
