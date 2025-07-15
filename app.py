import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import shap

# Load trained model
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset structure for SHAP
X_sample = pd.read_csv("heart.csv").drop("target", axis=1)

# Initialize SHAP Explainer
explainer = shap.Explainer(model.predict, X_sample)

# --- SESSION STATE LOGIN SYSTEM ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Dummy login credentials
USER_CREDENTIALS = {
    "admin": "1234",
    "test": "pass"
}

def login():
    st.set_page_config(page_title="🫀 Heart Disease Predictor", layout="centered")
    st.markdown("### 🔐 Login to Heart Disease Predictor")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

if not st.session_state.authenticated:
    login()
    st.stop()

# --- MAIN APP ---
st.title("🫀 Heart Disease Prediction App")
st.markdown("<p style='font-size:18px'>Enter the patient details to predict the risk of heart disease</p>", unsafe_allow_html=True)

# Patient Form
with st.form("form"):
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
    submit = st.form_submit_button("🔍 Predict")

# Encoding Maps
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs_map = {"Yes": 1, "No": 0}
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

if submit:
    if not name:
        name = "Unknown"

    input_data = pd.DataFrame([[
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
    ]], columns=X_sample.columns)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("📋 Prediction Result")
    st.success("🫀 Positive for Heart Disease" if prediction == 1 else "💚 No Heart Disease Detected")

    st.markdown("#### 💡 Risk Level")
    st.progress(int(proba * 100))
    st.write(f"🔴 **Estimated Risk: {proba * 100:.2f}%**")

    # Save to CSV
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history = pd.DataFrame({
        "Name": [name], "Age": [age], "Sex": [sex], "Chest Pain": [cp], "BP": [trestbps], "Cholesterol": [chol],
        "Fasting Sugar": [fbs], "ECG": [restecg], "Max HR": [thalach], "Angina": [exang], "ST Depression": [oldpeak],
        "Slope": [slope], "Vessels": [ca], "Thalassemia": [thal],
        "Prediction": ["Yes" if prediction == 1 else "No"], "Risk %": [f"{proba*100:.2f}%"], "Timestamp": [timestamp]
    })

    if not os.path.exists("prediction_history.csv"):
        history.to_csv("prediction_history.csv", index=False)
    else:
        history.to_csv("prediction_history.csv", mode='a', header=False, index=False)

    st.download_button("📥 Download Report", history.to_csv(index=False), file_name="report.csv", mime="text/csv")

    # SHAP Explanation
    st.markdown("## 🧠 Feature Impact (SHAP)")
    st.info("This chart shows which features (like age, cholesterol, etc.) pushed the model towards 'heart disease' prediction or away from it.")
    shap_values = explainer(input_data)
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(plt.gcf())

st.markdown("## 📊 Prediction Summary")
if os.path.exists("prediction_history.csv"):
    df_chart = pd.read_csv("prediction_history.csv")
    
    # Count Yes/No predictions
    yes_count = (df_chart["Prediction"] == "Yes").sum()
    no_count = (df_chart["Prediction"] == "No").sum()
    total = yes_count + no_count

    if total > 0:
        yes_percent = (yes_count / total) * 100
        no_percent = (no_count / total) * 100

        st.write(f"🟥 **Heart Disease (Yes)**: {yes_percent:.1f}%")
        st.write(f"🟩 **No Heart Disease (No)**: {no_percent:.1f}%")
    
    # Plot bar chart
if os.path.exists("prediction_history.csv"):
    df_chart = pd.read_csv("prediction_history.csv")
    
    if not df_chart.empty:
        total = len(df_chart)
        positive = df_chart[df_chart['Prediction'] == 'Yes'].shape[0]
        negative = df_chart[df_chart['Prediction'] == 'No'].shape[0]
        
        pos_percent = (positive / total) * 100
        neg_percent = (negative / total) * 100

        st.markdown("## 📊 Overall Prediction Summary")
        fig, ax = plt.subplots()
        sns.countplot(x="Prediction", data=df_chart, palette="Set2", ax=ax)

        # 🔢 Add percentage labels on top
        for p in ax.patches:
            count = p.get_height()
            percent = (count / total) * 100
            ax.annotate(f'{percent:.1f}%', (p.get_x() + 0.2, count + 0.1))

        st.pyplot(fig)
    else:
        st.info("No prediction data available to display.")
else:
    st.info("No prediction data file found.")



# Sidebar History
st.sidebar.title("📜 Prediction History")
if os.path.exists("prediction_history.csv"):
    df_hist = pd.read_csv("prediction_history.csv")
    search = st.sidebar.text_input("🔎 Search by Name")
    if search:
        df_hist = df_hist[df_hist['Name'].str.contains(search, case=False)]
    st.sidebar.dataframe(df_hist.tail(10), height=300)
    if st.sidebar.button("🧹 Clear History"):
        os.remove("prediction_history.csv")
        st.sidebar.success("History cleared.")
else:
    st.sidebar.info("No history yet.")

