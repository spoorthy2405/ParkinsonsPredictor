import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Disease Prediction System")

# Load and train model
@st.cache_data
def load_model():
    df = pd.read_csv("Parkinsson disease.csv")
    X = df.drop(columns=["name", "status"])
    y = df["status"]
    model = SVC(probability=True)  # Enable probability predictions
    model.fit(X, y)
    accuracy = round(accuracy_score(y, model.predict(X)) * 100, 2)
    return model, X.columns.tolist(), accuracy

model, feature_columns, accuracy = load_model()

# Feature descriptions
feature_descriptions = {
    "MDVP:Fo(Hz)": "Average vocal pitch frequency",
    "MDVP:Fhi(Hz)": "Maximum vocal frequency",
    "MDVP:Flo(Hz)": "Minimum vocal frequency",
    "MDVP:Jitter(%)": "Pitch variation (percentage)",
    "MDVP:Jitter(Abs)": "Pitch variation (absolute)",
    "MDVP:RAP": "Relative pitch variation",
    "MDVP:PPQ": "Period perturbation quotient",
    "Jitter:DDP": "Three-point pitch difference",
    "MDVP:Shimmer": "Amplitude variation",
    "MDVP:Shimmer(dB)": "Amplitude variation (dB)",
    "Shimmer:APQ3": "3-point amplitude perturbation",
    "Shimmer:APQ5": "5-point amplitude perturbation",
    "MDVP:APQ": "Average amplitude perturbation",
    "Shimmer:DDA": "Average absolute amplitude difference",
    "NHR": "Noise-to-Harmonics Ratio",
    "HNR": "Harmonics-to-Noise Ratio",
    "RPDE": "Signal complexity",
    "DFA": "Signal fractal scaling",
    "spread1": "Voice signal variation (1)",
    "spread2": "Voice signal variation (2)",
    "D2": "Signal dimensionality",
    "PPE": "Pitch variation entropy"
}

# Sample input values
sample_values_positive = {
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP": 0.0037,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.0313,
    "MDVP:APQ": 0.02971,
    "Shimmer:DDA": 0.06545,
    "NHR": 0.02211,
    "HNR": 21.033,
    "RPDE": 0.414783,
    "DFA": 0.815285,
    "spread1": -4.813031,
    "spread2": 0.266482,
    "D2": 2.301442,
    "PPE": 0.284654
}

sample_values_negative = {
    "MDVP:Fo(Hz)": 237.226,
    "MDVP:Fhi(Hz)": 247.326,
    "MDVP:Flo(Hz)": 225.227,
    "MDVP:Jitter(%)": 0.00298,
    "MDVP:Jitter(Abs)": 0.00001,
    "MDVP:RAP": 0.00169,
    "MDVP:PPQ": 0.00182,
    "Jitter:DDP": 0.00507,
    "MDVP:Shimmer": 0.01752,
    "MDVP:Shimmer(dB)": 0.164,
    "Shimmer:APQ3": 0.01035,
    "Shimmer:APQ5": 0.01024,
    "MDVP:APQ": 0.01133,
    "Shimmer:DDA": 0.03104,
    "NHR": 0.0074,
    "HNR": 22.736,
    "RPDE": 0.305062,
    "DFA": 0.654172,
    "spread1": -7.31055,
    "spread2": 0.098648,
    "D2": 2.416838,
    "PPE": 0.095032
}

# Mode selection
mode = st.radio("Choose Prediction Mode", ["📦 Batch Prediction (CSV)", "👤 Single Patient Prediction"])

# -----------------------
# Batch Prediction Mode
# -----------------------
if mode == "📦 Batch Prediction (CSV)":
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if all(col in df.columns for col in feature_columns):
                st.success("✅ File format verified.")

                df["prediction"] = model.predict(df[feature_columns])
                df["prediction_label"] = df["prediction"].apply(
                    lambda x: "Parkinson's Detected" if x == 1 else "No Parkinson's"
                )

                st.dataframe(df.head())
                output_filename = "Parkinson_Batch_Prediction.csv"
                df.to_csv(output_filename, index=False)

                with open(output_filename, "rb") as f:
                    st.download_button("📥 Download Predictions CSV", f, file_name=output_filename, mime="text/csv")
            else:
                st.error("❌ Missing required feature columns in the uploaded CSV file.")
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------
# Single Prediction Mode
# -----------------------
else:
    st.subheader("🧍‍♂️ Enter Patient Details")

    name = st.text_input("Patient Name", "John Doe")
    age = st.text_input("Age", "60")
    gender = st.selectbox("Gender", ["Male", "Female"])

    # Session state for auto-fill values
    if "filled_values" not in st.session_state:
        st.session_state.filled_values = {col: "0.0" for col in feature_columns}

    st.markdown("#### 🎲 For Demo:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 Auto-Fill: Parkinson's Positive"):
            st.session_state.filled_values = {k: str(v) for k, v in sample_values_positive.items()}
    with col2:
        if st.button("💪 Auto-Fill: No Parkinson's"):
            st.session_state.filled_values = {k: str(v) for k, v in sample_values_negative.items()}

    st.markdown("### 🔬 Input Voice Feature Values:")
    input_data = []
    for col in feature_columns:
        label = feature_descriptions.get(col, "Unknown feature")
        value_str = st.text_input(f"{col} ({label})", value=st.session_state.filled_values.get(col, "0.0"), key=col)
        try:
            value = float(value_str)
        except:
            st.warning(f"⚠️ Invalid input for {col}. Using 0.0.")
            value = 0.0
        input_data.append(value)

    if st.button("🔍 Predict"):
        try:
            instance = pd.DataFrame([input_data], columns=feature_columns)
            prediction = model.predict(instance)[0]
            probs = model.predict_proba(instance)[0]

            diagnosis = "Parkinson's Detected. Please consult a neurologist." if prediction == 1 else "No Parkinson's disease detected."
            st.success(f"🧠 Diagnosis: {diagnosis}")
            st.info(f"📊 Confidence - Parkinson: {round(probs[1]*100, 2)}% | No Parkinson: {round(probs[0]*100, 2)}%")
            st.write(f"🎯 Model Accuracy: **{accuracy}%**")

            # PDF Report
            class PDF(FPDF):
                def header(self):
                    self.set_font("Arial", "B", 14)
                    self.cell(0, 10, "Parkinson's Disease Prediction Report", ln=True, align="C")
                    self.ln(10)

                def footer(self):
                    self.set_y(-15)
                    self.set_font("Arial", "I", 8)
                    self.cell(0, 10, f'Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'C')

            pdf = PDF()
            pdf.add_page()
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Name: {name}", ln=True)
            pdf.cell(0, 10, f"Age: {age}", ln=True)
            pdf.cell(0, 10, f"Gender: {gender}", ln=True)
            pdf.cell(0, 10, f"Diagnosis: {diagnosis}", ln=True)
            pdf.cell(0, 10, f"Model Accuracy: {accuracy}%", ln=True)
            pdf.cell(0, 10, f"Confidence - Parkinson: {round(probs[1]*100, 2)}%, No Parkinson: {round(probs[0]*100, 2)}%", ln=True)
            pdf.ln(10)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Input Features:", ln=True)
            pdf.set_font("Arial", "", 10)
            for i, col in enumerate(feature_columns):
                label = feature_descriptions.get(col, col)
                pdf.cell(0, 8, f"{col} ({label}): {input_data[i]}", ln=True)

            pdf_path = f"{name.replace(' ', '_')}_Report.pdf"
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download PDF Report", f, file_name=pdf_path, mime="application/pdf")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
