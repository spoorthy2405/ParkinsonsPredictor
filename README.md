🧠 Parkinson's Disease Prediction System

An intelligent web-based application built using **Streamlit** and **Scikit-learn** to predict the likelihood of Parkinson's Disease based on biomedical voice measurements. This tool enables both **batch processing** via CSV and **individual patient evaluation** with detailed PDF reports.

---

🚀 Features

🎯 Trained SVM model for high-accuracy predictions  
📁 Batch Prediction Mode: Upload CSV and get instant results  
👤 Single Patient Mode: Enter data manually and get a downloadable PDF report  
📉 Confidence levels and model accuracy display  
📄 Auto-generated PDF diagnosis report  
🧪 Demo data auto-filled for testing  

---

📊 Tech Stack
**Frontend & UI:** Streamlit  
**Machine Learning Model:** Support Vector Machine (SVM) using Scikit-learn  
**Data Handling:** Pandas  
**PDF Report Generation:** FPDF  
**Dataset:** Parkinson’s Disease dataset (UCI Machine Learning Repository)  

---

📝 Sample Use Cases
- Can be used by researchers and doctors for early-stage Parkinson’s screening  
- Useful in academic demos and healthcare ML projects  
- Shows how AI can support medical decision making  

---

📂 Folder Structure

📦 Parkinsons-Prediction-App/
├── app.py                    # Main Streamlit application
├── Parkinsson disease.csv   # Dataset used for model training
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

---
📦 Dependencies
Below are the Python packages used in this project:
streamlit
pandas
scikit-learn
fpdf

You can install them all with:
```bash
pip install -r requirements.txt
```
---

⚙️ Setup Instructions
1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/parkinsons-prediction-app.git
cd parkinsons-prediction-app
```
2️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```
3️⃣ Run the Streamlit App
```bash
python -m streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`
---

🧪 Sample Inputs
*🧠 **Auto-Fill Parkinson’s Positive**: Populates form with voice features from a diagnosed case
*💪 **Auto-Fill No Parkinson’s**: Populates form with healthy voice data
These buttons are available in the "Single Patient Prediction" mode for quick testing
---
📄 Output PDF Report
Each diagnosis generates a downloadable PDF report containing:
* Patient details (Name, Age, Gender)
* Prediction result with confidence percentages
* Model accuracy
* Full breakdown of all 22 input features

---

✅ Model Details
**Algorithm Used:** Support Vector Machine (SVM)
**Accuracy Achieved:** \~88%
**Prediction Type:** Binary Classification (Parkinson's vs No Parkinson's)
**Input Features:** 22 voice-related biomedical attributes

---
🧠 Voice Features Used
Some example features:

* `MDVP:Fo(Hz)` – Average vocal pitch
* `MDVP:Jitter(%)` – Pitch variation percentage
* `HNR` – Harmonics-to-Noise Ratio
* `RPDE` – Signal complexity
* `DFA` – Fractal scaling exponent
* `PPE` – Pitch period entropy

> Each feature includes a tooltip in the app to help users understand the medical terminology.
