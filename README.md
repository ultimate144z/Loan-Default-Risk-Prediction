# 🏦 Loan Default Predictor (Decision Tree)

This is a web-based ML application that predicts whether a customer will default on a loan using a trained **Decision Tree Classifier**. It supports **manual input** and **CSV file uploads** for batch predictions.

Built with:
- Python (Flask for backend)
- HTML + CSS (for frontend)
- scikit-learn (Decision Tree model)
- pandas, numpy, joblib (data handling)
- Jinja2 (template rendering)

---

## 📌 Project Overview

This app was built as a demonstration of deploying a machine learning model with a clean frontend interface. It does the following:

- Accepts **4 numerical features** for each customer:
  - `AMT_INCOME_TOTAL`
  - `AMT_CREDIT`
  - `AMT_ANNUITY`
  - `CNT_FAM_MEMBERS`
- Predicts whether the person will **default (1)** or **not (0)**.
- Supports:
  - **Option A**: Single prediction through manual form input
  - **Option B**: Uploading a **CSV file** with multiple entries
- Shows predicted **probability** and **default label**
- CSV results are saved and downloadable via link (if enabled)

---

## 🚀 Setup Instructions

Follow the steps below to run this project locally on your machine.

### 1. Clone the Project

```bash
git clone https://github.com/yourusername/loan-default-predictor.git
cd loan-default-predictor
```

Replace `yourusername` with your actual GitHub username if you plan to host it there.

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Project Structure

```
├── app.py                       # Flask backend logic
├── application_train_cleaned.csv (optional, for dev ref)
├── best_decision_tree.pkl      # Trained Decision Tree model
├── feature_cols.json           # List of selected features
├── feature_medians.json        # Medians for missing value imputation
├── optimal_threshold.npy       # Threshold for classification
├── templates/
│   └── index.html              # Frontend template
├── uploads/                    # Stores uploaded CSVs
├── results/                    # Stores CSV prediction results
├── demo_screenshots/           # Optional screenshots for preview
├── requirements.txt
└── README.md
```

### 5. How to Run the Project

```bash
python app.py
```

### 6. Testing the Prediction

**NOT DEFAULT Example:**
```
AMT_INCOME_TOTAL   600000
AMT_CREDIT         150000
AMT_ANNUITY         10000
CNT_FAM_MEMBERS          2
EXT_SOURCE_1        0.70
EXT_SOURCE_2        0.78
EXT_SOURCE_3        0.62
DAYS_BIRTH        -17500    # ~48 years old
DAYS_EMPLOYED      -5200    # ~14 years employed
CODE_GENDER_M           0   # female
```

**DEFAULT Example:**
```
AMT_INCOME_TOTAL   120000
AMT_CREDIT         600000
AMT_ANNUITY         35000
CNT_FAM_MEMBERS          5
EXT_SOURCE_1        0.05
EXT_SOURCE_2        0.09
EXT_SOURCE_3        0.07
DAYS_BIRTH         -9000    # ~24.7 years old
DAYS_EMPLOYED       -300    # < 1 year employed
CODE_GENDER_M           1   # male
```

---

## 🧠 Model Info

- **Algorithm**: DecisionTreeClassifier
- **Evaluation metric**: ROC AUC
- **Training logic**: Included in separate Jupyter Notebook (not part of this repo by default)

---

## 📝 Features

- **Real-time predictions** for loan default risk
- **Batch processing** via CSV upload
- **Clean, responsive UI** with modern styling
- **Downloadable results** for CSV predictions
- **Error handling** for invalid inputs and file formats

---

## 🛠️ Requirements

```txt
Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
```

---

## 📊 Usage

1. **Single Prediction**: Fill in the form with customer details and click "Predict"
2. **Batch Prediction**: Upload a CSV file with the required columns and get results for all entries
3. **View Results**: See probability scores and default/no-default classifications
4. **Download Results**: For CSV uploads, download the results file with predictions

---

## 🔧 Customization

- Modify `app.py` to add new features or change the model
- Update `templates/index.html` to customize the UI
- Replace `best_decision_tree.pkl` with your own trained model
- Adjust thresholds in `optimal_threshold.npy` for different classification criteria

---

## 📈 Future Enhancements

- Add more sophisticated models (Random Forest, XGBoost)
- Implement feature importance visualization
- Add data validation and preprocessing steps
- Include confidence intervals for predictions
- Add user authentication and prediction history