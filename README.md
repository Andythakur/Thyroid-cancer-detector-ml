# 🧬 Thyroid Cancer Recurrence Predictor

![Model](https://img.shields.io/badge/Model-RandomForest-blue) ![Python](https://img.shields.io/badge/Python-3.12-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

A machine learning model that predicts whether a **thyroid cancer survivor** may experience a **cancer recurrence** based on medical history, physical findings, and diagnostic results.

---

## 🎯 Objective

To build an AI-powered diagnostic assistant that uses clinical data to assess whether a patient treated for thyroid cancer is at risk of recurrence.

---

## 📁 Dataset Overview

The dataset includes features such as:

- Demographics: `Age`, `Gender`, `Smoking`, `Hx Smoking`
- Treatment history: `Hx Radiotherapy`
- Thyroid health: `Thyroid Function`, `Physical Examination`, `Adenopathy`
- Diagnosis: `Pathology`, `Focality`, `Risk`, `T`, `N`, `M`, `Stage`, `Response`
- 🎯 Target variable: `Recurred` (0 = No, 1 = Yes)

---

## 🔧 Tech Stack

- **Python 3.12**
- **Pandas** and **NumPy** for data manipulation
- **Seaborn** and **Matplotlib** for data visualization
- **scikit-learn** for machine learning

---

## ⚙️ How It Works

1. Load and clean the data
2. Encode categorical variables
3. Scale features
4. Train a `RandomForestClassifier`
5. Evaluate using classification metrics
6. Visualize feature importance

---

## 🚀 Results

```bash
Accuracy: 0.83
Confusion Matrix:
[[86  4]
 [11 59]]

Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.96      0.92        90
           1       0.94      0.84      0.89        70

    accuracy                           0.91       160
   macro avg       0.92      0.90      0.91       160
weighted avg       0.91      0.91      0.91       160
```

---

## 📊 Feature Importance (Top Predictors)

- `Pathology`
- `Risk`
- `Adenopathy`
- `Response`
- `Stage`

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/thyroid-cancer-recurrence-predictor.git
   cd thyroid-cancer-recurrence-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python script:
   ```bash
   python "Thyroid Cancer Detector.py"
   ```

---

## 📈 Future Improvements

- Hyperparameter optimization (GridSearchCV, RandomizedSearchCV)
- Deep learning model integration (e.g., Keras/TensorFlow)
- Deploy as a web app using Flask or Streamlit
- Real-time prediction from user inputs

---

## 📜 License

Licensed under the [MIT License](LICENSE)

---

## 🙌 Contribute

Pull requests are welcome! Please feel free to fork and enhance this project.
