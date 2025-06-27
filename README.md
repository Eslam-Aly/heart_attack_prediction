# 🫀 Heart Attack Prediction with Machine Learning

This repository contains a machine learning pipeline to predict heart attack risk using Logistic Regression and Random Forest models. It includes data preprocessing, model training, evaluation, and model saving using `joblib`.

---

## 📁 Repository Contents

<code>
├── Heart_Attack_Logistic_Regression.ipynb   # Logistic Regression workflow
├── heart_attack_random_forest.ipynb         # Random Forest model workflow
├── heart_attack_model.pkl                   # Saved Logistic Regression model
├── scaler.pkl                               # StandardScaler used for input scaling
├── requirements.txt                         # Python package dependencies
├── .gitignore                               # Files to ignore in version control
└── README.md                                # Project documentation (this file)
</code>

---

## 📊 Project Summary

- **Goal**: Predict the probability of a person having a heart attack.
- **Dataset**: Includes features like Age, Sex, Blood Pressure, Cholesterol, Exercise, Diet, Previous Heart Problems, etc.
- **Models Used**:
  - Logistic Regression (with GridSearchCV + class balancing)
  - Random Forest (with SMOTE for handling imbalance)

---

## ⚙️ Steps Performed

1. **Data Cleaning**
   - Handled missing values
   - Dropped irrelevant features
   - Encoded categorical variables

2. **Feature Scaling**
   - Standardized using `StandardScaler`

3. **Class Balancing**
   - Applied **SMOTE** to address imbalance in heart attack risk classes

4. **Model Training**
   - Grid Search used for Logistic Regression tuning
   - Random Forest trained with default and tuned hyperparameters

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix and classification report

---

## 🧪 Requirements

<code>
Install dependencies using:

bash
pip install -r requirements.txt
Main packages:
	•	pandas, numpy
	•	scikit-learn
	•	seaborn, matplotlib
	•	imbalanced-learn (for SMOTE)
 </code>

💾 Using the Trained Model

To use the trained model and scaler for predictions:

<code>
import joblib

model = joblib.load('heart_attack_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example new input
new_data = [[45, 1, 220, 0, 1, 1, 0]]  # replace with real values

# Scale input
scaled_input = scaler.transform(new_data)

# Predict
prediction = model.predict(scaled_input)
print("Prediction:", prediction)
</code>
