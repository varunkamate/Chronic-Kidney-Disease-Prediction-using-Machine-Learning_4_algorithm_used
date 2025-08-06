# Chronic-Kidney-Disease-Prediction-using-Machine-Learning_4_algorithm_used

🧠 Chronic Kidney Disease Prediction using Machine Learning
A machine learning-based solution to predict the presence of Chronic Kidney Disease (CKD) using a structured dataset. This project utilizes 4 different classification algorithms to evaluate model performance and identify the best predictive model.

📂 kidney-disease-prediction/
│
├── kidney_disease_dataset.csv           # Cleaned CKD dataset
├── kidney_disease_with_4algoritms.ipynb # Jupyter Notebook with full ML pipeline
├── model.pkl                            # Best-performing saved model (Pickle format)
└── README.md                            # Project documentation

📊 Dataset Overview
File: kidney_disease_dataset.csv

Rows: 400+

Target Variable: classification (ckd / notckd)

Features Include:

Demographics: Age, Gender

Clinical: Blood pressure, Blood glucose, Serum creatinine, Hemoglobin, Albumin, etc.

Categorical indicators: Hypertension, Diabetes, Anemia, Appetite

🧪 Machine Learning Pipeline
📘 Notebook: kidney_disease_with_4algoritms.ipynb

✅ Steps:
Exploratory Data Analysis (EDA):

Missing value handling

Outlier detection

Feature correlation

Visualizations

Preprocessing:

Encoding categorical features

Normalization/Scaling

Train-test split

Model Training (with evaluation):

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Model Evaluation Metrics:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

ROC Curve (where applicable)

Best Model: Saved as model.pkl for deployment or further use.

💡 Future Improvements
Hyperparameter Tuning (GridSearchCV)

Feature Engineering (Domain-specific)

Deployment using Flask or Streamlit

Add cross-validation for robustness

Address potential overfitting in high-performing models

🛠 Requirements
Python 3.7+

pandas, numpy, matplotlib, seaborn

scikit-learn

joblib or pickle (for saving models)

<img width="1510" height="683" alt="Screenshot 2025-08-05 233916" src="https://github.com/user-attachments/assets/e04a9e08-e584-45d5-876f-10fdef08b521" />

<img width="1161" height="676" alt="Screenshot 2025-08-05 233942" src="https://github.com/user-attachments/assets/2aecfa0b-eddf-4f61-a95a-435291e74e5a" />

<img width="444" height="242" alt="Screenshot 2025-08-05 234014" src="https://github.com/user-attachments/assets/6c885703-cf61-4d2a-b7a1-616f524a833b" />

