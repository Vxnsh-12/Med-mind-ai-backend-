import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
url = "enhanced_fever_medicine_recommendation (1).csv"
df = pd.read_csv(url)
df.columns = ["Temperature","Fever_Severity","Age","Gender","BMI","Headache","Body_Ache","Fatigue","Chronic_Conditions","Allergies","Smoking_History","Alcohol_Consumption","Humidity","AQI","Physical_Activity","Diet_Type","Heart_Rate","Blood_Pressure","Previous_Medication","Recommended_Medication"]

print(df.head)
print(df.info)

X = df.drop("Recommended_Medication", axis=1)
y = df["Recommended_Medication"]

from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

