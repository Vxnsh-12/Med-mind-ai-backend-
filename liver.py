import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

url = "indian_liver_patient.csv"
df = pd.read_csv(url)

df.columns = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio","Dataset"]
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})
df = df.dropna()
df.head()
df.info()
df.describe()
df.isnull().sum()

plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Dataset", data=df)

plt.title("Liver Disease Diagnosis Recognition")
plt.xlabel("Liver Disease(0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width()/2, height), ha='center', va='bottom')

plt.tight_layout()
# plt.show() # Commented out to avoid blocking execution

X = df.drop("Dataset", axis=1)
y = df["Dataset"]

X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size= 0.2, random_state= 42
)

# Create a pipeline with StandardScaler and LogisticRegression
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit the pipeline
model.fit(X_train, y_train)

# Predict using the pipeline (scaling is applied automatically)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, -1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

# Save the model to a pickle file
with open('liver_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved to liver_model.pkl")

# Verify loading
# with open('liver_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
#     print("Model loaded successfully")
