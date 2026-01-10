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
url = "gastrointestinal_disease_dataset.csv"

df= pd.read_csv(url)
df.columns = ["Age","Gender","BMI","Body_Weight","Obesity_Status","Ethnicity","Family_History","Genetic_Markers","Microbiome_Index","Autoimmune_Disorders","H_Pylori_Status","Fecal_Calprotectin","Occult_Blood_Test","CRP_ESR","Endoscopy_Result","Colonoscopy_Result","Stool_Culture","Diet_Type","Food_Intolerance","Smoking_Status","Alcohol_Use","Stress_Level","Physical_Activity","Abdominal_Pain","Bloating","Diarrhea","Constipation","Rectal_Bleeding","Appetite_Loss","Weight_Loss","Bowel_Habits","Bowel_Movement_Frequency","NSAID_Use","Antibiotic_Use","PPI_Use","Medications","Disease_Class"]

df.head()
df.info()
df.describe()
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
categorical_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop("Disease_Class", axis=1)
y = df["Disease_Class"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,-1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled), multi_class="ovr", average="weighted")
print("ROC-AUC Score:", roc_auc)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("GERD Confusion Matrix")
plt.show()


