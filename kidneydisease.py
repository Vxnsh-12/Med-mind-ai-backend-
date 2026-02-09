import pandas as pd
import os
import pickle
from sklearn.pipeline import Pipeline
df = pd.read_csv('kidney_disease.csv')

df = df.drop(columns=['id'])

import seaborn as sns
sns.heatmap(df.isnull())
df.info()
df.isnull().sum()
df['bu'] = df['bu'].fillna(df['bu'].mean())
df['sc'] = df['sc'].fillna(df['sc'].mean())
df['age'] = df['age'].fillna(df['age'].mean())

import matplotlib.pyplot as plt
sns.histplot(df['age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

bins = [0, 20, 40, 65, 95]
labels = ['0-20', '20-40', '40-65', '65-95']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
print("Age groups created successfully.")

mean_bp_by_age_group = df.groupby('age_group')['bp'].mean()
print("Mean 'bp' values calculated for each age group:")
print(mean_bp_by_age_group)

for group, mean_bp in mean_bp_by_age_group.items():
    df.loc[(df['age_group'] == group) & (df['bp'].isnull()), 'bp'] = mean_bp
print("Missing 'bp' values filled using mean 'bp' for each age group.")

df = df.drop(columns=['age_group'])

df['ane'] = df['ane'].fillna(df['ane'].mode()[0])
df['pe'] = df['pe'].fillna(df['pe'].mode()[0])
df['appet'] = df['appet'].fillna(df['appet'].mode()[0])
df['cad'] = df['cad'].fillna(df['cad'].mode()[0])
df['dm'] = df['dm'].fillna(df['dm'].mode()[0])
df['htn'] = df['htn'].fillna(df['htn'].mode()[0])
df['pcc'] = df['pcc'].fillna(df['pcc'].mode()[0])
df['ba'] = df['ba'].fillna(df['ba'].mode()[0])

df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

correlation_matrix = df.select_dtypes(include=['number']).corr()

plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

df['wc'] = df['wc'].fillna(df['wc'].mean())
df['hemo'] = df['hemo'].fillna(df['hemo'].mean())

print("Missing values after cleaning and imputation:")
print(df[['wc', 'hemo']].isnull().sum())

from sklearn.linear_model import LinearRegression

df_rc_notnull = df[df['rc'].notnull()]
df_rc_null = df[df['rc'].isnull()]

df['rc'] = pd.to_numeric(df['rc'], errors='coerce')
df['hemo'] = pd.to_numeric(df['hemo'], errors='coerce')

if not df_rc_notnull.empty and not df_rc_null.empty:
    X_train = df_rc_notnull[['hemo']]
    y_train = df_rc_notnull['rc']

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_predict = df_rc_null[['hemo']]
    df.loc[df['rc'].isnull(), 'rc'] = model.predict(X_predict)

print("Missing 'rc' values filled using linear regression with 'hemo'.")
print("Remaining missing values in 'rc':", df['rc'].isnull().sum())

df['sg'] = df['sg'].fillna(df['sg'].mean())
df['al'] = df['al'].fillna(df['al'].mean())
df['sod'] = df['sod'].fillna(df['sod'].mean())
df['pot'] = df['pot'].fillna(df['pot'].mean())
df['bgr'] = df['bgr'].fillna(df['bgr'].mean())

df['pc'] = df['pc'].fillna(df['pc'].mode()[0])

df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['pcv'] = df['pcv'].fillna(df['pcv'].median())

print("Missing values after final imputation:")
print(df.isnull().sum())

categorical_cols_to_encode = ['htn', 'dm', 'cad', 'appet', 'pe', 'ane' , 'pcc' , 'ba', 'classification']

print("Unique values before encoding:")
for col in categorical_cols_to_encode:
    print(f"'{col}': {df[col].unique()}")

dm_mapping = {'\tno':0, ' yes': 1, '\tyes': 1 , 'yes': 1 , 'no':0}
df['dm'] = df['dm'].map(dm_mapping)

cad_mapping = {'\tno':0 , 'yes': 1 , 'no':0}
df['cad'] = df['cad'].map(cad_mapping)

appet_mapping = {'good': 1 , 'poor':0}
df['appet'] = df['appet'].map(appet_mapping)

pe_mapping = {'yes': 1 , 'no':0}
df['pe'] = df['pe'].map(pe_mapping)

ane_mapping = {'yes': 1 , 'no':0}
df['ane'] = df['ane'].map(ane_mapping)

htn_mapping = {'yes': 1 , 'no':0}
df['htn'] = df['htn'].map(htn_mapping)

pc_mapping = {'normal': 0, 'abnormal': 1}
df['pc'] = df['pc'].map(pc_mapping)

pcc_mapping = {'notpresent': 0, 'present': 1}
df['pcc'] = df['pcc'].map(pcc_mapping)

ba_mapping = {'notpresent': 0, 'present': 1}
df['ba'] = df['ba'].map(ba_mapping)

classification_mapping = {'notckd': 0, 'ckd': 1, 'ckd\t': 1}
df['classification'] = df['classification'].map(classification_mapping)
categorical_cols_to_encode = ['htn', 'dm', 'cad', 'appet', 'pe', 'ane' ,  'pcc' , 'ba', 'classification']

print("Unique values after encoding:")
for col in categorical_cols_to_encode:
    print(f"'{col}': {df[col].unique()}")

df.isnull().sum()

sns.histplot(df['su'], kde=True)
plt.title('Distribution of su')
plt.xlabel('su')
plt.ylabel('Frequency')
plt.show()

df = df.drop(columns=['su'])
df = df.drop(columns=['rbc'])

sns.histplot(df['bgr'], kde=True)
plt.title('Distribution of bgr')
plt.xlabel('bgr')
plt.ylabel('Frequency')
plt.show()

from sklearn.model_selection import train_test_split

X = df.drop('classification', axis=1)
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Data split into training and testing sets successfully.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np

# Create a pipeline with scaling, PCA, and KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=min(X_train.shape[1], X_train.shape[0]))),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Pipeline (StandardScaler + PCA + KNN) Performance (k=5):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Save the model
filename = 'kidney_model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))
print(f"Model saved as {filename}")

# Optional: verify optimal k with cross-validation on pipeline (simplified)
# Note: Cross-validation on pipeline is robust but might be slow. 
# Since the goal is mainly to save the model, checking performance with k=5 is enough as initial step.
