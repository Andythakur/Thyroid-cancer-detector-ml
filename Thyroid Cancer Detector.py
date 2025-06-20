import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("dataset.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
df = df.dropna()
categorical_cols = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function',
                    'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk',
                    'T', 'N', 'M', 'Stage', 'Response', 'Recurred']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('Recurred', axis=1)
y = df['Recurred']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance for Thyroid Cancer Recurrence Prediction")
plt.show()
