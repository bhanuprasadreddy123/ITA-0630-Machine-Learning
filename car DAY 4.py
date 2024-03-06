# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data = {
    'Income': [50000, 60000, 40000, 80000, 55000],
    'Age': [35, 42, 28, 55, 30],
    'Debt': [2000, 3000, 1000, 5000, 2500],
    'Credit_Score': ['Good', 'Excellent', 'Fair', 'Poor', 'Good']
}

credit_data = pd.DataFrame(data)


X = credit_data.drop(columns=['Credit_Score'])  
y = credit_data['Credit_Score']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

