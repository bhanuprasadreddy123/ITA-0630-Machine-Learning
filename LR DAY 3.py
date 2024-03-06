import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])  


model = LogisticRegression()


model.fit(X, y)


new_data = np.array([[9, 10]])
predictions = model.predict(new_data)


print(f"Predicted class: {predictions[0]}")


probabilities = model.predict_proba(new_data)
print(f"Probabilities: {probabilities[0]}")
