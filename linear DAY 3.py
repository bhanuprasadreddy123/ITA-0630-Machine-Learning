import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)


model = LinearRegression()


model.fit(X_poly, y)


new_data = np.array([[5]])
new_data_poly = poly.transform(new_data)
predictions = model.predict(new_data_poly)


print(f"Predicted value: {predictions[0]}")
