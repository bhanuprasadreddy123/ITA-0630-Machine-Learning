import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 * X**2 + 5 * X + 2 + np.random.randn(100, 1)


degree = 2  
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)


model = LinearRegression()
model.fit(X_poly, y)


X_new = np.linspace(0, 2, 100).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_new = model.predict(X_new_poly)

print("Predicted values:")
print(y_new)
