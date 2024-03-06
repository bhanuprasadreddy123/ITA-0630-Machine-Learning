import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            distances = []
            for j in range(len(self.X_train)):
                dist = self.euclidean_distance(X_test[i], self.X_train[j])
                distances.append((self.X_train[j], dist, self.y_train[j]))
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:self.k]
            class_votes = {}
            for neighbor in neighbors:
                label = neighbor[2]
                class_votes[label] = class_votes.get(label, 0) + 1
            sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
            predictions.append(sorted_votes[0][0])
        return predictions

X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
y_train = np.array([0, 0, 1, 1])

knn = KNN(k=3)
knn.fit(X_train, y_train)

X_test = np.array([[4, 5], [1, 1]])
predictions = knn.predict(X_test)
print("Predictions:", predictions)
