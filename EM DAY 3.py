import numpy as np
from scipy.stats import multivariate_normal

class EM_GMM:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        self.pi = np.ones(self.n_clusters) / self.n_clusters
        self.mu = np.random.randn(self.n_clusters, n_features)
        self.sigma = np.array([np.eye(n_features)] * self.n_clusters)
        for _ in range(self.max_iter):
            gamma = self._expectation(X)
            self._maximization(X, gamma)
        return self

    def _expectation(self, X):
        gamma = np.array([self.pi[k] * multivariate_normal.pdf(X, mean=self.mu[k], cov=self.sigma[k]) for k in range(self.n_clusters)]).T
        return gamma / np.sum(gamma, axis=1, keepdims=True)

    def _maximization(self, X, gamma):
        N_k = np.sum(gamma, axis=0)
        self.pi = N_k / X.shape[0]
        self.mu = np.dot(gamma.T, X) / N_k[:, np.newaxis]
        self.sigma = np.array([np.dot((X - self.mu[k]).T, (gamma[:, k] * (X - self.mu[k]).T).T) / N_k[k] for k in range(self.n_clusters)])

    def predict(self, X):
        return np.argmax(self._expectation(X), axis=1)
