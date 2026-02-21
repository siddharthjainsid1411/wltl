# pibb.py
import numpy as np

class PIBB:
    def __init__(self, theta_init, lambda_init=0.1, lambda_min=1e-4, lambda_max=10.0):
        self.theta = theta_init
        self.Sigma = lambda_init * np.eye(len(theta_init))
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def update(self, samples, costs, h=10.0):
        J_min = np.min(costs)
        J_max = np.max(costs)

        if J_max - J_min < 1e-8:
            return

        weights = np.exp(-h * (costs - J_min) / (J_max - J_min))
        weights /= np.sum(weights)

        theta_new = np.sum(weights[:,None] * samples, axis=0)

        Sigma = np.zeros_like(self.Sigma)
        for m in range(len(samples)):
            diff = samples[m] - self.theta
            Sigma += weights[m] * np.outer(diff, diff)

        # Eigenvalue bounding
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.clip(eigvals, self.lambda_min, self.lambda_max)
        self.Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self.theta = theta_new