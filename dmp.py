# dmp.py
import numpy as np

class DMP:
    def __init__(self, n_dims, n_basis, tau=1.0, alpha_z=25.0, beta_z=6.25, alpha_s=4.0):
        self.n_dims = n_dims
        self.n_basis = n_basis
        self.tau = tau
        self.alpha_z = alpha_z
        self.beta_z = beta_z
        self.alpha_s = alpha_s

        # Basis centers in phase space
        self.centers = np.linspace(0, 1, n_basis)
        self.widths = np.ones(n_basis) * (1.0 / n_basis**2)

    def basis_function(self, s):
        psi = np.exp(-0.5 * ((s - self.centers)**2) / self.widths)
        return psi

    def forcing_term(self, s, theta, g, y0):
        psi = self.basis_function(s)
        if np.sum(psi) < 1e-10:
            return np.zeros(self.n_dims)
        psi_normalized = psi / np.sum(psi)
        f = np.zeros(self.n_dims)
        for d in range(self.n_dims):
            f[d] = np.dot(psi_normalized * s * (g[d] - y0[d]), theta[d])
        return f

    def rollout(self, theta, y0, g, T=1000, dt=0.01):
        y = np.zeros((T, self.n_dims))
        z = np.zeros((T, self.n_dims))
        s = np.zeros(T)

        y[0] = y0
        s[0] = 1.0

        for t in range(T-1):
            f = self.forcing_term(s[t], theta, g, y0)

            z_dot = self.alpha_z * (self.beta_z * (g - y[t]) - z[t]) + f
            y_dot = z[t]
            s_dot = -self.alpha_s * s[t]

            z[t+1] = z[t] + (z_dot / self.tau) * dt
            y[t+1] = y[t] + (y_dot / self.tau) * dt
            s[t+1] = s[t] + (s_dot / self.tau) * dt

        return y