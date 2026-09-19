import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data in raw Cartesian space D = (x1, x2)
np.random.seed(42)
N = 300

# Class A: Inner Ring
r_inner = np.random.randn(N) * 0.1 + 2.0
theta_inner = np.random.uniform(0, 2 * np.pi, N)
X_inner = np.stack([r_inner * np.cos(theta_inner),
                    r_inner * np.sin(theta_inner)], axis=1)

# Class B: Outer Ring
r_outer = np.random.randn(N) * 0.1 + 5.0
theta_outer = np.random.uniform(0, 2 * np.pi, N)
X_outer = np.stack([r_outer * np.cos(theta_outer),
                    r_outer * np.sin(theta_outer)], axis=1)

# 2. Define the secondary representation mapping (polar transformation)
def transform_to_secondary_space(X):
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    angle = np.arctan2(X[:, 1], X[:, 0])
    return np.stack([radius, angle], axis=1)

r_hat_inner = transform_to_secondary_space(X_inner)
r_hat_outer = transform_to_secondary_space(X_outer)

# 3. Visualize raw space vs. secondary representation space
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

ax[0].scatter(X_inner[:, 0], X_inner[:, 1], color='crimson', label='Class A (Inner)')
ax[0].scatter(X_outer[:, 0], X_outer[:, 1], color='navy', label='Class B (Outer)')
ax[0].set_title("Raw Data Space $\\mathcal{D}$ (Cartesian)")
ax[0].set_xlabel("Feature $x_1$"); ax[0].set_ylabel("Feature $x_2$")
ax[0].grid(True, linestyle='--', alpha=0.6); ax[0].legend()

ax[1].scatter(r_hat_inner[:, 0], r_hat_inner[:, 1], color='crimson', label='Class A')
ax[1].scatter(r_hat_outer[:, 0], r_hat_outer[:, 1], color='navy', label='Class B')
ax[1].axvline(x=3.5, color='black', linestyle='--', linewidth=2,
              label='Linear Decision Boundary')
ax[1].set_title("Secondary Space $\\hat{r}$ (Polar)")
ax[1].set_xlabel("Radius ($r$)"); ax[1].set_ylabel("Angle ($\\theta$)")
ax[1].grid(True, linestyle='--', alpha=0.6); ax[1].legend()

plt.tight_layout()
plt.show()
