# main.py
import numpy as np
import matplotlib.pyplot as plt
from dmp import DMP
from robustness import robustness_case2
from robustness import robustness_case2_hard
from robustness import robustness_case1

from pibb import PIBB

# ----------------------------
# Hyperparameters (SET HERE)
# ----------------------------

k1 = 40.0      # smooth min sharpness
k2 = 40.0      # smooth max sharpness
h = 10.0       # PIBB eliteness parameter
M = 50        # number of samples (20–40 recommended)
iterations = 350  # 100–200 recommended

# ----------------------------

n_dims = 2
n_basis = 25
T = 300

dmp = DMP(n_dims, n_basis)

theta_init = np.random.randn(n_dims * n_basis) * 0.1
optimizer = PIBB(theta_init)

regions = {
    'A': (np.array([3.2, 2.0]), 0.2),
    'B': (np.array([3.0, 3.5]), 0.2),
    'G': (np.array([4.0, 4.0]), 0.2)
}

obstacles = {
    'O1': (np.array([2.0, 1.0]), 0.5),
    'O2': (np.array([3.0, 2.5]), 0.5)
}

y0 = np.array([1.0, 1.0])
g  = np.array([4.0, 4.0])

stored_trajs = {}
USE_CASE = 1

# regions = {
#     'A': (np.array([0.5, 0.8]), 0.2),
#     'B': (np.array([0.8, 0.2]), 0.2),
#     'G': (np.array([1.0, 1.0]), 0.2)
# }

# obstacle = (np.array([0.6, 0.6]), 0.15)
# weights = [1.0, 2.0]

# y0 = np.array([0.0, 0.0])
# g = np.array([1.0, 1.0])

for it in range(iterations):

    samples = []
    costs = []

    for m in range(M):

        eps = np.random.multivariate_normal(
            np.zeros(len(optimizer.theta)),
            optimizer.Sigma
        )

        theta_sample = optimizer.theta + eps
        theta_reshaped = theta_sample.reshape(n_dims, n_basis)

        traj = dmp.rollout(theta_reshaped, y0, g, T)

        if USE_CASE == 1:
            rho = robustness_case1(
                traj,
                regions,
                obstacles,
                k1=k1,
                k2=k2
            )
        else:
            rho = robustness_case2(
                traj,
                regions,
                obstacle,
                weights,
                k1=k1,
                k2=k2
            )

        #rho_true = robustness_case2_hard(traj, regions, obstacle, weights)

        print(f"Rho: {rho:.4f}") if rho > 0 else None
        #print(f"Rho True: {rho_true:.4f} at itr: {m}") 

        J = -rho if rho < 0 else 0

        samples.append(theta_sample)
        costs.append(J)

    samples = np.array(samples)
    costs = np.array(costs)

    optimizer.update(samples, costs, h=h)

    if it in [0,10,50,100,150,200,250,300]:
        stored_trajs[it] = traj.copy()

    print(f"Iter {it} | Mean Cost {np.mean(costs):.4f}")

# Final trajectory
plt.figure(figsize=(6,6))

# Plot stored intermediate trajectories
for k, v in stored_trajs.items():
    plt.plot(v[:,0], v[:,1], label=f"Update {k}")

# Plot final trajectory thicker
plt.plot(traj[:,0], traj[:,1], 'k', linewidth=3, label="Final")

# Plot regions
for name, (center, radius) in regions.items():
    circle = plt.Circle(center, radius, fill=False)
    plt.gca().add_patch(circle)
    plt.text(center[0], center[1], name)

# Plot obstacles
for name, (center, radius) in obstacles.items():
    circle = plt.Circle(center, radius, color='red', alpha=0.3)
    plt.gca().add_patch(circle)

plt.scatter(*y0, label="Start")
plt.legend()
plt.axis("equal")
plt.savefig("case1_evolution.png")
print("Saved case1_evolution.png")

# theta_final = optimizer.theta.reshape(n_dims, n_basis)
# traj = dmp.rollout(theta_final, y0, g, T)

# plt.plot(traj[:,0], traj[:,1])
# plt.scatter(*regions['A'][0], c='r', label='A')
# plt.scatter(*regions['B'][0], c='g', label='B')
# plt.scatter(*regions['G'][0], c='b', label='G')
# plt.scatter(*obstacle[0], c='k', label='Obstacle')
# plt.legend()

# plt.savefig("trajectory.png")
# print("Saved trajectory.png")