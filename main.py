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



USE_CASE = 2
if USE_CASE == 1:
    k1 = 25.0      # smooth min sharpness
    k2 = 25.0      # smooth max sharpness
    h = 6.0       # PIBB eliteness parameter
    M = 80        # number of samples (20–40 recommended)
    iterations = 400  # 100–200 recommended
    lambda_init = 1.0
    # ----------------------------

    n_dims = 2
    n_basis = 45 
    T = 500
    # regions = {
    #     'A': (np.array([3.2, 1.5]), 0.35),  #A': (np.array([3.2, 2.0]), 0.2),
    #     'B': (np.array([3.0, 3.5]), 0.3),  #B': (np.array([3.0, 3.5]), 0.2),
    #     'G': (np.array([4.0, 4.0]), 0.3)  #G': (np.array([4.0, 4.0]), 0.2)
    # }

    # obstacles = {
    #     'O1': (np.array([2.0, 1.0]), 0.4),  #O1': (np.array([2.0, 1.0]), 0.4),
    #     'O2': (np.array([3.0, 2.5]), 0.25)  #O2': (np.array([3.0, 2.5]), 0.25)
    # }

    # y0 = np.array([1.0, 1.0])
    # g  = np.array([4.0, 4.0])

    #Relaxed regions and obstacles for better convergence
    regions = {
        'A': (np.array([3.2, 1.5]), 0.35),  #A': (np.array([3.2, 2.0]), 0.2),
        'B': (np.array([4.5, 4.5]), 0.3),  #B': (np.array([3.0, 3.5]), 0.2),
        'G': (np.array([6.0, 6.0]), 0.3)  #G': (np.array([4.0, 4.0]), 0.2)
    }

    obstacles = {
        'O1': (np.array([2.1, 0.8]), 0.2),  #O1': (np.array([2.0, 1.0]), 0.4),
        'O2': (np.array([3.0, 2.5]), 0.25)  #O2': (np.array([3.0, 2.5]), 0.25)
    }

    y0 = np.array([1.0, 1.0])
    g  = np.array([6.0, 6.0])


    dmp = DMP(n_dims, n_basis, tau=1.0, alpha_z=20.0, beta_z=5.0,  alpha_s=5.0)  # beta_z = alpha_z / 4 for critical damping
    
else:
    k1 = 40.0      # smooth min sharpness
    k2 = 40.0      # smooth max sharpness
    h = 10.0       # PIBB eliteness parameter
    M = 50        # number of samples (20–40 recommended)
    iterations = 350  # 100–200 recommended
    lambda_init = 0.1
    # ----------------------------

    n_dims = 2
    n_basis = 15
    T = 300
    regions = {
        'A': (np.array([0.5, 0.8]), 0.05),
        'B': (np.array([0.8, 0.2]), 0.05),
        'G': (np.array([1.0, 1.0]), 0.05)
    }

    obstacle = (np.array([0.6, 0.6]), 0.15)
    weights = [1.0, 4.0]

    y0 = np.array([0.0, 0.0])
    g = np.array([1.0, 1.0])

    dmp = DMP(n_dims, n_basis)




theta_init = np.random.randn(n_dims * n_basis) * 0.1
optimizer = PIBB(theta_init, lambda_init = lambda_init)  
stored_trajs = {}

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
            rho, terms = robustness_case1(
                traj,
                regions,
                obstacles,
                k1=k1,
                k2=k2
            )
        else:
            rho, terms = robustness_case2(
                traj,
                regions,
                obstacle,
                weights,
                k1=k1,
                k2=k2
            )
            #rho_true = robustness_case2_hard(traj, regions, obstacle, weights)
            #print(f"Rho True: {rho_true:.4f} at itr: {it}") if rho_true >= 0 else None

        if it % 20 == 0 and m == 0:
            print("Term values:", terms)

        print(f"Rho: {rho:.4f}") if rho > 0 else None
        

        J = -rho if rho < 0 else 0

        samples.append(theta_sample)
        costs.append(J)

    samples = np.array(samples)
    costs = np.array(costs)

    optimizer.update(samples, costs, h=h)

    if it in [0,10,50,100,150,200,250,300]:
        stored_trajs[it] = traj.copy()

    

    print(f"Iter {it} | Mean Cost {np.mean(costs):.4f}")
    if np.mean(costs) < 1e-4:
        print("Converged!")
        break


# Final trajectory
plt.figure(figsize=(8,8))

if USE_CASE == 1:
    # Plot stored intermediate trajectories
    for k, v in stored_trajs.items():
        plt.plot(v[:,0], v[:,1], label=f"Update {k}")

    # Plot final trajectory thicker
    theta_final = optimizer.theta.reshape(n_dims, n_basis)
    traj_final = dmp.rollout(theta_final, y0, g, T)

    plt.plot(traj_final[:,0], traj_final[:,1], 'k', linewidth=3, label="Final")

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
    plt.savefig("case1_evolution6_relaxed_itr600_m80_k25_45.png")
    print("Saved case1_evolution6_relaxed_itr600_m80_k25_45.png")
else:
    theta_final = optimizer.theta.reshape(n_dims, n_basis)
    traj = dmp.rollout(theta_final, y0, g, T)

    # ---- Save Cartesian trajectory for MuJoCo IK ----
    filename = "case2_xy_trajectory.csv"
    # Save only x,y
    np.savetxt(
        filename,
        traj,
        delimiter=",",
        header="x,y",
        comments=""
    )

    print(f"Saved XY trajectory to {filename}")

    # ---- Plot trajectory ----
    plt.figure(figsize=(6,6))

    plt.plot(traj[:,0], traj[:,1], linewidth=2, label="Trajectory")

    # ---- Plot regions as circles ----
    for name, (center, radius) in regions.items():
        circle = plt.Circle(center, radius, fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(center[0], center[1], name)
        
    # ---- Plot obstacle as filled circle ----
    center, radius = obstacle
    circle = plt.Circle(center, radius, color='black', alpha=0.3)
    plt.gca().add_patch(circle)

    # ---- Plot start and goal ----
    plt.scatter(*y0, marker='o', label='Start')
    plt.scatter(*g, marker='x', label='Goal')

    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    plt.savefig("trajectory.png")
    print("Saved trajectory.png")