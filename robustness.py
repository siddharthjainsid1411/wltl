# robustness.py
import numpy as np
from smooth import smooth_min, smooth_max
from wtltl import weighted_and, weighted_or

# ---- Predicates ----

def predicate_region(y, center, radius):
    return radius - np.linalg.norm(y - center)

def predicate_obstacle(y, center, radius):
    return np.linalg.norm(y - center) - radius


# -------- CASE 1: Sequential A -> B -> G --------

def robustness_case1(traj, regions, obstacles,
                     k1=20.0, k2=20.0):

    rho_A = np.array([predicate_region(y, regions['A'][0], regions['A'][1]) for y in traj])
    rho_B = np.array([predicate_region(y, regions['B'][0], regions['B'][1]) for y in traj])
    rho_G = np.array([predicate_region(y, regions['G'][0], regions['G'][1]) for y in traj])

    rho_O = np.array([
        min(
            predicate_obstacle(y, obstacles['O1'][0], obstacles['O1'][1]),
            predicate_obstacle(y, obstacles['O2'][0], obstacles['O2'][1])
        )
        for y in traj
    ])

    # Eventually A
    rho_event_A = smooth_max(rho_A, k=k2)
    #rho_event_A *= 1.2  

    # Eventually B
    rho_event_B = smooth_max(rho_B, k=k2)

    # Eventually G
    rho_event_G = smooth_max(rho_G, k=k2)

    # Not B until A
    rho_notB = -rho_B
    rho_until_A = smooth_until(rho_notB, rho_A, k1=k1, k2=k2)

    # Not G until B
    rho_notG = -rho_G
    rho_until_B = smooth_until(rho_notG, rho_B, k1=k1, k2=k2)

    # Always avoid obstacles
    rho_always_O = smooth_min(rho_O, k=k1)

    # Full conjunction
    terms = {
        "event_A": rho_event_A,
        "event_B": rho_event_B,
        "event_G": rho_event_G,
        "until_A": rho_until_A,
        "until_B": rho_until_B,
        "always_O": rho_always_O
    }

    rho_total = smooth_min(list(terms.values()), k=k1)

    return rho_total, terms

    
# ---- UNTIL operator (corrected) ----

def smooth_until(rho_phi, rho_psi, k1=20.0, k2=20.0):
    """
    Implements:
    rho_e(phi U psi)
    = smooth_max_t'
        smooth_min(
            rho_psi(t'),
            smooth_min_{t'' < t'} rho_phi(t'')
        )
    """

    T = len(rho_phi)
    values = []

    for t_prime in range(T):

        if t_prime == 0:
            min_before = np.inf
        else:
            min_before = smooth_min(rho_phi[:t_prime], k=k1)

        inner = smooth_min([rho_psi[t_prime], min_before], k=k1)
        values.append(inner)

    return smooth_max(values, k=k2)

# ---- Full Case 2 robustness ----

def robustness_case2(traj, regions, obstacle, weights,
                     k1=20.0, k2=20.0):

    rho_A = np.array([predicate_region(y, regions['A'][0], regions['A'][1]) for y in traj])
    rho_B = np.array([predicate_region(y, regions['B'][0], regions['B'][1]) for y in traj])
    rho_G = np.array([predicate_region(y, regions['G'][0], regions['G'][1]) for y in traj])
    rho_O = np.array([predicate_obstacle(y, obstacle[0], obstacle[1]) for y in traj])

    # (A ∨_w B)
    rho_AB = np.array([
        weighted_or(weights, [rho_A[t], rho_B[t]])
        for t in range(len(traj))
    ])

    # Eventually visit A or B
    rho_event_AB = smooth_max(rho_AB, k=k2)

    # Eventually visit G
    rho_event_G = smooth_max(rho_G, k=k2)

    # Always avoid obstacle
    rho_always_O = smooth_min(rho_O, k=k1)

    # NOT G until (A or B)
    rho_notG = -rho_G
    rho_until = smooth_until(rho_notG, rho_AB, k1=k1, k2=k2)

    # Full conjunction
    terms = {
        "event_AB": rho_event_AB,
        "event_G": rho_event_G,
        "until_AB": rho_until,
        "always_O": rho_always_O
    }

    rho_total = smooth_min(list(terms.values()), k=k1)

    return rho_total, terms

# -------- HARD ROBUSTNESS (no smoothing) --------

def hard_until(rho_phi, rho_psi):
    """
    Implements exact:
    rho(phi U psi)
    = max_{t'} min(
        rho_psi(t'),
        min_{t'' < t'} rho_phi(t'')
      )
    """

    T = len(rho_phi)
    values = []

    for t_prime in range(T):

        if t_prime == 0:
            min_before = rho_phi[0]
        else:
            min_before = np.min(rho_phi[:t_prime])

        inner = min(rho_psi[t_prime], min_before)
        values.append(inner)

    return np.max(values)


def robustness_case2_hard(traj, regions, obstacle, weights):

    rho_A = np.array([predicate_region(y, regions['A'][0], regions['A'][1]) for y in traj])
    rho_B = np.array([predicate_region(y, regions['B'][0], regions['B'][1]) for y in traj])
    rho_G = np.array([predicate_region(y, regions['G'][0], regions['G'][1]) for y in traj])
    rho_O = np.array([predicate_obstacle(y, obstacle[0], obstacle[1]) for y in traj])

    # weighted OR
    rho_AB = np.array([
        weighted_or(weights, [rho_A[t], rho_B[t]])
        for t in range(len(traj))
    ])

    rho_event_AB = np.max(rho_AB)
    rho_event_G = np.max(rho_G)
    rho_always_O = np.min(rho_O)

    rho_notG = -rho_G
    rho_until = hard_until(rho_notG, rho_AB)

    rho_total = min(rho_event_AB, rho_event_G, rho_until, rho_always_O)

    return rho_total