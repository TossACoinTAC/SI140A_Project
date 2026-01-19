import json
import numpy as np
import scipy.spatial.distance as ssd
import os

# Set seed for reproducibility
np.random.seed(42)


def load_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return np.array(data)


def energy_distance(u, v):
    """
    Compute the energy distance between two distributions u and v.
    d^2(u, v) = 2 E||u - v|| - E||u - u'|| - E||v - v'||

    If inputs are very large, we subsample to keep computation feasible.
    """
    # Subsample if too large to avoid O(N^2) memory/time issues
    # Limit to e.g. 2000 samples for the distance matrix calculation
    max_samples = 2000

    if len(u) > max_samples:
        indices = np.random.choice(len(u), max_samples, replace=False)
        u = u[indices]

    if len(v) > max_samples:
        indices = np.random.choice(len(v), max_samples, replace=False)
        v = v[indices]

    # Efficient pairwise distance calculation
    d_uv = ssd.cdist(u, v, "euclidean")
    d_uu = ssd.cdist(u, u, "euclidean")
    d_vv = ssd.cdist(v, v, "euclidean")

    E_uv = np.mean(d_uv)
    E_uu = np.mean(d_uu)
    E_vv = np.mean(d_vv)

    result = 2 * E_uv - E_uu - E_vv
    return result


def energy_distance_permutation_test(u, v, n_permutations=100):
    """
    Perform a permutation test to get a p-value for the Energy Distance.
    H0: u and v are from the same distribution.
    """
    # Subsample for speed if needed (keep consistent with energy_distance call)
    max_samples = 1000  # smaller for permutation loop
    if len(u) > max_samples:
        u = u[np.random.choice(len(u), max_samples, replace=False)]
    if len(v) > max_samples:
        v = v[np.random.choice(len(v), max_samples, replace=False)]

    n_u = len(u)
    n_v = len(v)
    combined = np.vstack([u, v])

    # Precompute distance matrix
    d_mat = ssd.cdist(combined, combined, "euclidean")

    def get_e_stat(indices_u, indices_v):
        d_uu = d_mat[np.ix_(indices_u, indices_u)]
        d_vv = d_mat[np.ix_(indices_v, indices_v)]
        d_uv = d_mat[np.ix_(indices_u, indices_v)]

        return 2 * np.mean(d_uv) - np.mean(d_uu) - np.mean(d_vv)

    # Observed
    obs_stat = get_e_stat(np.arange(n_u), np.arange(n_u, n_u + n_v))

    count_greater = 0
    all_indices = np.arange(n_u + n_v)

    for _ in range(n_permutations):
        shuffled = np.random.permutation(all_indices)
        idx_u = shuffled[:n_u]
        idx_v = shuffled[n_u:]

        perm_stat = get_e_stat(idx_u, idx_v)
        if perm_stat >= obs_stat:
            count_greater += 1

    p_value = (count_greater + 1) / (n_permutations + 1)
    return obs_stat, p_value


def peacock_test(u, v):
    """
    Multidimensional Kolmogorov-Smirnov test (Fasano-Franceschini, 1987).
    Often referred to as Peacock's Algorithm (which was the 2D precursor).

    D = max | P(u in orthant) - P(v in orthant) |
    over all orthants defined by all data points.
    """

    # Subsample for feasibility
    # The number of orthants is 2^d. For d=6, 64 orthants.
    # We iterate over points to define the origin.
    # If N is large, this is O(N^2). We limit N.
    n_limit = 500  # limit reference points (centers)

    # We can use all points for calculating probabilities, but limit the centers.
    # However, to be consistent and fast, we'll subsample the data used for counting as well
    # if it's extremely large, but 1000-2000 is fine for counting.

    # Let's keep v (generated) large-ish but subsampled, and u (real) as is if small.
    # Subsampling
    u_sub = u
    v_sub = v

    # Limit the "points" we check (the grid centers) to a manageable union
    # We combine u and v, then sample from them to get test centers.

    combined = np.vstack([u, v])
    if len(combined) > n_limit:
        center_indices = np.random.choice(len(combined), n_limit, replace=False)
        centers = combined[center_indices]
    else:
        centers = combined

    # For accuracy, we should check counts against the full (or large subsample) sets
    # But evaluating 30000 points 64 times for 1000 centers is 1.9e9 ops. Too slow for python loop.
    # Vectorization is key.

    # Reduce v to a manageable size for density estimation, e.g. 2000
    if len(v_sub) > 2000:
        v_sub = v_sub[np.random.choice(len(v_sub), 2000, replace=False)]

    # Dimensions
    dims = u.shape[1]
    n_u = len(u_sub)
    n_v = len(v_sub)

    max_d = 0.0

    # Vectorized implementation
    # For each center, we have 2^dims orthants.
    # Instead of iterating 2^dims, we can just process the relationship of all points to a center.
    # A point p is in orthant O_k of center c if (p > c) matches the bool signature of k.

    for i, center in enumerate(centers):
        # Broadcast comparison: (N, D) > (1, D) -> (N, D) boolean
        # u_bool: (N_u, D), true if u_coord > center_coord
        diff_u = u_sub > center  # shape (N_u, D)
        diff_v = v_sub > center  # shape (N_v, D)

        # We need to group these by orthant.
        # An orthant is defined by the tuple of booleans.
        # efficiently, we can convert boolean vector to integer index for orthant
        # e.g. [True, False, True] -> 101 -> 5

        # Powers of 2 for conversion
        powers = 2 ** np.arange(dims)

        # (N, D) * (D,) -> (N,) sum over axis 1
        orthant_indices_u = (diff_u.astype(int) * powers).sum(axis=1)
        orthant_indices_v = (diff_v.astype(int) * powers).sum(axis=1)

        # Count occurrences of each orthant 0..2^d-1
        # using bincount is fast
        counts_u = np.bincount(orthant_indices_u, minlength=2**dims)
        counts_v = np.bincount(orthant_indices_v, minlength=2**dims)

        # Convert to frequencies
        freq_u = counts_u / n_u
        freq_v = counts_v / n_v

        # Max diff for this center
        d_center = np.max(np.abs(freq_u - freq_v))

        if d_center > max_d:
            max_d = d_center

    return max_d


def peacock_permutation_test(u, v, n_permutations=100):
    """
    Perform a permutation test to get a p-value for the Multidimensional KS statistic.
    """
    # Observed statistic
    obs_d = peacock_test(u, v)

    n_u = len(u)
    n_v = len(v)
    combined = np.vstack([u, v])
    all_indices = np.arange(n_u + n_v)

    count_greater = 0

    for k in range(n_permutations):
        shuffled = np.random.permutation(all_indices)
        idx_u = shuffled[:n_u]
        idx_v = shuffled[n_u:]

        u_perm = combined[idx_u]
        v_perm = combined[idx_v]

        perm_d = peacock_test(u_perm, v_perm)

        if perm_d >= obs_d:
            count_greater += 1

    p_value = (count_greater + 1) / (n_permutations + 1)
    return obs_d, p_value


def perform_tests():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wechat_path = os.path.join(base_dir, "Database", "Wechat_Samples.json")
    generated_path = os.path.join(base_dir, "Database", "DDPM_Generated_Samples.json")

    if not os.path.exists(wechat_path) or not os.path.exists(generated_path):
        print("Data files not found.")
        return

    wechat_data = load_data(wechat_path)
    generated_data = load_data(generated_path)

    # -----------------------------------------------------
    # EVALUATION: All 6 Dimensions (Standard)
    # -----------------------------------------------------
    if wechat_data.shape[1] >= 6 and generated_data.shape[1] >= 6:
        print("\n" + "=" * 60)
        print("DISTRIBUTION HYPOTHESIS TESTING (6 DIMENSIONS)")
        print(
            "H0: The generated samples come from the same distribution as the real data."
        )
        print("Alpha Level: 0.05")
        print("=" * 60)

        w_6d = wechat_data[:, :6]
        g_6d = generated_data[:, :6]

        # 1. Peacock's Test
        print("\n[Test 1] Peacock's Multidimensional KS Test")
        ks_stat_6d, ks_p_6d = peacock_permutation_test(
            w_6d, g_6d, n_permutations=200
        )  # Increased for better resolution
        print("-" * 40)
        print(f"  Statistic : {ks_stat_6d:.6f}")
        print(f"  P-value   : {ks_p_6d:.4f}")
        print(
            f"  Result    : {'REJECT H0 (Distributions Different)' if ks_p_6d < 0.05 else 'FAIL TO REJECT H0 (Distributions Consistent)'}"
        )

        # 2. Energy Distance Test
        print("\n[Test 2] Energy Distance Test")
        ed_6d, ed_p_6d = energy_distance_permutation_test(
            w_6d, g_6d, n_permutations=200
        )
        print("-" * 40)
        print(f"  Statistic : {ed_6d:.6f}")
        print(f"  P-value   : {ed_p_6d:.4f}")
        print(
            f"  Result    : {'REJECT H0 (Distributions Different)' if ed_p_6d < 0.05 else 'FAIL TO REJECT H0 (Distributions Consistent)'}"
        )

        print("\n" + "=" * 60)
    else:
        print("\nError: Data has fewer than 6 dimensions.")


if __name__ == "__main__":
    perform_tests()
