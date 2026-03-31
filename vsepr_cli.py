"""
Command-line interface for generating VSEPR configurations. Uses little GPU and can run multiple processes at once.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

LONE_CONSTANT = 1.23257467


def gen_points(amt: int):
    rng = np.random.default_rng()
    # Generate points in rectangular coordinates to prevent point clustering caused by spherical coordinates
    pts_rect = rng.uniform(-1.0, 1.0, (amt, 3))
    pts_rect = pts_rect / np.linalg.norm(pts_rect, axis=1, keepdims=True)

    # Now convert those coordinates to spherical coordinates (theta, phi)
    t = np.arctan2(pts_rect[:, 1], pts_rect[:, 0])
    p = np.arccos(np.clip(pts_rect[:, 2], -1, 1))

    return np.column_stack([t, p])


def find_points(
    bonded_p: int,
    lone_p: int,
    temp: np.float64,
    temp_min=0.00001,
    decay=0.999,
    bonded_points_arr: np.ndarray = None,
    lone_points_arr: np.ndarray = None,
):
    total_p = bonded_p + lone_p

    # Take the first bonded_p points to be the points that are "bonded" and the last lone_p points to be "lone" points
    points = gen_points(total_p)

    STEP = 0.3 / np.sqrt(total_p)
    NOISE_AMP = 0.1 / np.sqrt(total_p)
    NOISE_AMT = (total_p // 5) + 1

    temp_i = temp * total_p
    temp = temp_i
    best_e = np.float64('inf')
    best_pts: np.ndarray = points.copy()
    curr_e = calc_energy_state(points, bonded_p)
    total_steps = 0
    steps = 0
    accepted = 0

    start = time.time()
    # Phase A
    # Utilizing simulated annealing to work towards global minimum
    # See https://cp-algorithms.com/num_methods/simulated_annealing.html
    while temp > temp_min:
        grad = calc_grad_arr(points, bonded_p)

        # We want to attempt to apply the negative gradient AND some noise (in around 1/5 of points) to the
        # current configuration, then check if it will accept using the PAF (probability acceptance function).
        next_pts = points - STEP * grad

        noise_i = np.random.choice(total_p, NOISE_AMT, replace=False)
        noise = gen_points(NOISE_AMT)
        next_pts[noise_i] += NOISE_AMP * (temp / temp_i) * noise

        next_e = calc_energy_state(next_pts, bonded_p)

        if paf(curr_e, next_e, temp):
            points = next_pts
            curr_e = next_e
            accepted += 1

            if next_e < best_e:
                best_e = next_e
                best_pts = next_pts.copy()

        temp *= decay
        steps += 1

        # Update the STEP if the program is being greedy or too conservative
        if steps % 100 == 0:
            rate = accepted / 100
            if rate > 0.6:
                STEP *= 1.1
            elif rate < 0.4:
                STEP *= 0.9

            # But step can't be too high or low...
            STEP = np.clip(STEP, 1e-4 / np.sqrt(total_p), 1.0 / np.sqrt(total_p))
            accepted = 0

    # The program can explore higher-energy point configs, make sure to reset to the best known configuration
    points = best_pts

    # Phase O
    # Gradient descent applied to all points
    temp = temp_i
    STEP = 0.1 / np.sqrt(total_p)
    total_steps += steps
    steps = 0
    accepted = 0

    while temp > temp_min:
        grad = calc_grad_arr(points, bonded_p)
        next_pts = points - STEP * grad
        next_e = calc_energy_state(next_pts, bonded_p)

        if next_e < best_e:
            points = next_pts
            best_e = next_e
            accepted += 1

        temp *= decay
        steps += 1

        # Update the STEP if the program is being greedy or too conservative
        if steps % 100 == 0:
            rate = accepted / 100
            if rate > 0.6:
                STEP *= 1.1
            elif rate < 0.4:
                STEP *= 0.9

            # But step can't be too high or low...
            STEP = np.clip(STEP, 1e-7 / np.sqrt(total_p), 0.5 / np.sqrt(total_p))
            accepted = 0

    # Phase R
    # Negative gradient applied to one point only to further refine energy level
    temp = temp_i
    STEP = 0.05 / np.sqrt(total_p)
    total_steps += steps
    steps = 0
    accepted = 0

    # First calculate the distance array once, then update the array incrementally to keep a O(n) complexity
    dist_arr = calc_dist_arr(points)

    while temp > temp_min:
        i = np.random.randint(total_p)
        w = weights_row(bonded_p, total_p, i)

        old_dist = dist_arr[i, :].copy()
        grad = calc_grad_row(points, bonded_p, old_dist, i, w)

        old_pts = points[i, :].copy()
        points[i, 0] = points[i, 0] - STEP * grad[0]
        points[i, 1] = points[i, 1] - STEP * grad[1]

        # Instead of calculating the total energy of the system, calculate only the energy potential of
        # the old set of points and the new set of points, and compare those potentials
        new_dist = calc_dist_row(points, i)
        old_energy = np.sum(w[old_dist > 0] / old_dist[old_dist > 0])
        new_energy = np.sum(w[new_dist > 0] / new_dist[new_dist > 0])
        next_e = best_e - old_energy + new_energy

        if next_e < best_e:
            best_e = next_e
            accepted += 1
            # Apply the change to the dist array
            dist_arr[i, :] = new_dist
            dist_arr[:, i] = new_dist
        else:
            # Revert the previous change
            points[i, :] = old_pts

        temp *= decay
        steps += 1

        # Update the STEP if the program is being greedy or too conservative
        if steps % 100 == 0:
            rate = accepted / 100
            if rate > 0.6:
                STEP *= 1.1
            elif rate < 0.4:
                STEP *= 0.9

            # But step can't be too high or low...
            STEP = np.clip(STEP, 1e-7 / np.sqrt(total_p), 0.5 / np.sqrt(total_p))
            accepted = 0

    total_steps += steps
    end = time.time()

    # Print results to terminal console
    print([total_p, best_e, end - start, total_steps])
    return (best_e, points)


def run(
    bonded_p: int, lone_p: int, temp: np.float64, decay: np.float64, runs: int = 10
):
    with ProcessPoolExecutor(max_workers=runs) as executor:
        procs = [
            executor.submit(find_points, bonded_p, lone_p, temp, decay=decay)
            for _ in range(runs)
        ]
        results = [p.result() for p in procs]

    output = {
        'bonded_points': bonded_p,
        'lone_points': lone_p,
        'runs': [
            {
                'energy': energy,
                'points': [
                    {'type': 'bonded' if i < bonded_p else 'lone', 'theta': t, 'phi': p}
                    for i, (t, p) in enumerate(pts)
                ],
            }
            for (energy, pts) in results
        ],
    }

    filename = 'data/results_vsepr_cli.json'

    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump([], f)
    with open(filename, 'r+') as f:
        data = json.load(f)
        data.append(output)
        f.seek(0)
        json.dump(data, f, indent=4)


def calc_dist_arr(pts: np.ndarray):
    t = pts[:, 0]
    p = pts[:, 1]
    dt = t[:, np.newaxis] - t[np.newaxis, :]

    dist_sq = (
        2
        - 2 * np.sin(p[:, np.newaxis]) * np.sin(p[np.newaxis, :]) * np.cos(dt)
        - 2 * np.cos(p[:, np.newaxis]) * np.cos(p[np.newaxis, :])
    )

    # Ensure same distance to same points (the diagonal) is 0
    np.fill_diagonal(dist_sq, 0)
    return np.sqrt(np.clip(dist_sq, 0, None))


def calc_grad_arr(pts: np.ndarray, bonded_points: int) -> np.ndarray:
    t = pts[:, 0]
    p = pts[:, 1]
    dt = t[:, np.newaxis] - t[np.newaxis, :]
    weight = weights(pts, bonded_points)

    den = calc_dist_arr(pts) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_den = np.where(den > 0, 1.0 / den, 0.0)

    # Partial derivatives per point
    d_t = np.sum(
        (-np.sin(p[:, np.newaxis]) * np.sin(p[np.newaxis, :]) * np.sin(dt))
        * inv_den
        * weight,
        axis=1,
    )
    d_p = np.sum(
        (
            np.cos(p[:, np.newaxis]) * np.sin(p[np.newaxis, :]) * np.cos(dt)
            - np.sin(p[:, np.newaxis]) * np.cos(p[np.newaxis, :])
        )
        * inv_den
        * weight,
        axis=1,
    )

    return np.column_stack([d_t, d_p])


def calc_energy_state(pts: np.ndarray, bonded_points: int) -> np.double:
    dist_arr = calc_dist_arr(pts)
    # Only need the upper triangle of matrix so distances aren't counted duplicate
    # Where i < j, dist_arr[i][j] is kept
    weight = weights(pts, bonded_points)
    upper_mask = np.triu(dist_arr > 0, k=1)

    # Divide the upper array by the lone pair energy constants (if there are lone pairs)
    # Since the reciprocal of this value will be taken, it will essentially be "multiplied"

    return np.sum(weight[upper_mask] / dist_arr[upper_mask])


def calc_dist_row(pts: np.ndarray, i: int):
    t_i, p_i = pts[i, 0], pts[i, 1]
    dt = t_i - pts[:, 0]

    dist_sq = (
        2
        - 2 * np.sin(p_i) * np.sin(pts[:, 1]) * np.cos(dt)
        - 2 * np.cos(p_i) * np.cos(pts[:, 1])
    )

    # Set the distance to the same point (i) 0
    dist_sq[i] = 0

    return np.sqrt(np.clip(dist_sq, 0, None))


def calc_grad_row(
    pts: np.ndarray,
    bonded_points: int,
    dist_row: np.ndarray,
    i: int,
    weight: np.ndarray,
):
    t_i, p_i = pts[i, 0], pts[i, 1]
    dt = t_i - pts[:, 0]
    den = dist_row**3

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_den = np.where(den > 0, 1.0 / den, 0.0)

    d_t = np.sum(-np.sin(p_i) * np.sin(pts[:, 1]) * np.sin(dt) * inv_den * weight)
    d_p = np.sum(
        (np.cos(p_i) * np.sin(pts[:, 1]) * np.cos(dt) - np.sin(p_i) * np.cos(pts[:, 1]))
        * inv_den
        * weight
    )

    return np.array([d_t, d_p])


def weights(pts: np.ndarray, bonded_points: int) -> np.ndarray:
    dim = pts.shape[0]
    weight = np.ones((dim, dim))
    weight[bonded_points:, :] *= LONE_CONSTANT
    weight[:, bonded_points:] *= LONE_CONSTANT
    return weight


def weights_row(bonded_points: int, total: int, i: int) -> np.ndarray:
    weight = np.ones(total)
    weight[bonded_points:] *= LONE_CONSTANT
    if i >= bonded_points:
        weight *= LONE_CONSTANT
    return weight


def paf(e: np.float64, e_n: np.float64, t: int) -> bool:
    if e_n < e:
        return True

    return np.random.random() <= np.exp((e - e_n) / t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='VSEPR_cli', description='Calculates values for VSEPR configurations.'
    )
    parser.add_argument('bonded', type=int, help='Number of bonded points')
    parser.add_argument('lone', type=int, help='Number of lone points')
    parser.add_argument(
        '--temp', type=float, default=10000, help='Initial temperature [default: 10000]'
    )
    parser.add_argument(
        '--decay', type=float, default=0.999, help='Decay [default: 0.999]'
    )
    parser.add_argument(
        '--runs', type=int, default=10, help='Number of Runs to Execute [default: 10]'
    )

    args = parser.parse_args()
    run(args.bonded, args.lone, args.temp, args.decay, args.runs)
