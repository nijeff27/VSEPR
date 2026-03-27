"""
Command-line interface for generating VSEPR configurations. Uses little GPU and can run multiple processes at once.
"""

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RNG = np.random.default_rng()


def gen_points(amt: int):
    # Generate points in rectangular coordinates to prevent point clustering caused by spherical coordinates
    pts_rect = RNG.uniform(-1.0, 1.0, (amt, 3))
    pts_rect = pts_rect / np.linalg.norm(pts_rect, axis=1, keepdims=True)

    # Now convert those coordinates to spherical coordinates (theta, phi)
    t = np.arctan2(pts_rect[:, 1], pts_rect[:, 0])
    p = np.arccos(np.clip(pts_rect[:, 2], -1, 1))

    return np.column_stack([t, p])


def find_points(
    bonded_p: int,
    lone_p: int,
    temp: float,
    lone_const: np.float64,
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

    temp_i = temp
    best_e = float('inf')
    best_pts: np.ndarray = points.copy()
    curr_e = calc_energy_state(points, bonded_p, lone_const)
    total_steps = 0
    steps = 0
    accepted = 0

    start = time.time()
    # Phase A
    # Utilizing simulated annealing to work towards global minimum
    # See https://cp-algorithms.com/num_methods/simulated_annealing.html
    while temp > temp_min:
        grad = calc_grad_arr(points, bonded_p, lone_const)

        # We want to attempt to apply the negative gradient AND some noise (in around 1/5 of points) to the
        # current configuration, then check if it will accept using the PAF (probability acceptance function).
        next_pts = points - STEP * grad

        noise_i = np.random.choice(total_p, NOISE_AMT, replace=False)
        noise = gen_points(NOISE_AMT)
        next_pts[noise_i] += NOISE_AMP * (temp / temp_i) * noise

        next_e = calc_energy_state(next_pts, bonded_p, lone_const)

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
        grad = calc_grad_arr(points, bonded_p, lone_const)
        next_pts = points - STEP * grad
        next_e = calc_energy_state(next_pts, bonded_p, lone_const)

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
            STEP = np.clip(STEP, 1e-5 / np.sqrt(total_p), 0.5 / np.sqrt(total_p))
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
        w = weights_row(bonded_p, total_p, i, lone_const)

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
            STEP = np.clip(STEP, 1e-5 / np.sqrt(total_p), 0.5 / np.sqrt(total_p))
            accepted = 0

    total_steps += steps
    end = time.time()

    # Print results to terminal console
    print([total_p, best_e, end - start, total_steps])
    return points


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


def calc_grad_arr(
    pts: np.ndarray, bonded_points: int, lone_const: np.float64
) -> np.ndarray:
    t = pts[:, 0]
    p = pts[:, 1]
    dt = t[:, np.newaxis] - t[np.newaxis, :]
    weight = weights(pts, bonded_points, lone_const)

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


def calc_energy_state(
    pts: np.ndarray, bonded_points: int, lone_const: np.float64
) -> np.double:
    # Only need the upper triangle of matrix so distances aren't counted duplicate
    # Where i < j, dist_arr[i][j] is kept
    dist = calc_dist_arr(pts)
    weight = weights(pts, bonded_points, lone_const)
    upper_mask = np.triu(dist > 0, k=1)

    # Divide the upper array by the lone pair energy constants (if there are lone pairs)
    # Since the reciprocal of this value will be taken, it will essentially be "multiplied"

    return np.sum(weight[upper_mask] / dist[upper_mask])


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


def weights(pts: np.ndarray, bonded_points: int, lone_const: np.float64) -> np.ndarray:
    dim = pts.shape[0]
    weight = np.ones((dim, dim))
    weight[bonded_points:, :] *= lone_const
    weight[:, bonded_points:] *= lone_const
    return weight


def weights_row(
    bonded_points: int, total: int, i: int, lone_const: np.float64
) -> np.ndarray:
    weight = np.ones(total)
    weight[bonded_points:] *= lone_const
    if i >= bonded_points:
        weight *= lone_const
    return weight


def paf(e: float, e_n: float, t: int) -> bool:
    if e_n < e:
        return True

    return np.random.random() <= np.exp((e - e_n) / t)


if __name__ == '__main__':

    def get_angle(pts: np.ndarray) -> np.float64:
        rect_1 = np.array(
            [
                np.cos(pts[0][1]) * np.sin(pts[0][0]),
                np.sin(pts[0][1]) * np.sin(pts[0][0]),
                np.cos(pts[0][0]),
            ]
        )
        rect_2 = np.array(
            [
                np.cos(pts[1][1]) * np.sin(pts[1][0]),
                np.sin(pts[1][1]) * np.sin(pts[1][0]),
                np.cos(pts[1][0]),
            ]
        )
        return np.arccos(np.dot(rect_1, rect_2))

    def calc_angles(pts: np.ndarray) -> np.ndarray:
        t, p = pts[:, :, 0], pts[:, :, 1]
        rects = np.stack(
            [np.cos(t) * np.sin(p), np.sin(t) * np.sin(p), np.cos(p)], axis=-1
        )

        dots = np.einsum('nik,njk->nij', rects, rects)
        return np.arccos(dots[:, 0, 1])

    expansion: np.float64 = 0.0
    # Using water's bond angle, from experiments, of 104.5 degrees
    # Specific to 10 decimal places
    for i in range(10):
        with ProcessPoolExecutor(max_workers=10) as executor:
            procs = {
                j: executor.submit(
                    find_points, 2, 2, 10000, expansion + j * (10 ** (-i))
                )
                for j in range(10)
            }
            results = {}
            for j, p in procs.items():
                results[j] = p.result()

        pts = np.array([results[j] for j in range(10)])
        # Calculate bond angle of first and second point
        angles = calc_angles(pts)
        valid = angles[angles >= np.deg2rad(104.5)]
        next_digit = np.where(angles == valid.min())[0][0]

        expansion += next_digit * (10 ** (-i))

    # 1.244217559

    # Using ammonia's bond angle, from experiments, of 107 degrees
    # Specific to 10 decimal places
    # for i in range(10):
    #     with ProcessPoolExecutor(max_workers=10) as executor:
    #         procs = {
    #             j: executor.submit(
    #                 find_points, 3, 1, 10000, expansion + j * (10 ** (-i))
    #             )
    #             for j in range(10)
    #         }
    #         results = {}
    #         for j, p in procs.items():
    #             results[j] = p.result()

    #     pts = np.array([results[j] for j in range(10)])
    #     # Calculate bond angle of first and second point
    #     angles = calc_angles(pts)
    #     valid = angles[angles >= np.deg2rad(107)]
    #     next_digit = np.where(angles == valid.min())[0][0]

    #     expansion += next_digit * (10 ** (-i))

    # 1.220931786

    print(expansion)
