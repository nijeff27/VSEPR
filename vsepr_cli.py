"""
Command-line interface for generating VSEPR configurations. Uses little GPU and can run multiple processes at once.
"""

import threading
import time

import numpy as np

RNG = np.random.default_rng()
LONE_CONSTANT = 1


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
    temp_min=0.00001,
    decay=0.999,
    bonded_points_arr: np.ndarray = None,
    lone_points_arr: np.ndarray = None,
):
    bonded_points = gen_points(bonded_p)
    lone_points = gen_points(lone_p)

    total_p = bonded_p + lone_p
    start = time.time()
    temp_i = temp

    STEP = 0.3 / np.sqrt(total_p)
    NOISE_AMP = 0.1 / np.sqrt(total_p)
    NOISE_AMT = (total_p // 5) + 1

    best_e = float('inf')
    best_pts: np.ndarray = []
    curr_e = calc_energy_state(bonded_points, lone_points)
    total_steps = 0
    steps = 0
    accepted = 0

    # Phase A
    # Utilizing simulated annealing to work towards global minimum
    # See https://cp-algorithms.com/num_methods/simulated_annealing.html
    while temp > temp_min:
        pts = self.points.copy()
        grad = self.calc_grad_arr(pts)

        # We want to attempt to apply the negative gradient AND some noise (in around 1/5 of points) to the
        # current configuration, then check if it will accept using the PAF (probability acceptance function).
        next_pts = pts - STEP * grad

        noise_i = np.random.choice(n_p, NOISE_AMT, replace=False)
        noise = self.gen_points(NOISE_AMT)
        next_pts[noise_i] += NOISE_AMP * (temp / temp_i) * noise

        next_e = self.calc_energy_state(next_pts)

        if self.paf(curr_e, next_e, temp):
            self.points = next_pts
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
            STEP = np.clip(STEP, 1e-4 / np.sqrt(n_p), 1.0 / np.sqrt(n_p))
            accepted = 0

        if steps % UPDATE_INTERVAL == 0:
            gui.Application.instance.post_to_main_thread(
                self.window, self.update_point_meshes
            )
            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda t=temp, e=curr_e, s=steps: self.log_info(
                    f'[A] Step {s}: T = {t:.6f} E = {e:.6f}'
                ),
            )

    self.points = best_pts
    gui.Application.instance.post_to_main_thread(self.window, self.update_point_meshes)

    # Phase O
    # Gradient descent applied to all points
    temp = temp_i
    STEP = 0.1 / np.sqrt(n_p)
    total_steps += steps
    steps = 0
    accepted = 0

    while temp > temp_min:
        pts = self.points.copy()
        grad = self.calc_grad_arr(pts)

        next_pts = pts - STEP * grad

        next_e = self.calc_energy_state(next_pts)

        if next_e < best_e:
            self.points = next_pts
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
            STEP = np.clip(STEP, 1e-5 / np.sqrt(n_p), 0.5 / np.sqrt(n_p))
            accepted = 0

        if steps % UPDATE_INTERVAL == 0:
            gui.Application.instance.post_to_main_thread(
                self.window, self.update_point_meshes
            )
            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda t=temp, e=best_e, s=steps: self.log_info(
                    f'[R] Step {s}: T = {t:.6f} E = {e:.6f}'
                ),
            )

    # Phase R
    # Negative gradient applied to one point only to further refine energy level
    temp = temp_i
    STEP = 0.05 / np.sqrt(n_p)
    total_steps += steps
    steps = 0
    accepted = 0

    # First calculate the distance array once, then update the array incrementally to keep a O(n) complexity
    dist_arr = self.calc_dist_arr(self.points)

    while temp > temp_min:
        i = np.random.randint(n_p)

        old_dist = dist_arr[i, :].copy()
        grad = self.calc_grad_row(self.points, old_dist, i)

        old_pts = self.points[i, :].copy()
        self.points[i, 0] = self.points[i, 0] - STEP * grad[0]
        self.points[i, 1] = self.points[i, 1] - STEP * grad[1]

        # Instead of calculating the total energy of the system, calculate only the energy potential of
        # the old set of points and the new set of points, and compare those potentials
        new_dist = self.calc_dist_row(self.points, i)
        old_energy = np.sum(1.0 / old_dist[old_dist > 0])
        new_energy = np.sum(1.0 / new_dist[new_dist > 0])
        next_e = best_e - old_energy + new_energy

        if next_e < best_e:
            best_e = next_e
            accepted += 1
            # Apply the change to the dist array
            dist_arr[i, :] = new_dist
            dist_arr[:, i] = new_dist
        else:
            # Revert the previous change
            self.points[i, :] = old_pts

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
            STEP = np.clip(STEP, 1e-5 / np.sqrt(n_p), 0.5 / np.sqrt(n_p))
            accepted = 0

        if steps % UPDATE_INTERVAL == 0:
            gui.Application.instance.post_to_main_thread(
                self.window, self.update_point_meshes
            )
            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda t=temp, e=best_e, s=steps: self.log_info(
                    f'[O] Step {s}: T = {t:.6f} E = {e:.6f}'
                ),
            )

    gui.Application.instance.post_to_main_thread(
        self.window,
        lambda: self.log_info(f'Done in {steps} steps. Best E={best_e:.6f}'),
    )
    gui.Application.instance.post_to_main_thread(
        self.window, lambda: setattr(self.run_button, 'enabled', True)
    )

    total_steps += steps
    end = time.time()

    # Print results to terminal console
    print([n_p, best_e, end - start, total_steps])


def run():
    # To be run on a separate thread
    # Run the optimization loop in a separate thread to keep the GUI from freezing
    threading.Thread(target=annealing, daemon=True).start()


def calc_dist_arr(bonded_pts: np.ndarray, lone_pts: np.ndarray):
    pts = np.concat([bonded_pts, lone_pts], axis=0)
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


def calc_grad_arr(bonded_pts: np.ndarray, lone_pts: np.ndarray) -> np.ndarray:
    pts: np.ndarray = np.concat(bonded_pts, lone_pts, axis=0)
    t = pts[:, 0]
    p = pts[:, 1]
    dt = t[:, np.newaxis] - t[np.newaxis, :]
    weight = weights(bonded_pts, lone_pts)

    den = np.clip(calc_dist_arr(pts), 0, None) ** 1.5
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


def calc_energy_state(bonded_pts: np.ndarray, lone_pts: np.ndarray) -> np.double:
    # Only need the upper triangle of matrix so distances aren't counted duplicate
    # Where i < j, dist_arr[i][j] is kept
    dist = calc_dist_arr(bonded_pts, lone_pts)
    weight = weights(bonded_pts, lone_pts)
    upper_mask = np.triu(dist > 0, k=1)

    # Divide the upper array by the lone pair energy constants (if there are lone pairs)
    # Since the reciprocal of this value will be taken, it will essentially be "multiplied"

    return np.sum(weight[upper_mask] / dist[upper_mask])


def calc_dist_row(bonded_pts: np.ndarray, lone_pts: np.ndarray, i: int):
    pts: np.ndarray = np.concat(bonded_pts, lone_pts, axis=0)
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
    bonded_pts: np.ndarray, lone_pts: np.ndarray, dist_row: np.ndarray, i: int
):
    pts: np.ndarray = np.concat(bonded_pts, lone_pts, axis=0)
    t_i, p_i = pts[i, 0], pts[i, 1]
    dt = t_i - pts[:, 0]
    den = dist_row**3

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_den = np.where(den > 0, 1.0 / den, 0.0)

    d_t = np.sum(-np.sin(p_i) * np.sin(pts[:, 1]) * np.sin(dt) * inv_den)
    d_p = np.sum(
        (np.cos(p_i) * np.sin(pts[:, 1]) * np.cos(dt) - np.sin(p_i) * np.cos(pts[:, 1]))
        * inv_den
    )

    return np.array([d_t, d_p])


def paf(self, e: float, e_n: float, t: int) -> bool:
    if e_n < e:
        return True

    return np.random.random() <= np.exp((e - e_n) / t)


if __name__ == '__main__':
    pass


def weights(bonded_pts: np.ndarray, lone_pts: np.ndarray) -> np.ndarray:
    total = bonded_pts.shape[0] + lone_pts.shape[0]
    bonded_total = bonded_pts.shape[0]
    weight = np.ones((total, total))
    weight[bonded_total:, :] *= LONE_CONSTANT
    weight[:, bonded_total:] *= LONE_CONSTANT
    return weight
