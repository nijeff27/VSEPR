import threading
import time

import numpy as np
import open3d.geometry as geometry
import open3d.utility as utility
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

LONE_CONSTANT = 1.23257467


def gen_points(amt: int):
    # Generate points in rectangular coordinates to prevent point clustering caused by spherical coordinates
    rng = np.random.default_rng()
    pts_rect = rng.uniform(-1.0, 1.0, (amt, 3))
    pts_rect = pts_rect / np.linalg.norm(pts_rect, axis=1, keepdims=True)

    # Now convert those coordinates to spherical coordinates (theta, phi)
    t = np.arctan2(pts_rect[:, 1], pts_rect[:, 0])
    p = np.arccos(np.clip(pts_rect[:, 2], -1, 1))

    return np.column_stack([t, p])


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
    # Only need the upper triangle of matrix so distances aren't counted duplicate
    # Where i < j, dist_arr[i][j] is kept
    dist = calc_dist_arr(pts)
    weight = weights(pts, bonded_points)
    upper_mask = np.triu(dist > 0, k=1)
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


def paf(e: float, e_n: float, t: int) -> bool:
    if e_n < e:
        return True

    return np.random.random() <= np.exp((e - e_n) / t)


class VSEPR:
    def __init__(self):
        self.radius = 1.0
        self.points: np.ndarray = []
        self.info_lines: list[str] = []
        self.logs_are_saved = False
        self.last_click_time = 0.0
        self.recentered = False

        self.window = gui.Application.instance.create_window(
            'VSEPR Simulation', 1200, 700
        )

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        self.panel = gui.Vert(10, gui.Margins(10, 10, 10, 10))
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self.scene()
        self.inputs()

        # Prevents weird GUI glitches with the default mouse behavior in Open3D
        self.scene_widget.set_on_mouse(self.mouse_event)

    def _on_layout(self, layout_context):
        """Generates the layout for the window.

        Args:
            layout_context: The layout context
        """

        r = self.window.content_rect
        self.scene_widget.frame = gui.Rect(r.x, r.y, int(r.width * 0.7), r.height)
        self.panel.frame = gui.Rect(
            self.scene_widget.frame.get_right(), r.y, int(r.width * 0.3), r.height
        )

    def scene(self):
        self.scene_widget.scene.set_background([0.0, 0.0, 0.0, 1.0])

        # TriangleMesh won't be rendered into actual solid face
        # And instead LineSet will be utilized to create a wireframe sphere (visual preference)
        sphere_mesh = geometry.TriangleMesh.create_sphere(
            radius=self.radius, resolution=20
        )
        sphere_mesh.compute_vertex_normals()

        line_set = geometry.LineSet.create_from_triangle_mesh(sphere_mesh)
        lines = np.asarray(line_set.lines)
        lines_v_norms = np.asarray(sphere_mesh.vertex_normals)

        # Create the illusion of front and back by making the front "side" of sphere white, but the back "side" a dim blue-gray
        line_colors = []
        for seg in lines:
            norm_mid = (lines_v_norms[seg[0]] + lines_v_norms[seg[1]]) / 2.0
            norm = np.linalg.norm(norm_mid)

            if norm > 1e-9:
                norm_mid /= norm

            # Where's the z-component?
            t = (float(np.dot(norm_mid, [0.0, 0.0, 1.0])) + 1.0) / 2.0
            line_colors.append([0.15 + 0.85 * t, 0.18 + 0.82 * t, 0.28 + 0.72 * t])

        line_set.colors = utility.Vector3dVector(np.array(line_colors))
        self.wire_mesh = line_set
        self.base_mesh = sphere_mesh
        self.wire_material = rendering.MaterialRecord()
        self.wire_material.shader = 'unlitLine'
        self.wire_material.line_width = 1.4

        self.scene_widget.scene.add_geometry(
            'wireframe', self.wire_mesh, self.wire_material
        )

        self.scene_widget.scene.scene.enable_indirect_light(True)
        self.scene_widget.scene.scene.set_indirect_light_intensity(20000)
        self.scene_widget.scene.scene.enable_sun_light(True)
        self.scene_widget.scene.scene.set_sun_light(
            direction=[0, -1, -1], color=[1.0, 1.0, 1.0], intensity=100000
        )

        # Center of mass - used as a point of reference when the sphere is disabled
        self.center_mesh = geometry.TriangleMesh.create_sphere(radius=0.03)
        self.center_mesh.compute_vertex_normals()
        self.center_mesh.paint_uniform_color([0.0, 0.6, 1.0])
        self.center_material = rendering.MaterialRecord()
        self.center_material.shader = 'defaultLit'
        self.center_material.base_color = [0.0, 0.6, 1.0, 1.0]
        self.scene_widget.scene.add_geometry(
            'center', self.center_mesh, self.center_material
        )

        # Points (electrons), rendered as small spheres like the center of mass
        self.point_meshes = []
        self.point_material = rendering.MaterialRecord()
        self.point_material.shader = 'defaultLit'
        self.point_material.base_color = [0.4, 0.85, 1.0, 1.0]

        self._bounds = sphere_mesh.get_axis_aligned_bounding_box()
        self._sphere_center = self._bounds.get_center()
        self.scene_widget.setup_camera(60, self._bounds, self._sphere_center)

        self.orbit_r = self._bounds.get_extent().max() * 1.2

    def inputs(self):
        # Right-hand side of GUI
        # All sliders are accompanied with textboxes for both drag and type input

        def make_int_row(lo, hi, default):
            row = gui.Horiz(5)
            slider = gui.Slider(gui.Slider.INT)
            slider.set_limits(lo, hi)
            slider.int_value = default
            txt = gui.NumberEdit(gui.NumberEdit.INT)
            txt.set_limits(lo, hi)
            txt.set_value(default)
            slider.set_on_value_changed(txt.set_value)
            txt.set_on_value_changed(
                lambda val, s=slider: setattr(s, 'int_value', int(val))
            )
            row.add_child(slider)
            row.add_child(txt)
            return row, slider

        def make_double_row(lo, hi, default, precision=3):
            row = gui.Horiz(5)
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(lo, hi)
            slider.double_value = default
            txt = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            txt.set_limits(lo, hi)
            txt.set_value(default)
            txt.decimal_precision = precision
            slider.set_on_value_changed(txt.set_value)
            txt.set_on_value_changed(
                lambda val, s=slider: setattr(s, 'double_value', np.double(val))
            )
            row.add_child(slider)
            row.add_child(txt)
            return row, slider

        bonded_row, self.bonded_points = make_int_row(2, 200, 2)
        lone_row, self.lone_points = make_int_row(0, 50, 0)
        temp_row, self.temperature = make_int_row(1000, 20000, 5000)
        decay_row, self.decay = make_double_row(0.5, 0.999, 0.998)

        self.run_button = gui.Button('Run')
        self.run_button.set_on_clicked(self.run)
        reset_btn = gui.Button('Reset Camera')
        reset_btn.set_on_clicked(self.recenter_camera)
        self.checkbox = gui.Checkbox('Show Sphere')
        self.checkbox.checked = True
        self.checkbox.set_on_checked(self._toggle_sphere)

        info_btn = gui.Button('Show Info')
        info_btn.set_on_clicked(self.show_info)
        save_btn = gui.Button('Save Info To File')
        save_btn.set_on_clicked(self.save_info)
        self.info_status = gui.Label('')
        self.info_label = gui.Label("Press 'Show Info'...")

        for el in [
            gui.Label('Generation Settings'),
            gui.Label(''),
            gui.Label('Bonded pairs'),
            bonded_row,
            gui.Label('Lone pairs'),
            lone_row,
            gui.Label('Temperature'),
            temp_row,
            gui.Label('Decay (alpha)'),
            decay_row,
            self.run_button,
            reset_btn,
            self.checkbox,
            info_btn,
            save_btn,
            gui.Label('Runtime Info:'),
            self.info_status,
            self.info_label,
        ]:
            self.panel.add_child(el)

    def init_point_meshes(self):
        # Remove the existing rendered points
        for i in range(len(self.point_meshes)):
            self.scene_widget.scene.remove_geometry(f'pt_{i}')
        self.point_meshes = []

        # Then, re-add the updated points to GUI
        for i, (t, p) in enumerate(self.points):
            mesh = geometry.TriangleMesh.create_sphere(radius=0.03)
            mesh.compute_vertex_normals()
            # Different color based on bonded/lone pair
            mesh.paint_uniform_color(
                [0.4, 0.85, 1.0]
                if i < self.bonded_points.int_value
                else [1.0, 0.6, 0.1]
            )
            self.scene_widget.scene.add_geometry(f'pt_{i}', mesh, self.point_material)
            self.point_meshes.append(mesh)

    def update_point_meshes(self):
        for i, (t, p) in enumerate(self.points):
            # Calculate transform from current mesh center to new position
            T = np.eye(4)
            T[:3, 3] = [np.cos(t) * np.sin(p), np.sin(t) * np.sin(p), np.cos(p)]
            self.scene_widget.scene.set_geometry_transform(f'pt_{i}', T)

    def clear(self):
        self.points = []
        self.update_point_meshes()

    def recenter_camera(self):
        m = self.scene_widget.scene.camera.get_model_matrix()
        pos = m[:3, 3]
        up = m[:3, 1]

        # Normalize the current eye direction and place the eye at exactly the original orbit radius
        view_dir = pos - self._sphere_center
        dist = np.linalg.norm(view_dir)

        # Don't continue if distance isn't sufficient
        if dist < 1e-6:
            return

        corrected_pos = self._sphere_center + (view_dir / dist) * self.orbit_r
        self.scene_widget.look_at(self._sphere_center, corrected_pos, up)

    def mouse_event(self, event):
        """Mouse event for this project. When a double click happens in Open3D, by default
        it relocates the camera based on the click location. This method calls the recenter_camera()
        method when this happens to prevent this discrepancy.

        Args:
            event:

        Returns:
            EventCallbackresult: IGNORED, signifying the completion of the event.
        """
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            now = time.time()
            if now - self.last_click_time < 0.35:
                self.recentered = True
                self.last_click_time = 0.0
            else:
                self.last_click_time = now
            return gui.Widget.EventCallbackResult.IGNORED

        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self.recentered:
                self.recentered = False
                gui.Application.instance.post_to_main_thread(
                    self.window, self.recenter_camera
                )
            return gui.Widget.EventCallbackResult.IGNORED

        return gui.Widget.EventCallbackResult.IGNORED

    def show_info(self):
        self.info_label.text = '\n'.join(self.info_lines)

    def log_info(self, *lines: tuple[str]):
        for li in lines:
            self.info_lines.append(li)

    def save_info(self):
        if self.logs_are_saved:
            self.info_status.text = '! Already saved.'
            return

        txt = self.info_label.text

        if txt == "Press 'Show Info'...":
            self.info_status.text = "! Press 'Show Info' first!"
            return

        try:
            with open('vsepr_info.txt', 'a') as file:
                file.write(txt + '\n')
            self.logs_are_saved = True
            self.info_status.text = 'Appended and saved logs to vsepr_logs.txt.'
        except Exception as e:
            self.info_status.text = f'!! Error: {e}'

    def clear_info(self):
        self.info_lines = []
        self.info_label.text = "Press 'Show Info'..."
        self.logs_are_saved = False

    def run(self):
        # User should not be able to run the program again during algorithm run
        self.run_button.enabled = False

        # Reset program
        self.clear()
        self.clear_info()

        # To be run on a separate thread
        def find_points():
            start = time.time()

            bonded_p: int = self.bonded_points.int_value
            lone_p: int = self.lone_points.int_value
            total_p: int = bonded_p + lone_p
            temp: float = self.temperature.int_value * total_p
            temp_i = temp
            temp_min: float = 0.0001
            decay: np.double = self.decay.double_value

            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda b=bonded_p, l=lone_p, t=temp, d=decay: self.log_info(
                    f'Bonded:\t\t\t{b}',
                    f'Lone:\t\t\t{l}',
                    f'Temperature:\t{t}',
                    f'Decay:\t\t\t{d}',
                ),
            )

            STEP = 0.3 / np.sqrt(total_p)
            NOISE_AMP = 0.1 / np.sqrt(total_p)
            NOISE_AMT = (total_p // 5) + 1
            UPDATE_INTERVAL = max(500, total_p * 10)

            pts_ready = threading.Event()

            def set_points():
                self.points = gen_points(total_p)
                self.init_point_meshes()
                self.update_point_meshes()
                pts_ready.set()

            # Add random particles as a starting point
            gui.Application.instance.post_to_main_thread(self.window, set_points)
            pts_ready.wait()

            best_e = float('inf')
            best_pts: np.ndarray = self.points.copy()
            curr_e = calc_energy_state(self.points, bonded_p)
            total_steps = 0
            steps = 0
            accepted = 0

            # Phase A
            # Utilizing simulated annealing to work towards global minimum
            # See https://cp-algorithms.com/num_methods/simulated_annealing.html
            while temp > temp_min:
                pts = self.points.copy()
                grad = calc_grad_arr(pts, bonded_p)

                # We want to attempt to apply the negative gradient AND some noise (in around 1/5 of points) to the
                # current configuration, then check if it will accept using the PAF (probability acceptance function).
                next_pts = pts - STEP * grad

                noise_i = np.random.choice(total_p, NOISE_AMT, replace=False)
                noise = gen_points(NOISE_AMT)
                next_pts[noise_i] += NOISE_AMP * (temp / temp_i) * noise

                next_e = calc_energy_state(next_pts, bonded_p)

                if paf(curr_e, next_e, temp):
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
                    STEP = np.clip(
                        STEP, 1e-4 / np.sqrt(total_p), 1.0 / np.sqrt(total_p)
                    )
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
            gui.Application.instance.post_to_main_thread(
                self.window, self.update_point_meshes
            )

            # Phase O
            # Gradient descent applied to all points
            temp = temp_i
            STEP = 0.1 / np.sqrt(total_p)
            total_steps += steps
            steps = 0
            accepted = 0

            while temp > temp_min:
                pts = self.points.copy()
                grad = calc_grad_arr(pts, bonded_p)
                next_pts = pts - STEP * grad
                next_e = calc_energy_state(next_pts, bonded_p)

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
                    STEP = np.clip(
                        STEP, 1e-5 / np.sqrt(total_p), 0.5 / np.sqrt(total_p)
                    )
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

            # Phase R
            # Negative gradient applied to one point only to further refine energy level
            temp = temp_i
            STEP = 0.05 / np.sqrt(total_p)
            total_steps += steps
            steps = 0
            accepted = 0

            # First calculate the distance array once, then update the array incrementally to keep a O(n) complexity
            dist_arr = calc_dist_arr(self.points)

            while temp > temp_min:
                i = np.random.randint(total_p)
                w = weights_row(bonded_p, total_p, i)

                old_dist = dist_arr[i, :].copy()
                grad = calc_grad_row(self.points, bonded_p, old_dist, i, w)

                old_pts = self.points[i, :].copy()
                self.points[i, 0] = self.points[i, 0] - STEP * grad[0]
                self.points[i, 1] = self.points[i, 1] - STEP * grad[1]

                # Instead of calculating the total energy of the system, calculate only the energy potential of
                # the old set of points and the new set of points, and compare those potentials
                new_dist = calc_dist_row(self.points, i)
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
                    STEP = np.clip(
                        STEP, 1e-5 / np.sqrt(total_p), 0.5 / np.sqrt(total_p)
                    )
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
            print([total_p, best_e, end - start, total_steps])

        # Run the optimization loop in a separate thread to keep the GUI from freezing
        threading.Thread(target=find_points, daemon=True).start()

    def _toggle_sphere(self, checked):
        if checked:
            self.scene_widget.scene.add_geometry(
                'wireframe', self.wire_mesh, self.wire_material
            )
        else:
            self.scene_widget.scene.remove_geometry('wireframe')


if __name__ == '__main__':
    gui.Application.instance.initialize()
    app = VSEPR()
    gui.Application.instance.run()
