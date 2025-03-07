import numpy as np
from .transforms import compose_rt, invert_rigid
from .rasterize import project_points, should_cull_tri, get_silhouette_edges
from .core import to_34, to_44, to_hom
from .geometry_basic import project_point_to_segment
import cv2
from scipy.optimize import least_squares


class KalmanFilter(object):
    # x0 - initial guess of the state vector
    # P0 - initial guess of the covariance matrix of the state estimation error
    # A,B,C - system matrices describing the system model
    # Q - covariance matrix of the process noise
    # R - covariance matrix of the measurement noise
    def __init__(self, x0, P0, A, B, C, Q, R):
        # initialize vectors and matrices
        self.x0 = x0
        self.P0 = P0
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        ####### members for debuggin #######
        self.currentTimeStep = 0
        self.estimates_aposteriori = []
        self.estimates_aposteriori.append(x0)
        self.estimates_apriori = []
        self.estimationErrorCovarianceMatricesAposteriori = []
        self.estimationErrorCovarianceMatricesAposteriori.append(P0)
        self.estimationErrorCovarianceMatricesApriori = []
        self.gainMatrices = []
        self.errors = []

    def set_matrices(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

    # propagate x_{k-1}^{+} to compute x_{k}^{-}
    # this function also propagates P_{k-1}^{+} to compute P_{k}^{-}
    def predict(self, input_value, prev_xk_plus, prev_pk_plus):
        xk_minus = self.A * prev_xk_plus + self.B * input_value
        pk_minus = self.A * prev_pk_plus * (self.A.T) + self.Q
        return xk_minus, pk_minus

    # predict for k steps ahead (usefull for reducing latency in real systems)
    def predict_multi(self, inputValue, steps, prev_xk_plus):
        xk_minus = self.A * prev_xk_plus + self.B * inputValue
        for _ in range(steps - 1):
            xk_minus = self.A * xk_minus + self.B * inputValue
        return xk_minus

    # for debugging purposes, perdicts and stores the states as members of the class
    def predict_and_store(self, input_value):
        xk_minus = (
            self.A * self.estimates_aposteriori[self.currentTimeStep]
            + self.B * input_value
        )
        pk_minus = (
            self.A
            * self.estimationErrorCovarianceMatricesAposteriori[self.currentTimeStep]
            * (self.A.T)
            + self.Q
        )

        self.estimates_apriori.append(xk_minus)
        self.estimationErrorCovarianceMatricesApriori.append(pk_minus)

        self.currentTimeStep = self.currentTimeStep + 1

    # given a new measurement, perform the update state to fuse the measurement with the prediction
    def update(self, current_measurement, xk_minus, pk_minus):
        # gain matrix
        Kk = (
            pk_minus
            * (self.C.T)
            * np.linalg.inv(self.R + self.C * pk_minus * (self.C.T))
        )
        # prediction error
        error_k = current_measurement - self.C * xk_minus
        # a posteriori estimate
        xk_plus = xk_minus + Kk * error_k
        # a posteriori matrix update
        IminusKkC = np.matrix(np.eye(self.x0.shape[0])) - Kk * self.C
        pk_plus = IminusKkC * pk_minus * (IminusKkC.T) + Kk * (self.R) * (Kk.T)
        # return updated state
        return xk_plus, pk_plus

    # perform the update step and also stores the states as members of the class
    def update_and_store(self, current_measurement):
        Kk = (
            self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep - 1]
            * (self.C.T)
            * np.linalg.inv(
                self.R
                + self.C
                * self.estimationErrorCovarianceMatricesApriori[
                    self.currentTimeStep - 1
                ]
                * (self.C.T)
            )
        )

        # prediction error
        error_k = (
            current_measurement
            - self.C * self.estimates_apriori[self.currentTimeStep - 1]
        )
        # a posteriori estimate
        xk_plus = self.estimates_apriori[self.currentTimeStep - 1] + Kk * error_k

        # a posteriori matrix update
        IminusKkC = np.matrix(np.eye(self.x0.shape[0])) - Kk * self.C
        Pk_plus = IminusKkC * self.estimationErrorCovarianceMatricesApriori[
            self.currentTimeStep - 1
        ] * (IminusKkC.T) + Kk * (self.R) * (Kk.T)

        # update the lists that store the vectors and matrices
        self.gainMatrices.append(Kk)
        self.errors.append(error_k)
        self.estimates_aposteriori.append(xk_plus)
        self.estimationErrorCovarianceMatricesAposteriori.append(Pk_plus)


class HullTracker:
    def __init__(self, init_pose, params):
        self.params = params
        self.dist_coeff = np.zeros((5, 1)).astype(np.float32)
        # define the initial camera pose (using the gt object pose).
        # o2c = w2c @ o2w
        o2c = to_44(params["w2c"]) @ to_44(init_pose)
        # initial_rmat = np.eye(3).astype(np.float32)  # rand_rot_mat[0]
        o2c = to_34(o2c)
        initial_rmat = o2c[:3, :3]
        initial_rvec, _ = cv2.Rodrigues(initial_rmat)
        # initial_tvec = np.zeros(3).astype(np.float32)  # rand_rot_trans[0]
        initial_tvec = o2c[:, -1]
        # Assume an initial pose is given [rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z]
        self.poses = []
        self.corres = []
        self.cur_pose = np.concatenate((initial_rvec[:, 0], initial_tvec), axis=-1)

    def get_convex_hull_ideal(self, V):
        projected = project_points(V, self.params["K"], self.params["w2c"])[:, :2]
        hull = cv2.convexHull(projected)
        return hull[:, 0, :]

    def get_convex_hull(self, frame):
        # todo
        x = cv2.findContours(frame)
        hull = cv2.convexHull(x)
        return hull

    def extract_silhouette(self, projected_points):
        """
        For simplicity, we use the convex hull as the silhouette.
        Returns:
        silhouette: (M,2) array of 2D points on the hull.
        hull_indices: indices into the projected_points array.
        """
        pts = np.array(projected_points, dtype=np.float32).reshape(-1, 1, 2)
        hull_indices = cv2.convexHull(pts, returnPoints=False)
        hull_indices = hull_indices.flatten()
        silhouette = pts[hull_indices].reshape(-1, 2)
        return silhouette, hull_indices

    def find_closest_point(self, p, observed_silhouette):
        """
        For a given 2D point p, find the closest point on the observed silhouette.
        Returns:
        best_point: the closest point on an edge,
        best_tangent: the unit tangent of that edge.
        """
        best_distance = float("inf")
        best_point = None
        best_tangent = None
        N = observed_silhouette.shape[0]
        for i in range(N):
            a = observed_silhouette[i]
            b = observed_silhouette[(i + 1) % N]
            candidate = project_point_to_segment(p, a, b)
            d = np.linalg.norm(p - candidate)
            if d < best_distance:
                best_distance = d
                best_point = candidate
                tangent = b - a
                norm_t = np.linalg.norm(tangent)
                best_tangent = tangent / norm_t if norm_t > 0 else np.array([0, 0])
        return best_point, best_tangent

    def track(self, frame, cur_v, **kwargs):
        observed_silhouette = self.get_convex_hull_ideal(cur_v)
        # breakpoint()
        # observed_silhouette = self.get_convex_hull(frame)
        for i in range(self.params["iters_per_frame"]):
            # convert current o2c to o2w
            rvec = self.cur_pose[0:3]
            tvec = self.cur_pose[3:6]
            rmat, _ = cv2.Rodrigues(rvec)
            o2c = compose_rt(rmat[None, ...], tvec[None, :], square=True)[0]
            c2w = invert_rigid(to_44(self.params["w2c"])[None, ...], square=True)[0]
            o2w = to_34(c2w @ o2c)
            # apply o2w to object
            V_test = (o2w @ to_hom(self.params["v"]).T).T
            projected_points = project_points(
                V_test, self.params["K"], self.params["w2c"]
            )[:, :2]
            proj_silhouette, silhouette_indices = self.extract_silhouette(
                projected_points
            )
            correspondences = []
            for p in proj_silhouette:
                q, tangent = self.find_closest_point(p, observed_silhouette)
                correspondences.append((p, (q, tangent)))
            target_points = np.array([c[1][0] for c in correspondences])
            mean_error = np.mean(
                np.linalg.norm(proj_silhouette - target_points, axis=1)
            )
            # print(mean_error)
            success, rvec, tvec = cv2.solvePnP(
                self.params["v"][
                    silhouette_indices
                ],  # should be filtered to only use correspondences
                target_points,
                self.params["K"],
                self.dist_coeff,
                self.cur_pose[0:3],
                self.cur_pose[3:6],
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )  # finds best fitting w2c
            # breakpoint()
            if not success:
                raise RuntimeError("solvePnP failed during refinement.")
            self.cur_pose = np.concatenate((rvec, tvec), axis=-1)
            self.poses.append(self.cur_pose)
            self.corres.append(correspondences)

    def get_results(self):
        o2ws = []
        for pose in self.poses:
            rvec = pose[0:3]
            tvec = pose[3:6]
            rmat, _ = cv2.Rodrigues(rvec)
            o2c = compose_rt(rmat[None, ...], tvec[None, :], square=True)[0]
            c2w = invert_rigid(to_44(self.params["w2c"])[None, ...], square=True)[0]
            o2w = to_34(c2w @ o2c)
            o2ws.append(o2w)
        return o2ws, self.corres


class NaiveEdgeTracker:
    def __init__(self, init_o2w, params):
        # Assumes an initial object pose is given in world coordinates.
        # Also assumes camera is known.
        self.params = params
        # define a virtual camera pose (using the gt object pose) which will be optimized
        # in reality, it is o2c, but for this optimization, it plays the role of w2c
        init_o2c = self.o2w_to_o2c(self.params["w2c"], init_o2w)
        self.cur_cam_pose = self.SE3_to_se3(init_o2c)
        # sample points along edges of model in the init pose
        self.sampled_points = self.sample_edge_points(init_o2w, n_samples=10)
        self.poses = []

    def se3_to_SE3(self, pose):
        rvec = pose[0:3]
        tvec = pose[3:6]
        rmat, _ = cv2.Rodrigues(rvec)
        o2c = compose_rt(rmat[None, ...], tvec[None, ...], square=True)[0]
        return o2c

    def SE3_to_se3(self, o2c):
        # initial_rmat = np.eye(3).astype(np.float32)  # rand_rot_mat[0]
        # initial_tvec = np.zeros(3).astype(np.float32)  # rand_rot_trans[0]
        rmat = o2c[:3, :3]
        rvec, _ = cv2.Rodrigues(rmat)
        tvec = o2c[:, -1]
        return np.concatenate((rvec[:, 0], tvec), axis=-1)

    def o2w_to_o2c(self, w2c, o2w):
        # o2c = w2c @ o2w
        o2c = to_44(w2c) @ to_44(o2w)
        return o2c

    def o2c_to_o2w(self, o2c):
        # o2w = c2w @ o2c
        c2w = invert_rigid(to_44(self.params["w2c"])[None, ...], square=True)[0]
        o2w = to_34(c2w @ o2c)
        return o2w

    def sample_edge_points(self, o2w, n_samples=10):
        """
        Sample points uniformly along silhouette edges of the model.
        """
        cur_v = (o2w @ to_hom(self.params["v"]).T).T
        sil_edge_mask = get_silhouette_edges(
            cur_v,
            self.params["f"],
            self.params["e2f"],
            self.params["K"],
            self.params["w2c"],
        )
        points = self.params["v"][self.params["e"][sil_edge_mask]].transpose(
            1, 0, 2
        )  # (2, n_edges, 3)
        p0, p1 = points[0], points[1]
        # now interpolate between every pair of points n_samples
        t = np.linspace(0, 1, n_samples)[:, None]
        points = (p0[:, None, :] * (1 - t[None, :, :])) + (
            p1[:, None, :] * t[None, :, :]
        )
        points = points.reshape(-1, 3)
        return points
        # cam_pose = invert_rigid(to_44(self.params["w2c"])[None, :])[
        #     0, :3, -1
        # ]  # get cam pose
        # for i, (a, b) in enumerate(self.params["e"]):
        #     v0, v1, v2, _ = self.params["v"][self.params["f"][self.params["e2f"][i][0]]]
        #     cond1 = should_cull_tri(v0, v1, v2, cam_pose)
        #     v0, v1, v2, _ = self.params["v"][self.params["f"][self.params["e2f"][i][1]]]
        #     cond2 = should_cull_tri(v0, v1, v2, cam_pose)
        #     if cond1 and cond2:
        #         continue
        #     p0, p1 = self.params["v"][a], self.params["v"][b]
        #     # Sample n_samples points (including endpoints)
        #     for t in np.linspace(0, 1, n_samples):
        #         pt = (1 - t) * p0 + t * p1
        #         points.append(pt)
        # return np.array(points, dtype=np.float32)

    def track_project_points(self, points, params, K):
        w2c = to_34(self.se3_to_SE3(params))
        # rmat, _ = cv2.Rodrigues(rvec)
        # w2c = compose_rt(rmat[None, ...], tvec[None, ...])
        projected_points = project_points(points, K, w2c)[:, :2]
        return projected_points

    def cost_function(self, params, sampled_points, dist_transform, K):
        """
        Cost function that returns the distance from each projected sample point
        to the nearest image edge (via the distance transform).
        params: 6-vector (3 for rotation [Rodrigues], 3 for translation)
        """
        # rvec = params[0:3]
        # tvec = params[3:6]
        projected_pts = self.track_project_points(sampled_points, params, K)[:, :2]
        cost = []
        for pt in projected_pts:
            x, y = np.round(pt).astype(np.uint32)
            # Penalize if projected point is outside the image bounds
            if (
                x < 0
                or x >= dist_transform.shape[1]
                or y < 0
                or y >= dist_transform.shape[0]
            ):
                cost.append(np.array(100.0))
            else:
                cost.append(dist_transform[y, x])
        return np.array(cost).squeeze()

    def track(self, frame, **kwargs):
        # Preprocess the frame: grayscale and edge detection
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges_img = cv2.Canny(gray, 50, 150)
        # Invert edge image so that edges become zeros and background is nonzero.
        # _, inv_edges_img = cv2.threshold(edges_img, 50, 255, cv2.THRESH_BINARY_INV)
        inv_edges_img = 255 - ((frame > 0) * 255).astype(np.uint8)
        # Then compute the distance transform (distance from each pixel to the nearest edge).
        dist_transform = cv2.distanceTransform(inv_edges_img, cv2.DIST_L2, 5)
        # Optimize the pose parameters to align the model's projected edges with the image edges.
        for ii in range(self.params["iters_per_frame"]):
            res = least_squares(
                self.cost_function,
                self.cur_cam_pose,
                args=(self.sampled_points, dist_transform, self.params["K"]),
                verbose=0,
            )
            self.cur_cam_pose = res.x  # update pose for next iteration
            # and update sampled points
            o2c = self.se3_to_SE3(self.cur_cam_pose)
            o2w = self.o2c_to_o2w(o2c)
            self.sampled_points = self.sample_edge_points(o2w, n_samples=10)
            # save for display / analysis
            self.poses.append(self.cur_cam_pose)

    def get_results(self):
        # convert optimzed camera poses to object poses
        object_poses = []
        for i in range(len(self.poses)):
            o2c = self.se3_to_SE3(self.poses[i])
            o2w = self.o2c_to_o2w(o2c)
            object_poses.append(o2w)
        return np.array(object_poses), None


class RobustEdgeTracker:

    def __init__(self, init_pose, params):
        self.params = params
        # define the initial camera pose (using the gt object pose).
        # o2c = w2c @ o2w
        o2c = to_44(params["w2c"]) @ to_44(init_pose)
        # initial_rmat = np.eye(3).astype(np.float32)  # rand_rot_mat[0]
        initial_rmat = o2c[:3, :3]
        initial_rvec, _ = cv2.Rodrigues(initial_rmat)
        # initial_tvec = np.zeros(3).astype(np.float32)  # rand_rot_trans[0]
        initial_tvec = o2c[:, -1]
        # Get our 3D model and sample points along its edges
        self.sampled_points = self.sample_edge_points(n_samples=10)
        self.poses = []
        self.cur_pose = np.concatenate((initial_rvec[:, 0], initial_tvec), axis=-1)

    def skew(self, v):
        """Return the skew-symmetric matrix of a 3-vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def exp_se3(self, xi):
        """
        Exponential map from se(3) (6-d twist) to SE(3) (4x4 homogeneous transform).
        xi: a 6-element vector (first 3: translation, last 3: rotation)
        """
        v = xi[:3]
        omega = xi[3:]
        theta = np.linalg.norm(omega)
        T = np.eye(4)
        if theta < 1e-8:
            T[:3, 3] = v
            return T
        omega_hat = omega / theta
        K = skew(omega_hat)
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / (theta**2) * (K @ K)
        )
        t = V @ v
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def project_point(self, P, X):
        """
        Projects a 3D point X (in homogeneous coordinates) using projection matrix P.
        Returns the 2D image coordinates.
        """
        x = P @ X
        return x[:2] / x[2]

    def compute_cube_edge_samples_and_normals(
        self, pose, cube_vertices, cube_edges, camera_matrix
    ):
        """
        Given a cube model (vertices and edges) and the current pose,
        compute sample 3D points (midpoints of each edge) and their
        corresponding 2D edge normals.

        pose: 4x4 pose matrix.
        cube_vertices: (8,3) numpy array.
        cube_edges: list of tuples (i, j) representing edges.
        camera_matrix: 3x3 intrinsic matrix.

        Returns:
        points_3d: (n,3) array of sampled 3D points.
        normals_2d: (n,2) array of corresponding 2D normals.
        """
        R = pose[:3, :3]
        t = pose[:3, 3:4]
        extrinsic = np.hstack((R, t))
        P = camera_matrix @ extrinsic
        points_3d = []
        normals_2d = []
        for edge in cube_edges:
            i, j = edge
            v1 = cube_vertices[i]
            v2 = cube_vertices[j]
            midpoint = (v1 + v2) / 2.0
            # Project the endpoints into the image:
            proj1 = project_point(P, np.hstack((v1, 1)))
            proj2 = project_point(P, np.hstack((v2, 1)))
            tangent = proj2 - proj1
            if np.linalg.norm(tangent) < 1e-6:
                continue
            # Compute 2D normal: perpendicular to the tangent.
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / np.linalg.norm(normal)
            points_3d.append(midpoint)
            normals_2d.append(normal)
        return np.array(points_3d), np.array(normals_2d)

    # ------------------------------------
    # Helper function for bilinear interpolation
    # ------------------------------------
    def get_pixel(self, image, pt):
        """
        Returns the interpolated pixel intensity at a given (x, y) position.
        Performs bilinear interpolation.

        image: 2D numpy array (grayscale)
        pt: 2-element array (x, y)
        """
        x, y = pt
        # Calculate floor and ceil coordinates:
        x0 = int(np.floor(x))
        x1 = x0 + 1
        y0 = int(np.floor(y))
        y1 = y0 + 1

        # Clip to image boundaries:
        x0 = np.clip(x0, 0, image.shape[1] - 1)
        x1 = np.clip(x1, 0, image.shape[1] - 1)
        y0 = np.clip(y0, 0, image.shape[0] - 1)
        y1 = np.clip(y1, 0, image.shape[0] - 1)

        # Fractional parts:
        dx = x - x0
        dy = y - y0

        I00 = image[y0, x0]
        I01 = image[y0, x1]
        I10 = image[y1, x0]
        I11 = image[y1, x1]

        # Bilinear interpolation:
        I0 = I00 * (1 - dx) + I01 * dx
        I1 = I10 * (1 - dx) + I11 * dx
        I = I0 * (1 - dy) + I1 * dy
        return I

    def search_edge(
        self,
        image,
        initial_pt,
        normal,
        method="method1",
        threshold=50,
        search_range=range(-5, 6),
    ):
        """
        Performs a 1D search along the edge normal starting at initial_pt.

        For method1, returns the first candidate with intensity difference above threshold.
        For method2, returns the candidate with the maximum intensity difference.

        image: 2D numpy array (grayscale)
        initial_pt: starting 2D point (x, y)
        normal: 2D unit vector (edge normal)
        threshold: intensity difference threshold (for method1)
        search_range: iterable of offsets along the normal.
        """
        best_pt = np.array(initial_pt)
        if method == "method1":
            for j in search_range:
                candidate_pt = np.array(initial_pt) + j * np.array(normal)
                if (
                    abs(get_pixel(image, candidate_pt) - get_pixel(image, initial_pt))
                    > threshold
                ):
                    best_pt = candidate_pt
                    break
            return best_pt
        elif method == "method2":
            best_score = -np.inf
            for j in search_range:
                candidate_pt = np.array(initial_pt) + j * np.array(normal)
                score = abs(
                    get_pixel(image, candidate_pt) - get_pixel(image, initial_pt)
                )
                if score > best_score:
                    best_score = score
                    best_pt = candidate_pt
            return best_pt
        else:
            return best_pt

    # -------------------------------
    # Method 1: Lie Algebra Based Tracking
    # -------------------------------
    def method1_update(
        self,
        pose,
        points_3d,
        normals_2d,
        image,
        camera_matrix,
        weight_func,
        search_method="method1",
    ):
        """
        One-shot pose update using a Lie Algebra-based approach.

        pose: current 4x4 pose matrix.
        points_3d: (n,3) array of 3D model sample points.
        normals_2d: (n,2) array of corresponding 2D edge normals.
        image: current grayscale image.
        camera_matrix: 3x3 intrinsic parameters.
        weight_func: robust weighting function.
        search_method: determines the 1D search strategy ("method1" uses threshold search).

        Returns: updated 4x4 pose matrix.
        """
        n = points_3d.shape[0]
        R = pose[:3, :3]
        t = pose[:3, 3:4]
        extrinsic = np.hstack((R, t))
        P = camera_matrix @ extrinsic  # 3x4 projection matrix

        errors = []
        L_rows = []

        for i in range(n):
            X = np.hstack((points_3d[i], 1))
            proj = project_point(P, X)
            # Use the computed 2D normal to search along:
            correspondence = search_edge(
                image,
                proj,
                normals_2d[i],
                method=search_method,
                threshold=50,
                search_range=range(-5, 6),
            )
            error = np.dot(correspondence - proj, normals_2d[i])
            errors.append(error)

            # Compute a simplified interaction vector:
            X_cam = R @ points_3d[i].reshape(3, 1) + t  # 3D point in camera frame
            Z = X_cam[2, 0]
            x, y = proj
            # Using the normal's components directly in the simplified model:
            cos_theta, sin_theta = normals_2d[i]
            L_dp = np.array(
                [
                    -cos_theta / Z,
                    -sin_theta / Z,
                    (x * cos_theta + y * sin_theta) / Z,
                    x * y * cos_theta + (1 + y**2) * sin_theta,
                    -(1 + x**2) * cos_theta - x * y * sin_theta,
                    y * cos_theta - x * sin_theta,
                ]
            )
            L_rows.append(L_dp)

        errors = np.array(errors)
        L = np.array(L_rows)
        weights = weight_func(errors)
        W = np.diag(weights)

        A = L.T @ W @ L
        b = -L.T @ W @ errors
        try:
            alpha = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            alpha = np.linalg.pinv(A) @ b

        delta_pose = exp_se3(alpha)
        new_pose = delta_pose @ pose  # left multiplication update
        return new_pose

    # -------------------------------
    # Method 2: Virtual Visual Servoing (VVS)
    # -------------------------------
    def method2_update(
        self,
        pose,
        points_3d,
        normals_2d,
        image,
        camera_matrix,
        weight_func,
        lambda_gain=0.7,
        num_iterations=5,
        search_method="method2",
    ):
        """
        Iterative pose update using virtual visual servoing.

        pose: current 4x4 pose matrix.
        points_3d: (n,3) array of 3D model sample points.
        normals_2d: (n,2) array of corresponding 2D edge normals.
        image: current grayscale image.
        camera_matrix: 3x3 intrinsic parameters.
        weight_func: robust weighting function.
        lambda_gain: control gain.
        num_iterations: iterations per frame.
        search_method: "method2" uses maximum likelihood search.

        Returns: updated 4x4 pose matrix.
        """
        current_pose = pose.copy()
        for _ in range(num_iterations):
            n = points_3d.shape[0]
            R = current_pose[:3, :3]
            t = current_pose[:3, 3:4]
            extrinsic = np.hstack((R, t))
            P = camera_matrix @ extrinsic

            errors = []
            L_rows = []
            for i in range(n):
                X = np.hstack((points_3d[i], 1))
                proj = project_point(P, X)
                correspondence = search_edge(
                    image,
                    proj,
                    normals_2d[i],
                    method=search_method,
                    threshold=50,
                    search_range=range(-5, 6),
                )
                error = np.dot(correspondence - proj, normals_2d[i])
                errors.append(error)
                X_cam = R @ points_3d[i].reshape(3, 1) + t
                Z = X_cam[2, 0]
                x, y = proj
                cos_theta, sin_theta = normals_2d[i]
                L_dp = np.array(
                    [
                        -cos_theta / Z,
                        -sin_theta / Z,
                        (x * cos_theta + y * sin_theta) / Z,
                        x * y * cos_theta + (1 + y**2) * sin_theta,
                        -(1 + x**2) * cos_theta - x * y * sin_theta,
                        y * cos_theta - x * sin_theta,
                    ]
                )
                L_rows.append(L_dp)

            errors = np.array(errors)
            L = np.array(L_rows)
            weights = weight_func(errors)
            W = np.diag(weights)
            A = L.T @ W @ L
            b = -L.T @ W @ errors
            try:
                v = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                v = np.linalg.pinv(A) @ b
            v = lambda_gain * v
            delta_pose = exp_se3(v)
            current_pose = delta_pose @ current_pose
        return current_pose

    # -------------------------------
    # Robust weight function (Tukey-like)
    # -------------------------------
    def tukey_weight(self, errors, c=1.0):
        """
        Compute robust weights using a Tukey biweight function.
        """
        weights = np.ones_like(errors)
        for i, e in enumerate(errors):
            if np.abs(e) < c:
                weights[i] = (1 - (e / c) ** 2) ** 2
            else:
                weights[i] = 0
        return weights


# # -------------------------------
# # Example usage with synthetic data
# # -------------------------------
# if __name__ == "__main__":
#     # Initial pose (identity)
#     pose = np.eye(4)

#     # Camera intrinsics (example values)
#     camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

#     # Create a synthetic grayscale image.
#     # Here we generate a simple gradient image with an artificial edge.
#     image = np.tile(np.linspace(0, 255, 640), (480, 1)).astype(np.float32)
#     image[240:, :] = image[240:, :] + 50
#     image = np.clip(image, 0, 255)

#     # Compute sample points and 2D normals from the cube model using the current pose.
#     points_3d, normals_2d = compute_cube_edge_samples_and_normals(
#         pose, cube_vertices, cube_edges, camera_matrix
#     )

#     # Run one update using Method 1:
#     pose1 = method1_update(
#         pose,
#         points_3d,
#         normals_2d,
#         image,
#         camera_matrix,
#         tukey_weight,
#         search_method="method1",
#     )

#     # Run iterative update using Method 2:
#     pose2 = method2_update(
#         pose,
#         points_3d,
#         normals_2d,
#         image,
#         camera_matrix,
#         tukey_weight,
#         lambda_gain=0.7,
#         num_iterations=5,
#         search_method="method2",
#     )

#     print("Method 1 updated pose:")
#     print(pose1)
#     print("\nMethod 2 updated pose:")
#     print(pose2)
