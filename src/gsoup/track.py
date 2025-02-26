import numpy as np
from .transforms import compose_rt, invert_rigid
from .rasterize import project_points, should_cull_tri
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

    def get_convex_hull_ideal(self, V, K, w2c):
        projected = project_points(
            self.params["v"], self.params["K"], self.params["w2c"]
        )[:, :2]
        hull = cv2.convexHull(projected)
        return hull

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
        observed_silhouette = self.get_convex_hull_ideal(
            cur_v, params["K"], params["w2c"]
        )
        # observed_silhouette = self.get_convex_hull(frame)
        for iter in range(self.params["iters_per_frame"]):
            # convert current o2c to o2w
            rvec = self.cur_pose[0:3]
            tvec = self.cur_pose[3:6]
            rmat, _ = cv2.Rodrigues(rvec)
            o2c = compose_rt(rmat[None, ...], tvec[None, :, 0], square=True)[0]
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
            print(mean_error)
            success, rvec, tvec = cv2.solvePnP(
                V[silhouette_indices],  # should be filtered to only use correspondences
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
            self.cur_pose = np.concatenate((rvec[:, 0], tvec), axis=-1)
            self.poses.append(self.cur_pose)
            self.corres.append(correspondences)

    def get_results():
        o2ws = []
        for pose in self.poses:
            rvec = pose[0:3]
            tvec = pose[3:6]
            rmat, _ = cv2.Rodrigues(rvec)
            o2c = compose_rt(rmat[None, ...], tvec[None, :, 0], square=True)[0]
            c2w = invert_rigid(to_44(self.params["w2c"])[None, ...], square=True)[0]
            o2w = to_34(c2w @ o2c)
            o2ws.append(o2w)
        return o2ws, self.corres


class NaiveEdgeTracker:
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
        # breakpoint()
        # Assume an initial pose is given [rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z]
        self.poses = []
        self.cur_pose = np.concatenate((initial_rvec[:, 0], initial_tvec), axis=-1)

    def sample_edge_points(self, n_samples=10):
        """
        Sample points uniformly along each edge of the model.
        """
        points = []
        cam_pose = invert_rigid(to_44(self.params["w2c"])[None, :])[
            0, :3, -1
        ]  # get cam pose
        for i, (a, b) in enumerate(self.params["e"]):
            v0, v1, v2, _ = self.params["v"][self.params["f"][self.params["e2f"][i][0]]]
            cond1 = should_cull_tri(v0, v1, v2, cam_pose)
            v0, v1, v2, _ = self.params["v"][self.params["f"][self.params["e2f"][i][1]]]
            cond2 = should_cull_tri(v0, v1, v2, cam_pose)
            if cond1 and cond2:
                continue
            p0, p1 = self.params["v"][a], self.params["v"][b]
            # Sample n_samples points (including endpoints)
            for t in np.linspace(0, 1, n_samples):
                pt = (1 - t) * p0 + t * p1
                points.append(pt)
        return np.array(points, dtype=np.float32)

    def track_project_points(self, points, rvec, tvec, K):
        rmat, _ = cv2.Rodrigues(rvec)
        w2c = compose_rt(rmat[None, ...], tvec[None, ...])
        projected_points = project_points(points, K, w2c)[:, :2]
        return projected_points

    def cost_function(self, params, sampled_points, dist_transform, K):
        """
        Cost function that returns the distance from each projected sample point
        to the nearest image edge (via the distance transform).
        params: 6-vector (3 for rotation [Rodrigues], 3 for translation)
        """
        rvec = params[0:3]
        tvec = params[3:6]
        projected_pts = self.track_project_points(sampled_points, rvec, tvec, K)[:, :2]
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
                cost.append(np.array([100.0]))
            else:
                cost.append(dist_transform[y, x])
        return np.array(cost).squeeze()

    def track(self, frame):
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
                self.cur_pose,
                args=(self.sampled_points, dist_transform, self.params["K"]),
                verbose=0,
            )
            self.cur_pose = res.x  # update pose for next iteration
            self.poses.append(self.cur_pose)

    def get_results(self):
        object_poses = []
        for i in range(len(self.poses)):
            rvec = self.poses[i][0:3]
            rmat, _ = cv2.Rodrigues(rvec)
            tvec = self.poses[i][3:6]
            # breakpoint()
            o2c = compose_rt(rmat[None, ...], tvec[None, ...], square=True)[0]
            # c2o = gsoup.invert_rigid(o2c) # c2o = w2c @ o2w
            c2w = invert_rigid(to_44(self.params["w2c"])[None, ...], square=True)[0]
            o2w = to_34(c2w @ o2c)
            object_poses.append(o2w)
        return np.array(object_poses), None
