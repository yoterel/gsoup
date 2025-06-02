from .transforms import (
    compose_rt,
    invert_rigid,
)
from .rasterize import (
    project_points,
    should_cull_tri,
    get_silhouette_edges,
    get_visible_edges,
)
from .core import (
    to_34,
    to_44,
    to_hom,
    swap_columns,
)
from .geometry_basic import (
    project_point_to_segment,
    line_line_intersection,
    is_on_segment,
)
import numpy as np
import cv2
from scipy.optimize import least_squares
import time


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


class EdgeTracker:
    """
    a base class for model-based edge tracking using a 3D model and a camera.
    """

    def __init__(self, params, name):
        self.params = params
        self.name = name

    def get_name(self):
        return self.name

    def track(self, frame, **kwargs):
        # should track the object in the frame by estimating the object pose
        # returns the estimated o2w pose as a (4, 4) matrix, or None if failed to find a pose
        pass

    def get_results(self):
        # should return a tuple (o2w, correspondences)
        # n = number of frames
        # o2w is a (n, 3, 4) object to world transforms as computed by the tracker per frame
        # correspondences is a list of m tuples (p, pnorm, q, qnorm) correspondences where m is not necessarily the same for each entry
        # p, q are 2D points and pnorm, qnorm are their 2D normals
        # length of list must be n-1 (the last frame doesn't need any correspondences)
        # if no such info exists, should return None
        pass

    def se3_to_SE3(self, pose):
        """
        Convert (n, 6) se3 6-vector to (n, 4, 4) pose matrix.
        """
        SE3s = np.empty((pose.shape[0], 4, 4), dtype=pose.dtype)
        for i in range(len(pose)):
            rvec = pose[i, 0:3]
            tvec = pose[i, 3:6]
            rmat, _ = cv2.Rodrigues(rvec)
            SE3 = compose_rt(rmat[None, ...], tvec[None, ...], square=True)[0]
            SE3s[i] = SE3
        return SE3s

    def SE3_to_se3(self, o2c):
        """
        Convert (n, 4, 4) pose matrix to se3 6-vector.
        """
        # initial_rmat = np.eye(3).astype(np.float32)  # rand_rot_mat[0]
        # initial_tvec = np.zeros(3).astype(np.float32)  # rand_rot_trans[0]
        se3s = np.empty((o2c.shape[0], 6), dtype=o2c.dtype)
        for i in range(len(o2c)):
            rmat = o2c[i, :3, :3]
            rvec, _ = cv2.Rodrigues(rmat)
            tvec = o2c[i, :3, -1]
            se3 = np.concatenate((rvec[:, 0], tvec), axis=-1)
            se3s[i] = se3
        return se3s

    def o2w_to_o2c(self, w2c, o2w):
        """
        Convert object-to-world pose to object-to-camera pose.
        """
        # o2c = w2c @ o2w
        o2c = to_44(w2c) @ to_44(o2w)
        return o2c

    def o2c_to_o2w(self, o2c):
        """
        Convert object-to-camera pose to object-to-world pose.
        """
        # o2w = c2w @ o2c
        c2w = invert_rigid(to_44(self.params["w2c"]), square=True)
        o2w = to_34(c2w @ o2c)
        return o2w

    def sample_edge_points(self, o2w, w2c, n_samples=10, silhouette=True):
        """
        Sample points uniformly along edges of the model.
        o2w: 3x4 object-to-world pose matrix.
        w2c: 3x4 world-to-camera pose matrix.
        n_samples: number of points to sample along each edge.
        silhouette: if True, sample points on silhouette edges. Otherwise, sample visible edges.
        """
        cur_v = (o2w @ to_hom(self.params["v"]).T).T
        if silhouette:
            edge_mask = get_silhouette_edges(
                cur_v,
                self.params["f"],
                self.params["e2f"],
                self.params["K"],
                w2c,
            )
        else:
            edge_mask = get_visible_edges(
                cur_v,
                self.params["f"],
                self.params["e2f"],
                self.params["K"],
                w2c,
            )
        points = self.params["v"][self.params["e"][edge_mask]].transpose(
            1, 0, 2
        )  # (2, n_edges, 3)
        p0, p1 = points[0], points[1]
        # now interpolate between every pair of points n_samples
        t = np.linspace(0, 1, n_samples, dtype=np.float32)[:, None]
        points = (p0[:, None, :] * (1 - t[None, :, :])) + (
            p1[:, None, :] * t[None, :, :]
        )  # (n_edges, n_samples, 3)
        points = points.reshape(-1, 3)  # (n_edges * n_samples, 3)
        return points


class NaiveEdgeTracker(EdgeTracker):
    def __init__(self, init_o2w, params, name):
        super().__init__(params, name)
        # Assumes an initial object pose is given in world coordinates.
        # Also assumes camera is known.
        # define a virtual camera pose (using the gt object pose) which will be optimized
        # in reality, it is o2c, but for this optimization, it plays the role of w2c
        # init_o2c = self.o2w_to_o2c(self.params["w2c"], init_o2w)
        self.cur_pose = self.SE3_to_se3(init_o2w[None, ...])[0]
        self.poses = []
        self.errors = []
        self.corres = []

    def get_silhouette(self, frame):
        contours, _ = cv2.findContours(
            frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c = max(contours, key=cv2.contourArea)
        return c[:, 0, :].astype(np.float32)

    def find_best_point(self, p, observed_silhouette, use_normals=False):
        """
        finds the "best" point on the observed silhouette (2d contour) given p
        :param p: the 2D point (2,) used as query or the point and its normal (2, 2)
        :param observed_silouette: the (n, 2) endpoints of the contour
        :param use_normals: if True, will use normal for search (more robust usually), expects p.shape==(2,2)
        :return: (best_point, best_tangent): the best point on the contour, and optionally its tangent
        """
        best_tangent = np.array([np.inf, np.inf])
        # ab are the segments of the contour (target)
        a = observed_silhouette
        b = np.roll(observed_silhouette, -1, axis=0)
        if use_normals:
            assert p.shape == (2, 2)
            p, n = p[0], p[1]
            # with normal (find the closest point on the contour, but only in the direction of the normal)
            intersections = line_line_intersection(
                p[None, :], p[None, :] + n[None, :], a, b
            )
            mask = is_on_segment(intersections, a, b, collinear=True)
            if not mask.any():
                return best_tangent, best_tangent
            intersections = intersections[mask]
            d = np.linalg.norm(intersections - p, axis=-1)
            best_point = intersections[d.argmin()]
        else:
            # without normal (find closest point on contour)
            candidate = project_point_to_segment(p, a, b)
            d = np.linalg.norm(p - candidate, axis=-1)
            best_point = candidate[d.argmin()]
        return best_point, best_tangent

    def track(self, frames, frame_index, **kwargs):
        updated_pose = False
        # time1 = time.time()
        error_per_view = np.full(frames.shape[0], 1e3).astype(np.float32)
        cur_poses = np.vstack((self.cur_pose, self.cur_pose))
        for i, frame in enumerate(frames):
            observed_silhouette = self.get_silhouette(frame)
            # print("contour:", time.time() - time1)
            for ii in range(self.params["iters_per_frame"]):
                # save current pose for display / analysis
                if i == 0 and self.params["save_results"]:
                    self.poses.append(self.cur_pose.copy())
                # time1 = time.time()
                # convert current o2c to o2w
                o2w = self.se3_to_SE3(cur_poses[i : i + 1])
                o2c = self.o2w_to_o2c(self.params["w2c"][i : i + 1], o2w)
                cur_cam_pose = self.SE3_to_se3(o2c)[0]
                # o2c = self.se3_to_SE3(self.cur_cam_pose[None, :])
                # o2w = self.o2c_to_o2w(o2c)
                # print("se32SE3:", time.time() - time1)
                # time1 = time.time()
                sample_edge_points = self.sample_edge_points(
                    to_34(o2w[0]),
                    self.params["w2c"][i],
                    n_samples=self.params["sample_per_edge"],
                )
                correspondence_mask = np.zeros((len(sample_edge_points)), dtype=bool)
                # print("sample:", time.time() - time1)
                # time1 = time.time()
                projected_edge_points = project_points(
                    sample_edge_points,
                    self.params["K"],
                    to_34(o2c[0]),
                )[:, :2]
                # compute image space normals for each edge point
                if self.params["use_normals"]:
                    edge_points_normals = np.zeros_like(projected_edge_points)
                    points_per_edge = projected_edge_points.reshape(
                        -1, self.params["sample_per_edge"], 2
                    )
                    tangents = np.diff(points_per_edge, axis=1).mean(
                        axis=1
                    )  # (n_edges, 2)
                    normals = np.array([-tangents[:, 1], tangents[:, 0]]).T
                    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
                    normals = normals.repeat(self.params["sample_per_edge"], axis=0)
                # print("project:", time.time() - time1)
                # time1 = time.time()
                correspondences = []
                for iii, p in enumerate(projected_edge_points):
                    if self.params["use_normals"]:
                        cur_normal = normals[iii]
                        cur_p = np.array([p, cur_normal])
                    else:
                        cur_normal = np.array([np.inf, np.inf])
                        cur_p = p
                    q, tangent = self.find_best_point(
                        cur_p,
                        observed_silhouette,
                        use_normals=self.params["use_normals"],
                    )
                    if np.linalg.norm(p - q) < self.params["corres_dist_threshold"]:
                        correspondences.append(
                            np.array([p, cur_normal, q, tangent], dtype=np.float32)
                        )
                        correspondence_mask[iii] = True
                if self.params["save_results"]:
                    self.corres.append(correspondences)
                if np.count_nonzero(correspondence_mask) < 3:
                    print("tracker not enough correspondences: {}, {}".format(i, ii))
                    continue
                # print("correspondences:", time.time() - time1)
                # time1 = time.time()
                # print("append:", time.time() - time1)
                # time1 = time.time()
                target_points = np.array(correspondences)[:, 2]
                target_points = np.ascontiguousarray(target_points)
                mean_error = np.mean(
                    np.linalg.norm(
                        projected_edge_points[correspondence_mask] - target_points,
                        axis=1,
                    )
                )
                error_per_view[i] = mean_error
                if mean_error > 5.0:
                    print("tracker: mean error large {:.02f}".format(mean_error))
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        sample_edge_points[correspondence_mask],
                        target_points,
                        self.params["K"],
                        self.params["dist_coeffs"],
                        cur_cam_pose[0:3],
                        cur_cam_pose[3:6],
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )  # finds best fitting w2c
                    if not success:
                        print("tracker error: solvePnP returned false")
                        continue
                    # update pose for next iteration
                    o2c = self.se3_to_SE3(
                        np.concatenate((rvec, tvec), axis=-1)[None, ...]
                    )
                    o2w = self.o2c_to_o2w(o2c)
                    cur_poses[i] = self.SE3_to_se3(o2w)[0]
                    updated_pose = True
                except cv2.error as e:
                    breakpoint()
                    print("tracker error: solvePnP threw cv2.error")
                    print(e)
                    # usually happens if inputs are not contiguous or float32
                    continue
            # print("solve:", time.time() - time1)
        self.cur_pose = cur_poses[error_per_view.argmin()]
        self.errors.append(error_per_view)
        if updated_pose:
            # weights = (1 - (error_per_view**2 / error_per_view.max())) ** 2
            # o2c = self.se3_to_SE3(self.cur_cam_pose)
            # o2w = self.o2c_to_o2w(o2c)
            # o2w = (error_per_view @ o2w).squeeze()
            # o2w = o2w[error_per_view.argmin()]
            return self.se3_to_SE3(self.cur_pose[None, ...])[0]
        else:
            return None

    def get_results(self):
        if self.params["save_results"]:
            # push last pose into poses
            self.poses.append(self.cur_pose.copy())
            # convert optimzed camera poses to object poses
            object_poses = []
            for i in range(0, len(self.poses)):
                o2w = to_34(self.se3_to_SE3(self.poses[i][None, ...]))
                object_poses.append(o2w[0])
            corres = np.array(self.corres, dtype=object).reshape(
                -1, self.params["n_views"], self.params["iters_per_frame"]
            )
            corres = corres.transpose(1, 0, 2).reshape(self.params["n_views"], -1)
            return np.array(object_poses), corres
        else:
            return None, None


class NaiveEdgeTracker_2(EdgeTracker):
    def __init__(self, init_o2w, params, name):
        super().__init__(params, name)
        # Assumes an initial object pose is given in world coordinates.
        # Also assumes camera is known.
        # define a virtual camera pose (using the gt object pose) which will be optimized
        # in reality, it is o2c, but for this optimization, it plays the role of w2c
        init_o2c = self.o2w_to_o2c(self.params["w2c"], init_o2w)
        self.cur_cam_pose = self.SE3_to_se3(init_o2c)
        # sample points along edges of model in the init pose
        self.sampled_points = self.sample_edge_points(init_o2w, n_samples=10)
        self.poses = []
        self.corres = []

    def track_project_points(self, points, params, K):
        w2c = to_34(self.se3_to_SE3(params))
        # rmat, _ = cv2.Rodrigues(rvec)
        # w2c = compose_rt(rmat[None, ...], tvec[None, ...])
        projected_points = project_points(points, K, w2c)[:, :2]
        return projected_points

    def get_aprox_correspondence(
        self, params, sampled_points, dist_transform, labels, K
    ):
        projected_pts = self.track_project_points(sampled_points, params, K)
        corres = []
        for pt in projected_pts:
            x, y = np.round(pt).astype(np.uint32)
            # Penalize if projected point is outside the image bounds
            if not (
                x < 0
                or x >= dist_transform.shape[1]
                or y < 0
                or y >= dist_transform.shape[0]
            ):
                corres.append((pt, labels[y, x]))
        return np.array(corres)

    def cost_function(self, params, sampled_points, dist_transform, labels, K):
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
        dist_transform, labels = cv2.distanceTransformWithLabels(
            inv_edges_img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        yx = np.where(inv_edges_img == 0)
        actual_labels = np.array(yx).T[labels - 1]
        actual_labels = swap_columns(actual_labels, 0, 1)
        # Optimize the pose parameters to align the model's projected edges with the image edges.
        for ii in range(self.params["iters_per_frame"]):
            # save for display / analysis
            self.poses.append(self.cur_cam_pose)
            corres = self.get_aprox_correspondence(
                self.cur_cam_pose,
                self.sampled_points,
                dist_transform,
                actual_labels,
                self.params["K"],
            )
            self.corres.append(corres)
            res = least_squares(  # perform iteration
                self.cost_function,
                self.cur_cam_pose,
                args=(self.sampled_points, dist_transform, labels, self.params["K"]),
                verbose=0,
            )
            self.cur_cam_pose = res.x  # update pose for next iteration
            # and update sampled points
            o2c = self.se3_to_SE3(self.cur_cam_pose)
            o2w = self.o2c_to_o2w(o2c)
            self.sampled_points = self.sample_edge_points(o2w, n_samples=10)

    def get_current_pose(self):
        return self.cur_cam_pose

    def get_results(self):
        # push last pose into poses
        self.poses.append(self.cur_cam_pose)
        # convert optimzed camera poses to object poses
        object_poses = []
        for i in range(len(self.poses)):
            o2c = self.se3_to_SE3(self.poses[i])
            o2w = self.o2c_to_o2w(o2c)
            object_poses.append(o2w)
        return np.array(object_poses), self.corres


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
