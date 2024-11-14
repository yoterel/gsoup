import numpy as np
import cv2
from .gsoup_io import save_image, save_images, load_images, load_image
from .transforms import compose_rt
from .core import to_8b, to_hom, swap_columns
from .image import change_brightness
from .geometry_basic import ray_ray_intersection, point_line_distance
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull


def blend_intensity_multi_projectors(forward_maps, fgs, proj_whs, mode="ij"):
    """
    given an image of the overlapping region of two or more projectors, and dense mappings between camera and all projectors pixels, computes an alpha mask per projector for seamless projection
    based on: "Multiprojector Displays using Camera-Based Registration".
    :param forward_maps: a list of forward maps each of (proj_hi x proj_wi x 2) uint32 corresponding to projectors i resolution
    :param fgs: a list of foreground masks each of (proj_hi x proj_wi) bool corresponding to projectors i resolution
    :proj_whs: a list of projector resolutions (proj_wi, proj_hi) as tuples
    :param mode: "xy" or "ij" depending on the order of the last channel of forward_map (see GrayCode.decode)
    :return: a list of alpha masks to use per projector i of size (proj_hi x proj_wi x 1) uint8
    """
    cam_wh = forward_maps[0].shape[:2]
    multi_proj_map = np.ones(
        (len(forward_maps), cam_wh[1], cam_wh[0], 1), dtype=np.float
    )
    for i in range(len(forward_maps)):
        points = np.where(fgs[i])
        convex_hull = ConvexHull(points)
        hull_points = [points[simplex] for simplex in convex_hull.simplices] + [
            points[convex_hull.simplices[0]]
        ]
        hull_edges = np.array(
            [[hull_points[i], hull_points[i + 1]] for i in range(len(hull_points) - 1)]
        )
        distances = np.empty(len(hull_edges), len(points))
        for j, edge in enumerate(hull_edges):
            distances[j] = point_line_distance(points, edge[0], edge[1])
        distances = distances.min(axis=0)
        multi_proj_map[i, points] = distances
    multi_proj_map = multi_proj_map / multi_proj_map.sum(axis=0, keepdims=True)
    multi_proj_map = to_8b(multi_proj_map)
    masks = []
    for fmap, fg, proj_wh in zip(forward_maps, fgs, proj_whs):
        bmap = compute_backward_map(proj_wh, fmap, fg, mode=mode)
        masks.append(multi_proj_map[bmap])
    return masks


def warp_image(backward_map, desired_image, cam_wh=None, mode="xy", output_path=None):
    """
    given a 2D dense map of corresponding pixels between projector and camera, computes a warped image such that if projected, the desired image appears when observed using camera
    :param backward_map: 2D dense mapping between pixels from projector 2 camera (proj_h x proj_w x 2) uint32
    :param desired_image: path to desired image from camera, or float np array channels last (cam_h x cam_w x 3) uint8
    :param cam_wh: device1 (width, height) as tuple, if not supplied assumes desired_image is in the correct dimensions in relation to backward_map
    :param mode: "xy" or "ij" depending on the last channel of backward_map
    :param output_path: path to save warped image to
    :return: warped image (proj_h x proj_w x 3) uint8
    """
    if backward_map.ndim != 3:
        raise ValueError("backward_map must be 3D")
    if backward_map.shape[-1] != 2:
        raise ValueError("backward_map must have shape (cam_h, cam_w, 2)")
    if type(desired_image) == np.ndarray:
        if desired_image.ndim != 3:
            raise ValueError("desired_image must be 3D")
    else:
        if cam_wh is not None:
            desired_image = load_image(desired_image, resize_wh=cam_wh)
        else:
            desired_image = load_image(desired_image)
    if mode == "xy":
        warpped = desired_image[(backward_map[..., 1], backward_map[..., 0])]
    elif mode == "ij":
        warpped = desired_image[(backward_map[..., 0], backward_map[..., 1])]
    else:
        raise ValueError("mode must be 'xy' or 'ij'")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(warpped, output_path)
    return warpped


def compute_backward_map(
    proj_wh,
    forward_map,
    foreground,
    mode="ij",
    interpolate=True,
    output_dir=None,
    debug=False,
):
    """
    computes the inverse map of forward_map by piece-wise interpolating a triangulated version of it (see GrayCode.decode to understand forward_map)
    :param proj_wh: projector (width, height) as a tuple
    :param forward_map: forward map as a numpy array of shape (height, width, 2) of type int32
    :param foreground: a boolean mask of shape (height, width) where True indicates a valid pixel in forward_map
    :param interpolate: if False, output will not be interpolated (will have holes...)
    :param mode: the last channel order of forward_map. either "xy" (width first) or "ij" (height first).
    :param output_dir: directory to save the inverse map to
    :param debug: if True, saves a visualization of the backward map to output_dir (R=X, G=Y, B=0) where X increases from left to right and Y increases from top to bottom
    :return: backward map as a numpy array of shape (projector_height, projector_width, 2) of type int32, last channel encodes (height, width).
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    data = np.argwhere(foreground)  # always ij
    points = forward_map[foreground]
    if interpolate:
        interp = LinearNDInterpolator(points, data, fill_value=0.0)
        X, Y = np.meshgrid(np.arange(proj_wh[0]), np.arange(proj_wh[1]))
        result = interp(X, Y).transpose(1, 0, 2)  # eww
    else:
        sparse_map = np.zeros((proj_wh[1], proj_wh[0], 2), dtype=np.uint32)
        if mode == "xy":
            sparse_map[(points[:, 1]), (points[:, 0])] = data
        elif mode == "ij":
            sparse_map[(points[:, 0]), (points[:, 1])] = data
        result = sparse_map
    if output_dir:
        np.save(Path(output_dir, "backward_map.npy"), result)
        if debug:
            result_normalized = result / np.array(
                [forward_map.shape[0], forward_map.shape[1]]
            )
            result_normalized[..., [0, 1]] = result_normalized[..., [1, 0]]
            result_normalized_8b = to_8b(result_normalized)
            result_normalized_8b_3c = np.concatenate(
                (result_normalized_8b, np.zeros_like(result_normalized_8b[..., :1])),
                axis=-1,
            )
            save_image(result_normalized_8b_3c, Path(output_dir, "backward_map.png"))
    return result.round().astype(np.uint32)


def naive_color_compensate(
    target_image,
    all_white_image,
    all_black_image,
    cam_width,
    cam_height,
    brightness_decrease=-127,
    projector_gamma=2.2,
    output_path=None,
    debug=False,
):
    """
    color compensate a projected image such that it appears closer to a target image from the perspective of a camera
    based on "Embedded entertainment with smart projectors".
    :param target_image the desired image path from the perspective of the camera
    :param all_white_image a path to picture taken by camera when projector had all pixels fully on (float32)
    :param all_black_image a path to picture taken by camera when projector had all pixels fully off (float32)
    :param cam_width camera image width
    :param cam_height camera image height
    :param brightness_decrease a hyper parameter controlling how much the total brightness is decreased. without this, the result is saturated because of dividing by small numbers
    :param projector_gamma under normal circumstances the projector does not actually output linear values, so we need to compensate for that
    :output_path if passed, result will be saved to this path
    :debug if true, will save debug info into output_path parent directory
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    target_image = load_image(
        target_image, to_float=True, resize_wh=(cam_width, cam_height)
    )[..., :3]
    target_image = change_brightness(target_image, brightness_decrease)
    if debug:
        save_image(target_image, Path(output_path.parent, "decrease_brightness.png"))
    all_white_image = load_image(
        all_white_image, to_float=True, resize_wh=(cam_width, cam_height)
    )
    all_black_image = load_image(
        all_black_image, to_float=True, resize_wh=(cam_width, cam_height)
    )
    compensated = (target_image - all_black_image) / all_white_image
    compensated = np.power(compensated, (1 / projector_gamma))
    compensated = np.nan_to_num(compensated, nan=0.0, posinf=0.0, neginf=0.0)
    compensated = np.clip(compensated, 0, 1)
    if output_path:
        save_image(compensated, output_path)
    return compensated


def calibrate_procam(
    proj_wh,
    capture_dir,
    chess_vert=10,
    chess_hori=7,
    bg_threshold=10,
    chess_block_size=10.0,
    projector_orientation="lower_half",
    verbose=True,
    output_dir=None,
    debug=False,
):
    """
    calibrates a projection-camera pair using local homographies
    based on "Simple, accurate, and robust projector-camera calibration".
    note1: the calibration poses some reasonable constraints on the projector-camera pair:
    1) projector is assumed to have no distortion, and a square pixel aspect ratio.
    2) camera is assumed to have a square pixel aspect ratio, and principle axis is assumed to be the center of the image.
    3) both elements are assumed to have no tangential distortion.
    note2: recommendations for calibration:
    1) use a chessboard with as many blocks as possible, but make sure the chessboard is fully visible in the camera image and in focus.
    2) capture sessions should span the whole image plane, and should be as diverse as possible in terms of poses (tilt the checkerboard !)
    3) when tilting the checkerboard, do not tilt it too much or the projector pixels will get smeared and the gray code decoding will produce large errors.
    4) place the checkeboard only in the working zone of the projector, i.e. the area where the projector is in focus.
    5) make sure the full dynamic range of the camera is used, i.e. the blackest black and whitest white should be visible in the captured images.
    6) attach the checkerboard to a flat surface, make sure the final pattern is as flat as possible. any bending will cause large errors.
    7) capture as many sessions as possible, if a session is not good (use debug=True to check), discard it.
    8) the RMS is only a rough estimate of the calibration quality, but if it is below 1, the calibration should be good enough (units are pixels).
    :param proj_wh projector resolution (width, height) as a tuple
    :param chess_vert when holding the chessboard in portrait mode, this is the number of internal "crossing points" of chessboard in vertical direction (i.e. where two white and two black squares meet)
    :param chess_hori when holding the chessboard in portrait mode, this is the number of internal "crossing points" of chessboard in horizontal direction (i.e. where two white and two black squares meet)
    :param capture_dir directory containing the captured images when gray code patterns were projected, assumes structure as follows:
        capture_dir
            - folder1
                - 0000.png
                - 0001.png
                - ...
            - folder2
                - 0000.png
                - ...
    :param bg_threshold threshold for detecting foreground
    :param chess_block_size size of blocks of chessboard in real world in any unit of measurement (will only effect the translation componenet between camera and projector)
    :param projector_orientation -  effects initial guess for projector principal point.
                                    can be "upper_half" or "lower_half", or "none".
                                    tabletop projectors usually have their principal point in the lower half of the image while ceiling mounted projectors have it in the upper half.
                                    note: opencv convention has the y direction increasing downwards.
    :param verbose if true, will print out the calibration results
    :param output_dir will save debug results to this directory
    :param debug if true, will save debug info into output_dir
    :return camera intrinsics (3x3),
            camera distortion parameters (5x1),
            projector intrinsics (3x3),
            projector distortion parameters (5x1),
            proj_transform (4x4), a camera to projector (c2p) transformation matrix (to get p2w, you should invert this matrix and multiply from the left with the camera to world matrix i.e. p2w = c2w * p2c)
    """
    chess_shape = (chess_vert, chess_hori)
    capture_dir = Path(capture_dir)
    if not capture_dir.exists():
        raise FileNotFoundError("capture_dir was not found")
    dirnames = sorted([x for x in capture_dir.glob("*") if x.is_dir()])
    if len(dirnames) == 0:
        raise FileNotFoundError("capture_dir contains no subfolders")
    used_dirnames = []
    gc_fname_lists = []
    for dname in dirnames:
        gc_fnames = sorted(dname.glob("*"))
        if len(gc_fnames) == 0:
            continue
        used_dirnames.append(str(dname))
        gc_fname_lists.append([str(x) for x in gc_fnames])
    dirnames = used_dirnames
    objps = np.zeros((chess_shape[0] * chess_shape[1], 3), np.float32)
    objps[:, :2] = chess_block_size * np.mgrid[
        0 : chess_shape[0], 0 : chess_shape[1]
    ].T.reshape(-1, 2)
    graycode = GrayCode()
    patterns = graycode.encode(proj_wh)
    cam_shape = load_image(gc_fname_lists[0][0], as_grayscale=True)[:, :, 0].shape[
        ::-1
    ]  # width, height
    patch_size_half = int(
        np.ceil(cam_shape[0] / 180)
    )  # some magic number for patch size
    cam_corners_list = (
        []
    )  # will contain the corners of the chessboard in camera coordinates
    cam_objps_list = (
        []
    )  # will contain the corners of the chessboard in chessboard local coordinates (unit of measurement deduced from chess_block_size)
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    for dname, gc_filenames in zip(dirnames, gc_fname_lists):
        if verbose:
            print("processing: {}".format(dname))
        if len(gc_filenames) != len(patterns):
            raise ValueError("invalid number of images in " + dname)
        imgs = load_images(gc_filenames, as_grayscale=True)
        forwardmap, fg = graycode.decode(
            imgs,
            proj_wh,
            mode="xy",
            bg_threshold=bg_threshold,
            output_dir=output_dir,
            debug=debug,
        )
        black_img = imgs[-1]
        white_img = imgs[-2]
        imgs = imgs[:-2]
        res, cam_corners = cv2.findChessboardCorners(white_img, chess_shape)
        if not res:
            raise RuntimeError(
                "chessboard was not found in {}".format(gc_filenames[-2])
            )
        cam_objps_list.append(objps)
        cam_corners_list.append(cam_corners)
        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        for corner, objp in zip(cam_corners, objps):
            c_x = int(round(corner[0][0]))
            c_y = int(round(corner[0][1]))
            src_points = []
            dst_points = []
            # todo: vectorize these loops
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    if fg[y, x]:
                        proj_pix = forwardmap[y, x]
                        src_points.append((x, y))
                        dst_points.append(np.array(proj_pix))
            if len(src_points) < patch_size_half**2:
                if verbose:
                    print(
                        "corner {}, {} was skiped because too few decoded pixels found (check your images and thresholds)".format(
                            c_x, c_y
                        )
                    )
                continue
            h_mat, inliers = cv2.findHomography(
                np.array(src_points), np.array(dst_points)
            )
            point = h_mat @ np.array([corner[0][0], corner[0][1], 1]).transpose()
            point_pix = point[0:2] / point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
        if len(proj_corners) < 3:
            raise RuntimeError(
                "too few corners were found in {} (less than 3)".format(dname)
            )
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))
    if verbose:
        print(
            "total correspondence points: {}".format(
                sum([len(x) for x in proj_corners_list])
            )
        )
    # Initial solution of camera's intrinsic parameters
    # camera_intrinsics_init = np.array([[np.mean(cam_shape), 0, cam_shape[0]/2], [0, np.mean(cam_shape), cam_shape[1]/2], [0, 0, 1]])
    ret, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
        cam_objps_list,
        cam_corners_list,
        cam_shape,
        None,
        None,
        None,
        None,  # camera_intrinsics_init
        cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT,
    )  # + cv2.CALIB_ZERO_TANGENT_DIST  # cv2.CALIB_USE_INTRINSIC_GUESS
    if verbose:
        print("Camera calib. intrinsic parameters: {}".format(cam_int))
        print("Camera calib. distortion parameters: {}".format(cam_dist))
        print("Camera calib. reprojection error: {}".format(ret))

    # Initial solution of projector's parameters
    if projector_orientation == "none":
        cy_correction = 0
    elif projector_orientation == "lower_half":
        cy_correction = proj_wh[1] / 4
    elif projector_orientation == "upper_half":
        cy_correction = -proj_wh[1] / 4
    else:
        raise ValueError("invalid projector_orientation")
    projector_intrinsics_init = np.array(
        [
            [np.mean(proj_wh), 0, proj_wh[0] / 2],
            [0, np.mean(proj_wh), cy_correction + proj_wh[1] / 2],
            [0, 0, 1],
        ]
    )
    projector_ditortion_init = np.zeros((5, 1))
    ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list,
        proj_corners_list,
        proj_wh,
        projector_intrinsics_init,
        projector_ditortion_init,
        None,
        None,
        cv2.CALIB_USE_INTRINSIC_GUESS
        + cv2.CALIB_FIX_ASPECT_RATIO
        + cv2.CALIB_ZERO_TANGENT_DIST
        + cv2.CALIB_FIX_K1
        + cv2.CALIB_FIX_K2
        + cv2.CALIB_FIX_K3,
    )

    if verbose:
        print("Projector calib. intrinsic parameters: {}".format(proj_int))
        print("Projector calib. distortion parameters: {}".format(proj_dist))
        print("Projector calib. reprojection error: {}".format(ret))

    # Stereo calibration for final solution
    (
        ret,
        cam_int,
        cam_dist,
        proj_int,
        proj_dist,
        cam_proj_rmat,
        cam_proj_tvec,
        E,
        F,
    ) = cv2.stereoCalibrate(
        proj_objps_list,
        cam_corners_list2,
        proj_corners_list,
        cam_int,
        cam_dist,
        proj_int,
        proj_dist,
        None,
    )

    proj_transform = compose_rt(
        cam_proj_rmat[None, ...], cam_proj_tvec[None, :, 0], square=True
    )[0]
    if verbose:
        print("Stereo reprojection error: {}".format(ret))
        print("Stereo camera intrinsic parameters: {}".format(cam_int))
        print("Stereo camera distortion parameters: {}".format(cam_dist))
        print("Stereo projector intrinsic parameters: {}".format(proj_int))
        print("Stereo projector distortion parameters: {}".format(proj_dist))
        print("Stereo camera2projector transform): {}".format(proj_transform))
    if debug:
        # computes a histogram of camera reprojection errors, and not just average error
        cam_corners = np.array(cam_corners_list).squeeze()
        proj_corners = np.array(proj_corners_list).squeeze()
        obj_corners = np.array(cam_objps_list).squeeze()
        all_projected_cam_corners = []
        all_projected_proj_corners = []
        for i in range(len(cam_corners)):
            projected_cam_points, _ = cv2.projectPoints(
                obj_corners[i], cam_rvecs[i], cam_tvecs[i], cam_int, cam_dist
            )
            all_projected_cam_corners.append(projected_cam_points)
            projected_proj_points, _ = cv2.projectPoints(
                obj_corners[i], proj_rvecs[i], proj_tvecs[i], proj_int, proj_dist
            )
            all_projected_proj_corners.append(projected_proj_points)
        all_projected_cam_corners = np.array(all_projected_cam_corners).squeeze()
        cam_norms = np.linalg.norm(cam_corners - all_projected_cam_corners, axis=-1)
        cam_per_session_error = cam_norms.mean(axis=-1)
        worst_to_best_cam_session_ids = np.argsort(cam_per_session_error)[::-1]
        worst_to_best_cam_errors = cam_per_session_error[worst_to_best_cam_session_ids]
        print(
            "worst to best sessions ids for camera reprojection error: {}".format(
                worst_to_best_cam_session_ids
            )
        )
        print("and their associated errors: {}".format(worst_to_best_cam_errors))
        cam_hist = np.histogram(cam_norms)
        print(
            "camera reprojection error histogram: {} (should be similar to gaussian around 0)".format(
                cam_hist
            )
        )
        all_projected_proj_corners = np.array(all_projected_proj_corners).squeeze()
        proj_norms = np.linalg.norm(proj_corners - all_projected_proj_corners, axis=-1)
        per_session_projector_error = proj_norms.mean(axis=-1)
        worst_to_best_proj_session_ids = np.argsort(per_session_projector_error)[::-1]
        worst_to_best_proj_errors = per_session_projector_error[
            worst_to_best_proj_session_ids
        ]
        print(
            "worst to best sessions ids for projector reprojection error: {}".format(
                worst_to_best_proj_session_ids
            )
        )
        print("and their associated errors: {}".format(worst_to_best_proj_errors))
        proj_hist = np.histogram(proj_norms)
        print(
            "projector reprojection error histogram: {} (should be similar to gaussian around 0)".format(
                proj_hist
            )
        )
    ret = {
        "cam_intrinsics": cam_int,
        "cam_distortion": cam_dist,
        "proj_intrinsics": proj_int,
        "proj_distortion": proj_dist,
        "proj_transform": proj_transform,
    }
    return ret


def reconstruct_pointcloud(
    forward_map,
    fg,
    cam_transform,
    proj_transform,
    cam_int,
    cam_dist,
    proj_int,
    mode="ij",
    color_image=None,
    debug=False,
):
    """
    given a dense pixel correspondence map between a camera and a projector, and calibration results, reconstructs a 3D point cloud of the scene.
    :param forward_map: a dense pixel correspondence map between a camera and a projector (see GrayCode.decode)
    :param fg: a foreground mask of the scene (see GrayCode.decode)
    :param cam_transform: a 4x4 transformation matrix of the camera (cam to world)
    :param proj_transform: a 4x4 transformation matrix of the projector (proj to world)
    :param cam_int: camera's intrinsic parameters
    :param cam_dist: camera's distortion parameters
    :param proj_int: projector's intrinsic parameters
    :param mode: "xy" or "ij" which is the ordering of the last channel of the forward map (see GrayCode.decode)
    :param color_image: RGB image with the same spatial size as forwardmap. if supplied will return a colored point cloud (Nx6).
    :param debug: if True, will return debug information
    :return: a 3D point cloud of the scene (Nx3)
    """
    cam_pixels = swap_columns(np.argwhere(fg), 0, 1)
    # todo: account for distortion if supplied
    # undistorted_cam_points =  cv2.undistortPoints(cam_points, cam_int, cam_dist, P=proj_transform[0]).squeeze()  # equivalent to not setting P and doing K @ points outside
    cam_origins = cam_transform[None, :3, -1]
    cam_directions = (
        cam_transform[:3, :3] @ (np.linalg.inv(cam_int) @ to_hom(cam_pixels).T)
    ).T
    cam_directions = cam_directions / np.linalg.norm(
        cam_directions, axis=-1, keepdims=True
    )
    if mode == "xy":
        projector_pixels = forward_map[fg]
    elif mode == "ij":
        projector_pixels = swap_columns(forward_map[fg], 0, 1)
    else:
        raise ValueError("mode must be either 'xy' or 'ij'")
    projector_origins = proj_transform[None, :3, -1]
    projector_directions = (
        proj_transform[:3, :3] @ (np.linalg.inv(proj_int) @ to_hom(projector_pixels).T)
    ).T
    projector_directions = projector_directions / np.linalg.norm(
        projector_directions, axis=-1, keepdims=True
    )
    # note: using ray-ray is slightly naive. A better way is to minimize a reprojection error, and then ray-ray
    points, weight_factor = ray_ray_intersection(
        cam_origins, cam_directions, projector_origins, projector_directions
    )
    if color_image is not None:
        colors = (
            color_image[fg] if mode == "xy" else color_image[swap_columns(fg, 0, 1)]
        )
        points = np.concatenate([points, colors], axis=-1)
    if debug:
        return (
            points,
            weight_factor,
            cam_origins,
            cam_directions,
            projector_origins,
            projector_directions,
            cam_pixels,
            projector_pixels,
        )
    else:
        return points


class GrayCode:
    """
    a class that handles encoding and decoding graycode patterns
    """

    def encode1d(self, length):
        total_images = len(bin(length - 1)[2:])

        def xn_to_gray(n, x):
            # infer a coordinate gray code from its position x and index n (the index of the image out of total_images)
            # gray code is obtained by xoring the bits of x with itself shifted, and selecting the n-th bit
            return (x ^ (x >> 1)) & (1 << (total_images - 1 - n)) != 0

        imgs_code = 255 * np.fromfunction(
            xn_to_gray, (total_images, length), dtype=int
        ).astype(np.uint8)
        return imgs_code

    def encode(self, proj_wh, flipped_patterns=True):
        """
        encode projector's width and height into gray code patterns
        :param proj_wh: projector's (width, height) in pixels as a tuple
        :param flipped_patterns: if True, flipped patterns are also generated for better binarization
        :return: a 3D numpy array of shape (total_images, height, width) where total_images is the number of gray code patterns
        """
        width, height = proj_wh
        codes_width_1d = self.encode1d(width)[:, None, :]
        codes_width_2d = codes_width_1d.repeat(height, axis=1)
        codes_height_1d = self.encode1d(height)[:, :, None]
        codes_height_2d = codes_height_1d.repeat(width, axis=2)

        img_white = np.full((height, width), 255, dtype=np.uint8)[None, ...]
        img_black = np.full((height, width), 0, dtype=np.uint8)[None, ...]
        all_images = np.concatenate((codes_width_2d, codes_height_2d), axis=0)
        if flipped_patterns:
            all_images = np.concatenate((all_images, 255 - codes_width_2d), axis=0)
            all_images = np.concatenate((all_images, 255 - codes_height_2d), axis=0)
        all_images = np.concatenate((all_images, img_white, img_black), axis=0)
        return all_images[..., None]

    def binarize(self, captures, flipped_patterns=True, bg_threshold=10):
        """
        binarize a batch of images
        :param captures: a 4D numpy array of shape (n, height, width, 1) of captured images
        :param flipped_patterns: if true, patterns also contain their flipped version for better binarization
        :param bg_threshold: if the difference between the pixel in the white&black images is greater than this, pixel is foreground
        :return: a 4D numpy binary array for decoding (total_images, height, width, 1) where total_images is the number of gray code patterns
        and a binary foreground mask (height, width, 1)
        """
        if not -255 <= bg_threshold <= 255:
            raise ValueError("bg_threshold must be between -255 and 255")
        patterns, bw = captures[:-2], captures[-2:]
        foreground = bw[0].astype(np.int32) - bw[1].astype(np.int32) > bg_threshold
        if flipped_patterns:
            orig, flipped = (
                patterns[: len(patterns) // 2],
                patterns[len(patterns) // 2 :],
            )
            # valid = (orig.astype(np.int32) - flipped.astype(np.int32)) > bin_threshold
            binary = orig > flipped
            # foreground = foreground & np.all(valid, axis=0)  # only pixels that are valid in all images are foreground
        else:  # slightly more naive thresholding
            binary = patterns >= 0.5 * (bw[1] + bw[0])
        return binary, foreground

    def decode1d(self, gc_imgs):
        # gray code to binary
        n, h, w = gc_imgs.shape
        binary_imgs = gc_imgs.copy()
        for i in range(1, n):  # xor with previous image except MSB
            binary_imgs[i, :, :] = np.bitwise_xor(
                binary_imgs[i, :, :], binary_imgs[i - 1, :, :]
            )
        # decode binary
        cofficient = np.fromfunction(
            lambda i, y, x: 2 ** (n - 1 - i), (n, h, w), dtype=int
        )
        img_index = np.sum(binary_imgs * cofficient, axis=0)
        return img_index

    def decode(
        self,
        captures,
        proj_wh,
        flipped_patterns=True,
        bg_threshold=10,
        mode="ij",
        output_dir=None,
        debug=False,
    ):
        """
        decodes a batch of images encoded with gray code
        :param captures: a 4D numpy array of shape (n, height, width, c) of captured images (camera resolution is inferred from this)
        :param flipped_patterns: if true, patterns also contain their flipped version for better binarization
        :param bg_threshold: a threshold used for background detection using the all-white and all-black captures
        :param mode: "xy" or "ij" decides the order of last dimension coordinates in the output (ij -> height first, xy -> width first)
        :param output_dir: if not None, saves the decoded images to this directory
        :param debug: if True, visualizes the map in an image using the red and green channels where R=X, G=Y, B=0, X increases from left to right, Y increases from top to bottom
        :return: a 2D numpy array of shape (height, width, 2) uint32 mapping from camera pixels to projector pixels
        and a foreground mask (height, width) of booleans
        """
        if captures.ndim != 4:
            raise ValueError("captures must be a 4D numpy array")
        if captures.dtype != np.uint8:
            raise ValueError("captures must be uint8")
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        b, _, _, c = captures.shape
        b = b - 2  # dont count white and black images
        if flipped_patterns:
            b = b // 2  # dont count flipped patterns
        encoded = self.encode(
            proj_wh, flipped_patterns
        )  # sanity: encode with same arguments to verify enough captures are present
        if len(encoded) != len(captures):
            raise ValueError("captures must have length of {}".format(len(encoded)))
        if c != 1:  # naively convert to grayscale
            captures = captures.mean(axis=-1, keepdims=True).round().astype(np.uint8)
        imgs_binary, fg = self.binarize(captures, flipped_patterns, bg_threshold)
        imgs_binary = imgs_binary[:, :, :, 0]
        fg = fg[:, :, 0]
        x = self.decode1d(imgs_binary[: b // 2])
        y = self.decode1d(imgs_binary[b // 2 :])
        if mode == "ij":
            forward_map = np.concatenate((y[..., None], x[..., None]), axis=-1)
        elif mode == "xy":
            forward_map = np.concatenate((x[..., None], y[..., None]), axis=-1)
        else:
            raise ValueError("mode must be 'ij' or 'xy'")
        if output_dir is not None:
            np.save(Path(output_dir, "forward_map.npy"), forward_map)
            np.save(Path(output_dir, "fg.npy"), fg)
            if debug:
                save_images(imgs_binary[..., None], Path(output_dir, "imgs_binary"))
                save_image(fg, Path(output_dir, "foreground.png"))
                composed = forward_map * fg[..., None]
                if mode == "ij":
                    composed_normalized = composed / np.array([proj_wh[1], proj_wh[0]])
                    composed_normalized[..., [0, 1]] = composed_normalized[..., [1, 0]]
                elif mode == "xy":
                    composed_normalized = composed / np.array([proj_wh[0], proj_wh[1]])
                composed_normalized_8b = to_8b(composed_normalized)
                composed_normalized_8b_3c = np.concatenate(
                    (
                        composed_normalized_8b,
                        np.zeros_like(composed_normalized_8b[..., :1]),
                    ),
                    axis=-1,
                )
                save_image(
                    composed_normalized_8b_3c, Path(output_dir, "forward_map.png")
                )
        return forward_map, fg
