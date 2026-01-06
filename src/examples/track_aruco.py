import gsoup
import numpy as np
from pathlib import Path
import cv2
import time
import json


def create_aruco(marker_id, dst_path):
    """
    create AruCo marker image and possibly save to disk
    args:
        marker_id: int, id of marker to create
        dst_path: str or Path, directory to save marker image, if None, do not save
    returns:
        marker_image: (h,w,3) uint8 image of the marker
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    marker_size = 1024  # arbtitrary size in pixels
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    if dst_path is not None:
        gsoup.save_image(marker_image, Path(dst_path, "aruco{}.png".format(marker_id)))
    return marker_image


def detect_aruco_marker(image, id):
    """
    detect AruCo marker corners in image, return corners of given id
    0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
    args:
        image: (h,w,3) image
        id: int, marker id to detect
    returns:
        corners: 4x2 array of float32
    """
    gray = gsoup.to_gray(image)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.detectInvertedMarker = True
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    corners_filtered = corners[np.where(ids[:, 0] == id)[0][0]]
    return corners_filtered[0]


def rt_vec_to_mat(rvec, tvec):
    """
    standard conversion from rodrigues + translation to matrix
    """
    R, _ = cv2.Rodrigues(rvec)  # (3,3)
    T = np.eye(4)  # (4,4)
    T[:3, :3] = R
    T[:3, 3:4] = tvec
    return T


def plot_points(image, points):
    """
    debugging function to plot 2D points on image
    """
    cc = image.copy()
    for x, y in points:
        cv2.circle(
            cc, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA
        )
    return cc


def main(c1, c2, camera_params_path, dst_path):
    """
    Demonstrates how to using AruCo markers to track a planar surface pose
    In this setup we have:
    1) A pinhole camera observing a plane from the side
    2) A pinhole projector that projects an AruCo marker (id 42) onto the plane (projector is fronto-parallel to plane)
    3) The plane contains a printed AruCo marker (id 41) and is captured in two different poses by the camera (c1 and c2)
    - Camera and projector are calibrated (intrinsics, extrinsics known).
    - We want the projector to always project the AruCo marker (id 42) onto the plane
    such that the camera sees it as if it was a sticker on the plane.
    Output is the image that the projector should project at time of c2.
    Note: this *does not* leverage the projected marker id 42 in c2. That marker is projected there just to show how if we do nothing, it distorts.
    """
    cameras = json.load(open(camera_params_path, "r"))
    K_cam = np.array(cameras["camera"]["intrinsics"])
    K_proj = np.array(cameras["projector_camera"]["intrinsics"])
    c2w = np.array(cameras["camera"]["extrinsics"])
    p2w = np.array(cameras["projector_camera"]["extrinsics"])
    projector_resx = cameras["projector_camera"]["resx"]
    projector_resy = cameras["projector_camera"]["resy"]
    c2p = np.linalg.inv(p2w) @ c2w
    dist = np.zeros((5,))  # (5,)
    # ============================================================
    # STEP 1: offline procedure
    # ============================================================
    # 1.0: find AruCo markers' corners in camera-space
    marker41_c1 = detect_aruco_marker(c1, 41)
    marker42_c1 = detect_aruco_marker(c1, 42)
    # 1.1 find projected markers (42) corners in plane space
    s = 0.615  # important: set to match the printed marker size in same units as calibration
    marker41_plane = np.array(
        [
            [0, s, 0],  # top left
            [s, s, 0],  # top right
            [s, 0, 0],  # bottom right
            [0, 0, 0],  # bottom left
        ],
        dtype=np.float32,
    )  # (4,3) - crucial: Y increases upwards in world space. do not change order.
    # marker41_plane -= np.array([[s / 2, s / 2, 0]])  # center at origin
    ####################### visualize for debugging
    # success, rvec41, tvec41 = cv2.solvePnP(
    #     marker41_plane,
    #     marker41_c1,
    #     K_cam,
    #     dist,
    #     flags=cv2.SOLVEPNP_IPPE_SQUARE,
    # )
    # test = cv2.drawFrameAxes(c1.copy(), K_cam, None, rvec41, tvec41, 0.5)
    # gsoup.save_image(test, Path(dst_path, "marker41_pose.png"))
    #######################
    H, _ = cv2.findHomography(marker41_plane[:, :2], marker41_c1)  # (4,2)  # (4,2)
    H_inv = np.linalg.inv(H)  # (3,3)
    marker42_plane = (H_inv @ gsoup.to_hom(marker42_c1).T).T  # (4,3)
    marker42_plane = gsoup.homogenize(marker42_plane)  # (4,2)
    marker42_plane = np.hstack(
        [marker42_plane, np.zeros((4, 1), dtype=np.float32)]
    )  # (4,3)
    ####################### visualize for debugging
    # success, rvec42, tvec42 = cv2.solvePnP(
    #     marker41_plane,  # place marker bottom left at origin
    #     marker42_c1,
    #     K_cam,
    #     dist,
    #     flags=cv2.SOLVEPNP_IPPE_SQUARE,
    # )
    # test2 = cv2.drawFrameAxes(c1.copy(), K_cam, None, rvec42, tvec42, 0.5)
    # gsoup.save_image(test2, Path(dst_path, "marker42_pose.png"))
    ####################### visualize more for debugging
    # T_plane2c1 = rt_vec_to_mat(rvec41, tvec41)
    # marker41_c1_test = (T_plane2c1 @ gsoup.to_hom(marker41_plane).T).T  # (4,4)
    # marker41_c1_test = marker41_c1_test[:, :3]  # (4,3)
    # test, _ = cv2.projectPoints(
    #     marker41_c1_test,  # (4,3)
    #     np.zeros((3, 1)),  # rvec (identity)
    #     np.zeros((3, 1)),  # tvec (identity)
    #     K_cam,  # (3,3)
    #     dist,
    # )
    # test = test.reshape(-1, 2)  # (4,2)
    # gsoup.save_image(plot_points(c1, test), Path(dst_path, "reproject41.png"))
    ####################### visualize more for debugging
    # marker42_c1_test = (T_plane2c1 @ gsoup.to_hom(marker42_plane).T).T  # (4,4)
    # marker42_c1_test = marker42_c1_test[:, :3]  # (4,3)
    # test, _ = cv2.projectPoints(
    #     marker42_c1_test,  # (4,3)
    #     np.zeros((3, 1)),  # rvec (identity)
    #     np.zeros((3, 1)),  # tvec (identity)
    #     K_cam,  # (3,3)
    #     dist,
    # )
    # test = test.reshape(-1, 2)  # (4,2)
    # gsoup.save_image(plot_points(c1, test), Path(dst_path, "reproject42.png"))
    # ============================================================
    # STEP 2: online procedure (for every new frame)
    # ============================================================
    # 2.0: find AruCo marker (41) corners in camera-space
    marker41_c2 = detect_aruco_marker(c2, 41)
    # 2.1: estimate plane -> camera transform
    success, rvec_plane2c2, tvec_plane2c2 = cv2.solvePnP(
        marker41_plane,
        marker41_c2,
        K_cam,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,  # (4,3)  # (4,2)  # (3,3)  # (5,)
    )
    T_plane2c2 = rt_vec_to_mat(rvec_plane2c2, tvec_plane2c2)
    # 2.2: compute plane -> projector transform (using calibration)
    T_plane2p2 = c2p @ T_plane2c2  # (4,4)
    # 2.3: transform marker42 coords into projector space (using offline step)
    marker42_plane_h = gsoup.to_hom(marker42_plane)  # (4,4) h stands for homogeneous
    ####################### visualize for debugging
    # marker42_c2_test = (T_plane2c2 @ marker42_plane_h.T).T  # (4,4)
    # marker42_c2_test = marker42_c2_test[:, :3]  # (4,3)
    # test, _ = cv2.projectPoints(
    #     marker42_c2_test,  # (4,3)
    #     np.zeros((3, 1)),  # rvec (identity)
    #     np.zeros((3, 1)),  # tvec (identity)
    #     K_cam,  # (3,3)
    #     dist,
    # )
    # test = test.reshape(-1, 2)  # (4,2)
    # gsoup.save_image(plot_points(c2, test), Path(dst_path, "reproject42_c2.png"))
    ####################### visualize for debugging
    # marker41_c2_test = (T_plane2c2 @ gsoup.to_hom(marker41_plane).T).T  # (4,4)
    # marker41_c2_test = marker41_c2_test[:, :3]  # (4,3)
    # test, _ = cv2.projectPoints(
    #     marker41_c2_test,  # (4,3)
    #     np.zeros((3, 1)),  # rvec (identity)
    #     np.zeros((3, 1)),  # tvec (identity)
    #     K_cam,  # (3,3)
    #     dist,
    # )
    # test = test.reshape(-1, 2)  # (4,2)
    # gsoup.save_image(plot_points(c2, test), Path(dst_path, "reproject41_c2.png"))
    #######################
    marker42_p2_h = (T_plane2p2 @ marker42_plane_h.T).T  # (4,4)
    marker42_p2 = marker42_p2_h[:, :3]  # (4,3)
    # 2.4: project marker42_p2 into projector image plane
    marker42_p2_projected, _ = cv2.projectPoints(
        marker42_p2,  # (4,3)
        np.zeros((3, 1)),  # rvec (identity)
        np.zeros((3, 1)),  # tvec (identity)
        K_proj,  # (3,3)
        dist,
    )
    marker42_p2_projected = marker42_p2_projected.reshape(-1, 2)  # (4,2)
    # Now we know where the aruco should be projected in the projector image.
    # 2.5: synthesize the image to be projected, by deforming a pure AruCo marker
    aruco_patch = create_aruco(42, None)
    h, w = aruco_patch.shape[:2]
    square_pts = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    H_sq2p2, _ = cv2.findHomography(
        square_pts, marker42_p2_projected
    )  # patch -> projector
    warp = cv2.warpPerspective(
        aruco_patch,
        H_sq2p2,
        (int(projector_resx), int(projector_resy)),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[255, 255, 255],
    )
    # 2.6: save result (project this !)
    gsoup.save_image(warp, Path(dst_path, "warpped.png"))


if __name__ == "__main__":
    c1 = gsoup.load_image("tests/tests_resource/aruco_track/C1_t=1.png")
    c2 = gsoup.load_image("tests/tests_resource/aruco_track/C1_t=2.png")
    dst_path = "track_results_aruco"
    cams = Path("tests/tests_resource/aruco_track/cameras.json")
    main(c1, c2, cams, dst_path)
