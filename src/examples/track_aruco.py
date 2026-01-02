import gsoup
import numpy as np
from pathlib import Path
import cv2
import time


def create_aruco(dst_path):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    marker_id = 42
    marker_size = 1024  # Size in pixels
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    gsoup.save_image(marker_image, Path(dst_path, "aruco.png"))


def detect_aruco(image, detector=None):
    gray = gsoup.to_gray(image)
    # blur a bit to improve detection
    if detector is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.detectInvertedMarker = True
        # Create the ArUco detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)
    return detector, corners, ids, rejected


def get_default_cam(params):
    # Define camera intrinsics.
    height, width = params["resolution"], params["resolution"]
    f = height * 2
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    K = K.astype(np.float32)

    # Define camera extrinsics.
    if params["n_views"] == 1:
        cam_loc = np.array([1.0, 0.0, 0.0])[None, ...]
    elif params["n_views"] == 2:
        cam_loc = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    else:
        raise NotImplementedError
    cam_at = np.array([[0.0, 0.0, 0.0]])
    cam_up = np.array([[0.0, 0.0, 1.0]])
    c2w = gsoup.look_at_np(cam_loc, cam_at, cam_up)  # cam -> world
    w2c = gsoup.invert_rigid(c2w)  # world -> cam
    w2c = w2c.astype(np.float32)
    return height, width, K, w2c


def create_video(params, wireframe=False):
    frames = []
    o2ws = []
    height, width, K, w2c = get_default_cam(params)
    rand_qvec = gsoup.random_qvec(2).astype(np.float32)
    rand_trans = np.random.uniform(-0.1, 0.1, size=6).astype(np.float32).reshape(2, 3)
    t = np.linspace(0, 1, params["tmp_frames"], dtype=np.float32)
    if wireframe:
        colors = np.array([255, 255, 255])
    else:
        if params.get("texture", None) is not None:
            colors = params["texture"]
        else:
            colors = np.random.randint(0, 256, size=(len(params["f"]), 3))
    for i in range(params["tmp_frames"]):
        print("frame: {:04d}".format(i))
        cur_qvec = gsoup.qslerp(rand_qvec[0], rand_qvec[1], t[i])
        cur_t = rand_trans[0] * (t[i] - 1) + rand_trans[1] * t[i]
        cur_rmat = gsoup.qvec2mat(cur_qvec[None, ...])
        o2w = gsoup.compose_rt(cur_rmat, cur_t[None, ...], square=False)[0]
        o2ws.append(o2w)
    for i in range(len(w2c)):
        for ii in range(params["tmp_frames"]):
            V_cur = (o2ws[ii] @ gsoup.to_hom(params["v"]).T).T
            render = gsoup.render_mesh(
                V_cur,
                params["f"],
                K,
                w2c[i],
                colors,
                Vt=params.get("vt", None),
                Vf=params.get("vf", None),
                wireframe=wireframe,
                wireframe_occlude=True,
                wh=(width, height),
            )
            frames.append(render)
    frames = np.array(frames).reshape(
        params["n_views"], params["tmp_frames"], *frames[0].shape
    )  # (n_views, n_frames, h, w, 3)
    return frames, np.array(o2ws)


def save_video(videos, params, name):
    for i in range(len(videos)):
        video = videos[i]
        captions = ["frame: {:03d}".format(ii) for ii in range(len(video))]
        animation = gsoup.draw_text_on_image(
            video, captions, size=16, color=np.array([255, 255, 255])
        )
        dst = Path(params["dst_path"])
        dst.mkdir(parents=True, exist_ok=True)
        gsoup.save_video(
            animation,
            Path(dst, "{}_{:02d}.mp4".format(name, i)),
            int(1000 / params["ms_per_frame"]),
            "1000k",
        )


def main():
    print("Track AruCo example")
    params = {}
    mesh_name = "cube_tex"
    params["dst_path"] = "track_aruco_results"
    Path(params["dst_path"]).mkdir(exist_ok=True, parents=True)
    # settings for tracker
    params["save_results"] = True
    params["sample_per_edge"] = 10
    params["iters_per_frame"] = 1
    params["seed"] = 1234
    params["corres_dist_threshold"] = 5.0
    params["use_normals"] = True
    # settings for virtual camera
    params["resolution"] = 512
    params["n_views"] = 1  # (for now, support 1 or 2)
    height, width, K, w2c = get_default_cam(params)
    params["height"] = height
    params["width"] = width
    params["K"] = K
    params["w2c"] = w2c
    params["ms_per_frame"] = 100
    params["dist_coeffs"] = np.zeros((5, 1), dtype=np.float32)  # no lense distortion
    # set up geometry of scene
    vertices, faces, vt, vf = gsoup.load_mesh(
        "tests/tests_resource/{}.obj".format(mesh_name), return_vert_uvs=True
    )
    # vertices, faces = gsoup.structures.quad_cube()  # quad_cube(), icosehedron()
    # vertices[:, 2] *= 2
    # gsoup.save_mesh(vertices, faces, "tests/tests_resource/{}.obj".format(mesh_name))
    vertices = gsoup.normalize_vertices(vertices) / 10
    edges, _, e2f = gsoup.faces2edges_naive(faces)
    params["v"] = vertices
    params["e"] = edges
    params["e2f"] = e2f
    params["f"] = faces
    params["vt"] = vt
    params["vf"] = vf
    # set up texture
    # create_aruco(Path("tests/tests_resource/aruco.png"))
    params["texture"] = gsoup.load_image("tests/tests_resource/cube_tex.png")[..., :3]
    # video settings
    params["tmp_frames"] = 500  # n frames used to simulate motion
    params["n_frames"] = 20  # actual number of frames used for tracking
    params["force_recreate_video"] = False
    # create video
    video_path = Path(params["dst_path"], "video.npy")
    video_wireframe_path = Path(params["dst_path"], "video_wireframe.npy")
    gt_o2ws_path = Path(params["dst_path"], "gt_o2ws.npy")
    if (
        video_path.exists()
        and video_wireframe_path.exists()
        and gt_o2ws_path.exists()
        and not params["force_recreate_video"]
    ):
        print("loading video and poses...")
        videos = np.load(video_path)
        videos_wireframe = np.load(video_wireframe_path)
        # video_silhouette = np.load(video_silhouette_path)
        gt_o2ws = np.load(gt_o2ws_path)
    else:
        print("rendering gt video...")
        np.random.seed(params["seed"])
        videos, gt_o2ws = create_video(params, False)
        # print("rendering gt silhouette video...")
        # video_silhouette = [get_silhouette(frame[..., 0]) for frame in video]
        # video_silhouette = np.array(video_silhouette)
        print("rendering gt wireframe video...")
        np.random.seed(params["seed"])
        videos_wireframe, _ = create_video(params, True)
        np.save(video_path, videos)
        np.save(video_wireframe_path, videos_wireframe)
        # np.save(video_silhouette_path, video_silhouette)
        np.save(gt_o2ws_path, gt_o2ws)
        save_video(videos, params, "orig_video")
        save_video(videos_wireframe, params, "orig_video_wireframe")
        # save_video(video_silhouette, params, "orig_video_silhouette")

    ####################### TRACKING #########################
    ##########################################################
    # lets select two random views where the AruCo is visible.
    frame1 = videos[0, 256]
    frame2 = videos[0, 397]
    # create a "patch" to be applied onto the AruCo location for visualizing.
    patch_image = gsoup.generate_lollipop_pattern(200, 200)
    detector, c1, ids1, rejected1 = detect_aruco(frame1)
    _, c2, ids2, rejected2 = detect_aruco(frame2, detector)
    c1 = c1[0]
    c2 = c2[0]
    # find homography c1->c2
    H_1_to_2, _ = cv2.findHomography(c1, c2, method=0)
    h, w = patch_image.shape[:2]
    square_pts = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    # find homography square->c1
    H_sq_to_1, _ = cv2.findHomography(square_pts, c1)
    warp1 = cv2.warpPerspective(
        patch_image, H_sq_to_1, (frame1.shape[1], frame1.shape[0])
    )
    # visualize patch1 ontop of frame1
    frame1_walpha = gsoup.add_alpha(frame1, np.ones_like(frame1[..., :1]) * 127)
    final1 = gsoup.alpha_compose(frame1_walpha, warp1)
    gsoup.save_image(final1, Path(params["dst_path"], "final1.png"))
    # find homography square->c2
    H_sq_to_2 = H_1_to_2 @ H_sq_to_1
    warp2 = cv2.warpPerspective(
        patch_image, H_sq_to_2, (frame2.shape[1], frame2.shape[0])
    )
    # visualize patch2 ontop of frame2
    frame2_walpha = gsoup.add_alpha(frame2, np.ones_like(frame2[..., :1]) * 127)
    final2 = gsoup.alpha_compose(frame2_walpha, warp2)
    gsoup.save_image(final2, Path(params["dst_path"], "final2.png"))


if __name__ == "__main__":
    main()
