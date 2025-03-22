import numpy as np
import gsoup
from pathlib import Path
from gsoup.track import NaiveEdgeTracker1, NaiveEdgeTracker2
import cv2
import time


def get_default_cam(params):
    # Define camera intrinsics.
    height, width = params["resolution"], params["resolution"]
    f = height * 2
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    K = K.astype(np.float32)

    # Define camera extrinsics.
    cam_loc = np.array([1.0, 0.0, 0.0])
    cam_at = np.array([0.0, 0.0, 0.0])
    cam_up = np.array([0.0, 0.0, 1.0])
    c2w = gsoup.look_at_np(cam_loc, cam_at, cam_up)  # cam -> world
    w2c = gsoup.invert_rigid(c2w)[0]  # world -> cam
    w2c = w2c.astype(np.float32)
    return height, width, K, w2c


def create_video(V, F, n_frames, params, wireframe=False):
    frames = []
    o2ws = []
    height, width, K, w2c = get_default_cam(params)
    rand_qvec = gsoup.random_qvec(2).astype(np.float32)
    rand_trans = np.random.uniform(-0.1, 0.1, size=6).astype(np.float32).reshape(2, 3)
    t = np.linspace(0, 1, n_frames, dtype=np.float32)
    if wireframe:
        colors = np.array([255, 255, 255])
    else:
        colors = np.random.randint(0, 256, size=(len(F), 3))
    for i in range(n_frames):
        print("frame: {:04d}".format(i))
        cur_qvec = gsoup.qslerp(rand_qvec[0], rand_qvec[1], t[i])
        cur_t = rand_trans[0] * (t[i] - 1) + rand_trans[1] * t[i]
        cur_rmat = gsoup.qvec2mat(cur_qvec[None, ...])
        o2w = gsoup.compose_rt(cur_rmat, cur_t[None, ...], square=False)[0]
        V_cur = (o2w @ gsoup.to_hom(V).T).T
        render = gsoup.render_mesh(
            V_cur,
            F,
            K,
            w2c,
            colors,
            wireframe=wireframe,
            wireframe_occlude=True,
            wh=(width, height),
        )
        frames.append(render)
        o2ws.append(o2w)
    return np.array(frames), np.array(o2ws)


def draw_correspondences(image, corr):
    if len(corr) != 0:
        for c in corr:  # (p, pnorm, q, qnorm)
            source = c[0].round().astype(np.uint32)
            target = c[2].round().astype(np.uint32)
            # if (source != target).any():
            # source_normal = (c[0] + (c[1] * 10)).round().astype(np.uint32)
            # image = cv2.arrowedLine(image, source, source_normal, (255, 0, 0), 1)
            image = cv2.arrowedLine(image, source, target, (255, 255, 0), 1)
            image = cv2.circle(image, source, 2, (255, 0, 0), -1)
            image = cv2.circle(image, target, 2, (0, 0, 255), -1)
            # image[source[1], source[0]] = np.array([255, 0, 0])
            # image[target[1], target[0]] = np.array([0, 0, 255])
    return image


def render_video(video, params, name):
    captions = ["frame: {:03d}".format(i) for i in range(len(video))]
    animation = gsoup.draw_text_on_image(
        video, captions, size=16, color=np.array([255, 255, 255])
    )
    dst = Path(params["dst_path"])
    dst.mkdir(parents=True, exist_ok=True)
    gsoup.save_video(
        animation,
        Path(dst, "{}.mp4".format(name)),
        int(1000 / params["ms_per_frame"]),
        "1000k",
    )


def get_silhouette(frame):
    """
    Get silhouette from frame, returns a 3-channel image.
    """
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    silhouette = c[:, 0, :].astype(np.float32)
    a = silhouette
    b = np.roll(silhouette, -1, axis=0)
    new_frame = np.zeros(frame.shape + (3,), dtype=np.uint8)
    for i in range(len(a)):
        gsoup.draw_line(new_frame, a[i], b[i], np.array([255, 255, 255]))
    return new_frame


def main():
    print("building scene...")
    params = {}
    mesh_name = "fox"
    params["dst_path"] = "track_results_{}".format(mesh_name)
    # settings for tracker
    params["save_results"] = True
    params["sample_per_edge"] = 5
    params["iters_per_frame"] = 3
    params["seed"] = 43
    params["corres_dist_threshold"] = 5.0
    params["use_normals"] = True
    # settings for virtual camera
    params["resolution"] = 1024
    height, width, K, w2c = get_default_cam(params)
    params["height"] = height
    params["width"] = width
    params["K"] = K
    params["w2c"] = w2c
    params["ms_per_frame"] = 100
    params["dist_coeffs"] = np.zeros((5, 1), dtype=np.float32)  # no lense distortion
    # set up geometry of scene
    vertices, faces = gsoup.load_mesh("tests/tests_resource/{}.obj".format(mesh_name))
    # vertices, faces = gsoup.structures.quad_cube()  # quad_cube(), icosehedron()
    # vertices[:, 2] *= 2
    vertices = gsoup.normalize_vertices(vertices) / 10
    edges, _, e2f = gsoup.faces2edges_naive(faces)
    params["v"] = vertices
    params["e"] = edges
    params["e2f"] = e2f
    params["f"] = faces
    # video settings
    params["tmp_frames"] = 500  # n frames used to simulate motion
    params["n_frames"] = 500  # actual number of frames used for tracking
    params["force_recreate_video"] = True
    # create video
    Path(params["dst_path"]).mkdir(exist_ok=True, parents=True)
    video_path = Path(params["dst_path"], "video.npy")
    video_wireframe_path = Path(params["dst_path"], "video_wireframe.npy")
    video_silhouette_path = Path(params["dst_path"], "video_silhouette.npy")
    gt_o2ws_path = Path(params["dst_path"], "gt_o2ws.npy")
    if (
        video_path.exists()
        and video_wireframe_path.exists()
        and video_silhouette_path.exists()
        and gt_o2ws_path.exists()
        and not params["force_recreate_video"]
    ):
        print("loading video and poses...")
        video = np.load(video_path)
        video_wireframe = np.load(video_wireframe_path)
        video_silhouette = np.load(video_silhouette_path)
        gt_o2ws = np.load(gt_o2ws_path)
    else:
        print("rendering gt video...")
        np.random.seed(params["seed"])
        video, gt_o2ws = create_video(
            params["v"],
            params["f"],
            params["tmp_frames"],
            params,
            False,
        )
        print("rendering gt silhouette video...")
        video_silhouette = [get_silhouette(frame[..., 0]) for frame in video]
        video_silhouette = np.array(video_silhouette)
        print("rendering gt wireframe video...")
        np.random.seed(params["seed"])
        video_wireframe, _ = create_video(
            params["v"],
            params["f"],
            params["tmp_frames"],
            params,
            True,
        )
        np.save(video_path, video)
        np.save(video_wireframe_path, video_wireframe)
        np.save(video_silhouette_path, video_silhouette)
        np.save(gt_o2ws_path, gt_o2ws)
        render_video(video, params, "orig_video")
        render_video(video_wireframe, params, "orig_video_wireframe")
        render_video(video_silhouette, params, "orig_video_silhouette")
    sil_edges = []
    video = video[: params["n_frames"] + 1]
    video_wireframe = video_wireframe[: params["n_frames"] + 1]
    video_silhouette = video_silhouette[: params["n_frames"] + 1]
    gt_o2ws = gt_o2ws[: params["n_frames"] + 1]
    # initialize tracker
    print("init tracker...")
    tracker = NaiveEdgeTracker1(gt_o2ws[0], params, "NaiveEdgeTracker1")
    # tracker = NaiveEdgeTracker(gt_o2ws[0], params, "NaiveEdgeTracker2")
    print("starting tracking...")
    start_time = time.time()
    for i in range(len(video)):
        frame = video[i, :, :, 0]
        # gt_v = (gt_o2ws[i] @ gsoup.to_hom(params["v"]).T).T
        pose = tracker.track(frame, i)
        end_time = time.time()
        print(
            "frame: {:03d} / {:03d}, ms: {:.03f}".format(
                i, len(video), (end_time - start_time) * 1000
            )
        )
        start_time = end_time
    print("finished tracking...")
    # get results
    object_poses, correspondences = tracker.get_results()
    print("rendering results...")
    # render debug video
    render_results(
        object_poses[:-1],
        correspondences,
        tracker.get_name(),
        "track_debug",
        video_wireframe,
        params,
    )
    # render error video
    # render_results(
    #     object_poses[1:],
    #     None,
    #     tracker.get_name(),
    #     "track_error",
    #     video_wireframe,
    #     params,
    # )


def render_results(object_poses, correspondences, tracker_name, name, gt_video, params):
    animation = []
    for i in range(len(object_poses)):
        print("frame: {:04d}".format(i))
        iter_index = i % params["iters_per_frame"]
        frame_index = i // params["iters_per_frame"]
        # print("frame: {:03d}, iter: {:03d}".format(frame_index, iter_index))
        o2w = object_poses[i]
        # Draw the model: project each vertex and then draw each edge
        bg = gt_video[frame_index] * np.array([0, 1, 0])[None, None, :].astype(np.uint8)
        image = gsoup.render_mesh(
            (o2w @ gsoup.to_hom(params["v"]).T).T,
            params["f"],
            params["K"],
            params["w2c"],
            np.array([255, 255, 255]),
            wireframe=True,
            wireframe_occlude=True,
            image=bg,
            wh=(params["width"], params["height"]),
        )
        # image = get_silhouette(image[..., 0])
        # image = np.max([image, bg], axis=0)
        # draw correspondences
        if correspondences is not None:
            if i < len(correspondences):
                image = draw_correspondences(image, correspondences[i])
        # draw some info as text
        caption = "frame: {:03d}, iter: {:03d}".format(
            i // params["iters_per_frame"], i % params["iters_per_frame"]
        )
        image = gsoup.draw_text_on_image(image[None, ...], caption, size=16)
        image = gsoup.draw_text_on_image(
            image,
            "gt",
            loc=(0, 16),
            size=16,
            color=np.array([0, 255, 0]),
        )
        image = gsoup.draw_text_on_image(
            image,
            "estimate",
            loc=(0, 32),
            size=16,
            color=np.array([255, 255, 255]),
        )
        animation.append(image[0])
    # save resulting video
    dst = Path(params["dst_path"])
    dst.mkdir(parents=True, exist_ok=True)
    # gsoup.save_images(
    #     animation,
    #     Path(dst, "{}_{}".format(tracker_name, name)),
    # )
    gsoup.save_video(
        animation,
        Path(dst, "{}_{}.mp4".format(tracker_name, name)),
        int(1000 / params["ms_per_frame"]),
    )


if __name__ == "__main__":
    main()
