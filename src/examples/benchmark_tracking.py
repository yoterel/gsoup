import numpy as np
import gsoup
from pathlib import Path
from gsoup.track import NaiveEdgeTracker, HullTracker


def get_default_cam():
    # Define camera intrinsics.
    height, width = 512, 512
    f = 1024
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


def create_video(V, F, n_frames):
    frames = []
    o2ws = []
    height, width, K, w2c = get_default_cam()
    rand_qvec = gsoup.random_qvec(2).astype(np.float32)
    rand_trans = np.random.uniform(-0.1, 0.1, size=6).astype(np.float32).reshape(2, 3)
    t = np.linspace(0, 1, n_frames)
    for i in range(n_frames):
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
            np.array([255, 255, 255]),
            wireframe=True,
            wireframe_occlude=True,
            wh=(width, height),
        )
        frames.append(render)
        o2ws.append(o2w)
    return np.array(frames), np.array(o2ws)


def main():
    params = {}
    params["dst_path"] = "track_results"
    # set up camera (example values)
    height, width, K, w2c = get_default_cam()
    params["height"] = height
    params["width"] = width
    params["K"] = K
    params["w2c"] = w2c
    params["ms_per_frame"] = 100
    params["dist_coeffs"] = np.zeros(
        (5, 1), dtype=np.float32
    )  # assuming no lens distortion
    # set up geometry of scene
    vertices, faces = gsoup.structures.quad_cube()
    vertices = vertices / 10
    edges, _, e2f = gsoup.faces2edges_naive(faces)
    params["v"] = vertices
    params["e"] = edges
    params["e2f"] = e2f
    params["f"] = faces
    # general settings for tracker
    params["iters_per_frame"] = 1
    # video settings
    params["n_frames"] = 100
    params["tmp_frames"] = 100

    # create video
    video, gt_o2ws = create_video(params["v"], params["f"], params["tmp_frames"])
    video = video[: params["n_frames"]]
    gt_o2ws = gt_o2ws[: params["n_frames"]]
    # initialize tracker
    # tracker = NaiveEdgeTracker(gt_o2ws[0], params)
    tracker = HullTracker(gt_o2ws[0], params)
    for i in range(len(video)):
        frame = video[i, :, :, 0]
        gt_v = (gt_o2ws[i] @ gsoup.to_hom(params["v"]).T).T
        tracker.track(frame, cur_v=gt_v)
    # get results
    object_poses, correspondences = tracker.get_results()
    # render results
    animation = []
    for i in range(len(object_poses)):
        iter_index = i % params["iters_per_frame"]
        frame_index = i // params["iters_per_frame"]
        # print("frame: {:03d}, iter: {:03d}".format(frame_index, iter_index))
        o2w = object_poses[i]
        # Draw the model: project each vertex and then draw each edge
        bg = video[frame_index] * np.array([0, 1, 0])[None, None, :].astype(np.uint8)
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
        animation.append(image)
    # save resulting video
    captions = [
        "frame: {:03d}, iter: {:03d}".format(
            i // params["iters_per_frame"], i % params["iters_per_frame"]
        )
        for i in range(0, len(object_poses), 1)
    ]
    animation = gsoup.draw_text_on_image(np.array(animation), captions, size=16)
    dst = Path(params["dst_path"])
    dst.mkdir(parents=True, exist_ok=True)
    gsoup.save_animation(animation, Path(dst, "result.gif"), params["ms_per_frame"])


if __name__ == "__main__":
    main()
