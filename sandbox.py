import gsoup
from gsoup.viewer_drivers import poses_slide_view
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
#######################
# src = Path("D:/raw_vids/coloc.avi")
# dst = Path("D:/raw_vids/coloc_compressed.avi")
# gsoup.compress_video(src, dst)

######################
path = Path("D:/src/neural_projector/data")
paths = [x for x in path.glob("*.npz")]
cam_poses = []
proj_poses = []
proj_k = []
for path in sorted(paths):
    data = np.load(path)
    if "cam_rt" in data:
        cam_poses.append(data["cam_rt"][:, :, :3, :])
    if "proj_rt" in data:
        proj_poses.append(data["proj_rt"])
    if "proj_k" in data:
        proj_k.append(data["proj_k"])
if proj_k:
    proj_k = np.concatenate(proj_k, axis=0)
if cam_poses:
    cam_poses = np.concatenate(cam_poses, axis=0)
    cam_poses = gsoup.to_44(cam_poses)
    poses_slide_view(cam_poses)
#### change proj pose ###
test_loc = proj_poses[-1][-1][:3, 3]
test_loc[0] +=0.5
test_loc[2] +=0.5
new_pos = gsoup.look_at_np(test_loc, np.array([0, 0, 0.0]), np.array([0, 0, 1.0]))
proj_poses[-1][-1] = new_pos[0, :3, :]
#########################
if proj_poses:
    proj_poses = np.concatenate(proj_poses, axis=0)[:, None, :, :]
    proj_poses = gsoup.to_44(proj_poses)
    poses_slide_view(proj_poses)