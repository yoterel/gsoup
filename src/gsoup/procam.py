import numpy as np
import torch
import cv2
from .gsoup_io import save_image, save_images, load_images, load_image
from .core import to_8b
from .image import interpolate_multi_channel, change_brightness
from pathlib import Path

def warp_image(p2c, cam_image, cam_h=None, cam_w=None, output_path=None):
    """
    todo: explain p2c structure
    given a 2D dense mapping between pixels from optical device 1 to optical device 2,
    warp an image from optical device 1 to optical device 2
    :param p2c: 2D dense mapping between pixels from optical device 1 to optical device 2
    :param cam_image: path to image from optical device 1, or float np array channels last, or pytorch tensor channels last
    :param cam_h: camera height, if not supplied assumes cam_image is in the correct dimensions in relation to p2c
    :param cam_w: camera width, if not supplied assumes cam_image is in the correct dimensions in relation to p2c
    :param output_path: path to save warped image to
    :return: warped image np array uint8
    """
    if type(cam_image) == np.ndarray:
        unwarped = torch.tensor(cam_image)
    elif type(cam_image) == torch.Tensor:
        unwarped = cam_image
    else:
        unwarped = load_image(cam_image, to_float=True, to_torch=True, resize_wh=(cam_w, cam_h))
    if unwarped.dtype != torch.float32 and unwarped.dtype != torch.float64:
        raise ValueError("cam_image must be float32 or float64")
    if unwarped.ndim != 3:
        raise ValueError("cam_image must be 3D")
    if unwarped.shape[2] != 3:
        raise ValueError("cam_image must be a channels last image.")
    if cam_h is not None:
        if unwarped.shape[0] != cam_h:
            raise ValueError("cam_image must be of shape cam_h, cam_w, 3")
    if cam_w is not None:
        if unwarped.shape[1] != cam_w:
            raise ValueError("cam_image must be of shape cam_h, cam_w, 3")
    #print(p2c.shape)
    #p2c = Image.open(Path(interpolated_p2c_path))
    if type(p2c) != np.ndarray:
        p2c_data = np.load(p2c)
    else:
        p2c_data = p2c.copy()
    p2c_data = np.asarray(p2c_data)[:, :, :2]
    #p2c = p2c/255
    p2c_data[:, :, 0] = (p2c_data[:, :, 0] * 2) - 1
    p2c_data[:, :, 1] = (p2c_data[:, :, 1] * 2) - 1
    p2c_data[:, :, [1, 0]] = p2c_data[:, :, [0, 1]]
    # p2c = np.round(p2c).astype(np.int32)
    grid = torch.tensor(p2c_data.astype(np.float32)).unsqueeze(0)
    input = torch.tensor(unwarped).permute(2, 0, 1).unsqueeze(0)
    warped = torch.nn.functional.grid_sample(input, grid).squeeze().permute(1, 2, 0).numpy()
    warped_int = to_8b(warped)
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        save_image(warped_int, output_path)
    return warped_int

def generate_gray_code(height, width, step, output_dir=None):
    """
    generate gray code patterns for structured light scanning
    :param height: height of the pattern
    :param width: width of the pattern
    :param step: step size of the pattern (e.g. 2 means the patterns are half the resolution)
    :param output_dir: directory to save the patterns to
    :return: n x height x width array of patterns
    """
    gc_height = int((height-1)/step)+1
    gc_width = int((width-1)/step)+1
    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
    patterns = graycode.generate()[1]
    # decrease pattern resolution
    exp_patterns = []
    for pat in patterns:
        img = np.zeros((height, width), np.uint8)
        for y in range(height):
            for x in range(width):
                img[y, x] = pat[int(y/step), int(x/step)]
        exp_patterns.append(img)
    exp_patterns.append(255*np.ones((height, width), np.uint8))  # white
    exp_patterns.append(np.zeros((height, width), np.uint8))    # black
    exp_patterns = np.stack(exp_patterns)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        file_names = ["pattern_{:02d}.png".format(i) for i in range(len(exp_patterns))]
        save_images(exp_patterns[..., None], output_dir, file_names=file_names)
    return exp_patterns

def pix2pix_correspondence(proj_width, proj_height, step, captures,
                           BLACKTHR = 2, WHITETHR = 30, output_dir=None, verbose=True, debug=False):
    """
    finds dense pixel to pixel pix2pix_correspondence between a projector and a camera
    note: assumes gray code patterns used for projections were generated using generate_gray_code
    :param proj_width: width of projector
    :param proj_height: height of projector
    :param step: step factor used for gray code patterns (e.g. 2 means half resolution)
    :param captures: the actual n x cam_height x cam_width x 3 captured images, or a directory containing the images
    :param BLACKTHR: threshold for black pixels
    :param WHITETHR: threshold for white pixels
    :param output_dir: directory to save results to
    :param verbose: if True, prints progress and status
    :param debug: if True, saves debug images (must provide output_dir)
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    # prep decoder
    gc_width = int((proj_width-1)/step)+1
    gc_height = int((proj_height-1)/step)+1
    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
    graycode.setBlackThreshold(BLACKTHR)
    graycode.setWhiteThreshold(WHITETHR)
    correct_pattern_amount = graycode.getNumberOfPatternImages() + 2
    if type(captures) != np.ndarray:
        captures = load_images(captures)
    captures = captures.mean(axis=-1).astype(np.uint8)  # convert to grayscale
    if len(captures) != correct_pattern_amount:
        raise ValueError('Number of images is not right (right number is {})'.format(correct_pattern_amount))
    imgs = list(captures)
    black = imgs.pop()
    white = imgs.pop()
    cam_height = white.shape[0]
    cam_width = white.shape[1]
    if debug:
        diff_pic = white.astype(np.uint64) - black.astype(np.uint64)
        diff_pic[diff_pic < 0] = 0
        save_image(diff_pic[..., None].astype(np.uint8), Path(output_dir, "white_black_diff.png"))
        if verbose:
            print('camera image size :', white.shape)
    # initialize
    viz_c2p = np.zeros((cam_height, cam_width, 3), np.float32)
    ragged_p2c = np.empty((proj_height, proj_width), dtype=object)
    for i in np.ndindex(ragged_p2c.shape): ragged_p2c[i] = []
    missing_values_c2p = np.ones((cam_height, cam_width), np.bool8)
    # c2p
    c2p_list = [] # [((cam x, y), (proj x, y))]
    for y in range(cam_height):
        for x in range(cam_width):
            if int(white[y, x]) - int(black[y, x]) <= BLACKTHR:  # background
                continue
            err, proj_pix = graycode.getProjPixel(imgs, x, y)
            if not err:
                fixed_pix = step*(proj_pix[0]+0.5), step*(proj_pix[1]+0.5)  # x, y
                ragged_p2c[int(fixed_pix[1]), int(fixed_pix[0])].append([y, x])
                viz_c2p[y, x, :] = [fixed_pix[1] / proj_height, fixed_pix[0] / proj_width, 0.0]
                missing_values_c2p[y, x] = False
                c2p_list.append(((x, y), fixed_pix))
    # p2c
    total_size = ragged_p2c.size
    counter = 0
    for i in range(proj_height):
        for j in range(proj_width):
            if verbose:
                print("{} / {}".format(counter, total_size), end="\r", flush=True)
            val = np.mean(ragged_p2c[i, j], dtype=np.float32, axis=0)
            if np.isnan(val).any():
                val = np.zeros(2)
            else:
                val = np.round(val).astype(np.int32)
            ragged_p2c[i, j] = val
            counter += 1
    viz_p2c = np.vstack(ragged_p2c.reshape(-1)).reshape(proj_height, proj_width, 2).astype(np.uint32)
    viz_p2c = viz_p2c / np.array([cam_height, cam_width], dtype=np.float32)[None, None, :]
    viz_p2c = np.concatenate((viz_p2c, np.zeros_like(viz_p2c)[:, :, 0:1]), axis=-1)
    missing_values_p2c = (viz_p2c == 0).all(axis=-1)
    # interpolate missing values
    interpolated_c2p = interpolate_multi_channel(viz_c2p, missing_values_c2p)
    interpolated_p2c = interpolate_multi_channel(viz_p2c, missing_values_p2c)
    # save results
    if output_dir is not None:
        np.save(Path(output_dir, "c2p.npy"), interpolated_c2p) 
        np.save(Path(output_dir, "p2c.npy"), interpolated_p2c) 
        if debug:
            save_image(interpolated_c2p, Path(output_dir, "interpolated_c2p.png"))
            save_image(interpolated_p2c, Path(output_dir, "interpolated_p2c.png"))
            save_image(viz_c2p, Path(output_dir, "c2p.png"))
            save_image(viz_p2c, Path(output_dir, "p2c.png"))
            if verbose:
                print('Amount of c2p correspondences :', len(c2p_list))
    return interpolated_c2p, interpolated_p2c

def naive_color_compensate(target_image, all_white_image, all_black_image, cam_width, cam_height, brightness_decrease=-127, output_path=None, debug=False):
    """
    color compensate a projected image such that it appears closer to a target image from the perspective of a camera
    loosly based on "Embedded entertainment with smart projectors"
    :param target_image the desired image path from the perspective of the camera
    :param all_white_image a path to picture taken by camera when projector had all pixels fully on (float32)
    :param all_black_image a path to picture taken by camera when projector had all pixels fully off (float32)
    :param cam_width camera image width
    :param cam_height camera image height
    :param brightness_decrease a hyper parameter controlling how much the total brightness is decreased. without this, the result is saturated because of dividing by small numbers
    :output_path if passed, result will be saved to this path
    :debug if true, will save debug info into output_path parent directory
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    target_image = load_image(target_image, to_float=True, resize_wh=(cam_width, cam_height))[..., :3]
    target_image = change_brightness(target_image, brightness_decrease)
    if debug:
        save_image(target_image, Path(output_path.parent, "decrease_brightness.png"))
    all_white_image = load_image(all_white_image, to_float=True, resize_wh=(cam_width, cam_height))
    all_black_image = load_image(all_black_image, to_float=True, resize_wh=(cam_width, cam_height))
    #unwarped_image = np.power(unwarped_image, -2.2)
    compensated = (target_image - all_black_image) / all_white_image
    compensated = np.power(compensated, (1/2.2))
    compensated = np.nan_to_num(compensated, nan=0.0, posinf=0.0, neginf=0.0)
    compensated = np.clip(compensated, 0, 1)
    if output_path:
        save_image(compensated, output_path)
    return compensated

def calibrate_procam(proj_height, proj_width, graycode_step, capture_dir, 
                     chess_vert=10, chess_hori=7,
                     black_thr=40, white_thr=5, chess_block_size=10.0, verbose=True):
    """
    calibrates a projection-camera pair using local homographies
    :param proj_height projector pixel height
    :param proj_width projector pixel width
    :param chess_vert number of cross points of chessboard in vertical direction (not including the border, i.e. internal corners)
    :param chess_hori number of cross points of chessboard in horizontal direction (not including the border, i.e. internal corners)
    :param graycode_step factor used to downsample the graycode images (see generate_gray_code)
    :param capture_dir directory containing the captured images when gray code patterns were projected, assumes structure as follows:
        capture_dir
            - folder1
                - 0000.png
                - 0001.png
                - ...
            - folder2
                - 0000.png
                - ...
    :param black_thr threshold for detecting black pixels in the chessboard
    :param white_thr threshold for detecting white pixels in the chessboard
    :param chess_block_size size of blocks of chessboard in real world in any unit of measurement (will only effect the translation componenet between camera and projector)
    :param verbose if true, will print out the calibration results
    :return camera intrinsics, camera extrinsics, projector intrinsics, projector extrinsics, cam_proj_rmat, cam_proj_tvec
    """
    proj_shape = (proj_height, proj_width)
    chess_shape = (chess_vert, chess_hori)
    gc_step = graycode_step
    capture_dir = Path(capture_dir)
    if not capture_dir.exists():
        raise FileNotFoundError("capture_dir was not found")
    dirnames = sorted(capture_dir.glob('*'))
    if len(dirnames) == 0:
        raise FileNotFoundError("capture_dir contains no subfolders")
    used_dirnames = []
    gc_fname_lists = []
    for dname in dirnames:
        gc_fnames = sorted(dname.glob('*'))
        if len(gc_fnames) == 0:
            continue
        used_dirnames.append(str(dname))
        gc_fname_lists.append([str(x) for x in gc_fnames])
    dirnames = used_dirnames
    objps = np.zeros((chess_shape[0]*chess_shape[1], 3), np.float32)
    objps[:, :2] = chess_block_size * np.mgrid[0:chess_shape[0], 0:chess_shape[1]].T.reshape(-1, 2)
    gc_height = int((proj_shape[0]-1)/gc_step)+1
    gc_width = int((proj_shape[1]-1)/gc_step)+1
    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)
    cam_shape = load_image(gc_fname_lists[0][0], as_grayscale=True).shape
    patch_size_half = int(np.ceil(cam_shape[1] / 180))
    # print('  patch size :', patch_size_half * 2 + 1)

    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    for dname, gc_filenames in zip(dirnames, gc_fname_lists):
        if len(gc_filenames) != graycode.getNumberOfPatternImages() + 2:
            raise ValueError("invalid number of images in " + dname)

        imgs = []
        for fname in gc_filenames:
            img = load_image(fname, as_grayscale=True)
            if cam_shape != img.shape:
                raise ValueError("image size of {} does not match other images".format(fname))
            imgs.append(img)
        black_img = imgs.pop()
        white_img = imgs.pop()

        res, cam_corners = cv2.findChessboardCorners(white_img, chess_shape)
        if not res:
            raise RuntimeError("chessboard was not found in {}".format(gc_filenames[-2]))
        cam_objps_list.append(objps)
        cam_corners_list.append(cam_corners)

        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        # viz_proj_points = np.zeros(proj_shape, np.uint8)
        for corner, objp in zip(cam_corners, objps):
            c_x = int(round(corner[0][0]))
            c_y = int(round(corner[0][1]))
            src_points = []
            dst_points = []
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    if int(white_img[y, x]) - int(black_img[y, x]) <= black_thr:
                        continue
                    err, proj_pix = graycode.getProjPixel(imgs, x, y)
                    if not err:
                        src_points.append((x, y))
                        dst_points.append(gc_step*np.array(proj_pix))
            if len(src_points) < patch_size_half**2:
                if verbose:
                    print('corner {}, {} was skiped because too few decoded pixels found (check your images and threasholds)'.format(c_x, c_y))
                continue
            h_mat, inliers = cv2.findHomography(
                np.array(src_points), np.array(dst_points))
            point = h_mat@np.array([corner[0][0], corner[0][1], 1]).transpose()
            point_pix = point[0:2]/point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
            # viz_proj_points[int(round(point_pix[1])),
            #                 int(round(point_pix[0]))] = 255
        if len(proj_corners) < 3:
            raise RuntimeError("too few corners were found in {} (less than 3)".format(dname))
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))
        # cv2.imwrite('visualize_corners_projector_' +
        #             str(cnt) + '.png', viz_proj_points)
        # cnt += 1

    # Initial solution of camera's intrinsic parameters
    ret, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
        cam_objps_list, cam_corners_list, cam_shape, None, None, None, None)
    if verbose:
        print('Initial camera intrinsic parameters: {}'.format(cam_int))
        print('Initial camera distortion parameters: {}'.format(cam_dist))
        print('Initial camera RMS: {}'.format(ret))

    # Initial solution of projector's parameters
    ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, proj_shape, None, None, None, None)
    if verbose:
        print('Initial projector intrinsic parameters: {}'.format(proj_int))
        print('Initial projector distortion parameters: {}'.format(proj_dist))
        print('Initial projector RMS: {}'.format(ret))

    # Stereo calibration for final solution
    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_corners_list2, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None)
    
    if verbose:
        print('RMS: {}'.format(ret))
        print('Camera intrinsic parameters: {}'.format(cam_int))
        print('Camera distortion parameters: {}'.format(cam_dist))
        print('Projector intrinsic parameters: {}'.format(proj_int))
        print('Projector distortion parameters: {}'.format(proj_dist))
        print('Rotation matrix / translation vector from camera to projector (cam2proj transform): {}, {}'.format(cam_proj_rmat, cam_proj_tvec))
    return cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec