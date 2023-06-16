import numpy as np
import torch
import cv2
from .gsoup_io import save_image, save_images, load_images, load_image
from .core import to_8b
from .image import interpolate_multi_channel, change_brightness
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator

def warp_image(backward_map, desired_image, cam_wh=None, output_path=None):
    """
    given a 2D dense map of corresponding pixels between projector and camera, computes a warped image such that if projected, the desired image appears when observed using camera
    :param backward_map: 2D dense mapping between pixels from projector 2 camera (proj_h x proj_w x 2) uint32
    :param desired_image: path to desired image from camera, or float np array channels last (cam_h x cam_w x 3) uint8
    :param cam_wh: device1 (width, height) as tuple, if not supplied assumes desired_image is in the correct dimensions in relation to backward_map
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
    warpped = desired_image[(backward_map[..., 0], backward_map[..., 1])]
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(warpped, output_path)
    return warpped

def compute_backward_map(proj_wh, forward_map, foreground, output_dir=None, debug=False): 
    """
    computes the inverse map of forward_map by piece-wise interpolating a triangulated version of it.
    :param proj_wh: projector (width, height) as a tuple
    :param forward_map: forward map as a numpy array of shape (height, width, 2) of type int32
    :param output_dir: directory to save the inverse map to
    :param debug: if True, saves a visualization of the inverse map to output_dir
    :return: inverse map as a numpy array of shape (projector_height, projector_width, 2) of type int32
    """   
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    data = np.argwhere(foreground)
    points = forward_map[foreground]
    interp = LinearNDInterpolator(points, data, fill_value=0.0)
    X, Y = np.meshgrid(np.arange(proj_wh[1]), np.arange(proj_wh[0]))
    result = interp(X, Y).transpose(1, 0, 2)  # eww
    if output_dir:
        np.save(Path(output_dir, "backward_map.npy"), result)
        if debug:
            result_normalized = result / np.array([forward_map.shape[0], forward_map.shape[1]])
            result_normalized_8b = to_8b(result_normalized)
            result_normalized_8b_3c = np.concatenate((result_normalized_8b, np.zeros_like(result_normalized_8b[..., :1])), axis=-1)
            save_image(result_normalized_8b_3c, Path(output_dir, "backward_map.png"))
    return result.round().astype(np.uint32)

def naive_color_compensate(target_image, all_white_image, all_black_image, cam_width, cam_height, brightness_decrease=-127, projector_gamma=2.2, output_path=None, debug=False):
    """
    color compensate a projected image such that it appears closer to a target image from the perspective of a camera
    loosly based on "Embedded entertainment with smart projectors"
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
    target_image = load_image(target_image, to_float=True, resize_wh=(cam_width, cam_height))[..., :3]
    target_image = change_brightness(target_image, brightness_decrease)
    if debug:
        save_image(target_image, Path(output_path.parent, "decrease_brightness.png"))
    all_white_image = load_image(all_white_image, to_float=True, resize_wh=(cam_width, cam_height))
    all_black_image = load_image(all_black_image, to_float=True, resize_wh=(cam_width, cam_height))
    compensated = (target_image - all_black_image) / all_white_image
    compensated = np.power(compensated, (1/projector_gamma))
    compensated = np.nan_to_num(compensated, nan=0.0, posinf=0.0, neginf=0.0)
    compensated = np.clip(compensated, 0, 1)
    if output_path:
        save_image(compensated, output_path)
    return compensated

def calibrate_procam(proj_height, proj_width, graycode_step, capture_dir, 
                     chess_vert=10, chess_hori=7,
                     black_thr=40, chess_block_size=10.0, verbose=True):
    """
    calibrates a projection-camera pair using local homographies
    based on "Simple, accurate, and robust projector-camera calibration."
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
    graycode = GrayCode()
    patterns = graycode.encode((proj_shape[1], proj_shape[0]))
    cam_shape = load_image(gc_fname_lists[0][0], as_grayscale=True).shape
    patch_size_half = int(np.ceil(cam_shape[1] / 180))
    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    for dname, gc_filenames in zip(dirnames, gc_fname_lists):
        if len(gc_filenames) != len(patterns):
            raise ValueError("invalid number of images in " + dname)
        imgs = load_images(gc_filenames, as_grayscale=True)[..., None]
        forwardmap, fg = graycode.decode(imgs, (proj_shape[1], proj_shape[0]), mode="ij")
        black_img = imgs[-1]
        white_img = imgs[-2]
        imgs = imgs[:-2]
        res, cam_corners = cv2.findChessboardCorners(white_img, chess_shape)
        if not res:
            raise RuntimeError("chessboard was not found in {}".format(gc_filenames[-2]))
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
                    if int(white_img[y, x]) - int(black_img[y, x]) <= black_thr:
                        continue
                    if fg[y, x]:
                        proj_pix = forwardmap[y, x]  # backward map ?
                        src_points.append((x, y))
                        dst_points.append(gc_step*np.array(proj_pix))
            if len(src_points) < patch_size_half**2:
                if verbose:
                    print('corner {}, {} was skiped because too few decoded pixels found (check your images and thresholds)'.format(c_x, c_y))
                continue
            h_mat, inliers = cv2.findHomography(
                np.array(src_points), np.array(dst_points))
            point = h_mat@np.array([corner[0][0], corner[0][1], 1]).transpose()
            point_pix = point[0:2]/point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
        if len(proj_corners) < 3:
            raise RuntimeError("too few corners were found in {} (less than 3)".format(dname))
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))

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

class GrayCode:
    """
    a class that handles encoding and decoding graycode patterns
    """
    def encode1d(self, length):
        total_images = len(bin(length-1)[2:])

        def xn_to_gray(n, x):
            # infer a coordinate gray code from its position x and index n (the index of the image out of total_images)
            # gray code is obtained by xoring the bits of x with itself shifted, and selecting the n-th bit
            return (x^(x>>1))&(1<<(total_images-1-n))!=0
        
        imgs_code = 255*np.fromfunction(xn_to_gray, (total_images, length), dtype=int).astype(np.uint8)
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
        img_black = np.full((height, width),  0, dtype=np.uint8)[None, ...]
        all_images = np.concatenate((codes_width_2d, codes_height_2d), axis=0)
        if flipped_patterns:
            all_images = np.concatenate((all_images, 255-codes_width_2d), axis=0)
            all_images = np.concatenate((all_images, 255-codes_height_2d), axis=0)
        all_images = np.concatenate((all_images, img_white, img_black), axis=0)
        return all_images[..., None]
    
    def binarize(self, captures, flipped_patterns=True, bg_threshold=5, bin_threshold=5):
        """
        binarize a batch of images
        :param captures: a 4D numpy array of shape (n, height, width, 1) of captured images
        :param flipped_patterns: if true, patterns also contain their flipped version for better binarization
        :param bg_threshold: a threshold used for background detection using the all-white and all-black captures
        :param bin_threshold: a threshold used for binarization between the flipped and non-flipped patterns
        :return: a 4D numpy binary array for decoding (total_images, height, width, 1) where total_images is the number of gray code patterns
        and a binary foreground mask (height, width, 1)
        """
        captures, bw = captures[:-2], captures[-2:]
        foreground = np.abs(bw[0].astype(np.int32) - bw[1].astype(np.int32)) > bg_threshold
        # img_bin = np.zeros_like(captures, dtype=np.uint8)
        if flipped_patterns:
            captures, flipped = captures[:len(captures)//2], captures[len(captures)//2:]
            valid = np.abs(captures.astype(np.int32) - flipped.astype(np.int32)) > bin_threshold
            binary = captures > flipped
            foreground = foreground & np.all(valid, axis=0)  # do not use pixels that do not meet threshold in any of the images
        else:  # slightly naive threhsolding
            threhold = 0.5*(bw[1] + bw[0])
            binary = captures >= threhold
        return binary, foreground

    def decode1d(self, gc_imgs):
        # gray code to binary
        n, h, w = gc_imgs.shape
        binary_imgs = gc_imgs.copy()
        for i in range(1, n):  # xor with previous image except MSB
            binary_imgs[i, :, :] = np.bitwise_xor(binary_imgs[i, :, :], binary_imgs[i-1, :, :])
        # decode binary
        cofficient = np.fromfunction(lambda i,y,x: 2**(n-1-i), (n,h,w), dtype=int)
        img_index = np.sum(binary_imgs * cofficient, axis=0)
        return img_index

    def decode(self, captures, proj_wh,
               flipped_patterns=True, bg_threshold=10, bin_threshold=30,
               mode="xy", output_dir=None, debug=False):
        """
        decodes a batch of images encoded with gray code
        :param captures: a 4D numpy array of shape (n, height, width, c) of captured images
        :param flipped_patterns: if true, patterns also contain their flipped version for better binarization
        :param bg_threshold: a threshold used for background detection using teh all-white and all-black captures
        :param bin_threshold: a threshold used for binarization
        :param mode: "xy" or "ij" decides the order of last dimension coordinates (ij -> height first, xy -> width first)
        :return: a 2D numpy array of shape (height, width, 2) specifying the coordinates of decoded result, and foreground mask (height, width)
        """
        if captures.ndim != 4:
            raise ValueError("captures must be a 4D numpy array")
        if captures.dtype != np.uint8:
            raise ValueError("captures must be uint8")
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        b, _, _, c = captures.shape
        b = b - 2 # dont count white and black images
        if flipped_patterns:
            b = b // 2  # dont count flipped patterns
        encoded = self.encode(proj_wh, flipped_patterns)  # sanity: encode with same arguments to verify enough captures are present
        if len(encoded) != len(captures):
            raise ValueError("captures must have length of {}".format(len(encoded)))
        if c != 1:  # naively convert to grayscale
            captures = captures.mean(axis=-1, keepdims=True).astype(np.uint8)
        imgs_binary, fg = self.binarize(captures, flipped_patterns, bg_threshold, bin_threshold)
        imgs_binary = imgs_binary[:, :, :, 0]
        fg = fg[:, :, 0]
        x = self.decode1d(imgs_binary[:b // 2])
        y = self.decode1d(imgs_binary[b // 2:])
        if mode == "ij":
            forward_map = np.concatenate((y[..., None], x[..., None]), axis=-1)
        elif mode == "xy":
            forward_map = np.concatenate((x[..., None], y[..., None]), axis=-1)
        else:
            raise ValueError("mode must be 'ij' or 'xy'")
        if output_dir is not None:
            np.save(Path(output_dir, "forward_map.npy"), forward_map)
            if debug:
                save_images(imgs_binary[..., None], Path(output_dir, "imgs_binary"))
                composed = forward_map * fg[..., None]
                composed_normalized = composed / np.array([proj_wh[1], proj_wh[0]])
                composed_normalized_8b = to_8b(composed_normalized)
                composed_normalized_8b_3c = np.concatenate((composed_normalized_8b, np.zeros_like(composed_normalized_8b[..., :1])), axis=-1)
                save_image(composed_normalized_8b_3c, Path(output_dir, "forward_map.png"))
        return forward_map, fg
    