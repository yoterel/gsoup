import numpy as np
import torch
import cv2
from .gsoup_io import save_image, save_images, load_images, load_image
from .image import interpolate_multi_channel, change_brightness
from pathlib import Path
from PIL import Image

def warp_image(p2c, cam_image, output_path=None, BGR=False):
    """
    given a 2D dense mapping between pixels from optical device 1 to optical device 2,
    warp an image from optical device 1 to optical device 2
    :param p2c: 2D dense mapping between pixels from optical device 1 to optical device 2
    :param cam_image: path to image from optical device 1
    :param output_path: path to save warped image to
    :param BGR: if True, assumes input image is BGR instead of RGB
    """
    if type(cam_image) == np.ndarray:
        unwarped = cam_image
    else:
        pilImage = Image.open(Path(cam_image))
        unwarped = np.asarray(pilImage)
        if BGR:
            unwarped = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
        #unwarped = cv2.resize(unwarped, (cam_width, cam_height))
        unwarped = (unwarped / 255).astype(np.float32)
    #print(p2c.shape)
    #p2c = Image.open(Path(interpolated_p2c_path))
    p2c = np.asarray(p2c)[:, :, :2]
    #p2c = p2c/255
    p2c[:, :, 0] = (p2c[:, :, 0] * 2) - 1
    p2c[:, :, 1] = (p2c[:, :, 1] * 2) - 1
    p2c[:, :, [1, 0]] = p2c[:, :, [0, 1]]
    # p2c = np.round(p2c).astype(np.int32)
    grid = torch.tensor(p2c.astype(np.float32)).unsqueeze(0)
    input = torch.tensor(unwarped).permute(2, 0, 1).unsqueeze(0)
    warped = torch.nn.functional.grid_sample(input, grid).squeeze().permute(1, 2, 0).numpy()
    warped_int = (warped * 255).astype(np.uint8)
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
    :return: list of patterns
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
                           BLACKTHR = 2, WHITETHR = 30, output_dir=None, debug=False):
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
    :param debug: if True, saves debug images
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
            print('Amount of c2p correspondences :', len(c2p_list))
    return interpolated_c2p, interpolated_p2c

def naive_color_compensate(target_image, all_white_image, all_black_image, cam_width, cam_height, brightness_decrease=-127, output_dir=None, debug=False):
    """
    color compensate a projected image such that it appears closer to a target image from the perspective of a camera
    loosly based on ***insert citation***
    :param target_image the desired image from the perspective of the camera
    :param all_white_image a picture taken by camera when projector had all pixels fully on
    :param all_black_image a picture taken by camera when projector had all pixels fully off
    :param cam_width camera image width
    :param cam_height camera image height
    :param brightness_decrease a hyper parameter controlling how much the total brightness is decreased. without this, the result is saturated because of dividing by small numbers
    :output_dir if passed, result will be saved here
    :debug if true, will save debug info into outputdir 
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    target_image = load_image(target_image, to_float=True, resize_wh=(cam_width, cam_height))
    target_image = change_brightness(target_image, brightness_decrease)
    if debug:
        save_image(target_image, "target_decrease_brightness.png") 
    #unwarped_image = np.power(unwarped_image, -2.2)
    compensated = (target_image - all_black_image) / all_white_image
    compensated = np.power(compensated, (1/2.2))
    compensated = np.nan_to_num(compensated, nan=0.0, posinf=0.0, neginf=0.0)
    compensated = np.clip(compensated, 0, 1)
    if output_dir:
        save_image(compensated, "compensated.png")
    return compensated
