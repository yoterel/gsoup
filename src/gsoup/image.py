import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import interpolate, spatial
from .core import (
    to_8b,
    to_float,
    to_hom,
    homogenize,
    broadcast_batch,
    is_np,
    to_torch,
    to_np,
)
from .structures import get_gizmo_coords

def add_alpha(images, alphas):
    """
    adds an alpha channel to a batch of images
    :param images: numpy image b x h x w x 3 or h x w x 3
    :param alpha: alpha channel b x h x w x 1 or h x w x 1
    :return: numpy image b x h x w x (c+1) or h x w x (c+1)
    """
    if images.shape[-1] != 3:
        raise ValueError("images must have 3 channels")
    if images.ndim != 3 and images.ndim != 4:
        raise ValueError("image must be a 3D/4D array")
    if alphas.ndim != 3 and alphas.ndim != 4:
        raise ValueError("alphas must be a 3D/4D array")
    if images.ndim == 3 and alphas.ndim == 4:
        raise ValueError("images and alphas must have the same number of dimensions")
    if images.ndim == 4:
        if images.shape[0] != alphas.shape[0] and alphas.shape[0] != 1:
            raise ValueError("images and alpha must have the same batch size, or alphas batch size must equal 1")
        if images.shape[1:3] != alphas.shape[1:3]:
            raise ValueError("images and alphas must have the same spatial size")
        images, alphas = broadcast_batch(images, alphas)
    else:
        # both are ndim=3
        if images.shape[:2] != alphas.shape[:2]:
            raise ValueError("images and alphas must have the same spatial size")
    if is_np(images):
        return np.concatenate((images, alphas), axis=-1)
    else:
        return torch.cat((images, alphas), dim=-1)

def alpha_compose(images, backgrounds=None, bg_color=None):
    """
    composes a single or batch of RGBA images into a single or batch of RGB images.
    if backgrounds is provided, will blend them with the images, otherwise will blend with bg_color.
    if no backgrounds or bg_color is provided, the background is assumed to be black.
    :param image: b x H x W x 4 or H x W x 4
    :param background: b x H x W x 3 or H x W x 3
    :param bg_color: 3 or b x 3 float32 array
    :return: b x H x W x 3 or H x W x 3
    """
    if images.ndim != 3 and images.ndim != 4:
        raise ValueError("image must be 3 or 4 dimensional")
    if images.shape[-1] != 4:
        raise ValueError("image must have 4 channels")
    if backgrounds is not None:
        if images.shape[:-1] != backgrounds.shape[:-1]:
            raise ValueError("backgrounds must have same shape as images")
    if is_np(images):
        if bg_color is None:
            bg_color = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        if images.dtype != np.float32:
            images = to_float(images)
    else:
        if bg_color is None:
            bg_color = torch.tensor(
                [0.0, 0.0, 0.0], dtype=images.dtype, device=images.device
            )
        if images.dtype != torch.float32:
            images = to_float(images)
    if backgrounds is not None:
        if backgrounds.dtype != np.float32:
            backgrounds = to_float(backgrounds)
        bg_color = backgrounds
    alpha = images[..., 3:4]
    rgb = images[..., :3]
    return alpha * rgb + (1 - alpha) * bg_color


def draw_text_on_image(images, text_per_image, fill_white=True):
    """
    writes text on images given as np array (b x H x W x 3)
    :param images: (b x H x W x 3) numpy array
    :param text_per_image: b x 1 numpy array of strings
    :param fill_white: if True, text is white, otherwise black
    :return: new (b x H x W x 3) numpy array with text written
    """
    is_numpy = is_np(images)
    if not is_numpy:
        device = images.device
        images = to_np(images)
    is_float = images.dtype == np.float32
    if is_float:
        images = to_8b(images)
    rgbs = [Image.fromarray(x) for x in images]
    font = ImageFont.truetype("./FreeMono.ttf", 48)
    if fill_white:
        fill = "white"
    else:
        fill = "black"
    for i, rgb in enumerate(rgbs):
        text = text_per_image[i]
        ImageDraw.Draw(rgb).text((0, 0), text, fill=fill, font=font)
    rgbs = np.array([np.asarray(rgb) for rgb in rgbs])
    if is_float:
        rgbs = to_float(rgbs)
    if not is_numpy:
        rgbs = to_torch(rgbs, device=device)
    return rgbs


def draw_gizmo_on_image(np_images, w2c, opengl=False, scale=0.05):
    """
    adds a gizmo to a batch of np images.
    note: will broadcast np_images and w2c against eachother.
    :param np_images: b x H x W x 3
    :param w2c: b x 3 x 4 w2c transforms (opencv conventions)
    :param opengl: if True, the w2c transforms are assumed to be in OpenGL conventions, else OpenCV conventions
    for opengl, w2c should be a bx4x4 matrix converting from world to *CLIP* space.
    :param scale: scale of the gizmo
    :return: b x H x W x 3
    """
    new_images = []
    if np_images.ndim != 4:
        raise ValueError("np_images must be b x H x W x 3")
    if w2c.ndim != 3:
        raise ValueError("KRt must be b x 3 x 4")
    np_images, w2c = broadcast_batch(np_images, w2c)
    for i, np_image in enumerate(np_images):
        pil_image = Image.fromarray(to_8b(np_image))
        W, H = pil_image.size
        gizmo_cords, _, _ = get_gizmo_coords(scale)
        gizmo_hom = to_hom(gizmo_cords)
        verts_clip = (w2c[i] @ gizmo_hom.T).T
        verts_clip = homogenize(verts_clip)
        if opengl:
            verts_clip = verts_clip[:, :2]
            verts_screen = np.array([W, H]) * (verts_clip + 1) / 2
            verts_screen[:, 1] *= -1
        else:
            verts_screen = verts_clip
        desired_loc = np.array([W - 40, H - 40])
        verts_screen += desired_loc - verts_screen[0]
        draw = ImageDraw.Draw(pil_image)
        draw.line((tuple(verts_screen[0]), tuple(verts_screen[1])), fill="red", width=0)
        draw.line(
            (tuple(verts_screen[0]), tuple(verts_screen[2])), fill="green", width=0
        )
        draw.line(
            (tuple(verts_screen[0]), tuple(verts_screen[3])), fill="blue", width=0
        )
        new_images.append(np.array(pil_image))
    return (np.array(new_images) / 255.0).astype(np.float32)


def merge_figures_with_line(
    img1,
    img2,
    lower_intersection=0.6,
    angle=np.pi / 2,
    line_width=5,
    line_color=[255, 255, 255, 255],
):
    """
    merges two np images (H x W x 3) with a white line in between
    :param img1: (H x W x 3) numpy array
    :param img2: (H x W x 3) numpy array
    :param lower_intersection: lower intersection of the line with the images
    :param angle: angle of the line
    :param line_width: width of the line
    :param line_color: color of the line
    :return: new (H x W x 3) numpy array with line in between
    """
    if img1.dtype != np.uint8:
        line_color = np.array(line_color) / 255
    else:
        line_color = np.array(line_color, dtype=np.uint8)
    combined = np.ascontiguousarray(np.broadcast_to(line_color, img1.shape))
    y, x, _ = img1.shape
    yy, xx = np.mgrid[:y, :x]
    img1_positions = (xx - lower_intersection * x) * np.tan(angle) - line_width // 2 > (
        yy - y
    )
    img2_positions = (xx - lower_intersection * x) * np.tan(angle) + line_width // 2 < (
        yy - y
    )
    combined[img1_positions] = img1[img1_positions]
    combined[img2_positions] = img2[img2_positions]
    return combined


def generate_voronoi_diagram(height, width, num_cells=1000, bg_color="white", dst=None):
    """
    generate a voronoi diagram HxWx3 with random colored cells
    :param height: height of the image
    :param width: width of the image
    :param num_cells: number of cells
    :param bg_color: background color
    :param dst: if not None, the image is written to this path
    :return: (H x W x 3) numpy array
    """
    nx = np.random.rand(num_cells) * width
    ny = np.random.rand(num_cells) * height
    nxy = np.stack((nx, ny), axis=-1)
    img = Image.new("RGB", (width, height), bg_color)
    vor = spatial.Voronoi(nxy)
    polys = vor.regions
    vertices = vor.vertices
    for poly in polys:
        polygon = vertices[poly]
        if len(poly) > 0 and np.all(np.array(poly) > 0):
            img1 = ImageDraw.Draw(img)
            img1.polygon(
                list(map(tuple, polygon)),
                fill=tuple(np.random.randint(0, 255, size=(3,))),
            )
    if dst is not None:
        img.save(str(dst))
    return np.array(img)


def generate_dot_pattern(
    height, width, background="black", radius=5, spacing=50, dst=None
):
    """
    generates an image with colored circles in a grid
    :param height: height of the image
    :param width: width of the image
    :param background: background color
    :param radius: radius of the circles in pixels
    :param spacing: spacing between the circles in pixels
    :param dst: if not None, the image is written to this path
    :return: (H x W x 3) numpy array (uint8)
    """
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    for i in range(0, width, spacing):
        for j in range(0, height, spacing):
            img1.ellipse(
                [i, j, i + radius * 2, j + radius * 2],
                fill=tuple(np.random.randint(0, 255, size=(3,))),
            )
    if dst is not None:
        img.save(str(dst))
    return np.array(img)


def generate_checkerboard(h, w, blocksize):
    """
    generates a checkerboard pattern
    note: if blocksize is not a divisor of w or h, the pattern will have extra "crops" at the edges
    :param h: height of the image
    :param w: width of the image
    :param blocksize: size of the squares
    :return: (H x W x 1) numpy array (bool)
    """
    c0, c1 = 0, 1  # color of the squares, for binary these are just 0,1
    tile = (
        np.array([[c0, c1], [c1, c0]], dtype=bool)
        .repeat(blocksize, axis=0)
        .repeat(blocksize, axis=1)[..., None]
    )
    grid = np.tile(tile, (h // (2 * blocksize) + 1, w // (2 * blocksize) + 1, 1))
    return grid[:h, :w]


def generate_stripe_pattern(
    height,
    width,
    background="black",
    direction="vert",
    thickness=5,
    spacing=50,
    dst=None,
):
    """
    generates an image with colored stripes in a certain direction
    :param height: height of the image
    :param width: width of the image
    :param background: background color
    :param direction: direction of the stripes ("vert", "hor", "both")
    :param thickness: thickness of the stripes
    :param spacing: spacing between the stripes
    :param dst: if not None, the image is written to this path
    :return: (H x W x 3) numpy array (uint8)
    """
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    if direction == "vert":
        for i in range(0, width, spacing):
            img1.rectangle(
                [i, 0, i + thickness, height],
                fill=tuple(np.random.randint(0, 255, size=(3,))),
            )
    elif direction == "hor":
        for i in range(0, height, spacing):
            img1.rectangle(
                [0, i, width, i + thickness],
                fill=tuple(np.random.randint(0, 255, size=(3,))),
            )
    elif direction == "both":
        for i in range(0, width, spacing):
            img1.rectangle(
                [i, 0, i + thickness, height],
                fill=tuple(np.random.randint(0, 255, size=(3,))),
            )
        for i in range(0, height, spacing):
            img1.rectangle(
                [0, i, width, i + thickness],
                fill=tuple(np.random.randint(0, 255, size=(3,))),
            )
    else:
        raise ValueError("direction must be either 'vert', 'hor' or 'both'")
    if dst is not None:
        img.save(str(dst))
    return np.array(img)


def generate_lollipop_pattern(height, width, background="black", n=15, m=8, dst=None):
    """
    generates an image with a lollipop pattern
    :param height: height of the image
    :param width: width of the image
    :param background: background color
    :param n: number of circles in the pattern
    :param m: number of lines in the pattern
    :param dst: if not None, the image is written to this path
    :return: (H x W x 3) numpy array (uint8)
    """
    spacing_x = width // (2 * n)
    spacing_y = height // (2 * n)
    spacing_angle = 360 // m
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    for j in range(m):
        for i in range(n):
            x0 = spacing_x * i
            y0 = spacing_y * i
            start_angle = j * spacing_angle
            end_angle = 360
            img1.pieslice(
                [x0, y0, width - x0, height - y0],
                start=start_angle,
                end=end_angle,
                fill=tuple(np.random.randint(0, 255, size=(3,))),
            )
    if dst is not None:
        img.save(str(dst))
    return np.array(img)


def generate_concentric_circles(height, width, background="black", n=15, dst=None):
    """
    generates an image with colored concentric circles
    :param height: height of the image
    :param width: width of the image
    :param background: background color
    :param n: number of circles to draw
    :param dst: if not None, the image is written to this path
    """
    spacing_x = width // (2 * n)
    spacing_y = height // (2 * n)
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    for i in range(n):
        x0 = spacing_x * i
        y0 = spacing_y * i
        img1.ellipse(
            [x0, y0, width - x0, height - y0],
            fill=tuple(np.random.randint(0, 255, size=(3,))),
        )
    if dst is not None:
        img.save(str(dst))
    return np.array(img)


def generate_gray_gradient(
    height, width, grayscale=False, vertical=True, flip=False, bins=10
):
    """
    generate a gray gradient image HxWx3
    :param height: height of the image
    :param width: width of the image
    :param grayscale: if True, the image is grayscale
    :param vertical: if True, the gradient is vertical
    :param flip: if True, the gradient is flipped
    :param bins: number of bins
    :return: (H x W x 3) numpy array
    """
    bins = np.clip(bins, 1, 256)
    colors = np.linspace(0, 255, num=bins).astype(np.uint8)
    if vertical:
        n_bins = height // bins
        up_to_max_intensity = colors.repeat(n_bins)[:height]
        pad_size = height - up_to_max_intensity.shape[0]
        if pad_size > 0:
            channel = np.concatenate(
                (up_to_max_intensity, np.full(pad_size, 255, dtype=np.uint8))
            )
        else:
            channel = up_to_max_intensity
        img = channel[:, None].repeat(width, axis=1)
        if flip:
            img = np.flip(img, axis=0)
    else:
        n_bins = width // bins
        up_to_max_intensity = colors.repeat(n_bins)[:width]
        pad_size = width - up_to_max_intensity.shape[0]
        if pad_size > 0:
            channel = np.concatenate(
                (up_to_max_intensity, np.full(pad_size, 255, dtype=np.uint8))
            )
        else:
            channel = up_to_max_intensity
        img = channel[None, :].repeat(height, axis=0)
        if flip:
            img = np.flip(img, axis=1)
    if not grayscale:
        img = img[:, :, None].repeat(3, axis=-1)
    return img


def image_grid(images, rows, cols):
    """
    :param images: list of images
    :param rows: number of rows
    :param cols: number of cols
    :return: grid image
    """
    if images.ndim != 4:
        raise ValueError("images must be a 4D array")
    if len(images) != rows * cols:
        raise ValueError("number of images must be equal to rows * cols")
    tmp = images.reshape(rows, cols, images.shape[1], images.shape[2], -1)
    if type(tmp) == torch.Tensor:
        result = tmp.permute(0, 2, 1, 3, 4).reshape(
            rows * images.shape[1], cols * images.shape[2], -1
        )
    elif type(tmp) == np.ndarray:
        result = tmp.transpose(0, 2, 1, 3, 4).reshape(
            rows * images.shape[1], cols * images.shape[2], -1
        )
    return result


def resize_images_naive(images, H, W, channels_last=True, mode="mean"):
    """
    resize images to output_size, but only if the output size has a common divisor of the input size
    :param images: numpy array of images (N x H x W x C)
    :param H: output height size that has a common divisor with the input height size
    :param W: output width size that has a common divisor with the input width size
    :param channels_last: if True, the images are provided in channels last format (and so will the output)
    :param mode: one of "max", "mean"
    :return: np array of resized images
    """
    if images.ndim != 4:
        raise ValueError("images must be a 4D array")
    if images.shape[1] != images.shape[2]:
        raise ValueError("images must be square")
    if not channels_last:
        raise NotImplementedError("only channels last is supported")
    channels_size = images.shape[-1]
    input_size = np.array(images.shape[1:3])
    output_size = np.array([H, W])
    bin_size = input_size // output_size
    if mode == "max":
        small_images = (
            images.reshape(
                (
                    images.shape[0],
                    output_size[0],
                    bin_size[0],
                    output_size[1],
                    bin_size[1],
                    channels_size,
                )
            )
            .max(4)
            .max(2)
        )
    elif mode == "mean":
        small_images = (
            images.reshape(
                (
                    images.shape[0],
                    output_size[0],
                    bin_size[0],
                    output_size[1],
                    bin_size[1],
                    channels_size,
                )
            )
            .mean(4)
            .mean(2)
        )
        if images.dtype == np.uint8:
            small_images = small_images.astype(np.uint8)
    else:
        raise ValueError("mode must be one of 'max', 'mean'")
    return small_images


def pad_to_square(images, color=None):
    """
    pads a batch of images to a square shape
    note: smaller dimension is padded
    :param image: numpy image b x h x w x c
    :param color: color to pad with
    :return: padded image
    """
    if images.ndim != 4:
        raise ValueError("image must be a 4D array")
    diff = images.shape[1] - images.shape[2]
    if diff == 0:
        return images
    if diff > 0:
        return pad_to_res(images, images.shape[1], images.shape[1], color)
    else:
        return pad_to_res(images, images.shape[2], images.shape[2], color)


def crop_to_square(images):
    """
    crops a batch of images to a square shape
    note: bigger dimension is cropped
    :param img: numpy image h x w x c
    :return: the cropped square image
    """
    if images.ndim != 4:
        raise ValueError("image must be a 4D array")
    if images.shape[1] > images.shape[2]:
        s = int((images.shape[1] - images.shape[2]) / 2)
        return images[s : (s + images.shape[2])]
    else:
        s = int((images.shape[2] - images.shape[1]) / 2)
        return images[:, :, s : (s + images.shape[1])]


def pad_to_res(images, res_h, res_w, bg_color=None):
    """
    pads a batch of numpy images to a specific resolution
    :param image: numpy image b x h x w x c
    :param res_h: height of the output image
    :param res_w: width of the output image
    :param bg_color: background color c (defaults to black)
    :return: padded image b x res_h x res_w x c
    """
    if bg_color is None:
        if is_np(images):
            bg_color = np.zeros(images.shape[-1], dtype=images.dtype)
        else:
            bg_color = torch.zeros(
                images.shape[-1], dtype=images.dtype, device=images.device
            )
    if images.ndim != 4:
        raise ValueError("image must be a 4D array")
    b, h, w, c = images.shape
    if h > res_h or w > res_w:
        raise ValueError("images dimensions is larger than the output resolution")
    if h == res_h and w == res_w:
        return images
    if bg_color.shape[0] != c:
        raise ValueError(
            "background color must have the same number of channels as the image"
        )
    bg_color = bg_color[None, None, None, :]
    if is_np(images):
        output = np.zeros((b, res_h, res_w, c), dtype=images.dtype)
    else:
        output = torch.zeros(
            (b, res_h, res_w, c), dtype=images.dtype, device=images.device
        )
    output[:, :, :, :] = bg_color
    corner_left = (res_w - w) // 2
    corner_top = (res_h - h) // 2
    output[:, corner_top : corner_top + h, corner_left : corner_left + w, :] = images
    return output


def crop_center(images, dst_h, dst_w):
    """
    crops a batch of images to a specific resolution, but the crop comes from the center of the image
    :param image: numpy (or torch) array b x h x w x c
    :param dst_h: height of the output image
    :param dst_w: width of the output image
    :return: cropped image b x dst_h x dst_w x c
    """
    if images.ndim != 4:
        raise ValueError("image must be a 4D array")
    _, h, w, _ = images.shape
    if h < dst_h or w < dst_w:
        raise ValueError("images dimensions is smaller than the output resolution")
    if h == dst_h and w == dst_w:
        return images
    corner_left = (w - dst_w) // 2
    corner_top = (h - dst_h) // 2
    output = images[
        :, corner_top : corner_top + dst_h, corner_left : corner_left + dst_w, :
    ]
    return output


def mask_regions(images, start_h, end_h, start_w, end_w):
    """
    masks a batch of numpy image with black background outside of region of interest (roi)
    :param image: numpy image b x h x w x c
    :param start_h: where does the roi height start
    :param end_h: where does the roi height end
    :param start_w: where does the roi width start
    :param end_w: where does the roi width end
    :return: masked image
    """
    if images.ndim != 4:
        raise ValueError("image must be a 4D array")
    _, h, w, _ = images.shape
    if start_h < 0 or start_w < 0 or end_h > h or end_w > w:
        raise ValueError("Values exceed image resolution")
    output = np.zeros_like(images)
    output[:, start_h:end_h, start_w:end_w, :] = images[start_h:end_h, start_w:end_w, :]
    return output


def adjust_contrast_brightness(img, alpha, beta=None):
    """
    adjusts image contrast and brightness using naive gain and bias factors
    :param img: input image numpy array (n x h x w x 3), float values between 0 and 1
    :param alpha: gain factor ("contrast") between 0 and inf
    :param beta: bias factor ("brightness") between -inf and inf (but typically between -1 and 1)
    :return: the new image
    """
    if img.dtype != np.float32:
        raise ValueError("img must be a float32 numpy array (0-1)")
    if beta is None:  # if beta is not provided, set to factor of alpha
        beta = 0.5 - alpha / 2
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    return new_img.astype(np.uint8)


def change_brightness(input_img, brightness=0):
    """
    changes brightness of an image or batch of images
    :param input_img a numpy or torch tensor of float values between 0 and 1 (n x h x w x 3)
    :param brightness a number between -255 to 255 (0=no change)
    :return the new image
    """
    if input_img.dtype != np.float32 and input_img.dtype != torch.float32:
        raise ValueError("input_img must be a float32 array (0-1)")
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow / 255
    else:
        alpha_b = 1
        gamma_b = 0
    return input_img * alpha_b + gamma_b


def linear_to_srgb(linear):
    """
    converts linear RGB to sRGB, see https://en.wikipedia.org/wiki/SRGB.
    note: linear is expected to be in the range [0, 1]
    """
    eps = np.finfo(np.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb):
    """
    converts linear RGB to sRGB, see https://en.wikipedia.org/wiki/SRGB.
    note: srgb is expected to be in the range [0, 1]
    """
    eps = np.finfo(np.float32).eps
    linear0 = 25 / 323 * srgb
    linear1 = np.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
    return np.where(srgb <= 0.04045, linear0, linear1)
