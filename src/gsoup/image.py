import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import interpolate, spatial
from scipy.stats import wasserstein_distance
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
from pathlib import Path


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
            raise ValueError(
                "images and alpha must have the same batch size, or alphas batch size must equal 1"
            )
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


def draw_text_on_image(images, text_per_image, loc=(0, 0), size=48, fill_white=True):
    """
    writes text on images given as np array (b x H x W x 3)
    :param images: (b x H x W x 3) numpy array
    :param text_per_image: list or np array of strings
    :param loc: a tuple xy of anchor coordinates for the text (anchor is left-top of text)
    :param size: size of font
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
    resource_path = Path(__file__).parent.resolve()
    font = ImageFont.truetype(Path(resource_path, "FreeMono.ttf"), size)
    if fill_white:
        fill = "white"
    else:
        fill = "black"
    for i, rgb in enumerate(rgbs):
        text = text_per_image[i]
        ImageDraw.Draw(rgb).text(loc, text, fill=fill, font=font)
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


def generate_random_block_mask(size, block_size, batch_size=1):
    """
    Generates a random binary mask matrix with the specified size and block size.

    :param: size (int): Size of the square matrix (must be a power of 2).
    :param: block_size (int): Size of the square blocks (must be a power of 2 and ≤ size).

    :return: a binary np array of shape (size, size).
    """
    if not (size & (size - 1) == 0 and block_size & (block_size - 1) == 0):
        raise ValueError("Size and block_size must be powers of 2.")
    if block_size < 1:
        raise ValueError("Block size must be larger or equal to 1.")
    if block_size > size:
        raise ValueError("Block size must be less than or equal to the size.")

    num_blocks = size // block_size
    if batch_size > 1:
        random_blocks = np.random.randint(2, size=(batch_size, num_blocks, num_blocks))
        mask = np.kron(
            random_blocks, np.ones((block_size, block_size), dtype=bool)
        ).astype(bool)
    else:
        random_blocks = np.random.randint(2, size=(num_blocks, num_blocks))
        mask = np.kron(
            random_blocks, np.ones((block_size, block_size), dtype=bool)
        ).astype(bool)
    return mask


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


def generate_gaussian_image(height, width, center=(0, 0), sigma=(10, 10), theta=0):
    """
    generate an image of a gaussian
    :param: height: the height of the image
    :param: width: the width of the image
    :param: center: the x,y center of the gaussian (default is upper left corner)
    :param: theta: rotation of the gaussian
    :param: sigma: the sx,sy stdev in the x and y acis before rotation
    :return: the gaussian image
    """
    theta = 2 * np.pi * theta / 360
    x = np.arange(0, width, 1, np.float32)
    y = np.arange(0, height, 1, np.float32)
    y = y[:, np.newaxis]
    sx = sigma[0]
    sy = sigma[1]
    x0 = center[0]
    y0 = center[1]

    # rotation
    a = np.cos(theta) * x - np.sin(theta) * y
    b = np.sin(theta) * x + np.cos(theta) * y
    a0 = np.cos(theta) * x0 - np.sin(theta) * y0
    b0 = np.sin(theta) * x0 + np.cos(theta) * y0

    return np.exp(-(((a - a0) ** 2) / (2 * (sx**2)) + ((b - b0) ** 2) / (2 * (sy**2))))


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


def generate_concentric_circles(
    height, width, background="black", n=5, colors=None, dst=None
):
    """
    generates an image with colored concentric circles
    :param height: height of the image
    :param width: width of the image
    :param background: background color (as int, tuple of ints or name accepted by PIL)
    :param n: number of circles to draw
    :param colors: if not None, list of n colors of the circles where each color is a list of 3 \in [0,255]
    :param dst: if not None, the image is written to this path
    """
    spacing_x = width // (2 * n)
    spacing_y = height // (2 * n)
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    for i in range(n):
        x0 = spacing_x * i
        y0 = spacing_y * i
        if colors is None:
            img1.ellipse(
                [x0, y0, width - x0, height - y0],
                fill=tuple(np.random.randint(0, 255, size=(3,))),
            )
        else:
            img1.ellipse(
                [x0, y0, width - x0, height - y0],
                fill=tuple(colors[i]),
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
    :return: (H x W x 3) uint8 numpy array
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


def image_grid(images, rows, cols, pad=0, pad_color=None):
    """
    :param images: list of images
    :param rows: number of rows
    :param cols: number of cols
    :param pad: will pad images by this number of pixels with pad_color
    :param pad_color: a (3,) np array representing the pad color. if not provided pad will be black. must be same dtype as images.
    :return: grid image
    """
    if images.ndim != 4:
        raise ValueError("images must be a 4D array")
    if len(images) != rows * cols:
        raise ValueError("number of images must be equal to rows * cols")
    if pad > 0:
        images = pad_to_res(
            images, images.shape[1] + pad * 2, images.shape[2] + pad * 2, pad_color
        )
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


def resize(images, H, W, mode="bilinear"):
    """
    wrapper around torch interpolate (https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
    :param images: batch of np array (b, h, w, c) or torch tensors (b, c, h, w)
    :param H: output height
    :param W: output width
    :param mode: pass through for torch ()
    :return: same as input, with the new H and W
    """
    if images.ndim != 4:
        raise ValueError("images must be a 4D array")
    was_numpy = False
    if is_np(images):
        imgs_torch = to_torch(images).permute(2, 0, 1)
        was_numpy = True
    else:
        imgs_torch = images
    interpolated = torch.nn.functional.interpolate(
        imgs_torch,
        size=(H, W),
        scale_factor=None,
        mode=mode,
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    )
    if was_numpy:
        interpolated = to_np(interpolated.permute(1, 2, 0))
    return interpolated


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
    :param bg_color: background color sized (c,) (defaults to black)
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


def compute_color_distance(image1, image2, bin_per_dim=10):
    """
    computes a naive "color distance" between two images by binning the colors and computing the wasserstein distance per channel
    :param image1: numpy image h x w x 3
    :param image2: numpy image h x w x 3
    :return: the sum (over channels) of the wasserstein distances
    """
    if image1.shape != image2.shape:
        raise ValueError("images must have the same shape")
    if image1.ndim != 3:
        raise ValueError("images must be 3D arrays")
    if image1.dtype != np.uint8:
        raise ValueError("images must be uint8")
    spatial_size = image1.shape[0] * image1.shape[1]
    hist1_R, _ = np.histogram(image1[:, :, 0].reshape(-1), bins=bin_per_dim)
    hist2_R, _ = np.histogram(image2[:, :, 0].reshape(-1), bins=bin_per_dim)
    hist1_R = hist1_R / spatial_size
    hist2_R = hist2_R / spatial_size
    result_R = wasserstein_distance(hist1_R, hist2_R)
    ###
    hist1_G, _ = np.histogram(image1[:, :, 1].reshape(-1), bins=bin_per_dim)
    hist2_G, _ = np.histogram(image2[:, :, 1].reshape(-1), bins=bin_per_dim)
    hist1_G = hist1_G / spatial_size
    hist2_G = hist2_G / spatial_size
    result_G = wasserstein_distance(hist1_G, hist2_G)
    ###
    hist1_B, _ = np.histogram(image1[:, :, 2].reshape(-1), bins=bin_per_dim)
    hist2_B, _ = np.histogram(image2[:, :, 2].reshape(-1), bins=bin_per_dim)
    hist1_B = hist1_B / spatial_size
    hist2_B = hist2_B / spatial_size
    result_B = wasserstein_distance(hist1_B, hist2_B)
    result = result_R + result_G + result_B
    return result


def tonemap(
    hdr_image, exposure=0.0, offset=0.0, gamma=2.2, only_preproc=False, clip=True
):
    """
    maps an input image [-inf, inf] to [0, 1] using non-linear gamma correction.
    this slightly naive tonemapping was taken from https://github.com/Tom94/tev
    :param hdr_image: a numpy array or torch tensor (channels first or last, any float type)
    :param exposure: the image will be multiplied by 2*exposure prior to gamma correction
    :param offset: will be added to image (after multiplied by exposure, but prior to gamma correction)
    :param gamma: gamma to be used for non-linear correction
    :param only_preproc: if true, will not run gamma correction but only use exposure and offset
    :param clip: if true will clip result to [0.0, 1.0]
    :return: the tonemapped image, with same dtype and shape
    """
    if type(hdr_image) == np.ndarray:
        image = (hdr_image * np.power(2.0, exposure)) + offset
        if not only_preproc:
            image = np.sign(image) * np.power(np.abs(image), 1.0 / gamma)
        if clip:
            image = np.clip(image, 0.0, 1.0)
    elif type(hdr_image) == torch.Tensor:
        image = (hdr_image * np.power(2.0, exposure)) + offset
        if not only_preproc:
            image = torch.sign(image) * torch.pow(torch.abs(image), 1.0 / gamma)
        if clip:
            image = torch.clamp(image, 0.0, 1.0)
    else:
        raise TypeError("hdr_image must be either a numpy array or torch tensor")
    return image
