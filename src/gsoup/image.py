import numpy as np
import torch
import torch.nn.functional as F
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
    is_float,
)
from .structures import get_gizmo_coords
from pathlib import Path


def add_alpha(images, alphas):
    """Adds an alpha channel to a batch of images.

    Args:
        images: Numpy image array of shape (b, h, w, 3) or (h, w, 3).
        alphas: Alpha channel array of shape (b, h, w, 1) or (h, w, 1).

    Returns:
        Numpy image array of shape (b, h, w, c+1) or (h, w, c+1) with alpha channel added.

    Raises:
        ValueError: If images don't have 3 channels or if dimensions don't match.
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
    """Composes RGBA images into RGB images using alpha blending.

    Composes a single or batch of RGBA images into RGB images. If backgrounds
    is provided, will blend them with the images, otherwise will blend with
    bg_color. If no backgrounds or bg_color is provided, the background is
    assumed to be black.

    Args:
        images: RGBA image array of shape (b, h, w, 4) or (h, w, 4).
        backgrounds: Optional background image array of shape (b, h, w, 3) or (h, w, 3).
        bg_color: Optional background color array of shape (3,) or (b, 3) float32.

    Returns:
        RGB image array of shape (b, h, w, 3) or (h, w, 3).

    Raises:
        ValueError: If images don't have 4 channels or if dimensions don't match.
    """
    if images.ndim != 3 and images.ndim != 4:
        raise ValueError("image must be 3 or 4 dimensional")
    if images.shape[-1] != 4:
        raise ValueError("image must have 4 channels")
    if backgrounds is not None:
        if images.shape[:-1] != backgrounds.shape[:-1]:
            raise ValueError("backgrounds must have same shape as images")
    was_float = True
    if is_np(images):
        if bg_color is None:
            bg_color = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        if images.dtype != np.float32 and images.dtype != np.float64:
            images = to_float(images)
            was_float = False
    else:
        if bg_color is None:
            bg_color = torch.tensor(
                [0.0, 0.0, 0.0], dtype=images.dtype, device=images.device
            )
        if images.dtype != torch.float32 and images.dtype != torch.float64:
            images = to_float(images)
            was_float = False
    if backgrounds is not None:
        if backgrounds.dtype != np.float32:
            backgrounds = to_float(backgrounds)
        bg_color = backgrounds
    alpha = images[..., 3:4]
    rgb = images[..., :3]
    result = alpha * rgb + (1 - alpha) * bg_color
    if not was_float:
        result = to_8b(result)
    return result


def draw_text_on_image(images, text_per_image, loc=(0, 0), size=48, color=None):
    """Draws text on images.

    Writes text on images given as numpy array. Supports both numpy and torch
    tensors, with automatic conversion between formats.

    Args:
        images: Image array of shape (b, h, w, 3) numpy array or torch tensor.
        text_per_image: String, list of strings, or numpy array of strings.
        loc: Tuple (x, y) of anchor coordinates for the text (anchor is left-top of text).
        size: Font size in pixels.
        color: Optional color array of shape (1,), (3,), (4,), (b, 1), (b, 3), or (b, 4)
            representing the color of the text (+alpha). Defaults to white if None.

    Returns:
        New image array of shape (b, h, w, 3) with text written, same type as input.
    """
    is_numpy = is_np(images)
    if not is_numpy:
        device = images.device
        images = to_np(images)
    if type(text_per_image) == str:
        text_per_image = [text_per_image]
    text_per_image = np.array(text_per_image)
    if text_per_image.shape[0] == 1:
        text_per_image = np.tile(text_per_image, images.shape[0])
    is_float = images.dtype == np.float32
    if is_float:
        images = to_8b(images)
    rgbs = [Image.fromarray(x) for x in images]
    resource_path = Path(__file__).parent.resolve()
    font = ImageFont.truetype(Path(resource_path, "FreeMono.ttf"), size)
    if color is None:
        color = np.full((len(rgbs), 4), 255, dtype=np.uint8)
    else:
        if type(color) == int:
            color = np.array([color])
        if color.ndim == 0:
            color = np.array([color])
        if color.ndim == 1:
            color = np.tile(color[None, :], (len(rgbs), 1))
        if color.shape[-1] == 1:
            color = np.tile(color, (1, 4))
        if color.shape[-1] == 3:
            color = np.concatenate((color, np.full((len(rgbs), 1), 255)), axis=-1)
    for i, rgb in enumerate(rgbs):
        text = text_per_image[i]
        ImageDraw.Draw(rgb).text(loc, text, fill=tuple(color[i]), font=font)
    rgbs = np.array([np.asarray(rgb) for rgb in rgbs])
    if is_float:
        rgbs = to_float(rgbs)
    if not is_numpy:
        rgbs = to_torch(rgbs, device=device)
    return rgbs


def draw_gizmo_on_image(np_images, w2c, opengl=False, scale=0.05):
    """Adds a coordinate gizmo to a batch of images.

    Draws a 3D coordinate gizmo (red=X, green=Y, blue=Z axes) on images.
    Will broadcast np_images and w2c against each other.

    Args:
        np_images: Image array of shape (b, h, w, 3).
        w2c: World-to-camera transform matrices of shape (b, 3, 4) using OpenCV conventions.
        opengl: If True, w2c transforms are assumed to be in OpenGL conventions.
            For OpenGL, w2c should be a (b, 4, 4) matrix converting from world to CLIP space.
        scale: Scale factor for the gizmo size.

    Returns:
        Image array of shape (b, h, w, 3) with gizmo drawn.

    Raises:
        ValueError: If np_images is not 4D or w2c is not 3D.
    """
    new_images = []
    if np_images.ndim != 4:
        raise ValueError("np_images must be (b, h, w, 3)")
    if w2c.ndim != 3:
        raise ValueError("KRt must be (b, 3, 4)")
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
    """Merges two images with a diagonal line separating them.

    Combines two images by placing them on opposite sides of a diagonal line,
    creating a split-screen effect.

    Args:
        img1: First image array of shape (h, w, 3).
        img2: Second image array of shape (h, w, 3).
        lower_intersection: Lower intersection point of the line with the images (0.0 to 1.0).
        angle: Angle of the diagonal line in radians.
        line_width: Width of the separating line in pixels.
        line_color: Color of the line as RGBA values.

    Returns:
        New image array of shape (h, w, 3) with merged images and separating line.
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
    """Generates a Voronoi diagram with random colored cells.

    Creates a Voronoi diagram of specified dimensions with randomly colored cells.

    Args:
        height: Height of the generated image.
        width: Width of the generated image.
        num_cells: Number of Voronoi cells to generate.
        bg_color: Background color for the image.
        dst: Optional path to save the image. If None, image is not saved.

    Returns:
        Image array of shape (h, w, 3) as uint8 numpy array.
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
    """Generates an image with colored circles arranged in a grid pattern.

    Creates a pattern of randomly colored circles arranged in a regular grid.

    Args:
        height: Height of the generated image.
        width: Width of the generated image.
        background: Background color for the image.
        radius: Radius of the circles in pixels.
        spacing: Spacing between circle centers in pixels.
        dst: Optional path to save the image. If None, image is not saved.

    Returns:
        Image array of shape (h, w, 3) as uint8 numpy array.
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
    """Generates a random binary mask matrix with specified block structure.

    Creates a binary mask where blocks of specified size are randomly set to
    True or False. Useful for creating structured random patterns.

    Args:
        size: Size of the square matrix (must be a power of 2).
        block_size: Size of the square blocks (must be a power of 2 and ≤ size).
        batch_size: Number of masks to generate in batch.

    Returns:
        Binary numpy array of shape (size, size) or (batch_size, size, size).

    Raises:
        ValueError: If size or block_size are not powers of 2, or if block_size > size.
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
    """Generates a checkerboard pattern.

    Creates a binary checkerboard pattern with alternating black and white squares.

    Args:
        h: Height of the image.
        w: Width of the image.
        blocksize: Size of each square in the checkerboard.

    Returns:
        Binary image array of shape (h, w, 1) as boolean numpy array.

    Note:
        If blocksize is not a divisor of w or h, the pattern will have extra
        "crops" at the edges.
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
    """Generates an image with colored stripes in specified direction.

    Creates a pattern of randomly colored stripes in vertical, horizontal,
    or both directions.

    Args:
        height: Height of the generated image.
        width: Width of the generated image.
        background: Background color for the image.
        direction: Direction of stripes - "vert" (vertical), "hor" (horizontal), or "both".
        thickness: Thickness of the stripes in pixels.
        spacing: Spacing between stripes in pixels.
        dst: Optional path to save the image. If None, image is not saved.

    Returns:
        Image array of shape (h, w, 3) as uint8 numpy array.

    Raises:
        ValueError: If direction is not one of "vert", "hor", or "both".
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
    """Generates a 2D Gaussian image.

    Creates an image containing a 2D Gaussian function with specified parameters.

    Args:
        height: Height of the generated image.
        width: Width of the generated image.
        center: Tuple (x, y) of the Gaussian center coordinates.
        sigma: Tuple (sx, sy) of standard deviations in x and y axes before rotation.
        theta: Rotation angle of the Gaussian in degrees.

    Returns:
        Image array containing the Gaussian function values.
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
    """Generates an image with a lollipop pattern.

    Creates a pattern of concentric circles arranged in a radial pattern,
    resembling lollipops.

    Args:
        height: Height of the generated image.
        width: Width of the generated image.
        background: Background color for the image.
        n: Number of circles in each radial pattern.
        m: Number of radial lines/patterns.
        dst: Optional path to save the image. If None, image is not saved.

    Returns:
        Image array of shape (h, w, 3) as uint8 numpy array.
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
    """Generates an image with colored concentric circles.

    Creates a pattern of concentric circles with specified colors.

    Args:
        height: Height of the generated image.
        width: Width of the generated image.
        background: Background color (as int, tuple of ints, or PIL color name).
        n: Number of circles to draw.
        colors: Optional list of n colors for the circles, where each color is
            a list of 3 values in [0, 255]. If None, random colors are used.
        dst: Optional path to save the image. If None, image is not saved.

    Returns:
        Image array of shape (h, w, 3) as uint8 numpy array.
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
    """Generates a gray gradient image.

    Creates a gradient image transitioning from black to white (or vice versa).

    Args:
        height: Height of the generated image.
        width: Width of the generated image.
        grayscale: If True, returns single-channel grayscale image.
        vertical: If True, gradient is vertical; if False, horizontal.
        flip: If True, gradient direction is flipped.
        bins: Number of discrete intensity levels in the gradient.

    Returns:
        Image array of shape (h, w, 3) or (h, w, 1) as uint8 numpy array.
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


def grid_image(images, rows, cols, pad=0, pad_color=None):
    return image_grid(images, rows, cols, pad=0, pad_color=None)


def image_grid(images, rows, cols, pad=0, pad_color=None):
    """Arranges images in a grid layout.

    Takes a batch of images and arranges them in a grid with specified
    number of rows and columns.

    Args:
        images: Image array of shape (B, H, W, C) as numpy array or torch tensor.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        pad: Number of pixels to pad around each image.
        pad_color: Optional color array of shape (C,) for padding. If not provided,
            padding will be black. Must be same dtype as images.

    Returns:
        Grid image combining all input images.

    Raises:
        ValueError: If images is not 4D or if number of images doesn't match rows * cols.
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
    if type(tmp) == torch.Tensor:  # todo, treat torch as (b, c, h, w)
        result = tmp.permute(0, 2, 1, 3, 4).reshape(
            rows * images.shape[1], cols * images.shape[2], -1
        )
    elif type(tmp) == np.ndarray:
        result = tmp.transpose(0, 2, 1, 3, 4).reshape(
            rows * images.shape[1], cols * images.shape[2], -1
        )
    return result


def resize(images, H, W, mode="bilinear"):
    """Resizes images using PyTorch interpolation.

    Wrapper around torch.nn.functional.interpolate for resizing images.
    Supports both numpy arrays and torch tensors with automatic conversion.

    Args:
        images: Batch of images as numpy array (b, h, w, c) or torch tensor (b, c, h, w).
        H: Output height.
        W: Output width.
        mode: Interpolation mode passed to torch.nn.functional.interpolate.

    Returns:
        Resized images with same type as input and new dimensions (H, W).

    Raises:
        ValueError: If images is not 4D.
    """
    if images.ndim != 4:
        raise ValueError("images must be a 4D array")
    was_numpy = False
    if is_np(images):
        imgs_torch = to_torch(images).permute(
            0, 3, 1, 2
        )  # (b, h, w, c) -> (b, c, h, w)
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
        interpolated = to_np(
            interpolated.permute(0, 2, 3, 1)
        )  # (b, c, h, w) -> (b, h, w, c)
    return interpolated


def resize_images_naive(images, H, W, channels_last=True, mode="mean"):
    """Resizes images using naive binning method.

    Resizes images by binning pixels, but only works when output size has
    a common divisor with input size. More efficient than interpolation for
    downsampling.

    Args:
        images: Numpy array of images of shape (b, h, w, c).
        H: Output height (must have common divisor with input height).
        W: Output width (must have common divisor with input width).
        channels_last: If True, images are in channels-last format (and output will be too).
        mode: Resizing method - "max" for max pooling or "mean" for average pooling.

    Returns:
        Numpy array of resized images.

    Raises:
        ValueError: If images is not 4D, not square, or mode is invalid.
        NotImplementedError: If channels_last is False.
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
    """Pads images to square shape.

    Pads a batch of images to make them square by adding padding to the
    smaller dimension.

    Args:
        images: Image array as numpy array (b, h, w, c) or torch tensor (b, c, h, w).
        color: Optional padding color of size (c,) with same dtype as images.
            Defaults to black if None.

    Returns:
        Padded square image with same type as input.

    Raises:
        ValueError: If images is not 4D.
    """
    if images.ndim != 4:
        raise ValueError("images must be a 4D array")

    if is_np(images):
        # numpy format: (b, h, w, c)
        h, w = images.shape[1], images.shape[2]
        max_dim = max(h, w)
        if h == w:
            return images
        return pad_to_res(images, max_dim, max_dim, color)
    else:
        # torch format: (b, c, h, w)
        h, w = images.shape[2], images.shape[3]
        max_dim = max(h, w)
        if h == w:
            return images
        return pad_to_res(images, max_dim, max_dim, color)


def crop_to_square(images):
    """Crops images to square shape.

    Crops a batch of images to make them square by removing pixels from
    the larger dimension.

    Args:
        images: Image array as numpy array (b, h, w, c) or torch tensor (b, c, h, w).

    Returns:
        Cropped square image with same type as input.

    Raises:
        ValueError: If images is not 4D.
    """
    if images.ndim != 4:
        raise ValueError("images must be a 4D array")

    if is_np(images):
        # numpy format: (b, h, w, c)
        h, w = images.shape[1], images.shape[2]
        if h == w:
            return images
        if h > w:
            s = int((h - w) / 2)
            return images[:, s : (s + w), :, :]
        else:
            s = int((w - h) / 2)
            return images[:, :, s : (s + h), :]
    else:
        # torch format: (b, c, h, w)
        h, w = images.shape[2], images.shape[3]
        if h == w:
            return images
        if h > w:
            s = int((h - w) / 2)
            return images[:, :, s : (s + w), :]
        else:
            s = int((w - h) / 2)
            return images[:, :, :, s : (s + h)]


def pad_to_res(images, res_h, res_w, bg_color=None):
    """Pads images to specific resolution.

    Pads a batch of images to a specific resolution. Images will be centered
    in the output image.

    Args:
        images: Batch of images as numpy arrays (b, h, w, c) or torch tensors (b, c, h, w).
        res_h: Height of the output image.
        res_w: Width of the output image.
        bg_color: Optional background color of size (c,) with same dtype as images.
            Defaults to black if None.

    Returns:
        Padded image of shape (b, res_h, res_w, c) or (b, c, res_h, res_w).

    Raises:
        ValueError: If images is not 4D, images are larger than output resolution,
            or bg_color has wrong number of channels.
    """
    if images.ndim != 4:
        raise ValueError("image must be a 4D array")
    if is_np(images):
        was_numpy = True
        images = to_torch(images).permute(0, 3, 1, 2)
    else:
        was_numpy = False
    if bg_color is None:
        bg_color = torch.zeros(
            images.shape[1], dtype=images.dtype, device=images.device
        )
    if is_np(bg_color):
        bg_color = to_torch(bg_color, device=images.device, dtype=images.dtype)
    b, c, h, w = images.shape
    if h > res_h or w > res_w:
        raise ValueError("images dimensions is larger than the output resolution")
    if h == res_h and w == res_w:
        return images
    if bg_color.shape[0] != c:
        raise ValueError(
            "background color must have the same number of channels as the image"
        )
    bg_color = bg_color[None, :, None, None]
    output = torch.zeros((b, c, res_h, res_w), dtype=images.dtype, device=images.device)
    output[:, :, :, :] = bg_color
    corner_left = (res_w - w) // 2
    corner_top = (res_h - h) // 2
    output[:, :, corner_top : corner_top + h, corner_left : corner_left + w] = images
    if was_numpy:
        output = to_np(output.permute(0, 2, 3, 1))
    return output


def crop_center(images, dst_h, dst_w):
    """Crops images from center to specific resolution.

    Crops a batch of images to a specific resolution by taking the center
    portion of each image.

    Args:
        images: Image array as numpy or torch array of shape (b, h, w, c).
        dst_h: Height of the output image.
        dst_w: Width of the output image.

    Returns:
        Cropped image of shape (b, dst_h, dst_w, c).

    Raises:
        ValueError: If images is not 4D or images are smaller than output resolution.
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
    """Masks images to show only region of interest.

    Masks a batch of images with black background outside the specified
    region of interest (ROI).

    Args:
        images: Image array of shape (b, h, w, c).
        start_h: Starting row of the ROI.
        end_h: Ending row of the ROI.
        start_w: Starting column of the ROI.
        end_w: Ending column of the ROI.

    Returns:
        Masked image with same shape as input.

    Raises:
        ValueError: If images is not 4D or ROI coordinates exceed image bounds.
    """
    if images.ndim != 4:
        raise ValueError("image must be a 4D array")
    _, h, w, _ = images.shape
    if start_h < 0 or start_w < 0 or end_h > h or end_w > w:
        raise ValueError("Values exceed image resolution")
    output = np.zeros_like(images)
    output[:, start_h:end_h, start_w:end_w, :] = images[start_h:end_h, start_w:end_w, :]
    return output


def adjust_contrast_brightness(img, alpha=1.0, beta=0.0):
    """Adjusts image contrast and brightness.

    Applies linear contrast and brightness adjustment using gain and bias factors.
    Formula: new_img = img * alpha + beta

    Args:
        img: Input image array of shape (n, h, w, 3) with float values between 0 and 1.
        alpha: Gain factor for contrast adjustment. Range: [0.0, inf].
        beta: Bias factor for brightness adjustment. Range: [-1.0, 1.0].

    Returns:
        Adjusted image with same shape and type as input.

    Raises:
        ValueError: If img is not a float array.
    """
    if not is_float(img):
        raise ValueError("img must be a float array")
    else:
        new_img = img * alpha + beta
        if is_np(new_img):
            new_img = np.clip(new_img, 0.0, 1.0)
        else:
            new_img = torch.clamp(new_img, 0.0, 1.0)
        return new_img


def linear_to_srgb(linear):
    """Converts linear RGB to sRGB color space.

    Converts linear RGB values to sRGB color space using the standard
    sRGB transfer function. See https://en.wikipedia.org/wiki/SRGB.

    Args:
        linear: Linear RGB values in range [0, 1]. Supports both numpy arrays
            and torch tensors.

    Returns:
        sRGB values in range [0, 1] with same shape and type as input.
    """
    if is_np(linear):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        eps = torch.finfo(torch.float32).eps
        # broadcast eps to the same shape as linear
        eps = torch.full(linear.shape, eps, device=linear.device)
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb):
    """Converts sRGB to linear RGB color space.

    Converts sRGB values to linear RGB color space using the inverse
    sRGB transfer function. See https://en.wikipedia.org/wiki/SRGB.

    Args:
        srgb: sRGB values in range [0, 1]. Supports both numpy arrays
            and torch tensors.

    Returns:
        Linear RGB values in range [0, 1] with same shape and type as input.
    """
    if is_np(srgb):
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        eps = torch.finfo(torch.float32).eps
        # broadcast eps to the same shape as srgb
        eps = torch.full(srgb.shape, eps, device=srgb.device)
        linear0 = 25 / 323 * srgb
        linear1 = torch.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)


def linear_to_xyz(linear_img):
    """Converts linear RGB color space to XYZ color space.

    Args:
        linear_img: Linear RGB image [0,1] as torch tensor (B, C, H, W) or numpy array (B, H, W, C).
    Returns:
        XYZ image [0,1] as torch tensor (B, C, H, W) or numpy array (B, H, W, C).
    """
    was_numpy = False
    if is_np(linear_img):
        linear_img = to_torch(linear_img, permute_channels=True)
        was_numpy = True
    # sRGB to XYZ (D65)
    M = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        device=linear_img.device,
        dtype=linear_img.dtype,
    )
    # Flatten to apply matrix multiplication
    B, C, H, W = linear_img.shape
    linear_img_flat = linear_img.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz_flat = linear_img_flat @ M.T
    xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
    if was_numpy:
        xyz = to_np(xyz, permute_channels=True)
    return xyz


def xyz_to_lab(xyz):
    """
    Convert a batch of XYZ images to Lab (D65).

    Reference: https://en.wikipedia.org/wiki/CIELAB_color_space
    note: output is in the range [0,100] for L, [-110,110] for a, [-110,110] for b

    Args:
        xyz: XYZ image [0,1] as torch tensor (B, 3, H, W) or numpy array (B, H, W, 3).
    Returns:
        Lab image as torch tensor (B, 3, H, W) or numpy array (B, H, W, 3).
    """
    was_numpy = False
    if is_np(xyz):
        was_numpy = True
        xyz = to_torch(xyz, permute_channels=True)
    # Normalize by reference white (D65)
    white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype
    ).view(1, 3, 1, 1)
    xyz_scaled = (xyz / white).clamp(min=1e-6)

    delta = 6 / 29

    def f(t):
        t = torch.clamp(t, min=0)
        return torch.where(t > delta**3, t.pow(1 / 3), (t / (3 * delta**2)) + (4 / 29))

    fX, fY, fZ = f(xyz_scaled[:, 0]), f(xyz_scaled[:, 1]), f(xyz_scaled[:, 2])
    L = 116 * fY - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    Lab = torch.stack([L, a, b], dim=1)
    if was_numpy:
        Lab = to_np(Lab, permute_channels=True)
    return Lab


def linear_to_luminance(linear_img, keep_channels=False):
    """Converts linear RGB to luminance.

    Converts linear RGB image to luminance using Rec. 709 weights.

    Args:
        linear_img: Linear RGB image as torch tensor (B, C, H, W) or numpy array (B, H, W, C).
        keep_channels: If True, returns luminance with channel dimension
            (B, 1, H, W) or (B, H, W, 1). If False, returns without channel dimension (B, H, W).

    Returns:
        Luminance values using Rec. 709 weights.

    Raises:
        ValueError: If image is not 4D.
    """
    was_numpy = False
    if is_np(linear_img):
        was_numpy = True
        linear_img = to_torch(
            linear_img, permute_channels=True
        )  # (B, H, W, C) -> (B, C, H, W)
    if linear_img.ndim != 4:
        raise ValueError("image must be 4D")
    b, c, h, w = linear_img.shape
    r, g, b = linear_img[:, 0], linear_img[:, 1], linear_img[:, 2]
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b  # (B, H, W)
    L = L.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    if was_numpy:
        L = to_np(L, permute_channels=True)  # (B, 1, H, W) -> (B, H, W, 1)
        if not keep_channels:
            L = L.squeeze(-1)  # (B, H, W, 1) -> (B, H, W)
    else:
        if not keep_channels:
            L = L.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
    return L


def inset(
    base_image,
    inset_image,
    corner="bottom_right",
    percent=0.2,
    margin=0.02,
    pad=0,
    pad_color=None,
):
    """Embeds an inset image into a base image at specified corner.

    Places a smaller version of the inset image into one of the corners
    of the base image, maintaining aspect ratio.

    Args:
        base_image: Base image as numpy array (b, h, w, c) or torch tensor (b, c, h, w).
        inset_image: Image to embed as numpy array (b, h, w, c) or torch tensor (b, c, h, w).
        corner: Corner position - "top_left", "top_right", "bottom_left", or "bottom_right".
        percent: Size of inset as percentage of base image's longest dimension (0.0 to 1.0).
        margin: Margin from corner as percentage of base image's longest dimension (0.0 to 1.0).
        pad: Number of pixels to pad around the inset image.
        pad_color: Color to use for padding.

    Returns:
        Base image with inset embedded, same type and shape as base_image.

    Raises:
        ValueError: If corner is invalid, percent is out of range, or margin is invalid.
    """
    if corner not in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        raise ValueError(
            "corner must be one of 'top_left', 'top_right', 'bottom_left', 'bottom_right'"
        )
    if percent <= 0.0 or percent > 1.0:
        raise ValueError("percent must be between 0.0 and 1.0")
    if margin < 0.0 or margin >= 1.0:
        raise ValueError("margin must be between 0.0 and 1.0")

    # ensure inputs are 4D (batched)
    if base_image.ndim != 4:
        raise ValueError("base_image must be 4D (b x h x w x c)")
    if inset_image.ndim != 4:
        raise ValueError("inset_image must be 4D (b x h x w x c)")
    was_numpy = False
    if is_np(base_image):
        was_numpy = True
        base_image = to_torch(base_image).permute(
            0, 3, 1, 2
        )  # (b, h, w, c) -> (b, c, h, w)
    if is_np(inset_image):
        was_numpy = True
        inset_image = to_torch(inset_image).permute(
            0, 3, 1, 2
        )  # (b, h, w, c) -> (b, c, h, w)
    # use broadcasting to handle different batch sizes
    base_image, inset_image = broadcast_batch(base_image, inset_image)
    if pad > 0:
        pad = int(pad)
        inset_image = pad_to_res(
            inset_image,
            inset_image.shape[2] + pad * 2,
            inset_image.shape[3] + pad * 2,
            pad_color,
        )
    # get dimensions
    base_h, base_w = base_image.shape[2:]
    inset_h, inset_w = inset_image.shape[2:]

    # calculate inset size based on longest dimension of base image
    max_base_dim = max(base_h, base_w)
    inset_size = int(max_base_dim * percent)

    # resize inset image to fit within the calculated size while maintaining aspect ratio
    if inset_h > inset_w:
        new_inset_h = inset_size
        new_inset_w = int(inset_size * inset_w / inset_h)
    else:
        new_inset_w = inset_size
        new_inset_h = int(inset_size * inset_h / inset_w)

    # resize inset image - resize function expects 4D input
    resized_inset = resize(inset_image, new_inset_h, new_inset_w, mode="bilinear")

    # calculate margin in pixels
    margin_pixels = int(max_base_dim * margin)

    # calculate inset position
    if corner == "top_left":
        start_h = margin_pixels
        start_w = margin_pixels
    elif corner == "top_right":
        start_h = margin_pixels
        start_w = base_w - margin_pixels - new_inset_w
    elif corner == "bottom_left":
        start_h = base_h - margin_pixels - new_inset_h
        start_w = margin_pixels
    else:  # bottom_right
        start_h = base_h - margin_pixels - new_inset_h
        start_w = base_w - margin_pixels - new_inset_w

    # ensure inset fits within base image bounds
    start_h = max(0, min(start_h, base_h - new_inset_h))
    start_w = max(0, min(start_w, base_w - new_inset_w))

    # create output image (copy base image)
    if is_np(base_image):
        result = base_image.copy()
    else:
        result = base_image.clone()

    # embed inset image
    end_h = start_h + new_inset_h
    end_w = start_w + new_inset_w
    result[:, :, start_h:end_h, start_w:end_w] = resized_inset

    if was_numpy:
        result = to_np(result.permute(0, 2, 3, 1))
    return result


def compute_color_distance(image1, image2, bin_per_dim=10):
    """Computes color distance between two images.

    Computes a naive "color distance" between two images by binning the colors
    and computing the Wasserstein distance per channel.

    Args:
        image1: First image as numpy array of shape (h, w, 3).
        image2: Second image as numpy array of shape (h, w, 3).
        bin_per_dim: Number of bins per dimension for color histogram.

    Returns:
        Sum of Wasserstein distances over all channels.

    Raises:
        ValueError: If images have different shapes, are not 3D, or are not uint8.
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


def tonemap_reinhard(hdr_image, exposure=1.0, clip=True):
    """Applies Reinhard tonemapping to HDR image.

    Maps an input HDR image from [-inf, inf] to [0, 1] using Reinhard's
    tonemapping operator: x / (1 + x).

    Args:
        hdr_image: HDR image as numpy array or torch tensor (any float type).
        exposure: Exposure factor applied as e*x / (1 + e*x).
        clip: If True, clips result to [0.0, 1.0].

    Returns:
        Tonemapped image with same dtype and shape as input.

    Raises:
        TypeError: If hdr_image is not numpy array or torch tensor.
    """
    image = exposure * hdr_image
    image = image / (image + 1.0)
    if clip:
        if type(hdr_image) == np.ndarray:
            image = np.clip(image, 0.0, 1.0)
        elif type(hdr_image) == torch.Tensor:
            image = torch.clamp(image, 0.0, 1.0)
        else:
            raise TypeError("hdr_image must be either a numpy array or torch tensor")
    return image


def tonemap_tev(
    hdr_image, exposure=0.0, offset=0.0, gamma=2.2, only_preproc=False, clip=True
):
    """Applies Tev-style tonemapping to HDR image.

    Maps an input HDR image from [-inf, inf] to [0, 1] using non-linear gamma
    correction. This tonemapping was taken from https://github.com/Tom94/tev.

    Args:
        hdr_image: HDR image as numpy array or torch tensor (any float type).
        exposure: Image will be multiplied by 2^exposure prior to gamma correction.
        offset: Value added to image after exposure multiplication but before gamma correction.
        gamma: Gamma value for non-linear correction.
        only_preproc: If True, only applies exposure and offset without gamma correction.
        clip: If True, clips result to [0.0, 1.0].

    Returns:
        Tonemapped image with same dtype and shape as input.

    Raises:
        TypeError: If hdr_image is not numpy array or torch tensor.
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


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts image to patches using unfold operation.

    Splits an image into non-overlapping patches using PyTorch's unfold operation.

    Args:
        x: Input image tensor of shape (b, c, h, w).
        patch_size: Size of each patch (must divide h and w evenly).

    Returns:
        Patches tensor of shape (b, num_patches, c, ph, pw).

    Raises:
        ValueError: If patch_size doesn't divide image dimensions evenly.
    """
    b, c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            "current patchify/unpatchify does not support overlapping patches"
        )

    # unfold: (b, c*ph*pw, num_patches)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)

    # reshape to (b, num_patches, c, ph, pw)
    patches = patches.transpose(1, 2).reshape(b, -1, c, patch_size, patch_size)
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, h: int, w: int) -> torch.Tensor:
    """Reconstructs image from unfold-style patches.

    Reconstructs the original image from patches created by the patchify function.

    Args:
        patches: Patches tensor of shape (b, num_patches, c, ph, pw).
        patch_size: Size of each patch.
        h: Original image height.
        w: Original image width.

    Returns:
        Reconstructed image tensor of shape (b, c, h, w).

    Raises:
        ValueError: If patch dimensions don't match patch_size.
    """
    b, num_patches, c, ph, pw = patches.shape
    if ph != patch_size or pw != patch_size:
        raise ValueError(
            "current patchify/unpatchify does not support overlapping patches"
        )

    # back to (b, c*ph*pw, num_patches)
    patches = patches.reshape(b, num_patches, -1).transpose(1, 2)

    # fold: put patches back
    x = F.fold(patches, output_size=(h, w), kernel_size=patch_size, stride=patch_size)
    return x
