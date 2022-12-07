import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import interpolate, spatial
from .core import to_8b, to_float, to_hom, homogenize, broadcast_batch
from .structures import get_gizmo_coords

def alpha_compose(images, bg_color=None):
    """
    composes a single or batch of RGBA images into a single or batch of RGB images
    :param image: b x H x W x 4 or H x W x 4
    :param bg_color: 3 or b x 3
    :return: b x H x W x 3 or H x W x 3
    """
    if bg_color is None:
        bg_color = np.array([0., 0., 0.]).astype(np.float32)
    if images.ndim != 3 and images.ndim != 4:
        raise ValueError("image must be 3 or 4 dimensional")
    if images.shape[-1] != 4:
        raise ValueError("image must have 4 channels")
    if images.dtype != np.float32:
        images = to_float(images)
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
    is_float = images.dtype == np.float32
    if is_float:
        images = to_8b(images)
    rgbs = [Image.fromarray(x) for x in images]
    font = ImageFont.truetype("./FreeMono.ttf", 48)  # FreeSerif / FreeSans
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
    return rgbs

def draw_gizmo_on_image(np_images, w2c, isOpenGL=False, scale=.05):
    """
    adds a gizmo to a batch of np images.
    note: will broadcast np_images and w2c against eachother.
    :param np_images: b x H x W x 3
    :param w2c: b x 3 x 4 w2c transforms (opencv conventions)
    :param isOpenGL: if True, the w2c transforms are assumed to be in OpenGL conventions, else OpenCV conventions
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
        # W, H = image.shape[1], image.shape[0]
        gizmo_cords = get_gizmo_coords(scale)
        gizmo_hom = to_hom(gizmo_cords)  #  = np.concatenate((gizmo_cords, np.ones_like(gizmo_cords[:, 0:1])), axis=-1)
        verts_clip = (w2c[i] @ gizmo_hom.T).T
        verts_clip = homogenize(verts_clip) # verts_screen_xy = verts_screen[:, :2] / verts_screen[:, 2:3]
        if isOpenGL:
            raise NotImplementedError("OpenGL convention is not supported for now")
            # if not (np.abs(verts_screen_xy[:, 2]) <= 1).all():
            #    raise ValueError("OpenGL convention is not followed")
            # verts_clip = verts_clip[:, :2]
            # verts_screen = np.array([W, H]) * (verts_clip + 1) / 2
        else:
            verts_screen = verts_clip
        desired_loc = np.array([W - 40, H - 40])
        verts_screen += desired_loc - verts_screen[0]
        draw = ImageDraw.Draw(pil_image)
        draw.line((tuple(verts_screen[0]), tuple(verts_screen[1])), fill="red", width = 0)
        draw.line((tuple(verts_screen[0]), tuple(verts_screen[2])), fill="green", width = 0)
        draw.line((tuple(verts_screen[0]), tuple(verts_screen[3])), fill="blue", width = 0)
        new_images.append(np.array(pil_image))
    return (np.array(new_images) / 255.).astype(np.float32)

def merge_figures_with_line(img1, img2, lower_intersection=0.6, angle=np.pi/2, line_width=5):
    """
    merges two np images (H x W x 3) with a white line in between
    :param img1: (H x W x 3) numpy array
    :param img2: (H x W x 3) numpy array
    :param lower_intersection: lower intersection of the line with the images
    :param angle: angle of the line
    :param line_width: width of the line
    :return: new (H x W x 3) numpy array with line in between
    """
    combined = np.ones_like(img1)*255
    y, x, _ = img1.shape
    yy, xx = np.mgrid[:y, :x]
    img1_positions = (xx-lower_intersection*x)*np.tan(angle)-line_width//2>(yy-y)
    img2_positions = (xx-lower_intersection*x)*np.tan(angle)+line_width//2<(yy-y)
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
            img1.polygon(list(map(tuple, polygon)), fill=tuple(np.random.randint(0, 255, size=(3,))))
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def interpolate_single_channel(image: np.ndarray, mask: np.ndarray, method: str = "linear", fill_value: int = 0):
    """
    :param image: (H x W) numpy array
    :param mask: (H x W) boolean numpy array, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

def interpolate_multi_channel(image: np.ndarray, mask: np.ndarray):
    """
    given a multi channel image and a mask, interpolate the values where mask is true (per channel interpolation)
    :param image: (H x W x C) numpy array
    :param mask: (H x W) numpy array
    :return: (H x W x C) numpy array
    """
    if image.ndim != 3:
        raise ValueError("image must have atleast 1 channel")
    interpolated = np.zeros_like(image)
    for channel in range(image.shape[-1]):
        interpolated_channel = interpolate_single_channel(image[:, :, channel], mask)
        interpolated[:, :, channel] = interpolated_channel
    return interpolated