import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .core import to_8b, to_float, to_np
from scipy.spatial import Voronoi

def write_text_on_image(images, text_per_image, fill_white=True):
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
    font = ImageFont.truetype("../data/FreeMono.ttf", 48)  # FreeSerif / FreeSans
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

def generate_voronoi_diagram(height, width, num_cells=1000, dst=None):
    nx = np.random.rand(num_cells) * width
    ny = np.random.rand(num_cells) * height
    nxy = np.stack((nx, ny), axis=-1)
    img = Image.new("RGB", (width, height), "white")
    vor = Voronoi(nxy)
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
