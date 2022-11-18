import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .core import to_8b, to_float

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

