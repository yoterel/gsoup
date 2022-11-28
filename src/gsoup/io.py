import torch
import numpy as np
from pathlib import Path
import igl
from .core import to_8b, to_np
from PIL import Image

def save_animation(images, dst:Path):
    """
    saves a gif animation
    :param images: (b x H x W x 3) tensor
    :param dst: path to save animation to
    """
    if type(images) == torch.Tensor:
        images = to_np(images)
    if np.isnan(images).any():
        raise ValueError("Images must be finite")
    if images.dtype == np.float32:
        images = to_8b(images)
    if images.shape[-1] == 1:
        images = [Image.fromarray(image, mode="L") for image in images]
    else:
        images = [Image.fromarray(image) for image in images]
    images[0].save(str(dst), save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)


def save_images(images, dst: Path, force_grayscale=False):
    """
    saves images as png
    :param images: (b x H x W x 3) numpy array
    :param dst: path to save images to
    :param force_grayscale: if True, saves images as grayscale
    """
    if type(images) == torch.Tensor:
        images = to_np(images)
    if np.isnan(images).any():
        raise ValueError("Images must be finite")
    if images.dtype == np.float32:
        images = to_8b(images)
    for i, image in enumerate(images):
        if force_grayscale or images.shape[-1] == 1:
            if images.shape[-1] == 3:
                image = image.mean(axis=-1)
            pil_image = Image.fromarray(image, mode="L")
        else:
            pil_image = Image.fromarray(image)
        pil_image.save(str(Path(dst, "{:05d}.png".format(i))))

def load_images(path: Path, to_torch=False, device=None):
    """
    loads images from a folder
    :param path: path to folder
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :return: (b x H x W x 3) tensor
    """
    if not path.exists():
        raise ValueError("Path does not exist")
    if path.is_dir():
        images = []
        for image in path.iterdir():
            if image.suffix in [".png", ".jpg", ".jpeg"]:
                images.append(np.array(Image.open(str(image))))
        images = np.stack(images, axis=0)
        if to_torch and device is not None:
            images = torch.tensor(images, dtype=torch.float32, device=device)
    elif path.is_file():
        images = np.array(Image.open(str(path)))
        if to_torch and device is not None:
            images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images[None, ...]
    return images

def load_obj(path: Path, load_normals=False, to_torch=False, device=None):
    """
    needs explaining?
    use igl backend.
    """
    v, _, n, f, _, _ = igl.read_obj(str(path))
    if to_torch and device is not None:
        v = torch.tensor(v, dtype=torch.float, device=device)
        f = torch.tensor(f,dtype=torch.long, device=device)
        n = torch.tensor(n, dtype=torch.float, device=device)
    if load_normals:
        return v, f, n
    else:
        return v, f

def save_obj(path: Path, vertices, faces):
    """"
    saves a mesh as an obj file
    use igl backend.
    """
    filename = Path(filename)
    if filename.suffix not in [".obj", ".ply"]:
        raise ValueError("Only .obj and .ply are supported")
    if type(vertices) == torch.Tensor:
        vertices = vertices.detach().cpu().numpy()
    if type(faces) == torch.Tensor:
        faces = faces.detach().cpu().numpy()
    if (faces < 0).any():
        raise ValueError("Faces must be positive")
    if np.isnan(vertices).any():
        raise ValueError("Vertices must be finite")
    if np.isnan(faces).any():
        raise ValueError("Faces must be finite")
    if vertices.dtype != np.float32:
        raise ValueError("Vertices must be of type float32")
    if faces.dtype != np.int64:
        raise ValueError("Faces must be of type int64")
    igl.write_obj(str(path), vertices, faces)