import torch
import numpy as np
from pathlib import Path
import igl
from .core import to_8b, to_np
from PIL import Image

def save_animation(images, dst:Path):
    """
    saves a gif animation
    :param images: (b x H x W x C) tensor
    :param dst: path to save animation to
    """
    if type(images) == torch.Tensor:
        images = to_np(images)
    if np.isnan(images).any():
        raise ValueError("Images must be finite")
    if images.dtype == np.float32:
        images = to_8b(images)
    if images.shape[-1] == 1:
        images = [Image.fromarray(image[..., 0], mode="L").convert('P') for image in images]
    else:
        images = [Image.fromarray(image) for image in images]
    dst = Path(dst.parent, dst.stem)
    images[0].save(str(dst)+".gif", save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)


def save_image(image, dst: Path, file_name: str = None, force_grayscale: bool = False):
    """
    saves single image as png
    :param image: (H x W x C) tensor
    :param dst: path to save image to
    :param force_grayscale: if True, saves image as grayscale
    :param file_name: if provided, saves image with this name
    """
    if image.ndim != 3:
        raise ValueError("Image must be 3 dimensional")
    save_images(image[None, ...], dst, [file_name], force_grayscale)

def save_images(images, dst: Path, file_names: list = [], force_grayscale: bool = False):
    """
    saves images as png
    :param images: (b x H x W x C) tensor
    :param dst: path to save images to
    :param force_grayscale: if True, saves images as grayscale
    :param file_names: if provided, saves images with these names (list of length b)
    """
    if type(images) == torch.Tensor:
        images = to_np(images)
    if np.isnan(images).any():
        raise ValueError("Images must be finite")
    if images.dtype == np.float32 or images.dtype == np.float64:
        images = to_8b(images)
    if images.ndim != 4:
        raise ValueError("Images must be of shape (b x H x W x C)")
    if file_names:
        if images.shape[0] != len(file_names):
            raise ValueError("Number of images and length of file names list must match")
    for i, image in enumerate(images):
        if force_grayscale or images.shape[-1] == 1:
            if images.shape[-1] == 3:
                image = image.mean(axis=-1, keepdims=True)
            pil_image = Image.fromarray(image[..., 0], mode="L")
        else:
            pil_image = Image.fromarray(image)
        if file_names is not None:
            file_names = [Path(x).stem for x in file_names]  # remove suffix
            pil_image.save(str(Path(dst, "{}.png".format(file_names[i]))))
        else:
            pil_image.save(str(Path(dst, "{:05d}.png".format(i))))

def load_images(path: Path, return_paths=False, to_torch=False, device=None):
    """
    loads images from a folder or a single image from a file
    :param path: path to folder with images / file
    :param return_paths: if True, returns a list of file paths
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :return: (b x H x W x 3) tensor, and optionally a list of file names
    """
    if not path.exists():
        raise ValueError("Path does not exist")
    if path.is_dir():
        images = []
        file_paths = []
        for image in path.iterdir():
            if image.suffix in [".png", ".jpg", ".jpeg"]:
                images.append(np.array(Image.open(str(image))))
                file_paths.append(image)
        images = np.stack(images, axis=0)
        if to_torch and device is not None:
            images = torch.tensor(images, dtype=torch.float32, device=device)
    elif path.is_file():
        images = np.array(Image.open(str(path)))
        file_paths = [path]
        if to_torch and device is not None:
            images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images[None, ...]
    if return_paths:
        return images, file_paths
    else:
        return images

def load_obj(path: Path, load_normals=False, to_torch=False, device=None):
    """
    uses igl backend to read an obj file
    :param path: path to obj file
    :param load_normals: if True, loads normals
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :return: (V x 3) tensor, (F x 3) tensor, and optionally (V x 3) tensor
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
    uses igl backend to write an obj file
    :param path: path to save obj file to
    :param vertices: (n x 3) tensor of vertices
    :param faces: (m x 3) tensor of vertex indices
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