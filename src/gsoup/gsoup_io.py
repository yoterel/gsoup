import torch
import numpy as np
from pathlib import Path
from .core import to_8b, to_np
from PIL import Image
import json

def write_to_json(data, dst):
    """
    writes data to json file
    :param data: data to write
    :param dst: path to save json file to
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def save_animation(images, dst):
    """
    saves a gif animation
    :param images: (b x H x W x C) tensor
    :param dst: path to save animation to
    """
    dst = Path(dst)
    if type(images) == torch.Tensor:
        images = to_np(images)
    if np.isnan(images).any():
        raise ValueError("Images must be finite")
    if images.dtype != np.uint8:
        images = to_8b(images)
    if images.shape[-1] == 1:
        images = [Image.fromarray(image[..., 0], mode="L").convert('P') for image in images]
    else:
        images = [Image.fromarray(image) for image in images]
    dst = Path(dst.parent, dst.stem)
    images[0].save(str(dst)+".gif", save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

def save_image(image, dst, force_grayscale: bool = False, overwrite: bool = True, extension: str = "png"):
    """
    saves single image as png
    :param image: (H x W x C) tensor or (H x W) tensor
    :param dst: path to save image to (full path to destination, suffix not neccessary but allowed)
    :param force_grayscale: if True, saves image as grayscale
    :param file_name: if provided, saves image with this name
    """
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError("Image must be 2 or 3 dimensional")
    dst = Path(dst)
    save_images(image[None, ...], dst.parent, [dst.name], force_grayscale, overwrite, extension)

def save_images(images, dst, file_names: list = [], force_grayscale: bool = False, overwrite: bool = True, extension: str = "png"):
    """
    saves images as png
    :param images: (b x H x W x C) tensor
    :param dst: path to save images to (will create folder if it does not exist)
    :param force_grayscale: if True, saves images as grayscale
    :param file_names: if provided, saves images with these names (list of length b)
    """
    if type(images) == torch.Tensor:
        images = to_np(images)
    if np.isnan(images).any():
        raise ValueError("Images must be finite")
    if images.dtype == np.float32 or images.dtype == np.float64 or images.dtype == bool:
        images = to_8b(images)
    if images.dtype != np.uint8:
        raise ValueError("Images must be of type uint8 (or float32/64, which will be converted to uint8)")
    if images.ndim != 4:
        raise ValueError("Images must be of shape (b x H x W x C)")
    if file_names:
        if images.shape[0] != len(file_names):
            raise ValueError("Number of images and length of file names list must match")
        file_names = [Path(x).stem for x in file_names]  # remove suffix
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        if force_grayscale or images.shape[-1] == 1:
            if images.shape[-1] == 3:
                image = image.mean(axis=-1, keepdims=True).astype(np.uint8)
            pil_image = Image.fromarray(image[..., 0], mode="L")
        else:
            pil_image = Image.fromarray(image)
        if file_names:
            cur_dst = Path(dst, "{}.{}".format(file_names[i], extension))
        else:
            cur_dst = Path(dst, "{:05d}.{}".format(i, extension))
        if not overwrite:
            if cur_dst.exists():
                continue
        pil_image.save(str(cur_dst))

def load_image(path, to_float=False, channels_last=True, to_torch=False, device=None, resize_wh=None, as_grayscale=False):
    """
    loads an image from a single file
    :param path: path to file
    :param to_float: if True, converts image to float
    :param return_paths: if True, returns a list of file paths
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :param resize_wh: a tuple (w, h) to resize the image using nearest neighbor interpolation
    :param as_grayscale: if True, loads image as grayscale by averaging over the channels
    :return: (H x W x C) tensor, and optionally a list of file names
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Path does not exist")
    if path.is_dir():
        raise FileNotFoundError("Path must be a file")
    elif path.is_file():
        image = load_images(path, to_float=to_float, channels_last=channels_last, return_paths=False, to_torch=to_torch, device=device, resize_wh=resize_wh, as_grayscale=as_grayscale)
        return image[0]

def load_images(source, to_float=False, channels_last=True, return_paths=False, to_torch=False, device=None, resize_wh=None, as_grayscale=False):
    """
    loads images from a list of paths, a folder or a single file
    :param source: path to folder with images / path to image file / list of paths
    :param to_float: if True, converts images to float (and normalizes to [0, 1])
    :param return_paths: if True, returns a list of file paths
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :param resize_wh: a tuple (w, h) to resize all images using nearest neighbor interpolation
    :param as_grayscale: if True, loads images as grayscale by averaging over the channels
    :return: (b x H x W x C) tensor, and optionally a list of file names
    """
    supported_suffixes = [".png", ".jpg", ".jpeg"]
    if type(source) == list or type(source) == tuple or type(source) == np.ndarray:
        images = []
        file_paths = []
        for p in source:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError("Path does not exist")
            if p.suffix in supported_suffixes:
                im = Image.open(str(p))
                if resize_wh is not None:
                    im = im.resize(resize_wh)
                images.append(np.array(im))
                file_paths.append(p)
        images = np.stack(images, axis=0)
        if as_grayscale and images.ndim == 4:
            images = images.mean(axis=-1).astype(np.float32)
        if not channels_last and images.ndim == 4:
            images = np.moveaxis(images, -1, 1)
        if to_float:
            images = images.astype(np.float32) / 255
        else:
            images = images.astype(np.uint8)
        if to_torch:
            if device is None:
                device = torch.device("cpu")
            images = torch.tensor(images, device=device)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError("Path does not exist")
        if path.is_dir():
            images = []
            file_paths = []
            for image in sorted(path.iterdir()):
                if image.suffix in supported_suffixes:
                    im = Image.open(str(image))
                    if resize_wh is not None:
                        im = im.resize(resize_wh)
                    images.append(np.array(im))
                    file_paths.append(image)
            images = np.stack(images, axis=0)
            if as_grayscale and images.ndim == 4:
                images = images.mean(axis=-1).astype(np.float32)
            if not channels_last and images.ndim == 4:
                images = np.moveaxis(images, -1, 1)
            if to_float:
                images = images.astype(np.float32) / 255
            else:
                images = images.astype(np.uint8)
            if to_torch:
                if device is None:
                    device = torch.device("cpu")
                images = torch.tensor(images, device=device)
        elif path.is_file():
            im = Image.open(str(path))
            if resize_wh is not None:
                im = im.resize(resize_wh)
            images = np.array(im)
            if as_grayscale and images.ndim == 3:
                images = images.mean(axis=-1).astype(np.float32)
            if not channels_last and images.ndim == 3:
                images = np.moveaxis(images, -1, 0)
            if to_float:
                images = images.astype(np.float32) / 255
            else:
                images = images.astype(np.uint8)
            file_paths = [path]
            if to_torch:
                if device is None:
                    device = torch.device("cpu")
                images = torch.tensor(images, device=device)
            images = images[None, ...]
    if return_paths:
        return images, file_paths
    else:
        return images

def load_mesh(path: Path, to_torch=False, device=None, verbose=True):
    """
    loads a mesh from a file
    :param path: path to mesh file
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :return: (V x 3) tensor of vertices, (F x 3) tensor of faces, and optionally (V x 3) tensor of normals
    """
    path = Path(path)
    if path.suffix != ".obj":
        raise ValueError("Only .obj are supported")
    return load_obj(path, to_torch=to_torch, device=device, verbose=verbose)

def load_obj(path: Path, to_torch=False, device=None, verbose=True):
    """
    :param path: path to obj file
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :return: (V x 3) tensor, (F x 3) tensor, and optionally (V x 3) tensor
    """
    path = Path(path)
    if not path.exists():
        raise ValueError("Path does not exist")
    if not path.is_file():
        raise ValueError("Path must be a file")
    if path.suffix != ".obj":
        raise ValueError("Only .obj are supported")
    v, f = parse_obj(path, verbose=verbose)
    if to_torch and device is not None:
        v = torch.tensor(v, dtype=torch.float, device=device)
        f = torch.tensor(f,dtype=torch.long, device=device)
        # n = torch.tensor(n, dtype=torch.float, device=device)
    return v, f

def parse_obj(path: Path, verbose=True):
    """
    A simple obj parser
    currently supports vertex and triangular face elements only
    """
    path = Path(path)
    vertexList = []
    vertexTextureList = []
    vertexNormalList = []
    colorList = []
    faceList = []
    finite_flag = False
    with open(path, 'r') as objFile:
        for line in objFile:
            line = line.split("#")[0]  # remove comments
            split = line.split()
            if not len(split):  # skip empty lines
                continue
            if split[0] == "v":
                if 3 <= len(split[1:]) <= 4:  # x y z [w]
                    float_vertex = np.array([np.float64(x) for x in split[1:]])
                    if not np.isfinite(float_vertex).all():
                        finite_flag=True
                    vertexList.append(float_vertex)
                elif len(split[1:]) == 6:
                    float_vertex = np.array([np.float64(x) for x in split[1:4]])
                    float_color = np.array([np.float64(x) for x in split[4:]])
                    vertexList.append(float_vertex)
                    colorList.append(float_color)
                else:
                    raise ValueError("vertex {} has {} entries, but only 3 or 6 are supported".format(len(vertexList), len(split[1:])))
            elif split[0] == "f":
                if len(split[1:]) != 3:
                    raise ValueError("only triangular faces are supported")
                else:
                    face = [x.split("/")[0] for x in split[1:]]
                    int_face = np.array([int(x)-1 for x in face])
                    if np.any(int_face < 0):
                        raise ValueError("negative face indices are not supported")
                    faceList.append(int_face)
            elif split[0] == "vn":
                if len(split[1:]) != 3:  # xn yn zn
                    raise ValueError("vertex normal {} has {} entries, but only 3 are supported".format(len(vertexNormalList), len(split[1:])))
                else:
                    float_vertex_normal = np.array([np.float64(x) for x in split[1:]])
                    vertexNormalList.append(float_vertex_normal)
            elif split[0] == "vt":
                if 2 <= len(split[1:]) <= 3:  # u [v, w]
                    float_vertex_tex = np.array([np.float64(x) for x in split[1:]])
                    if (float_vertex_tex < 0).any():
                        raise ValueError("negative texture coordinates are not supported")
                    if (float_vertex_tex > 1).any():
                        raise ValueError("texture coordinates must be between 0 and 1")
                    vertexTextureList.append(float_vertex_tex)
                else:
                    raise ValueError("vertex normal {} has {} entries, but only 3 are supported".format(len(vertexNormalList), len(split[1:])))
            elif split[0] == "l":  # ignore polyline elements
                continue
            elif split[0] == "vp":  # ignore parameter space vertices
                continue
    v = np.stack(vertexList)
    f = np.stack(faceList)
    if finite_flag and verbose:
        print("Warning: some vertices in file were not finite")
    # vn = np.stack(vertexNormalList)
    # vt = np.stack(vertexTextureList)
    return v, f #, vn, vt

def save_obj(vertices, faces, path: Path):
    """"
    :param path: path to save obj file to
    :param vertices: (n x 3) tensor of vertices
    :param faces: (m x 3) tensor of vertex indices
    """
    path = Path(path)
    if path.suffix != ".obj":
        raise ValueError("Path must have suffix .obj")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if type(vertices) == torch.Tensor:
        vertices = to_np(vertices)
    if type(faces) == torch.Tensor:
        faces = to_np(faces)
    if (faces < 0).any():
        raise ValueError("Faces must be positive")
    if np.isnan(vertices).any():
        raise ValueError("Vertices must be finite")
    if np.isnan(faces).any():
        raise ValueError("Faces must be finite")
    if vertices.dtype != np.float32 and vertices.dtype != np.float64:
        raise ValueError("Vertices must be of type float32 / float64")
    if faces.dtype != np.int32 and faces.dtype != np.int64:
        raise ValueError("Faces must be of type int32 / int64")
    with open(str(path), "w") as file:
        for v in vertices:
            file.write("v {} {} {}\n".format(v[0], v[1], v[2]))  # write vertices
        for f in faces:
            file.write("f {} {} {}\n".format(f[0] + 1, f[1] + 1, f[2] + 1))  # obj indices start at 1

def save_ply(vertices, faces, path: Path):
    raise NotImplementedError

def save_mesh(vertices, faces, path):
    """
    saves a mesh to a file
    :param path: path to save mesh to
    :param vertices: (n x 3) tensor of vertices
    :param faces: (m x 3) tensor of vertex indices
    """
    path = Path(path)
    if path.suffix not in [".obj"]:
        raise ValueError("Only .obj is supported")
    else:
        if path.suffix == ".obj":
            save_obj(vertices, faces, path)
        # elif path.suffix == ".ply":
        #     save_ply(path, vertices, faces)

def save_pointcloud(vertices, path: Path):
    path = Path(path)
    if path.suffix != ".ply":
        raise ValueError("Only .ply are supported")
    else:
        save_ply(vertices, None, path)

def save_meshes(vertices, faces, path, file_names: list = []):
    """
    saves a list of meshes to a folder
    :param path: path to save meshes to
    :param vertices: (b x V x 3) tensor
    :param faces: (b x F x 3) tensor
    :param file_names: list of file names (of length b)
    """
    if vertices.ndim != 3 or faces.ndim != 3:
        raise ValueError("Vertices and faces must be 3D tensors")
    if vertices.shape[0] != faces.shape[0]:
        raise ValueError("Vertices and faces must have the same batch size")
    if file_names:
        if len(file_names) != vertices.shape[0]:
            raise ValueError("Number of file names must match batch size")
    path = Path(path)
    for i, (v, f) in enumerate(zip(vertices, faces)):
        if file_names:
            save_mesh(v, f, path / "{}.obj".format(file_names[i]))
        else:
            save_mesh(v, f, path / "{:05d}.obj".format(i))