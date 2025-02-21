import torch
import numpy as np
from pathlib import Path
from .core import to_8b, to_np, to_float
from .image import alpha_compose
from PIL import Image
import json
from struct import unpack, calcsize
from collections import OrderedDict
import OpenEXR


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


def save_animation(images, dst, ms_per_frame=100):
    """
    saves a gif animation
    :param images: (b x H x W x C) tensor
    :param dst: path to save animation to
    :param ms_per_frame: the display duration of each frame in ms
    """
    dst = Path(dst)
    if type(images) == torch.Tensor:
        images = to_np(images)
    if np.isnan(images).any():
        raise ValueError("Images must be finite")
    if images.dtype != np.uint8:
        images = to_8b(images)
    if images.shape[-1] == 1:
        images = [
            Image.fromarray(image[..., 0], mode="L").convert("P") for image in images
        ]
    else:
        images = [Image.fromarray(image) for image in images]
    dst = Path(dst.parent, dst.stem)
    images[0].save(
        str(dst) + ".gif",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=ms_per_frame,
        loop=0,
    )


def save_image(
    image,
    dst,
    force_grayscale: bool = False,
    overwrite: bool = True,
    extension: str = "png",
):
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
    save_images(
        image[None, ...], dst.parent, [dst.name], force_grayscale, overwrite, extension
    )


def save_images(
    images,
    dst,
    file_names: list = [],
    force_grayscale: bool = False,
    overwrite: bool = True,
    extension: str = "png",
):
    """
    saves images as png
    :param images: (b x H x W x C) np array, or list of (H X W X C)
    :param dst: path to save images to (will create folder if it does not exist)
    :param file_names: if provided, saves images with these names (list of length b)
    :param force_grayscale: if True, saves images as grayscale
    :param overwrite: if True, overwrites existing images
    :param extension: file extension to save images as
    """
    images = to_np(images)
    if np.isnan(images).any():
        raise ValueError("Images must be finite")
    if extension != "tiff":
        if (
            images.dtype == np.float32
            or images.dtype == np.float64
            or images.dtype == bool
        ):
            images = to_8b(images)
        if images.dtype != np.uint8:
            raise ValueError(
                "Images must be of type uint8 (or float32/64, which will be converted to uint8)"
            )
    if images.ndim != 4:
        raise ValueError("Images must be of shape (b x H x W x C)")
    if file_names:
        if images.shape[0] != len(file_names):
            raise ValueError(
                "Number of images and length of file names list must match"
            )
        file_names = [Path(x).stem for x in file_names]  # remove suffix
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        if force_grayscale or images.shape[-1] == 1:
            if images.shape[-1] == 3:
                image = image.mean(axis=-1, keepdims=True).astype(np.uint8)
            if extension == "tiff":
                pil_image = Image.fromarray(image[..., 0])
            else:
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


def load_image(
    path,
    as_float=False,
    channels_last=True,
    to_torch=False,
    device=None,
    resize_wh=None,
    as_grayscale=False,
):
    """
    loads an image from a single file
    :param path: path to file
    :param as_float: if True, converts image to float
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
        image = load_images(
            [path],
            as_float=as_float,
            channels_last=channels_last,
            return_paths=False,
            to_torch=to_torch,
            device=device,
            resize_wh=resize_wh,
            as_grayscale=as_grayscale,
        )
        return image[0]


def load_images(
    source,
    as_float=False,
    channels_last=True,
    return_paths=False,
    to_torch=False,
    device=None,
    resize_wh=None,
    as_grayscale=False,
):
    """
    loads images from a list of paths, a folder or a single file
    :param source: path to folder with images / path to image file / list of paths
    :param as_float: if True, converts images to float (and normalizes to [0, 1])
    :param return_paths: if True, returns a list of file paths
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :param resize_wh: a tuple (w, h) to resize all images using nearest neighbor interpolation
    :param as_grayscale: if True, loads images as grayscale by averaging over the channels
    :return: (b x H x W x C) tensor, and optionally a list of file names
    """
    supported_suffixes = [".png", ".jpg", ".jpeg", ".tiff"]
    images = []
    file_paths = []
    if type(source) == list or type(source) == tuple or type(source) == np.ndarray:
        for p in source:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError("Path does not exist: {}".format(p))
            if p.suffix in supported_suffixes:
                im = Image.open(str(p))
                if im.mode == "P":
                    im = im.convert("RGB")
                if resize_wh is not None:
                    im = im.resize(resize_wh)
                im_array = np.array(im)
                if im_array.ndim == 2:  # mode was "L"
                    im_array = im_array[:, :, None]
                images.append(im_array)
                file_paths.append(p)
    else:  # path to a folder
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError("Path does not exist: {}".format(path))
        if not path.is_dir():
            raise FileNotFoundError(
                "Path must be a folder or a list/tuple/array of paths"
            )
        for image in sorted(path.iterdir()):
            if image.suffix in supported_suffixes:
                im = Image.open(str(image))
                if im.mode == "P":
                    im = im.convert("RGB")
                if resize_wh is not None:
                    im = im.resize(resize_wh)
                im_array = np.array(im)
                if im_array.ndim == 2:  # mode was "L"
                    im_array = im_array[:, :, None]
                images.append(im_array)
                file_paths.append(image)
    images = np.stack(images, axis=0)
    if as_grayscale and images.shape[-1] != 1:
        if (
            images.shape[-1] == 4
        ):  # alpha compose before converting to grayscale as alpha is not affected by averaging
            images = alpha_compose(images)
        else:
            images = to_float(images)
        images = images.mean(axis=-1, keepdims=True)
        images = to_8b(images)
    if not channels_last:
        images = np.moveaxis(images, -1, 1)
    if as_float:
        images = to_float(images)
    if to_torch:
        if device is None:
            device = torch.device("cpu")
        images = torch.tensor(images, device=device)
    if return_paths:
        return images, file_paths
    else:
        return images


def load_mesh(
    path: Path,
    return_vert_uvs=False,
    return_vert_norms=False,
    return_vert_color=False,
    to_torch=False,
    device=None,
    verbose=True,
):
    """
    loads a mesh from a file
    :param path: path to mesh file
    :param return_vert_uvs: if True, returns a (V x 2) tensor of vertex uv coordinates
    :param return_vert_norms: if True, returns a (V x 3) tensor of vertex normals
    :param return_vert_color: if True, returns a (V x 3) tensor of vertex colors
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :param verbose: if True, prints out information about the mesh
    :return: (V x 3) tensor of vertices, (F x 3) tensor of faces, and more depending on flags
    """
    path = Path(path)
    supported = [".obj", ".ply"]
    if path.suffix not in supported:
        raise ValueError("Only {} formats are supported for loading".format(supported))
    if path.suffix == ".obj":
        return load_obj(
            path,
            return_vert_uvs=return_vert_uvs,
            return_vert_norms=return_vert_norms,
            return_vert_color=return_vert_color,
            to_torch=to_torch,
            device=device,
            verbose=verbose,
        )
    elif path.suffix == ".ply":
        if return_vert_uvs or return_vert_norms:
            raise ValueError(
                "current ply parser does not support vertex uvs or normals"
            )
        return load_ply_mesh(
            path,
            return_vert_color=return_vert_color,
            to_torch=to_torch,
            device=device,
            verbose=verbose,
        )
    else:
        raise ValueError("Only {} formats are supported for loading".format(supported))


def load_ply_mesh(
    path: Path, return_vert_color=False, to_torch=False, device=None, verbose=True
):
    """
    :param path: path to ply file
    :return_vert_color: if True, returns a (V x 3) tensor of vertex colors in addition to vertices and faces
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :return: (V x 3) tensor, (F x 3) tensor, and optionally (V x 3) tensor
    """
    path = Path(path)
    if not path.exists():
        raise ValueError("Path does not exist")
    if not path.is_file():
        raise ValueError("Path must be a file")
    if path.suffix != ".ply":
        raise ValueError("Only .ply are supported")
    result = parse_ply(path, verbose=verbose)
    if "vertex" in result:
        v_full = result["vertex"][0]
        v = v_full[0]
        if len(v_full) > 1:
            vc = v_full[1]
        else:
            vc = None
    else:
        v = None
    if "face" in result:
        f_full = result["face"][0]
        f = f_full[1]
    else:
        f = None
    if to_torch and device is not None:
        if v is not None:
            v = torch.tensor(v, dtype=torch.float, device=device)
        if f is not None:
            f = torch.tensor(f, dtype=torch.long, device=device)
        if return_vert_color and vc is not None:
            vc = torch.tensor(vc, dtype=torch.float, device=device)
    if return_vert_color:
        return v, f, vc
    else:
        return v, f


def load_obj(
    path: Path,
    return_vert_uvs=False,
    return_vert_norms=False,
    return_vert_color=False,
    to_torch=False,
    device=None,
    verbose=True,
):
    """
    :param path: path to obj file
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :return: (V x 3) tensor, (F x 3) tensor, and more depending on flags
    """
    path = Path(path)
    if not path.exists():
        raise ValueError("Path does not exist")
    if not path.is_file():
        raise ValueError("Path must be a file")
    if path.suffix != ".obj":
        raise ValueError("Only .obj are supported")
    v, f, vt, vn, vc, ft, fn = parse_obj(path, verbose=verbose)
    if to_torch and device is not None:
        v = torch.tensor(v, dtype=torch.float, device=device)
        f = torch.tensor(f, dtype=torch.long, device=device)
        # n = torch.tensor(n, dtype=torch.float, device=device)
        if return_vert_norms and vn is not None and fn is not None:
            vn = torch.tensor(vn, dtype=torch.float, device=device)
            fn = torch.tensor(fn, dtype=torch.long, device=device)
        if return_vert_uvs and vt is not None and ft is not None:
            vt = torch.tensor(vt, dtype=torch.float, device=device)
            ft = torch.tensor(ft, dtype=torch.long, device=device)
        if return_vert_color and vc is not None:
            vc = torch.tensor(vc, dtype=torch.float, device=device)
    # very ew. consider returning a dict / named tuple ?
    if return_vert_norms and return_vert_uvs and return_vert_color:
        return v, f, vt, ft, vn, fn, vc
    elif return_vert_uvs and return_vert_norms:
        return v, f, vt, ft, vn, fn
    elif return_vert_uvs and return_vert_color:
        return v, f, vt, ft, vc
    elif return_vert_norms and return_vert_color:
        return v, f, vn, fn, vc
    elif return_vert_uvs:
        return v, f, vt, ft
    elif return_vert_norms:
        return v, f, vn, fn
    elif return_vert_color:
        return v, f, vc
    else:
        return v, f


def get_ply_header(ply_path):
    """
    reads the header part of a ply file
    :param ply_path: path to ply file
    :return: list of strings per line in the header
    """
    ply_file = open(ply_path, "rb")
    header = []
    line = ply_file.readline().decode("ascii").strip()
    while line != "end_header":
        header.append(line)
        line = ply_file.readline().decode("ascii").strip()
    ply_file.close()
    return header + ["end_header"]


def parse_ply_header(raw_header):
    """
    parses the header of a ply file
    :param header: list of strings per line in the header
    :return:
    boolean if data is ascii,
    list of element types,
    list of element counts,
    per element list of property names,
    per element property type as string
    """
    header = [x for x in raw_header if "comment" not in x]
    line_index = 0
    if header[line_index] != "ply":
        raise ValueError(
            "ply file header corrupted ('ply' keyword not found in first line)"
        )
    line_index += 1
    if header[line_index] == "format ascii 1.0":
        data_is_ascii = True
    elif header[line_index] == "format binary_little_endian 1.0":
        data_is_ascii = False
    else:
        raise ValueError(
            "only ascii and binary_little_endian formats are supported (format not found)"
        )
    line_index += 1
    element_types = []
    elements_n = []
    property_names = []
    property_structs = []

    _, first_element_type, first_element_n = header[line_index].split()
    element_types.append(first_element_type)
    elements_n.append(int(first_element_n))
    line_index += 1
    line = header[line_index].split()
    while line[0] != "end_header":
        property_name = []
        property_struct = []
        while line[0] == "property":
            if line[1] == "float":
                property_name.append(line[2])
                property_struct.append("f")
            elif line[1] == "list":
                property_name.append(line[4])
                property_struct.append("1iii")
            elif line[1] == "uchar":
                property_name.append(line[2])
                property_struct.append("1")
            else:
                raise ValueError("unsupported property dtype")
            line_index += 1
            line = header[line_index].split()
            if line[0] == "end_header":
                break
        property_names.append(property_name)
        mystr = "".join(property_struct)
        if "1f" in mystr:  # this case is annoying and probably rare, discard
            raise ValueError("unsupported property type layout")
        property_structs.append(mystr)
        if line[0] == "element":
            _, element_type, element_n = line
            element_types.append(element_type)
            elements_n.append(int(element_n))
            line_index += 1
            line = header[line_index].split()
    return data_is_ascii, element_types, elements_n, property_names, property_structs


def dtype_from_letter(letter):
    if letter == "i":
        return np.int32
    if letter == "f":
        return np.float32
    if letter == "1":
        return np.uint8
    raise ValueError("unsupported dtype letter")


def parse_ply(ply_path, verbose=False):
    """
    A simple ply parser, supporting:
    ascii format, but only vertex and triangular faces (cannot handle very complex properties of vertices)
    binary format, but only for vertex and only float properties
    :param ply_path: path to ply file
    :param verbose: if True, prints out information during parsing
    :return: a dictionary of 2-element lists, the first is a numpy array, the second is the property names
    i.e. {"vertex": [np.array([[1.0, 1.0, 2.0], [2.0, 2.0, 0.5], ...]), ["x", "y", "z"]]}
    """
    verts = []
    faces = []
    vert_colors = []
    vert_norms = []
    n_faces = 0
    n_vertices = 0
    has_vert_color = False
    header = get_ply_header(Path(ply_path))
    data_is_ascii, element_types, elements_n, property_names, property_structs = (
        parse_ply_header(header)
    )
    if data_is_ascii:
        ply_file = open(ply_path, "r")
        data = [x.strip() for x in ply_file.readlines()]
        data = data[len(header) :]
        result = {}
        absolute_element = 0
        for i, element_type in enumerate(element_types):
            n_elements = elements_n[i]
            unique_types = list(OrderedDict.fromkeys(property_structs[i]).keys())
            if len(unique_types) == 1:
                single_data_type = True
                prealloc = [
                    np.empty((n_elements, len(property_structs[i])), dtype=np.float32)
                ]
            else:
                single_data_type = False
                split_key = "".join(unique_types)  # for example f1
                split_index = property_structs[i].index(split_key) + 1
                prealloc = [
                    np.empty(
                        (n_elements, split_index),
                        dtype=dtype_from_letter(unique_types[0]),
                    ),
                    np.empty(
                        (n_elements, len(property_structs[i]) - split_index),
                        dtype=dtype_from_letter(unique_types[1]),
                    ),
                ]
            result[element_type] = [prealloc, property_names[i]]
            for j in range(n_elements):
                raw_data = data[absolute_element].split()
                if single_data_type:
                    unpacked = np.array(raw_data, dtype=np.float32)
                    result[element_type][0][0][j] = unpacked
                else:
                    unpacked1 = np.array(
                        raw_data[:split_index], dtype=dtype_from_letter(unique_types[0])
                    )
                    unpacked2 = np.array(
                        raw_data[split_index:], dtype=dtype_from_letter(unique_types[1])
                    )
                    result[element_type][0][0][j] = unpacked1
                    result[element_type][0][1][j] = unpacked2
                absolute_element += 1
    else:
        ply_file = open(ply_path, "rb")
        for i in range(len(header)):
            ply_file.readline()
        result = {}
        for i, element_type in enumerate(element_types):
            n_elements = elements_n[i]
            prealloc = [
                np.empty((n_elements, len(property_names[i])), dtype=np.float32)
            ]
            result[element_type] = [prealloc, property_names[i]]
            line_size = calcsize(property_structs[i])
            for j in range(n_elements):
                raw_data = ply_file.read(line_size)
                unpacked = np.array(unpack(property_structs[i], raw_data))
                result[element_type][0][0][j] = unpacked
    return result
    vert_properties = []
    found_vertex_element = False
    found_face_element = False
    for line in header[2:]:
        split = line.split()
        if split[0] == "comment":
            continue
        if split[0] == "element":
            found_vertex_element = False
            found_face_element = False
            if split[1] == "vertex":
                n_vertices = int(split[2])
                found_vertex_element = True
            elif split[1] == "face":
                n_faces = int(split[2])
                found_face_element = True
            else:
                raise ValueError(
                    "ply file contains elements other than verticies or faces, which are not supported."
                )
        if split[0] == "property":
            if found_vertex_element:
                vert_properties.append((split[2], split[1]))
            elif found_face_element:
                pass  # todo: support face properties
            else:
                raise ValueError("ply heder currupted (property found before element)")
    if len(vert_properties) == 0:
        raise ValueError("ply file header corrupted (no vertex properties found)")
    vert_properties_names = [x[0] for x in vert_properties]
    if (
        "x" not in vert_properties_names
        or "y" not in vert_properties_names
        or "z" not in vert_properties_names
    ):
        raise ValueError(
            "ply file header corrupted (could not find xyz properties for vertices)"
        )
    if (
        "red" in vert_properties_names
        and "green" in vert_properties_names
        and "blue" in vert_properties_names
    ):
        has_vert_color = True
    n_vert_properties = len(vert_properties)
    result = {x[0]: [] for x in vert_properties}
    for line in data[:n_vertices]:
        split = line.split()
        if len(split) != n_vert_properties:
            raise ValueError(
                "ply file data corrupted (number of vertex properties does not match header)"
            )
        for i, prop in enumerate(vert_properties_names):
            result[prop].append(split[i])
    verts = np.stack(
        [
            np.array(result["x"], dtype=np.float32),
            np.array(result["y"], dtype=np.float32),
            np.array(result["z"], dtype=np.float32),
        ],
        axis=1,
    )
    if len(verts) != n_vertices:
        raise ValueError(
            "ply file data corrupted (number of vertices does not match header)"
        )
    if has_vert_color:
        vert_colors = np.stack(
            [
                np.array(result["red"], dtype=np.float32),
                np.array(result["green"], dtype=np.float32),
                np.array(result["blue"], dtype=np.float32),
            ],
            axis=1,
        )
        color_dtype = vert_properties[vert_properties_names.index("red")][1]
        if color_dtype == "uchar":
            vert_colors = vert_colors / 255.0
        if len(vert_colors) != n_vertices:
            raise ValueError(
                "ply file data corrupted (number of vertex colors does not match header)"
            )
    if n_faces > 0:
        for line in data[n_vertices:]:
            split = line.split()
            if split[0] != "3":
                raise ValueError("only triangular faces are supported")
            faces.append([int(x) for x in split[1:]])
        faces = np.stack(faces)
        if len(faces) != n_faces:
            raise ValueError(
                "ply file data corrupted (number of faces does not match header)"
            )
    return verts, faces, vert_norms, vert_colors


def parse_obj(path, verbose=True):
    """
    A simple obj parser
    currently supports vertex and triangular face elements only
    """
    path = Path(path)
    vertex_list = []
    vertex_texture_list = []
    vertex_normal_list = []
    color_list = []
    face_list = []
    face_texture_list = []
    face_normal_list = []
    finite_flag = False
    with open(path, "r") as objFile:
        for line in objFile:
            line = line.split("#")[0]  # remove comments
            split = line.split()
            if not len(split):  # skip empty lines
                continue
            if split[0] == "v":
                if 3 <= len(split[1:]) <= 4:  # x y z [w]
                    float_vertex = np.array([np.float64(x) for x in split[1:]])
                    if not np.isfinite(float_vertex).all():
                        finite_flag = True
                    vertex_list.append(float_vertex)
                elif len(split[1:]) == 6:
                    float_vertex = np.array([np.float64(x) for x in split[1:4]])
                    float_color = np.array([np.float64(x) for x in split[4:]])
                    if (float_color < 0.0).any() or (float_color > 1.0).any():
                        raise ValueError("color values must be in [0, 1]")
                    vertex_list.append(float_vertex)
                    color_list.append(float_color)
                else:
                    raise ValueError(
                        "vertex {} has {} entries, but only 3 or 6 are supported".format(
                            len(vertex_list), len(split[1:])
                        )
                    )
            elif split[0] == "f":
                if len(split[1:]) != 3:
                    raise ValueError("only triangular faces are supported")
                else:
                    face = np.array([x.split("/") for x in split[1:]])
                    if face.shape == (3, 1):  # v
                        int_face = face.squeeze().astype(np.int32) - 1
                        face_list.append(int_face)
                    elif face.shape == (3, 2):  # v/vt
                        int_face = face.astype(np.int32) - 1
                        face_list.append(int_face[:, 0])
                        face_texture_list.append(int_face[:, 1])
                    elif face.shape == (3, 3):  # v/vt/vn or v/vn
                        if np.all([len(x) == 0 for x in face[:, 1]]):  # v//vn
                            face = face[:, 0::2]
                            int_face = face.astype(np.int32) - 1
                            face_list.append(int_face[:, 0])
                            face_normal_list.append(int_face[:, 1])
                        else:  # v/vt/vn
                            if np.any([len(x) == 0 for x in face[:, 1]]):
                                raise ValueError(
                                    "face {} is corrupt".format(len(face_list))
                                )
                            else:
                                int_face = face.astype(np.int32) - 1
                                face_list.append(int_face[:, 0])
                                face_texture_list.append(int_face[:, 1])
                                face_normal_list.append(int_face[:, 2])
                    if np.any(int_face < 0):
                        raise ValueError("negative face indices are not supported")
            elif split[0] == "vn":
                if len(split[1:]) != 3:  # xn yn zn
                    raise ValueError(
                        "vertex normal {} has {} entries, but only 3 are supported".format(
                            len(vertex_normal_list), len(split[1:])
                        )
                    )
                else:
                    float_vertex_normal = np.array([np.float64(x) for x in split[1:]])
                    vertex_normal_list.append(float_vertex_normal)
            elif split[0] == "vt":
                if 2 <= len(split[1:]) <= 3:  # u [v, w]
                    float_vertex_tex = np.array([np.float64(x) for x in split[1:]])
                    if (float_vertex_tex < 0).any():
                        raise ValueError(
                            "negative texture coordinates are not supported"
                        )
                    if (float_vertex_tex > 1).any():
                        raise ValueError("texture coordinates must be between 0 and 1")
                    vertex_texture_list.append(float_vertex_tex)
                else:
                    raise ValueError(
                        "vertex normal {} has {} entries, but only 3 are supported".format(
                            len(vertex_normal_list), len(split[1:])
                        )
                    )
            elif split[0] == "l":  # ignore polyline elements
                continue
            elif split[0] == "vp":  # ignore parameter space vertices
                continue
    v = np.stack(vertex_list)
    f = np.stack(face_list)
    if finite_flag and verbose:
        print("Warning: some vertices in file were not finite")
    if len(vertex_normal_list) > 0:
        vn = np.stack(vertex_normal_list)
    else:
        vn = None
    if len(vertex_texture_list) > 0:
        vt = np.stack(vertex_texture_list)
    else:
        vt = None
    if len(color_list) > 0:
        vc = np.stack(color_list)
    else:
        vc = None
    if len(face_texture_list) > 0:
        ft = np.stack(face_texture_list)
    else:
        ft = None
    if len(face_normal_list) > 0:
        fn = np.stack(face_normal_list)
    else:
        fn = None
    return v, f, vt, vn, vc, ft, fn


def save_obj(vertices, faces, path):
    """ "
    :param vertices: (n x 3) tensor of vertices
    :param faces: (m x 3) or (m x 4) tensor of vertex indices
    :param path: path to save obj file to
    :param vertex_normals: optional (n x 3) tensor of vertex normals
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
            row = "f "
            for index in f:
                row += " {}".format(index + 1)  # obj indices start at 1
            row += "\n"
            file.write(row)


def save_ply(
    vertices,
    path,
    faces=None,
    vertex_colors=None,
    face_colors=None,
    vertex_normals=None,
):
    """
    saves a ply file in a human readable format
    :param vertices: (n, 3) np array or torch tensor of vertices float32/float64
    :param path: path to save ply file to
    :param faces: optional (m, x) np array or torch tensor of vertex indices np.int32/np.int64
    :param vertex_colors: optional (n, 3) np array or torch tensor of vertex colors np.uint8
    :param face_colors: optional (m, 3) np array or torch tensor of face colors np.uint8
    :param vertex_normals: optional (n, 3) np array or torch tensor of vertex normals np.float32/np.float64
    """
    path = Path(path)
    if path.suffix != ".ply":
        raise ValueError("Path must have suffix .ply")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if type(vertices) == torch.Tensor:
        vertices = to_np(vertices)
    if faces is not None:
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
    if vertex_colors is not None:
        if type(vertex_colors) == torch.Tensor:
            vertex_colors = to_np(vertex_colors)
        if vertex_colors.dtype != np.uint8:
            raise ValueError("Vertex colors must be of type uint8")
        if vertex_colors.shape != vertices.shape:
            raise ValueError("Vertex colors must be same shape as vertices")
        if np.isnan(vertex_colors).any():
            raise ValueError("Vertices colors must be finite")
    if face_colors is not None:
        if type(face_colors) == torch.Tensor:
            face_colors = to_np(face_colors)
        if face_colors.dtype != np.uint8:
            raise ValueError("Faces colors must be of type uint8")
        if face_colors.shape != faces.shape:
            raise ValueError("Faces colors must be same shape as vertices")
        if np.isnan(face_colors).any():
            raise ValueError("Faces colors must be finite")
    if vertex_normals is not None:
        if type(vertex_normals) == torch.Tensor:
            vertex_normals = to_np(vertex_normals)
        if vertex_normals.dtype != np.float32 and vertex_normals.dtype != np.float64:
            raise ValueError("Vertex normals must be of type float32 / float64")
        if vertex_normals.shape != vertices.shape:
            raise ValueError("Vertex normals must be same shape as vertices")
        if np.isnan(vertex_normals).any():
            raise ValueError("Vertices normals must be finite")
    with open(str(path), "w") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("comment generated by gsoup https://github.com/yoterel/gsoup\n")
        file.write("element vertex {}\n".format(len(vertices)))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        if vertex_colors is not None:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
        if vertex_normals is not None:
            file.write("property float nx\n")
            file.write("property float ny\n")
            file.write("property float nz\n")
        if faces is not None:
            file.write("element face {}\n".format(len(faces)))
            file.write("property list uchar int vertex_indices\n")
            if face_colors is not None:
                file.write("property uchar red\n")
                file.write("property uchar green\n")
                file.write("property uchar blue\n")
        file.write("end_header\n")
        for i, v in enumerate(vertices):
            file.write("{} {} {}".format(v[0], v[1], v[2]))
            if vertex_colors is not None:
                c = vertex_colors[i]
                file.write(" {} {} {}".format(c[0], c[1], c[2]))
            if vertex_normals is not None:
                n = vertex_normals[i]
                file.write(" {} {} {}".format(n[0], n[1], n[2]))
            file.write("\n")
        if faces is not None:
            for i, f in enumerate(faces):
                file.write("{}".format(len(f)))
                for index in f:
                    file.write(" {}".format(index))
                if face_colors is not None:
                    c = face_colors[i]
                    file.write(" {} {} {}".format(c[0], c[1], c[2]))
                file.write("\n")


def save_mesh(
    vertices, faces, path, vertex_normals=None, vertex_colors=None, face_colors=None
):
    """
    saves a mesh to a file
    :param vertices: (n x 3) tensor of vertices
    :param faces: (m x 3) tensor of vertex indices
    :param path: path to save mesh to
    :param vertex_normals: optional (n x 3) tensor of vertex normals
    :param vertex_colors: optional (n x 3) tensor of vertex colors
    :param face_colors: optional (m x 3) tensor of face colors
    """
    path = Path(path)
    if path.suffix not in [".obj", ".ply"]:
        raise ValueError("Only .obj or .ply are supported")
    else:
        if path.suffix == ".obj":
            if vertex_colors is not None or face_colors is not None:
                raise ValueError(
                    "obj does not officially support vertex or face colors"
                )
            save_obj(
                vertices, faces, path
            )  # will ignore vertex_normals/vertex_colors/face_colors
        elif path.suffix == ".ply":
            save_ply(
                vertices,
                path,
                faces=faces,
                vertex_normals=vertex_normals,
                vertex_colors=vertex_colors,
                face_colors=face_colors,
            )


def save_pointcloud(vertices, path, vertex_colors=None, vertex_normals=None):
    path = Path(path)
    if path.suffix != ".ply":
        raise ValueError("Only .ply are supported")
    if vertex_colors is not None:
        if vertex_colors.dtype != np.uint8:
            raise ValueError("Vertex colors must be of type uint8")
        if vertex_colors.shape != vertices.shape:
            raise ValueError("Vertex colors must have same shape as vertices")
    if vertex_normals is not None:
        if vertex_normals.shape != vertices.shape:
            raise ValueError("Vertex normals must have same shape as vertices")
    save_ply(vertices, path, vertex_colors=vertex_colors, vertex_normals=vertex_normals)


def load_pointcloud(
    path,
    return_vert_norms=False,
    return_vert_color=False,
    to_torch=False,
    device=None,
    verbose=True,
):
    """
    :param path: path to ply file
    :param to_torch: if True, returns a torch tensor
    :param device: device to load tensor to
    :return: (V x 3) np array (or torch tensor), and more depending on flags
    """
    path = Path(path)
    supported = [".ply"]
    if path.suffix not in supported:
        raise ValueError("Only {} formats are supported for loading".format(supported))
    if not path.exists():
        raise ValueError("Path does not exist")
    if not path.is_file():
        raise ValueError("Path must be a file")
    result = parse_ply(path, verbose=verbose)
    if "vertex" in result:
        v_full = result["vertex"][0]
        v = v_full[0]
        if len(v_full) > 1:
            vc = v_full[1]
        else:
            vc = None
        if v.shape[1] > 3:
            vn = v[:, 3:]  # vn will contain all other channels on the vertex after xyz
            v = v[:, :3]
        else:
            vn = None
    else:
        v = None
    if to_torch and device is not None:
        if v is not None:
            v = torch.tensor(v, dtype=torch.float, device=device)
        if return_vert_norms and vn is not None:
            vn = torch.tensor(vn, dtype=torch.float, device=device)
        if return_vert_color and vc is not None:
            vc = torch.tensor(vc, dtype=torch.float, device=device)
    # very ew. consider returning a dict / named tuple ?
    if return_vert_norms:
        if return_vert_color:
            return v, vc, vn
        else:
            return v, vn
    elif return_vert_color:
        return v, vc
    else:
        return v


def save_pointclouds(
    vertices, path, file_names: list = [], vertex_colors=None, vertex_normals=None
):
    """
    saves a batch of pointclouds to a folder
    :param path: path to save meshes to
    :param vertices: (b x V x 3) tensor
    :param file_names: list of file names of length b without suffix (suffix will be removed)
    :param vertex_normals: optional (b x V x 3) tensor of vertex normals
    :param vertex_colors: optional (b x V x 3) tensor of vertex colors
    """
    if vertices.ndim != 3:
        raise ValueError("Vertices must be a (batch, points, 3) np array")
    if file_names:
        if len(file_names) != vertices.shape[0]:
            raise ValueError("Number of file names must match batch size")
    path = Path(path)
    for i, v in enumerate(vertices):
        cur_vert_colors = None
        if vertex_colors is not None:
            cur_vert_colors = vertex_colors[i]
        cur_vert_norm = None
        if vertex_normals is not None:
            cur_vert_norm = vertex_normals[i]
        if file_names:
            file_name = Path(file_names[i]).stem
            save_pointcloud(
                v,
                path / "{}.ply".format(file_name),
                vertex_colors=cur_vert_colors,
                vertex_normals=cur_vert_norm,
            )
        else:
            save_pointcloud(
                v,
                path / "{:05d}.ply".format(i),
                vertex_colors=cur_vert_colors,
                vertex_normals=cur_vert_norm,
            )


def save_meshes(vertices, faces, path, file_names: list = []):
    """
    saves a batch of meshes to a folder
    :param path: path to save meshes to
    :param vertices: (b x V x 3) np array or torch tensor
    :param faces: (b x F x 3) np array or torch tensor
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


def write_exr(image, file_path, compression="ZIP", compression_level=6):
    """
    saves a RGB image to disk as exr (without losing precision).
    :param image: (h, w, 3) numpy array of either uint32, float16 or float32
    :param file_path: the location to save the file to. if parent folder doesn't exists, creates it.
    :param compression: which compression type to use (we just expose NONE, ZIPS, ZIP and PIZ)
    :param compression_level: what level of compression to use if suportted by compression algorithm
    :note see https://openexr.com/en/latest/ReadingAndWritingImageFiles.html#compression
    """
    channels = {"RGB": np.ascontiguousarray(image)}
    header = {}
    if compression == "NONE":
        header["compression"] = OpenEXR.NO_COMPRESSION
    elif compression == "ZIP":
        header["compression"] = OpenEXR.ZIP_COMPRESSION
        header["type"] = OpenEXR.scanlineimage
        header["zipCompressionLevel"] = compression_level
    elif compression == "ZIPS":
        header["compression"] = OpenEXR.ZIPS_COMPRESSION
    elif compression == "PIZ":
        header["compression"] = OpenEXR.PIZ_COMPRESSION
    else:
        raise NotImplementedError
    my_path = Path(file_path)
    if not my_path.parent.exists():
        parent_path.mkdir(parents=True, exist_ok=True)
    with OpenEXR.File(header, channels) as outputFile:
        outputFile.write(str(my_path))


def read_exr(file_path):
    """
    read exr into a numpy array (h, w, 3)
    note: assumes channels names in the exr are named "R" "G" and "B".
    :param file_path: path to exr file
    :return (h, w, 3) float16/float32 numpy array representing the image
    """
    typemap = {"UINT": np.uint32, "HALF": np.float16, "FLOAT": np.float32}
    # open the input file
    exr = OpenEXR.InputFile(str(file_path))
    # Compute the size
    dw = exr.header()["dataWindow"]
    w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    ##### a more principled way of accessing the data
    arr_maps = {}
    # Read the three color channels as 32-bit floats
    for ch_name, ch in exr.header()["channels"].items():
        exr_typename = ch.type.names[ch.type.v]
        np_type = typemap[exr_typename]
        bytestring = exr.channel(ch_name, ch.type)
        arr = np.frombuffer(bytestring, dtype=np_type).reshape(h, w, 1)
        arr_maps[ch_name] = arr
    # stack into matrix
    image = np.concatenate([arr_maps["R"], arr_maps["G"], arr_maps["B"]], axis=-1)
    ####################################################
    # FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    # R = np.frombuffer(exr.channel("R", FLOAT), dtype=np.float32).reshape((h, w, 1))
    # G = np.frombuffer(exr.channel("G", FLOAT), dtype=np.float32).reshape((h, w, 1))
    # B = np.frombuffer(exr.channel("B", FLOAT), dtype=np.float32).reshape((h, w, 1))
    # stack into matrix
    # image = np.concatenate((R, G, B), axis=-1)
    return image
