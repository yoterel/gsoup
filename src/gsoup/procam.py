import numpy as np
import cv2
from .gsoup_io import (
    save_image,
    save_images,
    load_images,
    load_image,
    save_mesh,
    write_exr,
)
from .transforms import compose_rt
from .core import to_8b, to_hom, swap_columns, make_monotonic, to_float, rgb_to_gray
from .image import (
    adjust_contrast_brightness,
    add_alpha,
    resize,
    tonemap_reinhard,
    linear_to_srgb,
)
from .geometry_basic import ray_ray_intersection, point_line_distance
from pathlib import Path, PosixPath
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import ConvexHull
from .projector_plugin_mitsuba import ProjectorPy
from itertools import product
import mitsuba as mi
import drjit


class ProjectorScene:
    """
    A class to create a Mitsuba scene for procam algorithms testing.
    """

    def __init__(self):
        # Set the Mitsuba variant
        # mi.set_variant(variant)
        # Register the plugin
        from .projector_plugin_mitsuba import ProjectorPy

        mi.register_emitter("projector_py", lambda props: ProjectorPy(props))
        self.scene = None
        self.proj_wh = None  # projector resolution (width, height)
        self.cam_wh = None  # camera resolution (width, height)

    def load_scene_from_file(self, file_path):
        self.scene = mi.load_file(file_path)

    def create_default_scene(
        self,
        cam_wh=(256, 256),
        cam_fov=45,
        cam_cx=0.5,
        cam_cy=0.5,
        cam_cv_K=None,
        proj_wh=(256, 256),
        proj_fov=45,
        proj_cx=0.5,
        proj_cy=0.5,
        proj_cv_K=None,
        proj_texture=None,
        proj_brightness=1.0,
        proj_response_mode="srgb",
        ambient_color=[0.01, 0.01, 0.01],
        mesh_file=None,
        mesh_scale=1.0,
        cb_tex_col1=[0.6, 0.9, 0.6],
        cb_tex_col2=[0.9, 0.6, 0.6],
        cb_tex_scale=5.0,
        spp=256,
    ):
        """
        Create a default Mitsuba scene with a projector, camera, constant ambient light and two diffuse screens perpendicular to the projector.
        :cam_wh: camera resolution as a tuple (width, height).
        :cam_fov: field of view of the camera in degrees.
        :cam_cx: x coord of principal point of camera in normalized coordinates (0.5=center).
        :cam_cy: y coord of principal point of camera in normalized coordinates (0.5=center).
        :cam_cv_K: opencv style intrinsics for camera (will override fov/cx/cy).
        :proj_wh: projector resolution as a tuple (width, height).
        :proj_fov: field of view of the projector in degrees.
        :proj_cx: x coord of principal point of projector in normalized coordinates (0.5=center).
        :proj_cy: y coord of principal point of projector in normalized coordinates (0.5=center).
        :proj_cv_K: opencv style intrinsics for projector (will override fov/cx/cy).
        :param proj_texture: texture to project as 3D numpy array (h, w, 3) or path to file. if None will shine all-white.
        :param proj_brightness: unitless brightness of the projector texture. the z=1 plane will have this brightness for an all-white texture.
        :param proj_response_mode: projector response function ("linear", "gamma" or "srgb").
        :param ambient_color: constant color for ambient light
        :param mesh_file: path to a mesh file to add to the scene. if None, a perpendicular wall will be added.
        :param mesh_scale: scale of the mesh in the scene.
        :cb_tex_col1: color of the first checkerboard square type.
        :cb_tex_col2: color of the second checkerboard square type.
        :cb_tex_scale: scale of the checkerboard texture.
        :spp: samples per pixel for the camera.
        For Mitsuba, sensors have:
        # Z: forward direction
        # Y: up direction
        # X: left
        :return: the Mitsuba scene object.
        """
        self.proj_wh = proj_wh
        self.cam_wh = cam_wh
        if proj_texture is None:
            proj_texture = np.ones(
                (proj_wh[0], proj_wh[1], 3), dtype=np.float32
            )  # white texture
        else:
            if type(proj_texture) == np.ndarray:
                assert proj_texture.ndim == 3, "proj_texture must be a 3D numpy array."
                if proj_texture.shape[-2::-1] != proj_wh:
                    proj_texture = resize(
                        proj_texture[None, ...], proj_wh[1], proj_wh[0]
                    )[0]
            elif type(proj_texture) == str or type(proj_texture) == PosixPath:
                proj_texture = load_image(
                    proj_texture, as_float=True, resize_wh=proj_wh
                )
            else:
                raise TypeError("proj_texture must be a numpy array or a file path.")
        if proj_cv_K is None:
            # mitsuba expects normalized coordinates relative to screen center rather than corner
            proj_cx = proj_cx - 0.5
            proj_cy = proj_cy - 0.5
        else:
            assert proj_cv_K.shape == (3, 3)
            # mitsuba expects normalized coordinates relative to screen center rather than corner
            proj_cx = (proj_cv_K[0, 2] / proj_wh[0]) - 0.5
            proj_cy = (proj_cv_K[1, 2] / proj_wh[1]) - 0.5
            # mistuba expects fov in degrees
            proj_fov = self.focal_length_to_fov(proj_cv_K[0, 0], proj_wh[0])
            # fy = focal_length_to_fov(proj_cv_K[1, 1], proj_wh[1])
        if cam_cv_K is None:
            # mitsuba expects normalized coordinates relative to screen center rather than corner
            cam_cx = cam_cx - 0.5
            cam_cy = cam_cy - 0.5
        else:
            assert cam_cv_K.shape == (3, 3)
            # mitsuba expects normalized coordinates relative to screen center rather than corner
            cam_cx = (cam_cv_K[0, 2] / cam_wh[0]) - 0.5
            cam_cy = (cam_cv_K[1, 2] / cam_wh[1]) - 0.5
            # mistuba expects fov in degrees
            cam_fov = self.focal_length_to_fov(cam_cv_K[0, 0], cam_wh[0])
            # fy = focal_length_to_fov(proj_cv_K[1, 1], proj_wh[1])
        scene_dict = {
            "type": "scene",
            "proj_texture": {
                "type": "bitmap",
            },
            "wall_texture": {
                "type": "checkerboard",
                "color0": {"type": "rgb", "value": cb_tex_col1},
                "color1": {"type": "rgb", "value": cb_tex_col2},
                "to_uv": mi.ScalarTransform4f().scale(
                    [cb_tex_scale, cb_tex_scale, cb_tex_scale]
                ),
            },
            "integrator": {
                "type": "path",
                "hide_emitters": True,
                "max_depth": 8,
            },
            "camera": {
                "type": "perspective",
                "fov": cam_fov,
                "principal_point_offset_x": cam_cx,
                "principal_point_offset_y": cam_cy,
                "near_clip": 0.01,
                "far_clip": 10000,
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[2.5, 0, 0.3],  # along +X axis
                    target=[0, 0, 0],
                    up=[0, 0, 1],  # Z-up
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": cam_wh[0],
                    "height": cam_wh[1],
                    "pixel_format": "rgba",
                    "rfilter": {"type": "box"},
                },
                "sampler": {
                    "type": "independent",
                    "sample_count": spp,  # number of samples per pixel
                },
            },
            "projector": {
                "type": "projector_py",
                "irradiance": {
                    "type": "ref",
                    "id": "proj_texture",
                    # "type": "bitmap",
                    # "filename": str(texture_file),
                    # "raw": True,  # assuming the image is in linear RGB
                },
                "scale": proj_brightness,
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[1.5, 0, 0],  # along +X axis
                    target=[0, 0, 0],
                    up=[0, 0, 1],  # Z-up
                ),
                "fov": proj_fov,
                "principal_point_offset_x": proj_cx,
                "principal_point_offset_y": proj_cy,
                "response_mode": proj_response_mode,
            },
            "ambient": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": ambient_color},
            },
        }
        # post process scene dict to add projector texture
        if type(proj_texture) == np.ndarray:
            scene_dict["proj_texture"][
                "data"
            ] = proj_texture  # mi.TensorXf(proj_texture)
        else:
            raise TypeError("proj_texture must be a numpy array.")
            # can't enter here, but if we were to use a file path, we would do:
            # scene_dict["proj_texture"]["filename"] = proj_texture
        # disk image aren't usually linear, but for consistency we assume the texture is linear RGB
        scene_dict["proj_texture"]["raw"] = True
        if mesh_file is not None:
            assert mesh_file.suffix in [".obj", ".ply"], "Unsupported mesh file format."
            scene_dict["geometry"] = {
                "type": mesh_file.suffix[1:],
                "filename": str(mesh_file),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": [1.0, 1.0, 1.0],
                    },
                },
                "to_world": mi.ScalarTransform4f().scale(
                    [mesh_scale, mesh_scale, mesh_scale]
                ),
            }
        else:
            v = np.array(
                [
                    [0, -1.0, -1.0],
                    [0, -1.0, 1.0],
                    [0, 1.0, 1.0],
                    [0, 1.0, -1.0],
                ],
                dtype=np.float32,
            )
            # v = v * quad_scale  # scale the vertices
            f = np.array(
                [
                    [0, 1, 2, 3],
                ],
                dtype=np.int32,
            )
            # uv = np.array(
            #     [
            #         [0, 0],
            #         [0, 1],
            #         [1, 1],
            #         [1, 0],
            #     ],
            #     dtype=np.float32,
            # )
            save_mesh(
                v, f, "resource/square.obj"
            )  # hack until mitsuba supports set_bsdf
            square_scale = np.tan((cam_fov / 2) * np.pi / 180)
            scene_dict["geometry"] = {
                "type": "obj",
                "filename": "resource/square.obj",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": [1.0, 1.0, 1.0],
                    },
                },
                "to_world": mi.ScalarTransform4f().scale(
                    [square_scale, square_scale, square_scale]
                ),
            }
            # scene_dict["geometry"] = {
            #     "type": "rectangle",
            #     "to_world": mi.ScalarTransform4f()
            #     .translate([-2.0, 0.0, 0.0])
            #     .rotate([0, 1, 0], 90),
            #     # "flip_normals": True,
            #     "bsdf": {
            #         "type": "diffuse",
            #         "reflectance": {
            #             "type": "ref",
            #             "id": "wall_texture",
            #             # "type": "rgb",
            #             # "value": [1.0, 1.0, 1.0],
            #         },
            #     },
            # }
            # quad_scale = np.tan((cam_fov / 2) * np.pi / 180)
            # wall_bsdf = mi.load_dict(
            #     {
            #         "type": "diffuse",
            #         "reflectance": {
            #             # "type": "ref",
            #             # "id": "wall_texture",
            #             "type": "rgb",
            #             "value": [1.0, 1.0, 1.0],
            #         },
            #     }
            # )
            # mesh = mi.Mesh(
            #     "wall",
            #     vertex_count=4,
            #     face_count=2,
            #     has_vertex_normals=False,
            #     has_vertex_texcoords=False,
            #     # bsdf=wall_bsdf,
            # )
            # mesh_params = mi.traverse(mesh)
            # mesh_params["vertex_positions"] = drjit.llvm.Float(v.reshape(-1))
            # mesh_params["faces"] = drjit.llvm.UInt(f.reshape(-1))
            # mesh_params["vertex_texcoords"] = drjit.llvm.Float(uv.reshape(-1))
            # # mesh_params["bsdf"] = {
            # #     "type": "diffuse",
            # #     "reflectance": {
            # #         "type": "ref",
            # #         "id": "wall_texture",
            # #         # "type": "rgb",
            # #         # "value": [1.0, 1.0, 1.0],
            # #     },
            # # }
            # # mesh_params["to_world"] = mi.ScalarTransform4f()
            # #     .translate([-2.0, 0.0, 0.0])
            # #     .rotate([0, 1, 0], 90)
            # mesh_params.update()
            # scene_dict["geometry"] = mesh
        self.scene = mi.load_dict(scene_dict)

    def focal_length_to_fov(self, focal_length_px, image_width):
        fov_rad = 2 * np.arctan(image_width / (2 * focal_length_px))
        fov_deg = np.rad2deg(fov_rad)
        return fov_deg

    def set_projector_texture(self, texture):
        """
        Set the projector texture from a numpy array / image file path.
        :param texture_numpy: a numpy array of shape (H, W, 3) or (H, W, 4) representing the texture or an image file path.
        """
        if self.scene is None:
            raise RuntimeError("Scene not created yet.")
        if type(texture) == str or type(texture) == PosixPath:
            texture = load_image(texture, as_float=True, resize_wh=self.proj_wh)
        elif type(texture) == np.ndarray:
            assert texture.ndim == 3, "proj_texture must be a 3D numpy array."
            if texture.shape[-2::-1] != self.proj_wh:
                texture = resize(texture[None, ...], self.proj_wh[1], self.proj_wh[0])[
                    0
                ]
        else:
            raise TypeError("texture must be a numpy array or a file path.")
        new_bitmap = mi.Bitmap(texture)
        new_bitmap = new_bitmap.convert(
            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False
        )
        params = mi.traverse(self.scene)
        params["projector.irradiance.data"] = new_bitmap
        params.update()

    def set_projector_transform(self, transform):
        """
        Set the projector transform.
        :param transform: a 4x4 numpy array representing the projector to world transformation.
        """
        if self.scene is None:
            raise RuntimeError("Scene not created yet.")
        assert transform.shape == (4, 4), "transform must be a 4x4 matrix."
        params = mi.traverse(self.scene)
        params["projector.to_world"] = transform
        params.update()

    def set_camera_transform(self, transform):
        """
        Set the camera transform.
        :param transform: a 4x4 numpy array representing the projector to world transformation.
        """
        if self.scene is None:
            raise RuntimeError("Scene not created yet.")
        assert transform.shape == (4, 4), "transform must be a 4x4 matrix."
        params = mi.traverse(self.scene)
        params["camera.to_world"] = transform
        params.update()

    def project_point_to_01(self, point_3d, sensor_id="camera"):
        """
        Given a Mitsuba 3 scene (Python form), a 3D point, and an optional sensor id,
        returns the pixel coordinates (x, y) where the point would appear in the sensor image.
        """
        # 1. Get the sensor object
        scene = mi.traverse(self.scene)
        to_world = scene["camera.to_world"]
        film_size = scene["camera.film.size"]
        crop_size = scene["camera.film.crop_size"]
        crop_offset = scene["camera.film.crop_offset"]
        x_fov = scene["camera.x_fov"]
        near_clip = scene["camera.near_clip"]
        far_clip = scene["camera.far_clip"]
        principal_x = scene["camera.principal_point_offset_x"]
        principal_y = scene["camera.principal_point_offset_y"]

        # 2. Transform world position to camera space
        point_world = mi.Point3f(point_3d)
        to_camera = to_world.inverse()
        point_cam = to_camera @ point_world
        # 3. Get perspective projection matrix
        persp_mat = mi.perspective_projection(
            film_size, crop_size, crop_offset, x_fov, near_clip, far_clip
        )
        # 4. Project to sample space (homogeneous divide)
        normalized_cords = (
            persp_mat @ point_cam
        )  # already in sample space (i.e. [0, 1])
        x_norm = normalized_cords.x + principal_x
        y_norm = normalized_cords.y + principal_y
        # ndc = ndc / ndc.w
        # 5. NDC to normalized [0,1], then add principal offset (in normalized units)
        # x_norm = (ndc.x + 1) * 0.5 + principal_x
        # y_norm = (ndc.y + 1) * 0.5 + principal_y
        # 6. Map NDC [-1,1] to pixel space [0,width-1], [0,height-1]
        # x_pix = (ndc.x + 1) * 0.5 * (film_size[0] - 1)
        # y_pix = (1 - (ndc.y + 1) * 0.5) * (
        #     film_size[1] - 1
        # )  # Mitsuba images are top-to-bottom
        return x_norm[0], y_norm[0]

    def transform_square_randomly(self, max_trials=1000):
        """
        generates a random transformation for the square geometry in the Mitsuba scene.
        The transformation is valid if the square is fully visible in the camera image and facing the camera.
        :param max_trials: maximum number of trials to find a valid transformation.
        :return: a valid transformation matrix (mi.ScalarTransform4f) that can be
        """
        if self.scene is None:
            raise RuntimeError("Scene not created yet.")
        params = mi.traverse(self.scene)
        # Rectangle model space corners (in [-1, 1] for unit rectangle centered at origin)
        # scale = 0.41
        # corners_local = np.array(
        #     [
        #         [0, -scale, -scale],
        #         [0, -scale, scale],
        #         [0, scale, scale],
        #         [0, scale, -scale],
        #     ]
        # )
        corners_local = np.array(params["geometry.vertex_positions"]).reshape(4, 3)
        # Surface normal of rectangle in world space
        normal_local = np.array([1, 0, 0])
        cam_to_world = params["camera.to_world"]
        world_to_cam = cam_to_world.inverse()

        rng = np.random.default_rng()
        for trial in range(max_trials):
            print(trial)
            # Sample random translation in camera space
            tx = rng.uniform(-1.0, 1.0)
            ty = rng.uniform(-1.0, 1.0)
            tz = rng.uniform(-1.0, 1.0)  # Must be in front of camera

            # Random small rotation around X and Y axes (keep it front-facing)
            rx = rng.uniform(-90, 90)  # degrees
            ry = rng.uniform(-90, 90)
            rz = rng.uniform(-90, 90)

            local_to_world = (
                mi.ScalarTransform4f().translate([tx, ty, tz])
                @ mi.ScalarTransform4f().rotate([1, 0, 0], rx)
                @ mi.ScalarTransform4f().rotate([0, 1, 0], ry)
                @ mi.ScalarTransform4f().rotate([0, 0, 1], rz)
            )
            # local_to_world = mi.ScalarTransform4f()

            # Transform to world space
            to_world = local_to_world
            # Transform corners to camera space for projection
            world_space_corners = [
                to_world @ mi.ScalarPoint3f(p) for p in corners_local
            ]
            # Project to normalized image coordinates
            projected = [self.project_point_to_01(p) for p in world_space_corners]
            # Check all points are inside [0, 1] x [0, 1]
            inside = all((0 <= p[0] <= 1) and (0 <= p[1] <= 1) for p in projected)
            # Check if surface normal is facing the camera (Z < 0 in camera space)
            normal_cam = (
                world_to_cam.matrix
                @ local_to_world.matrix
                @ mi.ScalarVector4f(
                    normal_local[0], normal_local[1], normal_local[2], 0.0
                )
            )
            facing_camera = normal_cam[2][0] < 0
            if inside and facing_camera:
                params["geometry.vertex_positions"] = drjit.ravel(
                    np.array(world_space_corners).T
                )
                params.update()
                return to_world

    def capture(self, raw=False):
        """
        captures an image with the virtual camera.
        :param raw: if true, will return the raw radiance values using the default units of Mitsuba
        otherwise, will tonemap the result and convert to srgb (gamma correction).
        """
        if self.scene is None:
            raise RuntimeError("Scene not created yet.")
        raw_render = mi.render(self.scene)
        # mi.util.write_bitmap("test.exr", raw_render)
        np_render = np.array(raw_render)
        # write_exr(np_render, "test.exr")
        if raw:
            final_image = np_render
        else:
            alpha = np_render[:, :, -1:]
            image = np_render[:, :, :3]
            # no_alpha_render = gsoup.alpha_compose(render)
            image = tonemap_reinhard(image, exposure=1.0)
            image = linear_to_srgb(image)
            final_image = add_alpha(image, alpha)
        return final_image


def estimate_color_mixing_matrix(
    off_image,
    red_image,
    green_image,
    blue_image,
    cam_inv_response=None,
):
    """
    estimates the color mixing matrix V per-pixel for a projector-camera pair.
    based on "A Projection System with Radiometric Compensation for Screen Imperfections"
    assumptions:
        - assumes projector and camera are geometrically calibrated.
        - assumes camera is photometrically calibrated, i.e. the camera response is linear.
        - assumes pixels are independant (no GI)
        - assumes monotonic response per channel.
    :param off_image: image taken when projector projects constant value of "bump_value_low"
    :param red_image: image taken when projector projects constant value of "bump_value_low", except red channel which is "bump_value_high"
    :param green_image: image taken when projector projects constant value of "bump_value_low", except green channel which is "bump_value_high"
    :param blue_image: image taken when projector projects constant value of "bump_value_low", except blue channel which is "bump_value_high"
    :param cam_inv_response: a list of inverse response functions per channel, each is a callable that maps radiance to input value (pass None for linear camera).
    :return: a 3D numpy array of shape (H, W, 3, 3) representing the color mixing matrix V.
    """
    H, W, _ = off_image.shape
    V = np.zeros((H, W, 3, 3), dtype=np.float32)

    # Helper: apply inverse response to image
    def apply_inverse_response(img):
        return np.stack([cam_inv_response[c](img[..., c]) for c in range(3)], axis=-1)

    if cam_inv_response is not None:
        # Linearize all images
        I_off = apply_inverse_response(off_image)
        I_r = apply_inverse_response(red_image)
        I_g = apply_inverse_response(green_image)
        I_b = apply_inverse_response(blue_image)
    else:
        # If no inverse response is provided, assume linear camera response
        I_off = off_image
        I_r = red_image
        I_g = green_image
        I_b = blue_image
    # I_off = to_float(I_off)
    # I_r = to_float(I_r)
    # I_g = to_float(I_g)
    # I_b = to_float(I_b)
    assert I_off.dtype == np.float32
    V = np.tile(np.eye(3)[None, None, ...], (H, W, 1, 1))
    dr = np.clip(I_r - I_off, 0.0, None)
    Vrg = dr[..., 1] / (dr[..., 0] + 1e-6)
    Vrb = dr[..., 2] / (dr[..., 0] + 1e-6)
    dg = np.clip(I_g - I_off, 0.0, None)
    Vgr = dg[..., 0] / (dg[..., 1] + 1e-6)
    Vgb = dg[..., 2] / (dg[..., 1] + 1e-6)
    db = np.clip(I_b - I_off, 0.0, None)
    Vbr = db[..., 0] / (db[..., 2] + 1e-6)
    Vbg = db[..., 1] / (db[..., 2] + 1e-6)
    V[:, :, 0, 1] = Vrg
    V[:, :, 0, 2] = Vrb
    V[:, :, 1, 0] = Vgr
    V[:, :, 1, 2] = Vgb
    V[:, :, 2, 0] = Vbr
    V[:, :, 2, 1] = Vbg

    # Solve 3x3 system per pixel
    inv_V = np.linalg.inv(V)
    return inv_V


def estimate_projector_inverse_response(
    measured_radiance,  # shape: (N, H, W) or (N, H, W, C)
    input_values=None,  # shape: (N,)
    fg_mask=None,  # shape: (H, W) or None
):
    """
    estimate inverse response per channel of a projector
    based on "A Projection System with Radiometric Compensation for Screen Imperfections"
    :param measured_radiance: np.ndarray of shape (N, H, W) or (N, H, W, C)
    :param input_values: np.ndarray of input intensity values. If None, assumes np.arange(N).
    :param fg_mask: foreground mask of shape (H, W) or None. If provided, only uses pixels where fg_mask is True.
    :return: np.ndarray (H, W, C) of interp1d functions. Each maps radiance -> input value.
    """
    # Validate input
    if measured_radiance.ndim == 3:
        # Grayscale: (N, H, W)
        measured_radiance = measured_radiance[..., np.newaxis]  # to (N, H, W, 1)
    elif measured_radiance.ndim != 4:
        raise ValueError("measured_radiance must be shape (N, H, W) or (N, H, W, C)")

    N, H, W, C = measured_radiance.shape
    if input_values is None:
        input_values = np.arange(N) / 255

    if len(input_values) != N:
        raise ValueError(
            "Length of input_values must match number of input images (N)."
        )

    if C == 4:  # discard alpha channel
        C = 3
    if fg_mask is None:
        fg_mask = np.ones((H, W), dtype=bool)
    inverse_maps = np.empty((H, W, C), dtype=object)
    for c in range(C):
        for x in range(W):
            for y in range(H):
                if fg_mask[y, x]:
                    radiance = measured_radiance[..., y, x, c]  # shape: (N, H, W)
                    radiance = make_monotonic(radiance, increasing=True)
                    interp_fn = interp1d(
                        radiance,
                        input_values,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(input_values[0], input_values[-1]),
                    )
                    inverse_maps[y, x, c] = interp_fn
                else:
                    inverse_maps[y, x, c] = None
    return inverse_maps  # list of callables, one per channel


def compute_compensation_image(
    orig_image,
    Vinv,
    cam_inverse_response=None,
    proj_inverse_response=None,
):
    """
    computes a compensation image, such that when projected, the camera will observe the original image.
    based on "A Projection System with Radiometric Compensation for Screen Imperfections"
    :param image: the desired image to be seen by camera (H,W,3)
    :param Vinv: the per-pixel inverse color mixing matrix (H,W,3,3)
    :param cam_inverse_response: a list of callables per channel that map camera pixel values to radiance.
    :param proj_inverse_response: a per-pixel callable that maps projector radiance to pixel values.
    """
    n_channels = orig_image.shape[-1]
    if cam_inverse_response is not None:
        C = np.zeros_like(orig_image, dtype=np.float32)
        for channel in n_channels:
            C[..., channel] = cam_inverse_response[channel](orig_image[..., channel])
    else:
        C = orig_image
    H, W, _ = C.shape
    # P = Vinv @ C
    P = np.matmul(Vinv.reshape(-1, 3, 3), C.reshape(-1, 3, 1)).reshape(H, W, 3)
    if proj_inverse_response is not None:
        I = np.zeros_like(orig_image, dtype=np.float32)
        for channel in range(n_channels):
            for x in np.arange(W):
                for y in np.arange(H):
                    if proj_inverse_response[y, x, channel] is not None:
                        I[y, x, channel] = proj_inverse_response[y, x, channel](
                            P[y, x, channel]
                        )
    else:
        I = P
    return I


def blend_intensity_multi_projectors(forward_maps, fgs, proj_whs, mode="ij"):
    """
    given an image of the overlapping region of two or more projectors, and dense mappings between camera and all projectors pixels, computes an alpha mask per projector for seamless projection
    based on: "Multiprojector Displays using Camera-Based Registration".
    :param forward_maps: a list of forward maps each of (proj_hi x proj_wi x 2) uint32 corresponding to projectors i resolution
    :param fgs: a list of foreground masks each of (proj_hi x proj_wi) bool corresponding to projectors i resolution
    :proj_whs: a list of projector resolutions (proj_wi, proj_hi) as tuples
    :param mode: "xy" or "ij" depending on the order of the last channel of forward_map (see GrayCode.decode)
    :return: a list of alpha masks to use per projector i of size (proj_hi x proj_wi x 1) uint8
    """
    cam_wh = forward_maps[0].shape[:2]
    multi_proj_map = np.ones(
        (len(forward_maps), cam_wh[1], cam_wh[0], 1), dtype=np.float
    )
    for i in range(len(forward_maps)):
        points = np.where(fgs[i])
        convex_hull = ConvexHull(points)
        hull_points = [points[simplex] for simplex in convex_hull.simplices] + [
            points[convex_hull.simplices[0]]
        ]
        hull_edges = np.array(
            [[hull_points[i], hull_points[i + 1]] for i in range(len(hull_points) - 1)]
        )
        distances = np.empty(len(hull_edges), len(points))
        for j, edge in enumerate(hull_edges):
            distances[j] = point_line_distance(points, edge[0], edge[1])
        distances = distances.min(axis=0)
        multi_proj_map[i, points] = distances
    multi_proj_map = multi_proj_map / multi_proj_map.sum(axis=0, keepdims=True)
    multi_proj_map = to_8b(multi_proj_map)
    masks = []
    for fmap, fg, proj_wh in zip(forward_maps, fgs, proj_whs):
        bmap = compute_backward_map(proj_wh, fmap, fg, mode=mode)
        masks.append(multi_proj_map[bmap])
    return masks


def warp_image(backward_map, desired_image, cam_wh=None, mode="xy", output_path=None):
    """
    given a 2D dense map of corresponding pixels between projector and camera, computes a warped image such that if projected, the desired image appears when observed using camera
    :param backward_map: 2D dense mapping between pixels from projector 2 camera (proj_h x proj_w x 2) int64, where -1 indicates a background pixel
    :param desired_image: path to desired image from camera, or float np array channels last (cam_h x cam_w x 3) uint8
    :param cam_wh: device1 (width, height) as tuple, if not supplied assumes desired_image is in the correct dimensions in relation to backward_map
    :param mode: "xy" or "ij" depending on the last channel of backward_map
    :param output_path: path to save warped image to
    :return: warped image (proj_h x proj_w x 3) uint8
    """
    if backward_map.ndim != 3:
        raise ValueError("backward_map must be 3D")
    if backward_map.shape[-1] != 2:
        raise ValueError("backward_map must have shape (cam_h, cam_w, 2)")
    if type(desired_image) == np.ndarray:
        if desired_image.ndim != 3:
            raise ValueError("desired_image must be 3D")
    else:
        if cam_wh is not None:
            desired_image = load_image(desired_image, resize_wh=cam_wh)
        else:
            desired_image = load_image(desired_image)
    if mode == "xy":
        warpped = desired_image[(backward_map[..., 1], backward_map[..., 0])]
    elif mode == "ij":
        warpped = desired_image[(backward_map[..., 0], backward_map[..., 1])]
    else:
        raise ValueError("mode must be 'xy' or 'ij'")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(warpped, output_path)
    return warpped


def compute_backward_map(
    proj_wh,
    forward_map,
    mode="ij",
    interpolate=True,
    output_dir=None,
    debug=False,
):
    """
    computes the inverse map of forward_map by piece-wise interpolating a triangulated version of it (see GrayCode.decode to understand forward_map)
    :param proj_wh: projector (width, height) as a tuple
    :param forward_map: forward map as a numpy array of shape (height, width, 2) of type int64, where -1 indicates a background pixel
    :param foreground: a boolean mask of shape (height, width) where True indicates a valid pixel in forward_map
    :param interpolate: if False, output will not be interpolated (will have holes...)
    :param mode: the last channel order of forward_map. either "xy" (width first) or "ij" (height first).
    :param output_dir: directory to save the inverse map to
    :param debug: if True, saves a visualization of the backward map to output_dir (R=X, G=Y, B=0) where X increases from left to right and Y increases from top to bottom
    :return: backward map as a numpy array of shape (projector_height, projector_width, 2) of type int32, last channel encodes (height, width).
    """
    #### TODO: return subpixel accurate map using median / averaging / both
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    fg = np.all(forward_map >= 0, axis=-1)
    data = np.argwhere(fg)  # always ij
    points = forward_map[fg]
    if interpolate:
        interp = LinearNDInterpolator(points, data, fill_value=0.0)
        X, Y = np.meshgrid(np.arange(proj_wh[0]), np.arange(proj_wh[1]))
        result = interp(X, Y).transpose(1, 0, 2)  # eww
    else:
        sparse_map = np.zeros((proj_wh[1], proj_wh[0], 2), dtype=np.uint32)
        if mode == "xy":
            sparse_map[(points[:, 1]), (points[:, 0])] = data
        elif mode == "ij":
            sparse_map[(points[:, 0]), (points[:, 1])] = data
        result = sparse_map
    if output_dir:
        np.save(Path(output_dir, "backward_map.npy"), result)
        if debug:
            result_normalized = result / np.array(
                [forward_map.shape[0], forward_map.shape[1]]
            )
            result_normalized[..., [0, 1]] = result_normalized[..., [1, 0]]
            result_normalized_8b = to_8b(result_normalized)
            result_normalized_8b_3c = np.concatenate(
                (result_normalized_8b, np.zeros_like(result_normalized_8b[..., :1])),
                axis=-1,
            )
            save_image(result_normalized_8b_3c, Path(output_dir, "backward_map.png"))
    return result.round().astype(np.uint32)


def naive_color_compensate(
    target_image,
    all_white_image,
    all_black_image,
    cam_width,
    cam_height,
    brightness_decrease=-0.5,
    projector_gamma=2.2,
    output_path=None,
    debug=False,
):
    """
    color compensate a projected image such that it appears closer to a target image from the perspective of a camera
    based on "Embedded entertainment with smart projectors".
    :param target_image the desired image path from the perspective of the camera
    :param all_white_image a path to picture taken by camera when projector had all pixels fully on (float32)
    :param all_black_image a path to picture taken by camera when projector had all pixels fully off (float32)
    :param cam_width camera image width
    :param cam_height camera image height
    :param brightness_decrease a hyper parameter controlling how much the total brightness is decreased. without this, the result is saturated because of dividing by small numbers
    :param projector_gamma under normal circumstances the projector does not actually output linear values, so we need to compensate for that
    :output_path if passed, result will be saved to this path
    :debug if true, will save debug info into output_path parent directory
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    target_image = load_image(
        target_image, as_float=True, resize_wh=(cam_width, cam_height)
    )[..., :3]
    target_image = adjust_contrast_brightness(target_image, 1.0, brightness_decrease)
    if debug:
        save_image(target_image, Path(output_path.parent, "decrease_brightness.png"))
    all_white_image = load_image(
        all_white_image, as_float=True, resize_wh=(cam_width, cam_height)
    )
    all_black_image = load_image(
        all_black_image, as_float=True, resize_wh=(cam_width, cam_height)
    )
    compensated = (target_image - all_black_image) / all_white_image
    compensated = np.power(compensated, (1 / projector_gamma))
    compensated = np.nan_to_num(compensated, nan=0.0, posinf=0.0, neginf=0.0)
    compensated = np.clip(compensated, 0, 1)
    if output_path:
        save_image(compensated, output_path)
    return compensated


def calibrate_procam(
    proj_wh,
    capture_dir,
    chess_vert=10,
    chess_hori=7,
    bg_threshold=10,
    chess_block_size=10.0,
    projector_orientation="lower_half",
    verbose=True,
    output_dir=None,
    debug=False,
):
    """
    calibrates a projector-camera pair using local homographies
    based on "Simple, accurate, and robust projector-camera calibration".
    note1: the calibration poses some reasonable constraints on the projector-camera pair:
    1) projector is assumed to have no distortion, and a square pixel aspect ratio.
    2) camera is assumed to have a square pixel aspect ratio, and principle axis is assumed to be the center of the image.
    3) both elements are assumed to have no tangential distortion.
    note2: recommendations for calibration:
    1) use a chessboard with as many blocks as possible, but make sure the chessboard is fully visible in the camera image and in focus.
    2) capture sessions should span the whole image plane, and should be as diverse as possible in terms of poses (tilt the checkerboard !)
    3) when tilting the checkerboard, do not tilt it too much or the projector pixels will get smeared and the gray code decoding will produce large errors.
    4) place the checkeboard only in the working zone of the projector, i.e. the area where the projector is in focus.
    5) make sure the full dynamic range of the camera is used, i.e. the blackest black and whitest white should be visible in the captured images.
    6) attach the checkerboard to a flat surface, make sure the final pattern is as flat as possible. any bending will cause large errors.
    7) capture as many sessions as possible, if a session is not good (use debug=True to check), discard it.
    8) the RMS is only a rough estimate of the calibration quality, but if it is below 1, the calibration should be good enough (units are pixels).
    :param proj_wh projector resolution (width, height) as a tuple
    :param chess_vert when holding the chessboard in portrait mode, this is the number of internal "crossing points" of chessboard in vertical direction (i.e. where two white and two black squares meet)
    :param chess_hori when holding the chessboard in portrait mode, this is the number of internal "crossing points" of chessboard in horizontal direction (i.e. where two white and two black squares meet)
    :param capture_dir directory containing the captured images when gray code patterns were projected, assumes structure as follows:
        capture_dir
            - folder1
                - 0000.png
                - 0001.png
                - ...
            - folder2
                - 0000.png
                - ...
    :param bg_threshold threshold for detecting foreground
    :param chess_block_size size of blocks of chessboard in real world in any unit of measurement (will only effect the translation componenet between camera and projector)
    :param projector_orientation -  effects initial guess for projector principal point.
                                    can be "upper_half" or "lower_half", or "none".
                                    tabletop projectors usually have their principal point in the lower half of the image while ceiling mounted projectors have it in the upper half.
                                    note: opencv convention has the y direction increasing downwards.
    :param verbose if true, will print out the calibration results
    :param output_dir will save debug results to this directory
    :param debug if true, will save debug info into output_dir
    :return camera intrinsics (3x3),
            camera distortion parameters (5x1),
            projector intrinsics (3x3),
            projector distortion parameters (5x1),
            proj_transform (4x4), a camera to projector (c2p) transformation matrix (to get p2w, you should invert this matrix and multiply from the left with the camera to world matrix i.e. p2w = c2w * p2c)
    """
    chess_shape = (chess_vert, chess_hori)
    capture_dir = Path(capture_dir)
    if not capture_dir.exists():
        raise FileNotFoundError("capture_dir was not found")
    dirnames = sorted([x for x in capture_dir.glob("*") if x.is_dir()])
    if len(dirnames) == 0:
        raise FileNotFoundError("capture_dir contains no subfolders")
    used_dirnames = []
    gc_fname_lists = []
    for dname in dirnames:
        gc_fnames = sorted(dname.glob("*"))
        if len(gc_fnames) == 0:
            continue
        used_dirnames.append(str(dname))
        gc_fname_lists.append([str(x) for x in gc_fnames])
    dirnames = used_dirnames
    objps = np.zeros((chess_shape[0] * chess_shape[1], 3), np.float32)
    objps[:, :2] = chess_block_size * np.mgrid[
        0 : chess_shape[0], 0 : chess_shape[1]
    ].T.reshape(-1, 2)
    graycode = GrayCode()
    patterns = graycode.encode(proj_wh)
    cam_shape = load_image(gc_fname_lists[0][0], as_grayscale=True)[:, :, 0].shape[
        ::-1
    ]  # width, height
    patch_size_half = int(
        np.ceil(cam_shape[0] / 180)
    )  # some magic number for patch size
    cam_corners_list = (
        []
    )  # will contain the corners of the chessboard in camera coordinates
    cam_objps_list = (
        []
    )  # will contain the corners of the chessboard in chessboard local coordinates (unit of measurement deduced from chess_block_size)
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    for dname, gc_filenames in zip(dirnames, gc_fname_lists):
        if verbose:
            print(f"processing: {dname}")
        if len(gc_filenames) != len(patterns):
            print(f"invalid number of images in {dname}, skipping")
            continue
        imgs = load_images(gc_filenames, as_grayscale=True)
        forward_map = graycode.decode(
            imgs,
            proj_wh,
            mode="xy",
            bg_threshold=bg_threshold,
            output_dir=output_dir,
            debug=debug,
        )
        fg = np.all(forward_map >= 0, axis=-1)
        black_img = imgs[-1]
        white_img = imgs[-2]
        imgs = imgs[:-2]
        res, cam_corners = cv2.findChessboardCorners(white_img, chess_shape)
        if not res:
            print(f"chessboard was not found in {gc_filenames[-2]}, skipping")
            continue
        cam_objps_list.append(objps)
        cam_corners_list.append(cam_corners)
        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        for corner, objp in zip(cam_corners, objps):
            c_x = int(round(corner[0][0]))
            c_y = int(round(corner[0][1]))
            src_points = []
            dst_points = []
            # todo: vectorize these loops
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    if fg[y, x]:
                        proj_pix = forward_map[y, x]
                        src_points.append((x, y))
                        dst_points.append(np.array(proj_pix))
            if len(src_points) < patch_size_half**2:
                if verbose:
                    print(
                        f"corner {c_x}, {c_y} was skiped because too few decoded pixels found (check your images and thresholds)"
                    )
                continue
            h_mat, inliers = cv2.findHomography(
                np.array(src_points), np.array(dst_points)
            )
            point = h_mat @ np.array([corner[0][0], corner[0][1], 1]).transpose()
            point_pix = point[0:2] / point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
        if len(proj_corners) < 3:
            raise RuntimeError(
                "too few corners were found in {} (less than 3)".format(dname)
            )
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))
    if verbose:
        print(
            "total correspondence points: {}".format(
                sum([len(x) for x in proj_corners_list])
            )
        )
    # Initial solution of camera's intrinsic parameters
    # camera_intrinsics_init = np.array([[np.mean(cam_shape), 0, cam_shape[0]/2], [0, np.mean(cam_shape), cam_shape[1]/2], [0, 0, 1]])
    ret, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
        cam_objps_list,
        cam_corners_list,
        cam_shape,
        None,
        None,
        None,
        None,  # camera_intrinsics_init
        cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT,
    )  # + cv2.CALIB_ZERO_TANGENT_DIST  # cv2.CALIB_USE_INTRINSIC_GUESS
    if verbose:
        print("Camera calib. intrinsic parameters: {}".format(cam_int))
        print("Camera calib. distortion parameters: {}".format(cam_dist))
        print("Camera calib. reprojection error: {}".format(ret))

    # Initial solution of projector's parameters
    if projector_orientation == "none":
        cy_correction = 0
    elif projector_orientation == "lower_half":
        cy_correction = proj_wh[1] / 4
    elif projector_orientation == "upper_half":
        cy_correction = -proj_wh[1] / 4
    else:
        raise ValueError("invalid projector_orientation")
    projector_intrinsics_init = np.array(
        [
            [np.mean(proj_wh), 0, proj_wh[0] / 2],
            [0, np.mean(proj_wh), cy_correction + proj_wh[1] / 2],
            [0, 0, 1],
        ]
    )
    projector_ditortion_init = np.zeros((5, 1))
    ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list,
        proj_corners_list,
        proj_wh,
        projector_intrinsics_init,
        projector_ditortion_init,
        None,
        None,
        cv2.CALIB_USE_INTRINSIC_GUESS
        + cv2.CALIB_FIX_ASPECT_RATIO
        + cv2.CALIB_ZERO_TANGENT_DIST
        + cv2.CALIB_FIX_K1
        + cv2.CALIB_FIX_K2
        + cv2.CALIB_FIX_K3,
    )

    if verbose:
        print("Projector calib. intrinsic parameters: {}".format(proj_int))
        print("Projector calib. distortion parameters: {}".format(proj_dist))
        print("Projector calib. reprojection error: {}".format(ret))

    # Stereo calibration for final solution
    (
        ret,
        cam_int,
        cam_dist,
        proj_int,
        proj_dist,
        cam_proj_rmat,
        cam_proj_tvec,
        E,
        F,
    ) = cv2.stereoCalibrate(
        proj_objps_list,
        cam_corners_list2,
        proj_corners_list,
        cam_int,
        cam_dist,
        proj_int,
        proj_dist,
        None,
    )

    proj_transform = compose_rt(
        cam_proj_rmat[None, ...], cam_proj_tvec[None, :, 0], square=True
    )[0]
    if verbose:
        print("Stereo reprojection error: {}".format(ret))
        print("Stereo camera intrinsic parameters: {}".format(cam_int))
        print("Stereo camera distortion parameters: {}".format(cam_dist))
        print("Stereo projector intrinsic parameters: {}".format(proj_int))
        print("Stereo projector distortion parameters: {}".format(proj_dist))
        print("Stereo camera2projector transform): {}".format(proj_transform))
    if debug:
        # computes a histogram of camera reprojection errors, and not just average error
        cam_corners = np.array(cam_corners_list).squeeze()
        proj_corners = np.array(proj_corners_list).squeeze()
        obj_corners = np.array(cam_objps_list).squeeze()
        all_projected_cam_corners = []
        all_projected_proj_corners = []
        for i in range(len(cam_corners)):
            projected_cam_points, _ = cv2.projectPoints(
                obj_corners[i], cam_rvecs[i], cam_tvecs[i], cam_int, cam_dist
            )
            all_projected_cam_corners.append(projected_cam_points)
            projected_proj_points, _ = cv2.projectPoints(
                obj_corners[i], proj_rvecs[i], proj_tvecs[i], proj_int, proj_dist
            )
            all_projected_proj_corners.append(projected_proj_points)
        all_projected_cam_corners = np.array(all_projected_cam_corners).squeeze()
        cam_norms = np.linalg.norm(cam_corners - all_projected_cam_corners, axis=-1)
        cam_per_session_error = cam_norms.mean(axis=-1)
        worst_to_best_cam_session_ids = np.argsort(cam_per_session_error)[::-1]
        worst_to_best_cam_errors = cam_per_session_error[worst_to_best_cam_session_ids]
        print(
            "worst to best sessions ids for camera reprojection error: {}".format(
                worst_to_best_cam_session_ids
            )
        )
        print("and their associated errors: {}".format(worst_to_best_cam_errors))
        cam_hist = np.histogram(cam_norms)
        print(
            "camera reprojection error histogram: {} (should be similar to gaussian around 0)".format(
                cam_hist
            )
        )
        all_projected_proj_corners = np.array(all_projected_proj_corners).squeeze()
        proj_norms = np.linalg.norm(proj_corners - all_projected_proj_corners, axis=-1)
        per_session_projector_error = proj_norms.mean(axis=-1)
        worst_to_best_proj_session_ids = np.argsort(per_session_projector_error)[::-1]
        worst_to_best_proj_errors = per_session_projector_error[
            worst_to_best_proj_session_ids
        ]
        print(
            "worst to best sessions ids for projector reprojection error: {}".format(
                worst_to_best_proj_session_ids
            )
        )
        print("and their associated errors: {}".format(worst_to_best_proj_errors))
        proj_hist = np.histogram(proj_norms)
        print(
            "projector reprojection error histogram: {} (should be similar to gaussian around 0)".format(
                proj_hist
            )
        )
    ret = {
        "cam_intrinsics": cam_int,
        "cam_distortion": cam_dist,
        "proj_intrinsics": proj_int,
        "proj_distortion": proj_dist,
        "proj_transform": proj_transform,
    }
    return ret


def reconstruct_pointcloud(
    forward_map,
    cam_transform,
    proj_transform,
    cam_int,
    cam_dist,
    proj_int,
    mode="ij",
    color_image=None,
    debug=False,
):
    """
    given a dense pixel correspondence map between a camera and a projector, and calibration results, reconstructs a 3D point cloud of the scene.
    :param forward_map: a dense pixel correspondence map between a camera and a projector (see GrayCode.decode)
    :param fg: a foreground mask of the scene (see GrayCode.decode)
    :param cam_transform: a 4x4 transformation matrix of the camera (cam to world)
    :param proj_transform: a 4x4 transformation matrix of the projector (proj to world)
    :param cam_int: camera's intrinsic parameters
    :param cam_dist: camera's distortion parameters
    :param proj_int: projector's intrinsic parameters
    :param mode: "xy" or "ij" which is the ordering of the last channel of the forward map (see GrayCode.decode)
    :param color_image: RGB image with the same spatial size as forwardmap. if supplied will return a colored point cloud (Nx6).
    :param debug: if True, will return debug information
    :return: a 3D point cloud of the scene (Nx3)
    """
    fg = np.all(forward_map >= 0, axis=-1)
    cam_pixels = swap_columns(np.argwhere(fg), 0, 1)
    # todo: account for distortion if supplied
    # undistorted_cam_points =  cv2.undistortPoints(cam_points, cam_int, cam_dist, P=proj_transform[0]).squeeze()  # equivalent to not setting P and doing K @ points outside
    cam_origins = cam_transform[None, :3, -1]
    cam_directions = (
        cam_transform[:3, :3] @ (np.linalg.inv(cam_int) @ to_hom(cam_pixels).T)
    ).T
    cam_directions = cam_directions / np.linalg.norm(
        cam_directions, axis=-1, keepdims=True
    )
    if mode == "xy":
        projector_pixels = forward_map[fg]
    elif mode == "ij":
        projector_pixels = swap_columns(forward_map[fg], 0, 1)
    else:
        raise ValueError("mode must be either 'xy' or 'ij'")
    projector_origins = proj_transform[None, :3, -1]
    projector_directions = (
        proj_transform[:3, :3] @ (np.linalg.inv(proj_int) @ to_hom(projector_pixels).T)
    ).T
    projector_directions = projector_directions / np.linalg.norm(
        projector_directions, axis=-1, keepdims=True
    )
    # note: using ray-ray is slightly naive. A better way is to minimize a reprojection error, and then ray-ray
    points, weight_factor = ray_ray_intersection(
        cam_origins, cam_directions, projector_origins, projector_directions
    )
    if color_image is not None:
        colors = (
            color_image[fg] if mode == "xy" else color_image[swap_columns(fg, 0, 1)]
        )
        points = np.concatenate([points, colors], axis=-1)
    if debug:
        return (
            points,
            weight_factor,
            cam_origins,
            cam_directions,
            projector_origins,
            projector_directions,
            cam_pixels,
            projector_pixels,
        )
    else:
        return points


class GrayCode:
    """
    a class that handles encoding and decoding binary graycode patterns
    """

    def encode1d(self, length):
        total_bits = np.ceil(
            np.log2(length)
        )  # how many bits are needed to represent the length
        x = np.arange(length, dtype=np.uint64)  # [0, 1, 2, ..., length-1]
        gray = x ^ (x >> 1)  # Gray code of each x
        shifts = np.arange(total_bits - 1, -1, -1, dtype=np.uint64)[
            :, None
        ]  # [[MSB], ..., [LSB]]
        bits = ((gray >> shifts) & 1).astype(
            np.uint8
        )  # represent as binary with shape (total_bits, length) -> a binary number per pixel
        return bits * 255

    def encode(self, proj_wh, flipped_patterns=True):
        """
        encode projector's width and height into gray code patterns
        :param proj_wh: projector's (width, height) in pixels as a tuple
        :param flipped_patterns: if True, flipped patterns are also generated for better binarization
        :return: numpy array of shape (total_images, height, width, 1) where total_images is the number of gray code patterns
        """
        width, height = proj_wh
        codes_width_1d = self.encode1d(width)[:, None, :]
        codes_width_2d = codes_width_1d.repeat(height, axis=1)
        codes_height_1d = self.encode1d(height)[:, :, None]
        codes_height_2d = codes_height_1d.repeat(width, axis=2)

        img_white = np.full((height, width), 255, dtype=np.uint8)[None, ...]
        img_black = np.full((height, width), 0, dtype=np.uint8)[None, ...]
        all_images = np.concatenate((codes_width_2d, codes_height_2d), axis=0)
        if flipped_patterns:
            all_images = np.concatenate((all_images, 255 - codes_width_2d), axis=0)
            all_images = np.concatenate((all_images, 255 - codes_height_2d), axis=0)
        all_images = np.concatenate((all_images, img_white, img_black), axis=0)
        return all_images[..., None]

    def binarize(self, captures, flipped_patterns=True, bg_threshold=10):
        """
        binarize a batch of images
        :param captures: a 4D numpy array of shape (n, height, width, 1) of captured images
        :param flipped_patterns: if true, patterns also contain their flipped version for better binarization
        :param bg_threshold: if the difference between the pixel in the white&black images is greater than this, pixel is foreground
        :return: a 4D numpy binary array for decoding (total_images, height, width, 1) where total_images is the number of gray code patterns
        and a binary background mask (height, width, 1)
        """
        if not 0 <= bg_threshold <= 255:
            raise ValueError("bg_threshold must be between 0 and 255")
        patterns, bw = captures[:-2], captures[-2:]
        bg = np.abs(bw[0].astype(np.int32) - bw[1].astype(np.int32)) <= bg_threshold
        if flipped_patterns:
            orig, flipped = (
                patterns[: len(patterns) // 2],
                patterns[len(patterns) // 2 :],
            )
            # valid = (orig.astype(np.int32) - flipped.astype(np.int32)) > bin_threshold
            binary = orig > flipped
            # foreground = foreground & np.all(valid, axis=0)  # only pixels that are valid in all images are foreground
        else:  # slightly more naive thresholding
            binary = patterns >= 0.5 * (bw[1] + bw[0])
        return binary, bg

    def decode1d(self, gc_imgs):
        # gc_imgs: shape (n, h, w), are boolean images
        n = gc_imgs.shape[0]
        # gray -> binary via cumulative xor along bit axis (MSB->LSB)
        binary = np.bitwise_xor.accumulate(gc_imgs, axis=0)
        # build 1D weights and broadcast (MSB weight = 2**(n-1))
        weights = (1 << np.arange(n - 1, -1, -1, dtype=np.int64))[:, None, None]
        # multiply and sum to get final index image
        result = np.sum(binary * weights, axis=0)  # shape (h, w)
        return result

    def decode(
        self,
        captures,
        proj_wh,
        flipped_patterns=True,
        bg_threshold=10,
        mode="ij",
        output_dir=None,
        debug=False,
    ):
        """
        decodes a batch of images encoded with gray code
        :param captures: a 4D numpy array of shape (n, height, width, c) of captured images (camera resolution is inferred from this)
        :param flipped_patterns: if true, patterns also contain their flipped version for better binarization
        :param bg_threshold: a threshold used for background detection using the all-white and all-black captures
        :param mode: "xy" or "ij" decides the order of last dimension coordinates in the output (ij -> height first, xy -> width first)
        :param output_dir: if not None, saves the decoded images to this directory
        :param debug: if True, visualizes the map in an image using the red and green channels where R=X, G=Y, B=0, X increases from left to right, Y increases from top to bottom
        :return: a 2D numpy array of shape (height, width, 2) int64 mapping from camera pixels to projector pixels (-1 for background)
        and a foreground mask (height, width) of booleans
        """
        if captures.ndim != 4:
            raise ValueError("captures must be a 4D numpy array")
        if captures.dtype != np.uint8:
            raise ValueError("captures must be uint8")
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        b, _, _, c = captures.shape
        b = b - 2  # dont count white and black images
        if flipped_patterns:
            b = b // 2  # dont count flipped patterns
        encoded = self.encode(
            proj_wh, flipped_patterns
        )  # sanity: encode with same arguments to verify enough captures are present
        if len(encoded) != len(captures):
            raise ValueError("captures must have length of {}".format(len(encoded)))
        if c != 1:  # naively convert to grayscale
            captures = rgb_to_gray(captures, keep_channels=True)
        imgs_binary, bg = self.binarize(captures, flipped_patterns, bg_threshold)
        imgs_binary = imgs_binary[:, :, :, 0]
        bg = bg[:, :, 0]
        x = self.decode1d(imgs_binary[: b // 2])
        maskx = x < proj_wh[0]
        x[~maskx] = (
            -1
        )  # mask out invalid x coordinates (the amount of bits we used is equal or larger than the width)
        y = self.decode1d(imgs_binary[b // 2 :])
        masky = y < proj_wh[1]
        y[~masky] = (
            -1
        )  # mask out invalid y coordinates (the amount of bits we used is equal or larger than the height)
        x[bg] = -1
        y[bg] = -1
        if mode == "ij":
            forward_map = np.concatenate((y[..., None], x[..., None]), axis=-1)
        elif mode == "xy":
            forward_map = np.concatenate((x[..., None], y[..., None]), axis=-1)
        else:
            raise ValueError("mode must be 'ij' or 'xy'")
        if output_dir is not None:
            np.save(Path(output_dir, "forward_map.npy"), forward_map)
            # np.save(Path(output_dir, "fg.npy"), fg)
            if debug:
                save_images(imgs_binary[..., None], Path(output_dir, "imgs_binary"))
                # save_image(fg, Path(output_dir, "foreground.png"))
                if mode == "ij":
                    composed_normalized = forward_map / np.array(
                        [proj_wh[1], proj_wh[0]]
                    )
                    composed_normalized[..., [0, 1]] = composed_normalized[..., [1, 0]]
                elif mode == "xy":
                    composed_normalized = forward_map / np.array(
                        [proj_wh[0], proj_wh[1]]
                    )
                composed_normalized[forward_map < 0] == 0
                composed_normalized_8b = to_8b(composed_normalized)
                composed_normalized_8b_3c = np.concatenate(
                    (
                        composed_normalized_8b,
                        np.zeros_like(composed_normalized_8b[..., :1]),
                    ),
                    axis=-1,
                )
                save_image(
                    composed_normalized_8b_3c, Path(output_dir, "forward_map.png")
                )
        return forward_map


class PhaseShifting:
    """
    A class that handles encoding and decoding phase-shifting sinusoidal patterns.
    Uses multiple phase-shifted sinusoidal patterns to encode pixel coordinates.
    Based on "Phase-shifting for structured light scanning".
    """

    def __init__(self, num_phases=4, cycles_x=None, cycles_y=None):
        """
        Initialize phase-shifting pattern generator.
        :param num_temporal_phases: number of phase shifts (typically 3-5)
        :param cycles_x: number of cycles across image width
        :param cycles_y: number of cycles across image height
        """
        self.num_temporal_phases = num_phases
        self.cycles_x = cycles_x
        self.cycles_y = cycles_y

    def encode_internal(self, proj_wh, cycles_x, cycles_y, num_temporal_phases, include_reference):
        width, height = proj_wh

        patterns = []

        # Generate x-direction phase-shifting patterns
        x_coords = np.arange(width, dtype=np.float32)
        DC = 128
        A = 127
        spatial_phase = 2 * np.pi * cycles_x * x_coords / width
        for phase_idx in range(num_temporal_phases):
            temporal_phase = 2 * np.pi * phase_idx / num_temporal_phases
            # Create sinusoidal pattern: I = A + B * cos(2πfx + φ)
            pattern_x = DC + A * np.cos(spatial_phase + temporal_phase)
            pattern_x = np.clip(pattern_x, 0, 255).astype(np.uint8)

            # Broadcast to full 2D pattern
            pattern_2d = np.tile(pattern_x[None, :], (height, 1))
            patterns.append(pattern_2d[..., None])  # Add channel dimension

        # Generate y-direction phase-shifting patterns
        y_coords = np.arange(height, dtype=np.float32)
        spatial_phase_y = 2 * np.pi * cycles_y * y_coords / height
        for phase_idx in range(num_temporal_phases):
            temporal_phase = 2 * np.pi * phase_idx / num_temporal_phases
            # Create sinusoidal pattern: I = A + B * cos(2πfy + φ)
            pattern_y = DC + A * np.cos(spatial_phase_y + temporal_phase)
            pattern_y = np.clip(pattern_y, 0, 255).astype(np.uint8)

            # Broadcast to full 2D pattern
            pattern_2d = np.tile(pattern_y[:, None], (1, width))
            patterns.append(pattern_2d[..., None])  # Add channel dimension

        if include_reference:
            # Add white and black reference images
            white_img = np.full((height, width, 1), 255, dtype=np.uint8)
            black_img = np.zeros((height, width, 1), dtype=np.uint8)
            patterns.extend([white_img, black_img])

        return np.array(patterns)

    def encode(self, proj_wh):
        """
        Encode projector coordinates into phase-shifting sinusoidal patterns.
        :param proj_wh: projector's (width, height) in pixels as a tuple
        :return: a 3D numpy array of shape (total_images, height, width, 1) for grayscale patterns
        """
        width, height = proj_wh
        # Set default cycle counts if not provided
        if self.cycles_x is None:
            self.cycles_x = width // 8  # 8 cycles across width
        if self.cycles_y is None:
            self.cycles_y = height // 8  # 8 cycles across height
        lowfreq_patterns = self.encode_internal(proj_wh, 1, 1, self.num_temporal_phases, False)
        highfreq_patterns = self.encode_internal(proj_wh, self.cycles_x, self.cycles_y, self.num_temporal_phases, True)
        all_patterns = np.concatenate([lowfreq_patterns, highfreq_patterns], axis=0)
        return all_patterns

    def compute_spatial_phase(self, intensity_values, num_temporal_phases):
        """
        Compute spatial phase from intensity values using phase-shifting algorithm.
        :param intensity_values: array of intensity values for different phases (n_phases/2,)
        :param num_temporal_phases: number of phase shifts
        :return: computed phase values
        """
        if len(intensity_values) != num_temporal_phases:
            raise ValueError(
                f"Expected {num_temporal_phases} intensity values, got {len(intensity_values)}"
            )

        # Convert to float for computation
        I = intensity_values.astype(np.float32)

        if num_temporal_phases == 3:
            # 3-step algorithm: φ = atan2(I3 - I1, 2*I2 - I1 - I3)
            numerator = I[2] - I[0]  # I3 - I1
            denominator = 2 * I[1] - I[0] - I[2]  # 2*I2 - I1 - I3
            spatial_phase = np.arctan2(numerator, denominator)
        elif num_temporal_phases == 4:
            # 4-step algorithm: φ = atan2(I4 - I2, I1 - I3)
            numerator = I[3] - I[1]  # I4 - I2
            denominator = I[0] - I[2]  # I1 - I3
            spatial_phase = np.arctan2(numerator, denominator)
        elif num_temporal_phases == 5:
            # 5-step algorithm: φ = atan2(2*(I4 - I2), 2*I3 - I1 - I5)
            numerator = 2 * (I[3] - I[1])  # 2*(I4 - I2)
            denominator = 2 * I[2] - I[0] - I[4]  # 2*I3 - I1 - I5
            spatial_phase = np.arctan2(numerator, denominator)
        else:
            # General N-step algorithm
            numerator = 0
            denominator = 0
            for k in range(num_temporal_phases):
                angle = 2 * np.pi * k / num_temporal_phases
                numerator += I[k] * np.sin(angle)
                denominator += I[k] * np.cos(angle)
            spatial_phase = np.arctan2(numerator, denominator)

        return spatial_phase

    def unwrap_spatial_phase(self, wrapped_spatial_phase, cycles, image_size, bins=None):
        """
        Unwrap phase to get absolute coordinates.
        :param wrapped_spatial_phase: wrapped phase values (-π to π)
        :param cycles: number of cycles across image dimension
        :param image_size: size of the image dimension
        :param bins: bins to use for unwrapping
        :return: unwrapped coordinates
        """
        # Convert phase to coordinates
        # spatial_phase = 2π * cycles * coord / image_size
        # coord = phase * image_size / (2π * cycles)
        coordinates = wrapped_spatial_phase * image_size / (2 * np.pi * cycles)
        # Handle phase wrapping by finding the correct cycle
        cycle_length = image_size / cycles
        coordinates = np.mod(coordinates, cycle_length)
        return coordinates

    def decode_lowfreq(self, captures, proj_wh, foreground, mode, output_dir, debug):
        width, height = proj_wh
        # Extract x-direction patterns (first num_phases images)
        x_patterns = captures[:self.num_temporal_phases, :, :, 0]  # (n_phases, h, w)

        # Extract y-direction patterns (next num_phases images)
        y_patterns = captures[self.num_temporal_phases:2*self.num_temporal_phases, :, :, 0]  # (n_phases, h, w)

        # Compute phase for x-direction
        x_phase = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if foreground[y, x]:
                    intensity_values = x_patterns[:, y, x]  # (n_phases,)
                    x_phase[y, x] = self.compute_spatial_phase(
                        intensity_values, self.num_temporal_phases
                    )

        # Compute phase for y-direction
        y_phase = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if foreground[y, x]:
                    intensity_values = y_patterns[:, y, x]
                    y_phase[y, x] = self.compute_spatial_phase(
                        intensity_values, self.num_temporal_phases
                    )

        # Unwrap phases to get coordinates
        x_coords = self.unwrap_spatial_phase(x_phase, 1, width)
        y_coords = self.unwrap_spatial_phase(y_phase, 1, height)

        # Convert to integer coordinates and clamp to valid range
        x_coords = np.clip(np.round(x_coords), 0, width - 1).astype(np.uint64)
        y_coords = np.clip(np.round(y_coords), 0, height - 1).astype(np.uint64)

        # Create forward mapping
        forward_map = np.zeros((height, width, 2), dtype=np.int64)
        forward_map[..., 0] = x_coords
        forward_map[..., 1] = y_coords

        # set negative values where foreground is false
        forward_map[~foreground] = -1
        forward_map[forward_map[..., 0] >= width] = -1
        forward_map[forward_map[..., 1] >= height] = -1
        # Apply coordinate mode
        if mode == "ij":
            forward_map = forward_map[..., [1, 0]]  # Swap to (y, x) order
        elif mode == "xy":
            forward_map = forward_map  # Already in (x, y) order
        else:
            raise ValueError("mode must be 'ij' or 'xy'")

        if output_dir is not None:
            np.save(Path(output_dir, "forward_map_coarse.npy"), forward_map)

            if debug:
                self.save_debug_info(x_phase, y_phase, foreground, forward_map, proj_wh, mode, output_dir, "coarse")

        return forward_map

    def decode_highfreq(self, captures, proj_wh,foreground, coarse_forwardmap, mode, output_dir, debug):
        width, height = proj_wh
        # Extract x-direction patterns (first num_phases images)
        x_patterns = captures[: self.num_temporal_phases, :, :, 0]  # (n_phases, h, w)

        # Extract y-direction patterns (next num_phases images)
        y_patterns = captures[
            self.num_temporal_phases : 2 * self.num_temporal_phases, :, :, 0
        ]  # (n_phases, h, w)

        # Compute phase for x-direction
        x_phase = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if foreground[y, x]:
                    intensity_values = x_patterns[:, y, x]  # (n_phases,)
                    x_phase[y, x] = self.compute_spatial_phase(
                        intensity_values, self.num_temporal_phases
                    )

        # Compute phase for y-direction
        y_phase = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if foreground[y, x]:
                    intensity_values = y_patterns[:, y, x]
                    y_phase[y, x] = self.compute_spatial_phase(
                        intensity_values, self.num_temporal_phases
                    )

        # Unwrap phases to get coordinates
        x_coords = self.unwrap_spatial_phase(x_phase, self.cycles_x, width)
        y_coords = self.unwrap_spatial_phase(y_phase, self.cycles_y, height)

        # Convert to integer coordinates and clamp to valid range
        x_coords = np.clip(np.round(x_coords), 0, width - 1).astype(np.uint64)
        y_coords = np.clip(np.round(y_coords), 0, height - 1).astype(np.uint64)

        # Use coarse forward map to unwrap phases
        if mode == "ij":
            xbins = coarse_forwardmap[..., 1]
            ybins = coarse_forwardmap[..., 0]
        elif mode == "xy":
            xbins = coarse_forwardmap[..., 0]
            ybins = coarse_forwardmap[..., 1]
        else:
            raise ValueError("mode must be 'ij' or 'xy'")

        # use the coarse bins to resolve phase wrapping (x_coords are "local" within a bin)
        # Calculate cycle length for high frequency patterns
        x_cycle_length = width / self.cycles_x
        y_cycle_length = height / self.cycles_y
        
        # For each pixel, determine which cycle it should be in based on coarse map
        # and adjust the high-frequency coordinates accordingly
        for y in range(height):
            for x in range(width):
                if foreground[y, x] and xbins[y, x] >= 0 and ybins[y, x] >= 0:
                    # Find which cycle the coarse bin corresponds to
                    coarse_x_cycle = int(xbins[y, x] // x_cycle_length)
                    coarse_y_cycle = int(ybins[y, x] // y_cycle_length)
                    
                    # Adjust high-frequency coordinates to be in the correct cycle
                    x_coords[y, x] = coarse_x_cycle * x_cycle_length + (x_coords[y, x] % x_cycle_length)
                    y_coords[y, x] = coarse_y_cycle * y_cycle_length + (y_coords[y, x] % y_cycle_length)

        # Create forward mapping
        forward_map = np.zeros((height, width, 2), dtype=np.int64)
        forward_map[..., 0] = x_coords
        forward_map[..., 1] = y_coords

        # set negative values where foreground is false
        forward_map[~foreground] = -1
        forward_map[forward_map[..., 0] >= width] = -1
        forward_map[forward_map[..., 1] >= height] = -1
        # Apply coordinate mode
        if mode == "ij":
            forward_map = forward_map[..., [1, 0]]  # Swap to (y, x) order
        elif mode == "xy":
            forward_map = forward_map  # Already in (x, y) order
        else:
            raise ValueError("mode must be 'ij' or 'xy'")

        if output_dir is not None:
            np.save(Path(output_dir, "forward_map_fine.npy"), forward_map)
            if debug:
                self.save_debug_info(x_phase, y_phase, foreground, forward_map, proj_wh, mode, output_dir, "fine")
        return forward_map

    def save_debug_info(self, x_phase, y_phase, foreground, forward_map, proj_wh, mode,output_dir, prefix):
        # Save phase maps
            x_phase_normalized = ((x_phase + np.pi) / (2 * np.pi) * 255).astype(
                np.uint8
            )
            y_phase_normalized = ((y_phase + np.pi) / (2 * np.pi) * 255).astype(
                np.uint8
            )
            save_image(x_phase_normalized, Path(output_dir, "x_phase_{}.png".format(prefix)))
            save_image(y_phase_normalized, Path(output_dir, "y_phase_{}.png".format(prefix)))
            # save_image(foreground, Path(output_dir, "foreground.png"))

            # Create visualization of forward map
            composed = forward_map * foreground[..., None]
            if mode == "ij":
                composed_normalized = composed / np.array([proj_wh[1], proj_wh[0]])
                composed_normalized[..., [0, 1]] = composed_normalized[..., [1, 0]]
            elif mode == "xy":
                composed_normalized = composed / np.array([proj_wh[0], proj_wh[1]])

            composed_normalized_8b = to_8b(composed_normalized)
            composed_normalized_8b_3c = np.concatenate(
                (
                    composed_normalized_8b,
                    np.zeros_like(composed_normalized_8b[..., :1]),
                ),
                axis=-1,
            )
            save_image(composed_normalized_8b_3c, Path(output_dir, "forward_map_{}.png".format(prefix)))

    def decode(
        self,
        captures,
        proj_wh,
        bg_threshold=10,
        mode="ij",
        output_dir=None,
        debug=False,
    ):
        """
        Decode captured phase-shifting patterns to extract forward mapping.
        :param captures: a 4D numpy array of shape (n, height, width, 1) of captured grayscale images
        :param proj_wh: projector's (width, height) in pixels as a tuple
        :param bg_threshold: threshold for background detection
        :param mode: "xy" or "ij" decides coordinate order in output
        :param output_dir: if not None, saves debug results
        :param debug: if True, saves debug visualizations
        :return: forward_map (height, width, 2) int64, negative values for invalid coordinates
        """
        if captures.ndim != 4:
            raise ValueError("captures must be a 4D numpy array")
        if captures.shape[-1] != 1:
            raise ValueError("captures must have 1 grayscale channel")
        if captures.dtype != np.uint8:
            raise ValueError("captures must be uint8")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        
        n_images = len(captures)

        # Expected images: x_patterns_lowfreq, y_patterns_lowfreq, x_patterns_highfreq, y_patterns_highfreq, white, black
        expected_images = 4 * self.num_temporal_phases + 2
        if n_images != expected_images:
            raise ValueError(
                f"captures must have {expected_images} images, got {n_images}"
            )

        # Extract reference images for background detection
        white_img = captures[-2]  # Second to last
        black_img = captures[-1]  # Last image

        # Create foreground mask based on white-black difference
        intensity_diff = white_img.astype(np.float32) - black_img.astype(np.float32)
        foreground = intensity_diff[..., 0] > bg_threshold

        coarse_forwardmap = self.decode_lowfreq(captures[:2*self.num_temporal_phases], proj_wh, foreground, mode, output_dir, debug)
        fine_forwardmap = self.decode_highfreq(captures[2*self.num_temporal_phases:4*self.num_temporal_phases], proj_wh, foreground, coarse_forwardmap, mode, output_dir, debug)
        return fine_forwardmap
