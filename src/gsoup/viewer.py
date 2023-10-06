import numpy as np
from . import structures
from .core import to_hom, homogenize
from .gsoup_io import write_to_json


class PolyscopeStub:
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            print("{} was called".format(name))

        return wrapper


class gviewer:
    """
    wrapper class for polyscope, invoked as a singelton ("from gsoup.viewer import gviewer")
    offers a stub for when polyscope is not installed, window resizing and a few other things
    """

    def __init__(self):
        self.is_init = False

    def init(
        self,
        up_dir="z_up",
        look_at=None,
        ground_mode="shadow_only",
        projection_mode="perspective",
        width=None,
        height=None,
    ):
        """
        given a ps instance, intializes it
        :param up_dir: up direction
        :param look_at: where to look at (3 tuples eye, at ,up)
        :param ground_mode: ground render mode
        :param projection_mode: projection_mode
        """
        if width is not None:
            if height is None:
                height = width
            poly_dict = {
                "windowHeight": height,
                "windowPosX": 50,
                "windowPosY": 50,
                "windowWidth": width,
            }
            write_to_json(poly_dict, "./.polyscope.ini")
        try:
            import polyscope as ps
            import polyscope.imgui as psim

            ps.init()
        except RuntimeError:
            ps = PolyscopeStub()
            psim = PolyscopeStub()
        self.ps = ps
        self.clean()
        self.psim = psim
        self.ps.set_up_dir(up_dir)
        if look_at is None:
            eye, at, up = (-1.0, 0.75, -1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        else:
            eye, at, up = look_at
        self.ps.look_at_dir(eye, at, up)
        self.ps.set_ground_plane_mode(ground_mode)
        self.ps.set_view_projection_mode(
            projection_mode
        )  # ps.set_view_projection_mode("orthographic")
        self.is_init = True

    def register_pointcloud(
        self,
        name,
        points,
        c=None,
        s=None,
        v=None,
        radius=1e-2,
        mode="quad",
        enabled=True,
    ):
        """
        register a point cloud to polyscope
        """
        ps_pointcloud = self.ps.register_point_cloud(
            name,
            points,
            radius=radius,
            point_render_mode=mode,
            enabled=enabled,
        )  # color=np.array([0.5, 0.1, 0.3])  # sphere / quad
        if c is not None:
            ps_pointcloud.add_color_quantity("colors", c, enabled=True)
        if s is not None:
            s = (s - np.min(s)) / (np.max(s) - np.min(s))
            s = np.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)
            ps_pointcloud.add_scalar_quantity(
                "scalar_value", s, enabled=True, vminmax=(0.0, 1.0), cmap="reds"
            )
        if v is not None:
            # v /= (np.linalg.norm(v, axis=-1) + 1e-6)
            ps_pointcloud.add_vector_quantity(
                "vecs", v, enabled=True, radius=0.002, length=0.1, color=(0.2, 0.5, 0.5)
            )
        return ps_pointcloud

    def register_camera(self, name, poses, edge_rad, group=True, alpha=1.0, scale=0.1):
        """
        register a camera structure to polyscope
        """
        v_cam, e_cam, c_cam = structures.get_camera_coords(scale)
        v_tot = []
        e_tot = []
        c_tot = []
        for i, pose in enumerate(poses):
            v = homogenize((pose @ to_hom(v_cam).T).T)
            if group:
                v_tot.append(v)
                c_tot.append(c_cam)
                e_tot.append(e_cam + (i * len(v_cam)))
            else:
                ps_net = self.ps.register_curve_network(
                    "{}_{}".format(name, i), v, e_cam, radius=edge_rad
                )
                ps_net.add_color_quantity(
                    "color", c_cam, defined_on="edges", enabled=True
                )
                ps_net.set_transparency(alpha)
        if group:
            v_tot = np.array(v_tot).reshape(-1, 3)
            e_tot = np.array(e_tot).reshape(-1, 2)
            c_tot = np.array(c_tot).reshape(-1, 3)
            ps_net = self.ps.register_curve_network(name, v_tot, e_tot, radius=edge_rad)
            ps_net.add_color_quantity("color", c_tot, defined_on="edges", enabled=True)
        ps_net.set_transparency(alpha)
        return v_tot, e_tot, c_tot

    def register_mesh(
        self,
        name,
        v,
        f,
        transparency=1.0,
        edge_width=0.0,
        color=[0.5, 0.5, 0.5],
        smooth_shade=True,
        s_vertices=None,
        c_vertices=None,
        v_vertices=None,
        s_faces=None,
        c_faces=None,
        cmap=None,
        enabled=True,
    ):
        """
        regiter a mesh to polyscope
        :param ps: polyscope instance
        :param name: name of the mesh
        :param v: vertices
        :param f: faces
        :param transparency: transparency of the mesh
        :param edge_width: edge width of the mesh
        :param smooth_shade: smooth shading
        :param s_vertices: vertex scalar values
        :param c_vertices: vertex scalar values
        :param v_vertices: vertex vectors
        :param s_faces: face scalar values
        :param c_faces: face colors
        """
        ps_mesh = self.ps.register_surface_mesh(
            name,
            v,
            f,
            edge_width=edge_width,
            transparency=transparency,
            color=color,
            smooth_shade=smooth_shade,
            enabled=enabled,
        )
        if cmap is None:
            cmap = "reds"
        if s_vertices is not None:
            s = (s_vertices - np.min(s_vertices)) / (
                np.max(s_vertices) - np.min(s_vertices)
            )
            ps_mesh.add_scalar_quantity(
                "scalar",
                s,
                defined_on="vertices",
                enabled=True,
                vminmax=(0.0, 1.0),
                cmap=cmap,
            )
        if c_vertices is not None:
            ps_mesh.add_color_quantity(
                "color", c_vertices, defined_on="vertices", enabled=True
            )
        if v_vertices is not None:
            ps_mesh.add_vector_quantity(
                "vecs",
                v_vertices,
                enabled=True,
                # vectortype='ambient',
                radius=0.01,
                length=0.1,
                color=(0.2, 0.5, 0.5),
            )
        if s_faces is not None:
            s = (s_faces - np.min(s_faces)) / (np.max(s_faces) - np.min(s_faces))
            ps_mesh.add_scalar_quantity(
                "fscalar",
                s,
                defined_on="faces",
                enabled=True,
                vminmax=(0.0, 1.0),
                cmap=cmap,
            )
        if c_faces is not None:
            ps_mesh.add_color_quantity(
                "fcolor", c_faces, defined_on="faces", enabled=True
            )
        return ps_mesh

    def show(self):
        self.ps.show()

    def clean(self):
        self.ps.remove_all_structures()

    def clear(self):
        self.ps.clean()

    def remove_all_structures(self):
        self.ps.clean()


gviewer = gviewer()
