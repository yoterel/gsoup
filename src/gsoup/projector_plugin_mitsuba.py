import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant(
    "llvm_ad_rgb"
    # "scalar_rgb"
)  # "llvm_ad_rgb", "scalar_rgb" # must set before defining the emitter


class ProjectorPy(mi.Emitter):
    def __init__(self, props):
        super().__init__(props)
        self.m_to_world = props.get("to_world", mi.ScalarTransform4f())
        if isinstance(self.m_to_world, mi.ScalarTransform4d):
            self.m_to_world = mi.Transform4d(self.m_to_world.matrix)
        self.m_intensity_scale = dr.opaque(mi.Float, props.get("scale", 1.0))
        # self.m_irradiance = props.texture_d65("irradiance", 1.0)
        if "irradiance" in props:
            self.m_irradiance = props["irradiance"]
        else:
            # Create a constant white RGB texture (D65 white)
            self.m_irradiance = mi.load_dict({"type": "rgb", "value": 1.0})
        # it_query = dr.zeros(mi.SurfaceInteraction3f)
        # it_query.uv = mi.Point2f(0.5, 0.5)
        # val = self.m_irradiance.eval(it_query, True)
        size = self.m_irradiance.resolution()
        self.m_x_fov = float(mi.parse_fov(props, size[0] / float(size[1])))
        self.parameters_changed()
        self.m_flags = mi.EmitterFlags.DeltaPosition

    def traverse(self, callback):
        super().traverse(callback)
        callback.put_parameter(
            "scale", self.m_intensity_scale, mi.ParamFlags.Differentiable
        )
        callback.put_object(
            "irradiance", self.m_irradiance, mi.ParamFlags.Differentiable
        )
        callback.put_parameter(
            "to_world", self.m_to_world, mi.ParamFlags.NonDifferentiable
        )

    def parameters_changed(self, keys=[]):
        if not keys or "irradiance" in keys:
            size = self.m_irradiance.resolution()
            self.m_camera_to_sample = mi.perspective_projection(
                size, size, 0, self.m_x_fov, 1e-4, 1e4
            )
            self.m_sample_to_camera = self.m_camera_to_sample.inverse()

            pmin = self.m_sample_to_camera @ mi.Point3f(0.0, 0.0, 0.0)
            pmax = self.m_sample_to_camera @ mi.Point3f(1.0, 1.0, 0.0)
            image_rect = mi.BoundingBox2f(mi.Point2f(pmin.x / pmin.z, pmin.y / pmin.z))
            image_rect.expand(mi.Point2f(pmax.x / pmax.z, pmax.y / pmax.z))
            self.m_sensor_area = image_rect.volume()

            dr.make_opaque(
                self.m_camera_to_sample,
                self.m_sample_to_camera,
                self.m_intensity_scale,
                self.m_sensor_area,
            )

        dr.make_opaque(self.m_intensity_scale)
        super().parameters_changed(keys)

    def sample_ray(
        self, time, wavelength_sample, spatial_sample, direction_sample, active=True
    ):
        # never actually called
        # 1. Sample position on film
        uv, pdf = self.m_irradiance.sample_position(direction_sample, active)

        # 2. Sample spectrum (weight includes irradiance eval)
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.t = 0.0
        si.time = time
        si.p = self.m_to_world.translation()
        si.uv = uv
        wavelengths, weight = self.sample_wavelengths(si, wavelength_sample, active)

        # 4. Compute the sample position on the near plane (local camera space).
        near_p = self.m_sample_to_camera @ mi.Point3f(uv[0], uv[1], 0.0)
        near_dir = dr.normalize(near_p)

        # 5. Generate transformed ray
        ray = mi.Ray3f()
        ray.time = time
        ray.wavelengths = wavelengths
        ray.o = si.p
        ray.d = self.m_to_world @ near_dir

        # Scaling factor to match sample_direction
        weight *= dr.pi * self.m_sensor_area

        return ray, mi.depolarizer(weight / pdf) & active

    def sample_direction(self, it, sample, active=True):
        # 1. Transform the reference point into the local coordinate system
        it_local = self.m_to_world.inverse().transform_affine(it.p)

        # 2. Map to UV coordinates
        uv = self.m_camera_to_sample @ it_local
        uv = mi.Point2f(uv.x, uv.y)
        active &= dr.all(uv >= 0) & dr.all(uv <= 1) & (it_local.z > 0)

        # 3. Query texture
        it_query = dr.zeros(mi.SurfaceInteraction3f)
        it_query.wavelengths = it.wavelengths
        it_query.uv = uv
        spec = self.m_irradiance.eval(it_query, active)

        # 4. Prepare DirectionSample record
        ds = mi.DirectionSample3f()
        ds.p = self.m_to_world.translation()
        tmp = mi.Transform4d().translate(-self.m_to_world.translation())
        tmp2 = tmp @ self.m_to_world
        ds.n = tmp2 @ mi.Vector3f(0, 0, 1)
        # look_dir = np.array(self.m_to_world.matrix)[:3, 2]
        # ds.n = mi.Vector3f(look_dir[0], look_dir[1], look_dir[2])
        # ds.n = self.m_to_world @ mi.Vector3f(0, 0, 1)
        ds.uv = uv
        ds.time = it.time
        ds.pdf = 1.0
        ds.delta = True
        # print(ds.n)
        # print(ds.d)
        # print(dr.dot(ds.n, ds.d))
        # breakpoint()
        if hasattr(mi, "EmitterPtr"):
            ds.emitter = mi.EmitterPtr(self)
        else:
            ds.emitter = self
        ds.d = ds.p - it.p
        dist_squared = dr.squared_norm(ds.d)
        ds.dist = dr.sqrt(dist_squared)
        ds.d *= dr.rcp(ds.dist)

        # Scale so that irradiance at z=1 is correct
        spec *= (
            dr.pi
            * self.m_intensity_scale
            / (dr.square(it_local.z) * -dr.dot(ds.n, ds.d))
        )

        return ds, mi.depolarizer(spec & active)

    def sample_position(self, time, sample, active=True):
        # never actually called
        center_dir = self.m_to_world @ mi.Vector3f(0.0, 0.0, 1.0)
        ps = mi.PositionSample3f(
            self.m_to_world.translation(), center_dir, mi.Point2f(0.5), time, 1.0, True
        )
        return ps, 1.0

    def sample_wavelengths(self, si, sample, active=True):
        # never actually called
        wav, weight = self.m_irradiance.sample_spectrum(
            si, mi.math.sample_shifted(sample), active
        )
        return wav, weight * self.m_intensity_scale

    def pdf_direction(self, it, ds, active=True):
        # called by the integrator to compute the PDF of a direction sample
        return 0.0

    def eval_direction(self, it, ds, active=True):
        # never actually called?
        it_local = self.m_to_world.inverse().transform_affine(it.p)

        it_query = dr.zeros(mi.SurfaceInteraction3f)
        it_query.wavelengths = it.wavelengths
        it_query.uv = ds.uv
        spec = self.m_irradiance.eval(it_query, active)

        spec *= (
            dr.pi
            * self.m_intensity_scale
            / (dr.square(it_local.z) * -dr.dot(ds.n, ds.d))
        )
        return mi.depolarizer(spec) & active

    def eval(self, si, active=True):
        # This method is called by the integrator to evaluate the emitter's contribution
        return 0.0

    def bbox(self):
        # This emitter does not occupy any particular region of space
        # never actually called?
        return mi.ScalarBoundingBox3f()

    def to_string(self):
        return (
            f"Projector[\n"
            f"  x_fov = {self.m_x_fov},\n"
            f"  irradiance = {self.m_irradiance},\n"
            f"  intensity_scale = {self.m_intensity_scale},\n"
            f"  to_world = {self.m_to_world}\n"
            f"]"
        )
