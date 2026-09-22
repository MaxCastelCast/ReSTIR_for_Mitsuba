"""Small Mitsuba scene viewer using NanoGUI interop."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import drjit as dr
import mitsuba as mi
import nanogui as ng
import numpy as np
from nanogui.interop import FrameStream, drjit_sink

import temp_reuse
import ris_di

VARIANTS = ("cuda_ad_rgb", "metal_ad_rgb", "llvm_ad_rgb")

# Upper end of the samples/frame setting. Past this the producer thread
# stops keeping up with camera input, which is the point of the viewer.
MAX_SPP = 64


@dataclass(frozen=True)
class IntegratorChoice:
    label: str
    plugin: str | None


INTEGRATORS = (
    IntegratorChoice("Scene integrator", None),
    IntegratorChoice("Path tracer", "path"),
    IntegratorChoice("RIS DI", "direct_ris"),
    IntegratorChoice("Temporal Reuse", "temp_reuse"),
)


@dataclass(frozen=True)
class IntegratorParam:
    """A plugin property together with the default it falls back to."""

    name: str
    default: bool | int | float | None


# C++ plugins cannot be introspected; list their properties explicitly.
BUILTIN_PARAMS: dict[str, tuple[IntegratorParam, ...]] = {
    "path": (IntegratorParam("max_depth", -1), IntegratorParam("rr_depth", 5)),
}


class _RecordingProperties(mi.Properties):
    """Property set that records every ``get`` with its default value."""

    def __init__(self) -> None:
        super().__init__()
        self.recorded: list[IntegratorParam] = []

    def get(self, name: str, default=None):
        self.recorded.append(IntegratorParam(name, default))
        return default


_PARAM_CACHE: dict[str, tuple[IntegratorParam, ...]] = {}
_CLASS_CACHE: dict[str, type] = {}


def plugin_class(plugin: str) -> type:
    """The class registered for ``plugin``, instantiated once to find it."""
    if plugin not in _CLASS_CACHE:
        _CLASS_CACHE[plugin] = type(mi.load_dict({"type": plugin}))
    return _CLASS_CACHE[plugin]


def integrator_params(plugin: str | None) -> tuple[IntegratorParam, ...]:
    """Return the properties accepted by ``plugin`` with their defaults.

    Python plugins are discovered by constructing them with a recording
    property set; C++ plugins come from :data:`BUILTIN_PARAMS`.
    """
    if plugin is None:
        return ()
    if plugin in BUILTIN_PARAMS:
        return BUILTIN_PARAMS[plugin]
    if plugin not in _PARAM_CACHE:
        cls = plugin_class(plugin)
        props = _RecordingProperties()
        try:
            cls(props)
        except Exception:
            traceback.print_exc()
        seen: dict[str, IntegratorParam] = {}
        for param in props.recorded:
            seen.setdefault(param.name, param)
        _PARAM_CACHE[plugin] = tuple(seen.values())
    return _PARAM_CACHE[plugin]


def integrator_dict(plugin: str, overrides: dict[str, object]) -> dict[str, object]:
    """Build the ``load_dict`` description of ``plugin`` with ``overrides``."""
    description: dict[str, object] = {"type": plugin}
    for param in integrator_params(plugin):
        value = overrides.get(param.name, param.default)
        if value is not None:
            description[param.name] = value
    return description


@dataclass(frozen=True)
class FilterChoice:
    label: str
    plugin: str | None


FILTERS = (
    FilterChoice("Scene filter", None),
    FilterChoice("Box", "box"),
    FilterChoice("Gaussian", "gaussian"),
    FilterChoice("Tent", "tent"),
    FilterChoice("Lanczos", "lanczos"),
    FilterChoice("Mitchell", "mitchell"),
    FilterChoice("Catmull–Rom", "catmullrom"),
)

# OptiX requires albedo whenever normal guidance is enabled.
OPTIX_GUIDES = (
    ("Color only", False, False),
    ("Albedo", True, False),
    ("Albedo + normals", True, True),
)

CONFIG_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    / "mitsuba-viewer"
    / "settings.json"
)


def optix_denoiser_available() -> bool:
    """Return whether the active Mitsuba variant can use OptiX denoising."""
    return mi.variant().startswith("cuda_") and bool(mi.MI_ENABLE_CUDA)


def jit_variants() -> list[str]:
    """Return Mitsuba JIT variants usable by the viewer."""
    return [
        variant
        for variant in mi.variants()
        if variant.startswith(("cuda_", "metal_", "llvm_"))
    ]


@dataclass
class Settings:
    """Persistent settings stored as JSON at :data:`CONFIG_PATH`."""

    invert_zoom: bool = False
    variant: str = ""
    spp: int = 1
    integrator_props: dict[str, dict[str, object]] = field(default_factory=dict)

    @staticmethod
    def load() -> "Settings":
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except (OSError, ValueError):
            return Settings()
        names = {field.name for field in fields(Settings)}
        return Settings(**{key: value for key, value in data.items() if key in names})

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2) + "\n")


HELP = (
    (
        "Orbit navigation",
        (
            ("Left drag", "Orbit around the pivot"),
            ("Middle/right drag", "Pan"),
            ("Scroll", "Zoom"),
            ("Click", "Select the shape under the cursor"),
            ("Double click", "Select and frame the shape"),
        ),
    ),
    (
        "First person navigation",
        (
            ("W/A/S/D or arrows", "Move (switches to this mode)"),
            ("Q/E", "Move down/up"),
            ("Left drag", "Look around"),
            ("Scroll", "Change the movement speed"),
            ("Shift (hold)", "Move faster"),
        ),
    ),
    (
        "Interface",
        (
            ("Drop a scene (.xml)", "Load it"),
            ("Tab", "Toggle the performance panel"),
            ("I", "Toggle the integrator settings"),
            ("H", "Toggle this help"),
            ("Escape", "Quit"),
        ),
    ),
)


def film_rays(sensor: mi.Sensor, width: int, height: int) -> mi.Ray3f:
    """Return rays through a pixel-center grid covering the sensor film."""
    index = dr.arange(mi.UInt, width * height)
    position = mi.Point2f(
        (mi.Float(index % width) + 0.5) / width,
        (mi.Float(index // width) + 0.5) / height,
    )
    return sensor.sample_ray(0.0, 0.0, position, mi.Point2f(0.5))[0]


@dr.freeze(state_fn=lambda _, sensor: tuple(int(v) for v in sensor.film().size()))
def probe_image(scene: mi.Scene, sensor: mi.Sensor) -> tuple[mi.TensorXf, mi.TensorXu]:
    """Return eye-space depth and hit attribution at each pixel center."""
    width, height = sensor.film().size()
    ray = film_rays(sensor, width, height)
    preliminary = scene.ray_intersect_preliminary(ray, coherent=True)
    to_world = sensor.world_transform()
    forward = to_world @ mi.Vector3f(0, 0, 1)
    origin_offset = ray.o - to_world @ mi.Point3f(0)
    depth = dr.select(
        preliminary.is_valid(),
        dr.dot(origin_offset + ray.d * preliminary.t, forward),
        dr.inf,
    )
    instance_index = (
        preliminary.instance_index
        if hasattr(preliminary, "instance_index")
        else dr.reinterpret_array(mi.UInt, preliminary.instance)
    )
    record = mi.Vector4u(
        dr.reinterpret_array(mi.UInt, depth),
        dr.reinterpret_array(mi.UInt, preliminary.shape),
        preliminary.prim_index,
        instance_index,
    )
    return (
        mi.TensorXf(depth, (height, width)),
        mi.TensorXu(dr.ravel(record), (height, width, 4)),
    )


@dr.freeze(state_fn=lambda sensor, *_: tuple(int(v) for v in sensor.film().size()))
def flow_image(
    sensor: mi.Sensor, prev_sensor: mi.Sensor, depth: mi.TensorXf
) -> mi.TensorXf:
    """Return per-pixel motion vectors for the OptiX temporal denoiser.

    Each pixel center of ``sensor`` is lifted to its surface point using the
    eye-space ``depth`` image and reprojected through ``prev_sensor``. OptiX
    expects the movement from the previous frame to the current one, so the
    previous position is the current pixel minus the flow. Pixels without a
    surface or outside the previous view get zero motion.
    """
    width, height = sensor.film().size()
    ray = film_rays(sensor, width, height)
    forward = sensor.world_transform() @ mi.Vector3f(0, 0, 1)
    z = mi.Float(depth.array)
    t = z / dr.dot(ray.d, forward)
    si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
    si.p = ray.o + ray.d * t
    ds, _ = prev_sensor.sample_direction(si, mi.Point2f(0.0))
    index = dr.arange(mi.UInt, width * height)
    current = mi.Point2f(mi.Float(index % width) + 0.5, mi.Float(index // width) + 0.5)
    valid = dr.isfinite(z) & (ds.pdf > 0)
    flow = dr.select(valid, current - mi.Point2f(ds.uv), mi.Vector2f(0))
    return mi.TensorXf(dr.ravel(flow), (height, width, 2))


def copy_pose(dst: mi.Sensor, src: mi.Sensor) -> None:
    """Copy the camera pose without the device readback of a parameter update."""
    if hasattr(dst, "set_world_transform_scalar"):
        dst.set_world_transform_scalar(src.world_transform_scalar())
    else:
        params = mi.traverse(dst)
        params["to_world"] = mi.traverse(src)["to_world"]
        params.update()


@dataclass
class PickCache:
    """CPU-readable hit attribution for one displayed camera state."""

    cam: ng.CameraState
    data: np.ndarray

    def lookup(self, u: float, v: float) -> tuple[int, int, int, int]:
        height, width, _ = self.data.shape
        y = min(max(int(v * height), 0), height - 1)
        x = min(max(int(u * width), 0), width - 1)
        data = self.data
        return data[y, x, 0], data[y, x, 1], data[y, x, 2], data[y, x, 3]


def replace_rgb(image: mi.TensorXf, rgb: mi.TensorXf) -> mi.TensorXf:
    """Return ``image`` with its RGB channels replaced and alpha preserved."""
    height, width, _ = image.shape
    index = dr.arange(mi.UInt, height * width * 4)
    pixel = index // 4
    channel = index % 4
    value = dr.select(
        channel == 3,
        dr.gather(mi.Float, image.array, index),
        dr.gather(mi.Float, rgb.array, pixel * 3 + dr.minimum(channel, 2)),
    )
    return mi.TensorXf(value, image.shape)


@dr.freeze(
    warn_after=100,
    state_fn=lambda _, sensor, integrator, *__, spp=1: (
        tuple(int(v) for v in sensor.film().size()),
        id(integrator),
        spp,
    ),
)
def render_frame(
    scene: mi.Scene,
    sensor: mi.Sensor,
    integrator: mi.Integrator,
    accum: mi.TensorXf,
    depth: mi.TensorXf,
    seed: mi.UInt,
    weight: mi.Float,
    spp: int = 1,
) -> tuple[mi.TensorXf, mi.TensorXf | None, mi.TensorXf | None]:
    """Accumulate RGB, merge depth, and return optional denoising guides.

    ``spp`` sizes the sample array the integrator traces, so it is part of the
    frozen kernel rather than an input to it: changing it records a new one.
    """
    image = integrator.render(scene, sensor, seed=seed, spp=spp)
    height, width, _ = accum.shape
    image_channels = image.shape[-1]
    index = dr.arange(mi.UInt, height * width * 4)
    pixel = index // 4
    channel = index % 4
    source_channel = dr.minimum(channel, min(image_channels, 3) - 1)
    source = dr.gather(mi.Float, image.array, pixel * image_channels + source_channel)
    previous = dr.gather(mi.Float, accum.array, index)
    value = dr.select(
        channel == 3,
        dr.gather(mi.Float, depth.array, pixel),
        dr.lerp(previous, source, weight),
    )
    output = mi.TensorXf(value, (height, width, 4))
    if image_channels >= 9:
        return output, image[..., 3:6], image[..., 6:9]
    return output, None, None


class MitsubaRenderer:
    """Progressive renderer owned by the NanoGUI producer thread."""

    def __init__(
        self,
        source: Path | dict,
        integrator: str | None = None,
        rfilter: str | None = None,
        integrator_props: dict[str, object] | None = None,
    ) -> None:
        self.source = source
        self.integrator_override = integrator
        self.rfilter_override = rfilter
        self.integrator_props = dict(integrator_props or {})
        self._load_scene()
        self.native_size = ng.Vector2i(self.sensor.film().size())
        self.seed = 0
        self.samples = 0
        self.denoise_enabled = False
        self.denoise_albedo = True
        self.denoise_normals = True
        self.denoise_temporal = False
        # Running mean over frames, or only the latest frame.
        self.accumulate = True
        # Samples per displayed frame.
        self.spp = 1
        self.denoiser = None
        self.denoiser_config: tuple[bool, bool, bool] | None = None
        self.previous_denoised: mi.TensorXf | None = None
        self.flow: mi.TensorXf | None = None
        self.zero_flow: mi.TensorXf | None = None
        self.prev_pose: mi.Sensor | None = None
        self.size = ng.Vector2i(0, 0)
        self.pick_cache: PickCache | None = None
        self.pending_probe = None
        self.projection: ng.Matrix4f | None = None
        self.shape_by_index: dict[int, mi.Shape] = {}
        self.initial = self.initial_state(mi.ScalarPoint3f(self.scene.bbox().center()))

    @staticmethod
    def _sensor_with_filter(sensor: mi.Sensor, rfilter: str) -> mi.Sensor:
        """Rebuild a standard camera when Mitsuba lacks ``Sensor.set_film``."""
        old_film = sensor.film()
        size = old_film.size()
        film = {
            "type": "hdrfilm",
            "width": int(size[0]),
            "height": int(size[1]),
            "rfilter": {"type": rfilter},
            "sample_border": old_film.sample_border(),
        }
        class_name = sensor.class_name()
        sensor_types = {
            "PerspectiveCamera": "perspective",
            "OrthographicCamera": "orthographic",
            "ThinLensCamera": "thinlens",
        }
        if class_name not in sensor_types:
            raise RuntimeError(
                f"filter overrides require Sensor.set_film() for {class_name}"
            )

        params = mi.traverse(sensor)
        description: dict[str, object] = {
            "type": sensor_types[class_name],
            "to_world": mi.ScalarTransform4f(
                dr.slice(sensor.world_transform().matrix, 0)
            ),
            "near_clip": sensor.near_clip(),
            "far_clip": sensor.far_clip(),
            "shutter_open": sensor.shutter_open(),
            "shutter_close": sensor.shutter_open() + sensor.shutter_open_time(),
            "sampler": sensor.sampler(),
            "film": film,
        }
        if "x_fov" in params:
            description["fov"] = float(dr.slice(params["x_fov"], 0))
            description["fov_axis"] = "x"
            description["principal_point_offset_x"] = float(
                dr.slice(params["principal_point_offset_x"], 0)
            )
            description["principal_point_offset_y"] = float(
                dr.slice(params["principal_point_offset_y"], 0)
            )
        if class_name == "ThinLensCamera":
            description["aperture_radius"] = sensor.aperture_radius()
            description["focus_distance"] = sensor.focus_distance()
        return mi.load_dict(description)

    def _load_scene(self, to_world: mi.ScalarTransform4f | None = None) -> None:
        """Load a fresh scene, optionally preserving the current camera pose."""
        self.scene = (
            mi.load_file(str(self.source))
            if isinstance(self.source, Path)
            else mi.load_dict(self.source)
        )
        self.sensor = self.scene.sensors()[0]
        if self.rfilter_override is not None:
            old_film = self.sensor.film()
            size = old_film.size()
            film = mi.load_dict(
                {
                    "type": "hdrfilm",
                    "width": int(size[0]),
                    "height": int(size[1]),
                    "rfilter": {"type": self.rfilter_override},
                    "sample_border": old_film.sample_border(),
                }
            )
            if hasattr(self.sensor, "set_film"):
                self.sensor.set_film(film)
            else:
                self.sensor = self._sensor_with_filter(
                    self.sensor, self.rfilter_override
                )
        self.sensor_params = mi.traverse(self.sensor)
        if to_world is not None:
            self.sensor_params["to_world"] = to_world
            self.sensor_params.update()
        self._build_integrator()
        self.shapes = self.scene.shapes()
        self.scene.sample_emitter_direction(
            dr.zeros(mi.SurfaceInteraction3f), mi.Point2f(0.5), False
        )
        if hasattr(self, "shape_by_index"):
            self.shape_by_index.clear()

    def _build_integrator(self) -> None:
        """Instantiate the active integrator from the override and properties."""
        self.integrator = (
            self.scene.integrator()
            if self.integrator_override is None
            else mi.load_dict(
                integrator_dict(self.integrator_override, self.integrator_props)
            )
        )
        self.aov_integrator = mi.load_dict(
            {
                "type": "aov",
                "aovs": "albedo:albedo,sh_normal:sh_normal",
                "image": self.integrator,
            }
        )

    def set_integrator_props(self, props: dict[str, object]) -> None:
        """Rebuild the integrator with new properties (producer thread).

        Nothing changes when the plugin rejects the properties.
        """
        previous = self.integrator_props
        self.integrator_props = dict(props)
        try:
            self._build_integrator()
        except Exception:
            self.integrator_props = previous
            raise
        self._working_props = previous
        render_frame.clear()

    def revert_integrator_props(self) -> bool:
        """Return to the properties that last rendered successfully.

        Returns ``False`` when there is nothing to fall back to.
        """
        previous = getattr(self, "_working_props", None)
        if previous is None or previous == self.integrator_props:
            return False
        self.set_integrator_props(previous)
        self._working_props = None
        return True

    def update_projection(self) -> None:
        film = self.sensor.film()
        projection_args = (
            film.size(),
            film.crop_size(),
            film.crop_offset(),
        )
        if "x_fov" in self.sensor_params:
            camera_to_sample = mi.perspective_projection(
                *projection_args,
                self.sensor_params["x_fov"],
                self.sensor.near_clip(),
                self.sensor.far_clip(),
            )
        else:
            camera_to_sample = mi.orthographic_projection(
                *projection_args,
                self.sensor.near_clip(),
                self.sensor.far_clip(),
            )
        self.projection = ng.Matrix4f.from_mitsuba_projection(
            ng.Matrix4f(dr.slice(camera_to_sample.matrix, 0))
        )
        self.projection_stale = False

    def initial_state(self, pivot: mi.ScalarPoint3f) -> ng.CameraState:
        to_world = mi.ScalarTransform4f(
            dr.slice(self.sensor.world_transform().matrix, 0)
        )
        eye = to_world @ mi.ScalarPoint3f(0, 0, 0)
        forward = dr.normalize(to_world @ mi.ScalarVector3f(0, 0, 1))
        up = dr.normalize(to_world @ mi.ScalarVector3f(0, 1, 0))
        distance = dr.dot(pivot - eye, forward)
        if not distance > 1e-4:
            distance = max(dr.norm(pivot - eye), 1.0)
        return ng.CameraState(eye, eye + forward * distance, up)

    def refine_pivot(self) -> bool:
        pivot = self.pick_pivot()
        if pivot is None:
            return False
        self.initial.target = self.initial_state(pivot).target
        return True

    def pick_pivot(self) -> mi.ScalarPoint3f | None:
        resolution = 128
        interaction = self.scene.ray_intersect(
            film_rays(self.sensor, resolution, resolution)
        )
        valid = interaction.is_valid()
        dr.schedule(valid, interaction.t, interaction.p)
        indices = dr.compress(valid)
        if dr.width(indices) == 0:
            return None
        points = dr.gather(mi.Point3f, interaction.p, indices)
        return mi.ScalarPoint3f(dr.slice(dr.mean(points), 0))

    def can_pick(self, cam: ng.CameraState) -> bool:
        cache = self.pick_cache
        return cache is not None and cache.cam is cam

    def pick(
        self, cam: ng.CameraState, u: float, v: float
    ) -> tuple[str, mi.ScalarBoundingBox3f] | None:
        _, shape_index, primitive_index, instance_index = self.pick_cache.lookup(u, v)
        if shape_index == 0:
            return None
        return self.describe_hit(
            self.shape(shape_index), primitive_index, instance_index
        )

    def describe_hit(
        self, shape: mi.Shape, primitive_index: int, instance_index: int
    ) -> tuple[str, mi.ScalarBoundingBox3f]:
        del primitive_index
        if instance_index == 0:
            return self.shape_name(shape), shape.bbox()
        instance = self.shape(instance_index)
        return self.shape_name(shape, instance), instance.bbox()

    def shape(self, index: int) -> mi.Shape:
        shape = self.shape_by_index.get(index)
        if shape is None:
            shape = dr.reinterpret_array(mi.ShapePtr, mi.UInt32(index))[0]
            self.shape_by_index[index] = shape
        return shape

    def depth_at(self, cam: ng.CameraState, u: float, v: float) -> float:
        cache = self.pick_cache
        if cache is None or cache.cam is not cam:
            return 0.0
        (depth,) = struct.unpack("<f", struct.pack("<I", cache.lookup(u, v)[0]))
        return 0.0 if math.isinf(depth) else depth

    def shape_name(self, shape: mi.Shape, instance: mi.Shape | None = None) -> str:
        if shape.id():
            return shape.id()
        try:
            index = self.shapes.index(shape if instance is None else instance)
        except ValueError:
            return shape.class_name()
        return f"{shape.class_name()} #{index}"

    def configure(self, size: ng.Vector2i) -> None:
        if size == self.size:
            return

        self.size = ng.Vector2i(size)
        film_size = mi.ScalarVector2u(*size)
        self.sensor_params["film.size"] = film_size
        self.sensor_params["film.crop_size"] = film_size
        self.sensor_params["film.crop_offset"] = mi.ScalarPoint2u(0)
        self.sensor_params.update()
        self.reset_denoiser()
        self.prev_pose = None
        self.zero_flow = dr.zeros(mi.TensorXf, (size[1], size[0], 2))
        self.projection_stale = True
        self.pick_cache = None
        self.pending_probe = None
        width, height = size[0], size[1]
        self.accum = dr.opaque(mi.TensorXf, 0.0, (height, width, 4))
        self.depth = dr.opaque(mi.TensorXf, 0.0, (height, width))

    def render(self, cam: ng.CameraState, restart: bool) -> mi.TensorXf:
        """Render one frame, restarting the running mean on camera changes."""
        if restart:
            self.samples = 0
            if self.denoise_temporal:
                self._remember_pose()
            to_world = mi.ScalarTransform4f.look_at(
                origin=list(cam.origin), target=list(cam.target), up=list(cam.up)
            )
            if hasattr(self.sensor, "set_world_transform_scalar"):
                self.sensor.set_world_transform_scalar(to_world)
            else:
                self.sensor_params["to_world"] = mi.Transform4f(to_world)
                self.sensor_params.update()
            self.depth, probe = probe_image(self.scene, self.sensor)
            self.pending_probe = (cam, probe)
            if self.denoise_temporal and self.prev_pose is not None:
                self.flow = flow_image(self.sensor, self.prev_pose, self.depth)
        elif self.pending_probe is not None:
            camera, probe = self.pending_probe
            self.pending_probe = None
            self.pick_cache = PickCache(camera, probe.numpy())

        self.samples += 1
        self.seed += 1
        # Snapshot the denoiser switches: the UI thread may flip them while
        # this frame is in flight, and the guide AOVs must match the call.
        denoise = self.denoise_enabled
        config = (self.denoise_albedo, self.denoise_normals, self.denoise_temporal)
        use_albedo, use_normals, temporal_mode = config
        render_integrator = (
            self.aov_integrator if denoise and use_albedo else self.integrator
        )
        self.accum, albedo, normals = render_frame(
            self.scene,
            self.sensor,
            render_integrator,
            self.accum,
            self.depth,
            dr.opaque(mi.UInt, self.seed),
            dr.opaque(mi.Float, self.frame_weight()),
            max(int(self.spp), 1),
        )
        if not denoise:
            return self.accum
        if self.denoiser is None or self.denoiser_config != config:
            self.denoiser = mi.OptixDenoiser(
                mi.ScalarVector2u(self.size[0], self.size[1]),
                albedo=use_albedo,
                normals=use_normals,
                temporal=temporal_mode,
            )
            self.denoiser_config = config
            self.previous_denoised = None
        noisy = self.accum[..., :3]
        temporal: dict[str, mi.TensorXf] = {}
        if temporal_mode:
            # OptiX recommends the noisy input as the "previous" frame when
            # there is none yet; the motion vectors are then meaningless too.
            previous = self.previous_denoised
            flow = self.flow if previous is not None else None
            temporal = {
                "flow": self.zero_flow if flow is None else flow,
                "previous_denoised": noisy if previous is None else previous,
            }
            self.flow = None
        if use_normals:
            denoised = self.denoiser(
                noisy,
                albedo,
                normals,
                self.sensor.world_transform().inverse(),
                **temporal,
            )
        elif use_albedo:
            denoised = self.denoiser(noisy, albedo, **temporal)
        else:
            denoised = self.denoiser(noisy, **temporal)
        if temporal_mode:
            self.previous_denoised = denoised
        return replace_rgb(self.accum, denoised)

    def frame_weight(self) -> float:
        """Blend weight of the new frame into the displayed image."""
        return 1.0 / self.samples if self.accumulate else 1.0

    def reset_denoiser(self) -> None:
        """Drop the denoiser and its temporal history (rebuilt lazily)."""
        self.denoiser = None
        self.previous_denoised = None
        self.flow = None

    def _remember_pose(self) -> None:
        """Keep a sensor at the pose that is about to be replaced."""
        if self.prev_pose is None:
            if "x_fov" not in self.sensor_params:
                return  # only perspective-like sensors are reprojected
            self.prev_pose = mi.load_dict(
                {
                    "type": "perspective",
                    "film": {
                        "type": "hdrfilm",
                        "width": int(self.size[0]),
                        "height": int(self.size[1]),
                    },
                }
            )
            params = mi.traverse(self.prev_pose)
            for key in params.keys():
                if key in self.sensor_params:
                    params[key] = self.sensor_params[key]
            params.update()
        copy_pose(self.prev_pose, self.sensor)


class Viewer(ng.Screen):
    """Interactive scene viewer backed by a progressive producer thread."""

    def __init__(
        self,
        scene: Path | dict | None,
        size: ng.Vector2i,
        settings: Settings | None = None,
        integrator: str | None = None,
        rfilter: str | None = None,
    ) -> None:
        super().__init__(
            size=size,
            caption=f"Mitsuba Viewer [{mi.variant()}]",
            float_buffer=True,
        )
        self.settings = settings if settings is not None else Settings.load()
        plugins = [choice.plugin for choice in INTEGRATORS]
        if integrator not in plugins:
            raise ValueError(f"unknown integrator: {integrator}")
        self.integrator_override = integrator
        filters = [choice.plugin for choice in FILTERS]
        if rfilter not in filters:
            raise ValueError(f"unknown reconstruction filter: {rfilter}")
        self.rfilter_override = rfilter
        self.denoise_enabled = False
        self.denoise_albedo = True
        self.denoise_normals = True
        self.denoise_temporal = False
        self.accumulate = True
        background = ng.Color(0.04, 0.045, 0.055, 1.0)
        self.set_background(background)

        self.render_pass = ng.RenderPass(color_targets=[self], depth_target=self)
        self.render_pass.set_clear_color(0, background)
        self.render_pass.set_depth_test(ng.RenderPass.DepthTest.LessEqual, True)

        self.quad = ng.TexturedQuad(self.render_pass)
        self.quad.set_linear(True)
        self.quad.set_depth_from_alpha(True)
        self.box = ng.LineRenderer(self.render_pass)
        self.box.set_color(ng.Color(1.0, 0.8, 0.2, 1.0))

        self.selection: tuple[str, mi.ScalarBoundingBox3f] | None = None
        self.pending_pick: tuple[float, float, bool, float] | None = None
        self.notice: tuple[str, float] | None = None
        self.displayed_cam: ng.CameraState | None = None
        self.renderer: MitsubaRenderer | None = None
        self.controller = ng.CameraController(ng.CameraState(), ng.Vector3f(0, 1, 0))
        self.controller.set_click_callback(self.click_event)
        self.controller.set_depth_callback(self.depth_event)
        self.zoom_step = self.controller.zoom_step
        self.apply_settings()
        self.phase: str | None = "Drag a Mitsuba scene (.xml) onto this window"
        self.stream: FrameStream | None = None
        self.adopt_size = False
        self.camera_moved = False

        self.window = window = ng.Window(self, "Viewer")
        window.set_position(ng.Vector2i(15, 15))
        window.set_layout(ng.GroupLayout())
        performance = ng.Widget(window)
        performance.set_layout(
            ng.GridLayout(ng.Orientation.Horizontal, 2, ng.Alignment.Fill, 0, 6)
        )
        self.stats = []
        for name in ("Render", "Display", "Resolution"):
            ng.Label(performance, name, "sans-bold")
            value = ng.Label(performance, "-")
            value.set_fixed_width(210)
            self.stats.append(value)

        integrator_row = ng.Widget(window)
        integrator_row.set_layout(
            ng.BoxLayout(ng.Orientation.Horizontal, ng.Alignment.Middle, 0, 8)
        )
        ng.Label(integrator_row, "Integrator", "sans-bold")
        self.integrator_selector = ng.ComboBox(
            integrator_row, [choice.label for choice in INTEGRATORS]
        )
        self.integrator_selector.set_fixed_width(230)
        self.integrator_selector.set_selected_index(plugins.index(integrator))
        self.integrator_selector.set_callback(self.select_integrator)
        self.accumulate_toggle = ng.CheckBox(
            integrator_row, "Average", self.set_accumulate
        )
        self.accumulate_toggle.set_tooltip(
            "Show the running mean of all frames instead of the latest one"
        )
        self.accumulate_toggle.set_checked(self.accumulate)
        ng.Label(integrator_row, "spp")
        self.spp_box = ng.IntBox(integrator_row, self.current_spp())
        self.spp_box.set_tooltip("Samples traced per displayed frame")
        self.spp_box.set_editable(True)
        self.spp_box.set_spinnable(True)
        self.spp_box.set_min_value(1)
        self.spp_box.set_max_value(MAX_SPP)
        self.spp_box.set_fixed_width(60)
        self.spp_box.set_callback(self.set_spp)

        filter_row = ng.Widget(window)
        filter_row.set_layout(
            ng.BoxLayout(ng.Orientation.Horizontal, ng.Alignment.Middle, 0, 8)
        )
        ng.Label(filter_row, "Filter", "sans-bold")
        self.filter_selector = ng.ComboBox(
            filter_row, [choice.label for choice in FILTERS]
        )
        self.filter_selector.set_fixed_width(230)
        self.filter_selector.set_selected_index(filters.index(rfilter))
        self.filter_selector.set_callback(self.select_filter)

        row = ng.Widget(window)
        row.set_layout(
            ng.BoxLayout(ng.Orientation.Horizontal, ng.Alignment.Middle, 0, 8)
        )
        ng.Label(row, "Exposure", "sans-bold")
        self.exposure_slider = slider = ng.Slider(row)
        slider.set_range((-5.0, 5.0))
        slider.set_value(0.0)
        slider.set_fixed_width(140)
        slider.set_callback(self.set_exposure)
        self.exposure_label = ng.Label(row, "EV +0.0")
        self.exposure_label.set_fixed_width(50)
        hdr = ng.Button(row, "HDR")
        hdr.set_flags(ng.Button.Flags.ToggleButton)
        hdr.set_pushed(True)
        hdr.set_font_size(14)
        hdr.set_padding(ng.Vector2i(12, 2))
        hdr.set_change_callback(self.quad.set_hdr)
        self.denoise_toggle = ng.CheckBox(row, "OptiX", self.set_denoise)
        self.denoise_toggle.set_enabled(optix_denoiser_available())
        self.denoise_temporal_toggle = ng.CheckBox(
            row, "Temporal", self.set_denoise_temporal
        )
        self.denoise_temporal_toggle.set_enabled(optix_denoiser_available())

        guides_row = ng.Widget(window)
        guides_row.set_layout(
            ng.BoxLayout(ng.Orientation.Horizontal, ng.Alignment.Middle, 0, 8)
        )
        ng.Label(guides_row, "OptiX guides", "sans-bold")
        self.denoise_guides = ng.ComboBox(
            guides_row, [label for label, _, _ in OPTIX_GUIDES]
        )
        self.denoise_guides.set_fixed_width(230)
        self.denoise_guides.set_selected_index(2)
        self.denoise_guides.set_callback(self.set_denoise_guides)
        self.denoise_guides.set_enabled(optix_denoiser_available())

        self.help: ng.Window | None = None
        self.settings_window: ng.Window | None = None
        self.integrator_window: ng.Window | None = None
        self.integrator_helper: ng.FormHelper | None = None
        self.integrator_props = self.settings.integrator_props
        self.pending_integrator_props: dict[str, object] | None = None
        self.integrator_panel_stale = False
        ng.Button(window.button_panel(), "", ng.icons.FA_SLIDERS_H).set_callback(
            self.toggle_integrator_settings
        )
        ng.Button(window.button_panel(), "", ng.icons.FA_COG).set_callback(
            self.toggle_settings
        )
        ng.Button(window.button_panel(), "", ng.icons.FA_QUESTION).set_callback(
            self.toggle_help
        )
        self.perform_layout()
        self.last_update = time.perf_counter()
        if scene is not None:
            self.load(scene)

    def load(
        self,
        scene: Path | dict,
        adopt_size: bool = False,
        label: str | None = None,
    ) -> None:
        if self.stream is not None:
            self.stream.close()
            render_frame.clear()
            probe_image.clear()
        self.scene_source = scene
        self.scene_label = label or (scene.name if isinstance(scene, Path) else "scene")
        self.renderer = None
        self.selection = None
        self.pending_pick = None
        self.displayed_cam = None
        self.adopt_size = adopt_size
        self.camera_moved = False
        self.phase = "Loading"
        self.stream = FrameStream(drjit_sink(mi.TensorXf), self.size())
        self.stream.start(self.producer)

    def set_accumulate(self, enabled: bool) -> None:
        """Display the running mean of all frames, or only the latest frame."""
        self.accumulate = bool(enabled)
        if self.renderer is not None:
            self.renderer.accumulate = self.accumulate
            self.stream.set_state(self.controller.state())

    def current_spp(self) -> int:
        """Samples every integrator traces per displayed frame."""
        return min(max(int(self.settings.spp), 1), MAX_SPP)

    def set_spp(self, value: int) -> None:
        """Set the samples the integrator traces per displayed frame.

        The change restarts the running mean: frames of different sample counts
        must not be averaged together with the equal weights
        :meth:`MitsubaRenderer.frame_weight` gives them.
        """
        spp = min(max(int(value), 1), MAX_SPP)
        if spp == self.current_spp():
            return
        self.settings.spp = spp
        self.settings.save()
        if self.renderer is not None:
            self.renderer.spp = spp
            self.stream.set_state(self.controller.state())

    def select_integrator(self, index: int) -> None:
        """Restart the active scene using the selected integrator."""
        self.integrator_override = INTEGRATORS[index].plugin
        self.pending_integrator_props = None
        if self.integrator_window is not None:
            self.toggle_integrator_settings()
            self.toggle_integrator_settings()
        if hasattr(self, "scene_source"):
            self.load(self.scene_source, label=self.scene_label)

    def current_integrator_props(self) -> dict[str, object]:
        """Return the stored property overrides of the active integrator."""
        plugin = self.integrator_override
        if plugin is None:
            return {}
        return self.integrator_props.setdefault(plugin, {})

    def integrator_props_changed(self) -> None:
        """Persist the overrides and rebuild the integrator on the next frame."""
        self.settings.save()
        self.pending_integrator_props = dict(self.current_integrator_props())

    def reject_integrator_props(self, renderer: MitsubaRenderer) -> None:
        """Restore the panel to the properties the renderer still uses."""
        overrides = self.current_integrator_props()
        overrides.clear()
        overrides.update(renderer.integrator_props)
        self.integrator_panel_stale = True
        self.notify("Integrator rejected the settings (see console)")

    def toggle_integrator_settings(self) -> None:
        if self.integrator_window is not None:
            self.integrator_window.dispose()
            self.integrator_window = None
            self.integrator_helper = None
            return

        plugin = self.integrator_override
        label = next(choice.label for choice in INTEGRATORS if choice.plugin == plugin)
        helper = ng.FormHelper(self)
        self.integrator_helper = helper
        self.integrator_window = window = helper.add_window(
            ng.Vector2i(0, 0), "Integrator settings"
        )
        ng.Button(window.button_panel(), "", ng.icons.FA_TIMES).set_callback(
            self.toggle_integrator_settings
        )
        helper.add_group(label)

        params = integrator_params(plugin)
        if not params:
            helper.add_widget("", ng.Label(window, "No adjustable properties"))
        overrides = self.current_integrator_props()

        def make_setter(param: IntegratorParam):
            def setter(value) -> None:
                # ``None`` defaults (unlimited) are entered as zero
                if param.default is None and value == 0:
                    overrides.pop(param.name, None)
                elif value == param.default:
                    overrides.pop(param.name, None)
                else:
                    overrides[param.name] = value
                self.integrator_props_changed()

            return setter

        def make_getter(param: IntegratorParam):
            def getter():
                value = overrides.get(param.name, param.default)
                return 0 if value is None else value

            return getter

        for param in params:
            caption = param.name.replace("_", " ")
            default = param.default
            if isinstance(default, bool):
                helper.add_bool_variable(
                    caption, make_setter(param), make_getter(param)
                )
            elif default is None or isinstance(default, int):
                if default is None:
                    caption += " (0 = unlimited)"
                box = helper.add_int_variable(
                    caption, make_setter(param), make_getter(param)
                )
                box.set_spinnable(True)
                box.set_fixed_width(110)
            elif isinstance(default, float):
                box = helper.add_double_variable(
                    caption, make_setter(param), make_getter(param)
                )
                box.set_spinnable(True)
                box.set_value_increment(max(abs(default) * 0.1, 1e-3))
                box.set_fixed_width(110)

        def reset() -> None:
            overrides.clear()
            self.integrator_props_changed()
            helper.refresh()

        if params:
            helper.add_button("Reset to defaults", reset)
        self.perform_layout()
        window.center()

    def select_filter(self, index: int) -> None:
        """Restart the active scene using the selected reconstruction filter."""
        self.rfilter_override = FILTERS[index].plugin
        if hasattr(self, "scene_source"):
            self.load(self.scene_source, label=self.scene_label)

    def set_denoise(self, enabled: bool) -> None:
        """Enable OptiX denoising when running a CUDA Mitsuba variant."""
        self.denoise_enabled = bool(enabled) and optix_denoiser_available()
        if self.renderer is not None:
            self.renderer.denoise_enabled = self.denoise_enabled

    def set_denoise_guides(self, index: int) -> None:
        """Choose the guide AOVs passed to the OptiX denoiser."""
        _, self.denoise_albedo, self.denoise_normals = OPTIX_GUIDES[index]
        if self.renderer is not None:
            self.renderer.denoise_albedo = self.denoise_albedo
            self.renderer.denoise_normals = self.denoise_normals
            self.renderer.reset_denoiser()

    def set_denoise_temporal(self, enabled: bool) -> None:
        """Feed motion vectors and the previous output to the OptiX denoiser."""
        self.denoise_temporal = bool(enabled) and optix_denoiser_available()
        if self.renderer is not None:
            self.renderer.denoise_temporal = self.denoise_temporal
            self.renderer.reset_denoiser()

    def camera_changed(self, state: ng.CameraState) -> None:
        self.camera_moved = True
        self.stream.set_state(state)

    def producer(self) -> None:
        stream = self.stream
        try:
            renderer = MitsubaRenderer(
                self.scene_source,
                integrator=self.integrator_override,
                rfilter=self.rfilter_override,
                integrator_props=self.current_integrator_props(),
            )
            renderer.denoise_enabled = self.denoise_enabled
            renderer.denoise_albedo = self.denoise_albedo
            renderer.denoise_normals = self.denoise_normals
            renderer.denoise_temporal = self.denoise_temporal
            renderer.accumulate = self.accumulate
            renderer.spp = self.current_spp()
        except Exception:
            self.phase = f"Could not load {self.scene_label}"
            traceback.print_exc()
            return
        renderer.configure(stream.size)
        self.renderer = renderer

        restart_next = False
        while stream.active:
            if stream.wait_if_reconfiguring():
                renderer.configure(stream.size)
                continue
            stream.pace()
            state, changed = stream.state()
            changed |= restart_next
            restart_next = False
            if state is None:
                state = renderer.initial
            pending = self.pending_integrator_props
            if pending is not None:
                self.pending_integrator_props = None
                try:
                    renderer.set_integrator_props(pending)
                    changed = True
                except Exception:
                    traceback.print_exc()
                    self.reject_integrator_props(renderer)
            try:
                frame = renderer.render(state, restart=changed)
            except Exception:
                # Settings that the plugin accepted may still fail when
                # rendering; fall back to the last working configuration.
                if not renderer.revert_integrator_props():
                    raise
                traceback.print_exc()
                self.reject_integrator_props(renderer)
                restart_next = True
                continue
            stream.submit(frame, state)
            if renderer.projection_stale:
                renderer.update_projection()

    def set_exposure(self, stops: float) -> None:
        self.quad.set_exposure(2.0**stops)
        self.exposure_label.set_caption(f"EV {stops:+.1f}")

    def apply_settings(self) -> None:
        zoom = self.zoom_step
        self.controller.zoom_step = 1 / zoom if self.settings.invert_zoom else zoom

    def toggle_settings(self) -> None:
        if self.settings_window is not None:
            self.settings_window.dispose()
            self.settings_window = None
            return
        settings = self.settings
        variants = ["auto"] + jit_variants()

        def set_invert(value: bool) -> None:
            settings.invert_zoom = value
            self.apply_settings()
            settings.save()

        def set_variant(index: int) -> None:
            settings.variant = variants[index] if index > 0 else ""
            settings.save()

        helper = ng.FormHelper(self)
        self.settings_window = window = helper.add_window(ng.Vector2i(0, 0), "Settings")
        ng.Button(window.button_panel(), "", ng.icons.FA_TIMES).set_callback(
            self.toggle_settings
        )
        helper.add_group("Navigation")
        helper.add_bool_variable(
            "Invert zoom", set_invert, lambda: settings.invert_zoom
        )
        helper.add_group("Renderer")
        helper.add_enum_variable(
            "Variant",
            set_variant,
            lambda: (
                variants.index(settings.variant) if settings.variant in variants else 0
            ),
        ).set_items(variants)
        helper.add_widget("", ng.Label(window, "Variant changes apply after a restart"))
        self.perform_layout()
        window.center()

    def toggle_help(self) -> None:
        if self.help is not None:
            self.help.dispose()
            self.help = None
            return
        self.help = window = ng.Window(self, "Viewport controls")
        window.set_layout(ng.GroupLayout())
        ng.Button(window.button_panel(), "", ng.icons.FA_TIMES).set_callback(
            self.toggle_help
        )
        for section, rows in HELP:
            ng.Label(window, section, "sans-bold", 18)
            grid = ng.Widget(window)
            grid.set_layout(
                ng.GridLayout(ng.Orientation.Horizontal, 2, ng.Alignment.Fill, 0, 4)
            )
            for keys, description in rows:
                ng.Label(grid, keys, "sans-bold").set_fixed_width(170)
                ng.Label(grid, description)
        self.perform_layout()
        window.center()

    def drop_event(self, filenames: list[str]) -> bool:
        for filename in filenames:
            path = Path(filename)
            if path.suffix.lower() == ".xml":
                self.load(path, adopt_size=True)
                return True
        return False

    def update_stats(self) -> None:
        renderer = self.renderer
        if renderer is None:
            return
        producer_rate = self.stream.producer_rate
        consumer_rate = self.stream.consumer_rate
        values = (
            f"{producer_rate} [Host: {producer_rate.busy() * 1e3:.1f} ms]",
            str(consumer_rate),
            f"{renderer.size[0]} x {renderer.size[1]} @ {renderer.samples} spp",
        )
        for label, value in zip(self.stats, values):
            label.set_caption(value)

    def draw_contents(self) -> None:
        now = time.perf_counter()
        if self.phase == "Loading" and self.renderer is not None:
            renderer = self.renderer
            if self.adopt_size and renderer.native_size != self.size():
                self.set_size(renderer.native_size)
                self.stream.resize(renderer.native_size)
            self.adopt_size = False
            self.controller.set_world_up(renderer.initial.up, snap=True)
            self.controller.set_state(renderer.initial)
            self.controller.set_callback(self.camera_changed)
            bbox = renderer.scene.bbox()
            self.controller.scene_scale = max(float(dr.norm(bbox.max - bbox.min)), 1e-3)
            self.phase = "Compiling"

        self.controller.update()

        if self.integrator_panel_stale:
            self.integrator_panel_stale = False
            self.settings.save()
            if self.integrator_helper is not None:
                self.integrator_helper.refresh()

        if now - self.last_update > 0.25:
            self.update_stats()
            self.last_update = now

        if self.stream is not None:
            with self.stream.present(self.render_pass) as (texture, camera):
                self.displayed_cam = camera
                if (
                    texture is not None
                    and self.renderer is not None
                    and self.renderer.projection is not None
                ):
                    self.phase = None
                    self.draw_frame(texture, camera)
        else:
            with self.render_pass:
                pass

        if self.pending_pick is not None:
            u, v, focus, deadline = self.pending_pick
            camera = self.displayed_cam
            if (
                self.renderer is not None
                and camera is not None
                and self.renderer.can_pick(camera)
            ):
                self.pick(camera, u, v, focus)
            elif now > deadline:
                self.pending_pick = None
        self.redraw()

    def draw_frame(self, texture: ng.Texture, cam: ng.CameraState | None) -> None:
        render_pass = self.render_pass
        renderer = self.renderer
        projection = renderer.projection
        self.controller.set_projection(projection, self.size())

        render_pass.set_depth_test(ng.RenderPass.DepthTest.LessEqual, True)
        self.quad.set_depth_projection(projection, 1.01)
        self.quad.set_texture(texture)
        self.quad.draw()

        if self.selection and cam:
            self.box.set_mvp(projection @ cam.view_matrix())
            self.box.set_width(2.5 * self.pixel_ratio())
            render_pass.set_depth_test(ng.RenderPass.DepthTest.LessEqual, False)
            self.box.draw()

    def notify(self, text: str) -> None:
        self.notice = (text, time.perf_counter() + 2.0)

    def draw(self, context: ng.nanovg.NVGcontext) -> None:
        if self.notice is not None:
            text, expiry = self.notice
            remaining = expiry - time.perf_counter()
            if remaining <= 0:
                self.notice = None
            else:
                context.FontFace("sans-bold")
                context.FontSize(18)
                context.FillColor(ng.nanovg.RGBAf(1, 1, 1, min(remaining / 0.25, 1.0)))
                context.TextAlign(ng.nanovg.ALIGN_LEFT | ng.nanovg.ALIGN_BOTTOM)
                context.Text(15, self.size()[1] - 12, text)
        if self.phase is not None:
            text = self.phase
            if text in ("Loading", "Compiling"):
                text += " " + "." * (1 + int(time.perf_counter() * 3) % 3)
            context.FontFace("sans-bold")
            context.FontSize(26)
            context.FillColor(ng.nanovg.RGBAf(1, 1, 1, 1))
            context.TextAlign(ng.nanovg.ALIGN_LEFT | ng.nanovg.ALIGN_MIDDLE)
            bounds = context.TextBounds(0, 0, self.phase)
            width, height = self.size()
            context.Text((width - (bounds[2] - bounds[0])) / 2, height / 2, text)
        super().draw(context)

    def resize_event(self, size: ng.Vector2i) -> bool:
        self.render_pass.resize(self.framebuffer_size())
        return super().resize_event(size)

    def resize_end_event(self, size: ng.Vector2i) -> bool:
        if self.stream is not None:
            self.stream.resize(size)
        return True

    def mouse_button_event(
        self, p: ng.Vector2i, button: int, down: bool, modifiers: int
    ) -> bool:
        return super().mouse_button_event(
            p, button, down, modifiers
        ) or self.controller.mouse_button_event(p, button, down, modifiers)

    def click_event(self, p: ng.Vector2i, double_click: bool) -> None:
        camera, size = self.displayed_cam, self.size()
        if self.renderer is None or camera is None:
            return
        self.pick(
            camera,
            (float(p[0]) + 0.5) / max(int(size[0]), 1),
            (float(p[1]) + 0.5) / max(int(size[1]), 1),
            double_click,
        )

    def depth_event(self, p: ng.Vector2f) -> float:
        camera = self.displayed_cam
        if self.renderer is None or camera is None:
            return 0.0
        return self.renderer.depth_at(camera, float(p[0]), float(p[1]))

    def pick(self, cam: ng.CameraState, u: float, v: float, focus: bool) -> None:
        if not self.renderer.can_pick(cam):
            self.pending_pick = (u, v, focus, time.perf_counter() + 2.0)
            return
        self.pending_pick = None
        self.selection = hit = self.renderer.pick(cam, u, v)
        if hit is None:
            return
        name, bbox = hit
        self.notify(f"Selected {name}")
        lower, upper = ng.Vector3f(bbox.min), ng.Vector3f(bbox.max)
        self.box.set_box(lower, upper)
        if focus:
            self.controller.frame(lower, upper)

    def mouse_motion_event_f(
        self, p: ng.Vector2f, rel: ng.Vector2f, button: int, modifiers: int
    ) -> bool:
        return super().mouse_motion_event_f(
            p, rel, button, modifiers
        ) or self.controller.mouse_motion_event(p, rel, button, modifiers)

    def scroll_event(self, p: ng.Vector2i, rel: ng.Vector2f, flags: int) -> bool:
        return super().scroll_event(p, rel, flags) or self.controller.scroll_event(
            p, rel, flags
        )

    def keyboard_event(
        self, key: int, scancode: int, action: int, modifiers: int
    ) -> bool:
        if super().keyboard_event(key, scancode, action, modifiers):
            return True
        if key == ng.glfw.KEY_ESCAPE and action == ng.glfw.PRESS:
            if self.help is not None:
                self.toggle_help()
            else:
                self.set_visible(False)
            return True
        if key == ng.glfw.KEY_H and action == ng.glfw.PRESS:
            self.toggle_help()
            return True
        if key == ng.glfw.KEY_I and action == ng.glfw.PRESS:
            self.toggle_integrator_settings()
            return True
        if key == ng.glfw.KEY_TAB and action == ng.glfw.PRESS:
            self.window.set_visible(not self.window.visible())
            return True
        return self.controller.keyboard_event(key, scancode, action, modifiers)

    def focus_event(self, focused: bool) -> bool:
        self.controller.focus_event(focused)
        return super().focus_event(focused)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scene",
        type=Path,
        nargs="?",
        default=None,
        help="scene to load (scenes can also be dropped onto the window)",
    )
    parser.add_argument(
        "--integrator",
        choices=["scene", *[choice.plugin for choice in INTEGRATORS if choice.plugin]],
        default="scene",
        help="integrator override (default: use the scene integrator)",
    )
    parser.add_argument(
        "--filter",
        choices=["scene", *[choice.plugin for choice in FILTERS if choice.plugin]],
        default="scene",
        help="reconstruction filter override (default: use the scene filter)",
    )
    parser.add_argument(
        "--size",
        default="1024x768",
        metavar="WxH",
        help="window size in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=jit_variants(),
        help=f"Mitsuba variant (default: try {', '.join(VARIANTS)})",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    settings = Settings.load()
    if args.variant:
        mi.set_variant(args.variant)
    else:
        mi.set_variant(*dict.fromkeys((*filter(None, [settings.variant]), *VARIANTS)))

    ng.init()
    width, height = (int(value) for value in args.size.lower().split("x"))
    scene = args.scene.expanduser() if args.scene is not None else None
    integrator = None if args.integrator == "scene" else args.integrator
    rfilter = None if args.filter == "scene" else args.filter
    viewer = Viewer(
        scene,
        ng.Vector2i(width, height),
        settings,
        integrator=integrator,
        rfilter=rfilter,
    )
    viewer.set_visible(True)
    try:
        ng.run()
    finally:
        if viewer.stream is not None:
            viewer.stream.close()
        ng.shutdown()


if __name__ == "__main__":
    run()
