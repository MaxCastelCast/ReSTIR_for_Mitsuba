import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field

mi.set_variant("llvm_ad_rgb")

POSITION_THRESHOLD = 0.05
NORMAL_THRESHOLD = 0.9
MAX_PATH_LENGTH = 2


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float, n_a: float, n_b: float, power: int = 2) -> mi.Float:
    """Compute the MIS weight using the power heuristic."""

    a = n_a * pdf_a
    b = n_b * pdf_b

    # Dr.Jit convert automaticaly the ** operator to dr.power
    a = a ** power
    b = b ** power
    
    w = a / (a + b)

    return dr.select(dr.isfinite(w), w, 0.0)


@dataclass
class Sample:
    # The position of the SI associated to the Sample
    surface_position: mi.Point3f = field(default_factory=mi.Point3f)
    
    # The surface normal of the SI associated to the Sample
    surface_normal: mi.Normal3f = field(default_factory=mi.Normal3f)

    # Sampled emitter associated with the sample
    direction_sample: mi.DirectionSample3f = field(default_factory=mi.DirectionSample3f)

    # Original pdf of that sample
    pdf: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # unoccluded radiance contribution associated with that light sample
    radiance: mi.Spectrum = field(default_factory=lambda: mi.Spectrum(0.0))

    @staticmethod
    def make_empty_sample(width):
        return Sample(
            surface_position=dr.zeros(mi.Point3f, width),
            surface_normal=dr.zeros(mi.Normal3f, width),
            direction_sample=dr.zeros(mi.DirectionSample3f, width),
            pdf=dr.zeros(mi.Float, width),
            radiance=dr.zeros(mi.Spectrum, width)
        )


def select_sample(mask: mi.Mask, a: Sample, b: Sample) -> Sample:
    return Sample(
        surface_position=dr.select(mask, a.surface_position, b.surface_position),
        surface_normal=dr.select(mask, a.surface_normal, b.surface_normal),
        direction_sample=dr.select(mask, a.direction_sample, b.direction_sample),
        pdf=dr.select(mask, a.pdf, b.pdf),
        radiance=dr.select(mask, a.radiance, b.radiance)
    )

@dataclass
class Path:
    # Fixed-size Python container:
    # each entry itself contains Dr.Jit arrays
    vertices: list[Sample] = field(default_factory=list)

    # Number of valid vertices for every JIT lane
    length: mi.UInt32 = field(default_factory=lambda: mi.UInt32(0))

    # PT/GI shift metadata
    reconnection_index: mi.UInt32 = field(default_factory=lambda: mi.UInt32(0))

    @staticmethod
    def empty(width):
        return Path(
            vertices=[Sample.make_empty_sample(width) for _ in range(MAX_PATH_LENGTH)],
            length=dr.zeros(mi.UInt32, width),
            reconnection_index=dr.zeros(mi.UInt32, width),
        )

def select_path(mask: mi.Mask, a: Path, b: Path) -> Path:
    return Path(
        vertices=[select_sample(mask, a.vertices[i], b.vertices[i]) for i in range(MAX_PATH_LENGTH)],
        length=dr.select(mask, a.length, b.length),
        reconnection_index=dr.select(mask, a.reconnection_index, b.reconnection_index)
    )

@dataclass
class Reservoir:
    # selected sample
    select: Path = field(default_factory = Path)

    # sum of all reservoir weights
    w_sum: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # total number of samples
    M: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # contribution weight associated with the reservoir's selected Sample
    W: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    @staticmethod
    def empty(width):
        return Reservoir(
            select=Path.empty(width),
            w_sum=dr.zeros(mi.Float, width),
            M=dr.zeros(mi.Float, width),
            W=dr.zeros(mi.Float, width)
        )

    # Adding a sample to the reservoir
    def add_path(self, sample: Path, reservoir_weight: mi.Float, rnd1D: mi.Float, active: mi.Mask):

        # Get the mask for currently active Jit lanes with valid weights
        active_path = active & (reservoir_weight > 0)

        # Increment the Sample number counter for all valid lanes
        self.M = dr.select(active_path, self.M + 1, self.M)

        # Update the total weight of all valid Samples seen so far
        self.w_sum = dr.select(active_path, self.w_sum + reservoir_weight, self.w_sum)

        # Replace the Sample only for elements that are active and succeed the random test
        replace = active_path & (rnd1D < reservoir_weight/self.w_sum)

        # Replace the selected Sample with the new Sample
        self.select = select_path(replace, sample, self.select)

    # Final computation of the reservoir, must be called before merging reservoirs or getting the selected and W arguments
    def finalize(self, p_hat_selected: mi.Float, active: mi.Mask):

        active_r = active & (self.M > 0) & (p_hat_selected > 0)
        safe_M = dr.select(active_r, self.M, 1.0)
        safe_p_hat = dr.select(active_r, p_hat_selected, 1.0)

        self.W = dr.select(active_r, self.w_sum / (safe_M * safe_p_hat) ,0.0)


# Combines an arbitrary number of reservoir and returns the combined reservoirs
def combine_reservoirs(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, sampler: mi.Sampler, active: mi.Mask, *reservoirs_mask_pairs : tuple[Reservoir, mi.Mask]) -> Reservoir:
    # Initialize the return reservoire
    width = dr.width(current_si.p)
    s = Reservoir.empty(width)
    M_total = dr.zeros(mi.Float, width)

    # Combine the reservoirs by adding their selected candidates and the associate reservoir weight
    for r, valid_r in reservoirs_mask_pairs:

        # Reservoir exists only where it contains samples
        active_r = active & valid_r & (r.M > 0)

        # We need to evaluate p_hat for the current pixel/surface_interaction so that the candidates can be used for the current context 
        p_hat = eval_p_hat(scene, current_si, r.select, bsdf, bsdf_ctx, sampler, active_r)

        # Compute their reservoir weight which is r_w = r.w_sum * (p_hat_current / p_hat_previous) : determine how this weight is important in this context compared to its original one
        reservoir_weight = p_hat * r.W * r.M

        # Add this candidate in the new reservoir
        s.add_path(r.select, reservoir_weight, sampler.next_1d(active_r), active_r)

        # Add the nb of samples of this reservoir to our combined reservoir nb
        M_total += dr.select(active_r, r.M, 0.0)

    # Set the total number of samples for the combined reservoir
    s.M = M_total

    active_s = active & (s.M > 0)

    # Compute the new contribution weight for the selected sample of the combined reservoir
    p_hat_select = eval_p_hat(scene, current_si, s.select, bsdf, bsdf_ctx, sampler, active_s)
    active_s &= p_hat_select > 0
    s.finalize(p_hat_select, active_s)

    return s    


def gather_sample(sample: Sample, index: mi.UInt32, active: mi.Mask) -> Sample:
    return Sample(
        surface_position=dr.gather(mi.Point3f, sample.surface_position, index, active),
        surface_normal=dr.gather(mi.Normal3f, sample.surface_normal, index, active),
        direction_sample=dr.gather(mi.DirectionSample3f, sample.direction_sample, index, active),
        pdf=dr.gather(mi.Float, sample.pdf, index, active),
        radiance=dr.gather(mi.Spectrum, sample.radiance, index, active)
    )

def gather_path(path: Path, index: mi.UInt32, active: mi.Mask) -> Path:
    return Path(
        vertices=[gather_sample(path.vertices[i], index, active) for i in range(MAX_PATH_LENGTH)],
        length=dr.gather(mi.UInt32, path.length, index, active),
        reconnection_index=dr.gather(mi.UInt32, path.reconnection_index, index, active)
    )

# Gathers the reservoirs associated with the indexes
def gather_reservoir(r_set: Reservoir, index: mi.UInt32, active: mi.Mask) -> Reservoir:
    path_set = gather_path(r_set.select, index, active)
    w_sum_set = dr.gather(dtype=mi.Float, source=r_set.w_sum, index=index, active=active)
    M_set = dr.gather(dtype=mi.Float, source=r_set.M, index=index, active=active)
    W_set = dr.gather(dtype=mi.Float, source=r_set.W, index=index, active=active)

    return Reservoir(select=path_set, w_sum=w_sum_set, W=W_set, M=M_set)


def scatter_sample(target: Sample, source: Sample, index: mi.UInt32, active: mi.Mask):
    dr.scatter(target.surface_position, source.surface_position, index, active=active)
    dr.scatter(target.surface_normal, source.surface_normal, index, active=active)
    dr.scatter(target.direction_sample, source.direction_sample, index, active=active)
    dr.scatter(target.pdf, source.pdf, index, active=active)
    dr.scatter(target.radiance, source.radiance, index, active=active)

def scatter_path(target: Path, source: Path, index: mi.UInt32, active: mi.Mask):
    for i in range(MAX_PATH_LENGTH):
        scatter_sample(target.vertices[i], source.vertices[i], index, active)

    dr.scatter(target.length, source.length, index, active=active)
    dr.scatter(target.reconnection_index, source.reconnection_index, index, active=active)


# Store resulting reservoir for next frame
# dr.scatter() modifies its target in place
def scatter_reservoir(previous: Reservoir, current: Reservoir, index: mi.UInt32, active: mi.Mask):

    scatter_path(previous.select, current.select, index, active)
    dr.scatter(previous.w_sum, current.w_sum, index, active=active)
    dr.scatter(previous.M, current.M, index, active=active)
    dr.scatter(previous.W, current.W, index, active=active)


# Evaluate candidate samples that was from an other sample space in the current sample space
# (For the GI implementation Only)
"""
Evaluate the spectral target contribution

Note:
This function follows the Mitsuba convention of returning p_hat(x) / pdf(x),
so it does not return the pure ReSTIR target function p_hat.
"""
def eval_p_hat_spectrum(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, candidate: Path, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, sampler: mi.Sampler, active: mi.Mask) -> mi.Spectrum:

    secondary = candidate.vertices[1]
    active_p = (active & (candidate.length >= 2) & (secondary.pdf > 0.0))

    # Reconnect current x1 to stored x2
    delta = secondary.surface_position - current_si.p
    dist = dr.norm(delta)

    active_p &= dist > 0.0

    safe_dist = dr.select(active_p, dist, 1.0)
    d = delta / safe_dist

    # Evaluate current x1 BSDF toward x2
    wo = current_si.to_local(d)
    bsdf_val, _ = bsdf.eval_pdf(bsdf_ctx, current_si, wo, active_p)
    bsdf_val = current_si.to_world_mueller(bsdf_val, -wo, current_si.wi)

    # Stored indirect radiance coming from x2
    Li = secondary.radiance
    p_hat_spectrum = bsdf_val * Li
    safe_pdf = dr.select(active_p, secondary.pdf, 1.0)
    result = dr.select(active_p, p_hat_spectrum / safe_pdf, 0.)
    
    return dr.select(active_p, result, 0.0)

def eval_p_hat(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, candidate: Path, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, sampler: mi.Sampler, active: mi.Mask) -> mi.Float:
    # Evaluate p_hat_spectrum to get the spectrum
    p_hat_spectrum = eval_p_hat_spectrum(scene, current_si, candidate, bsdf, bsdf_ctx, sampler, active)

    # Get a float value with the luminance of the spectrum
    p_hat_lum = mi.luminance(mi.unpolarized_spectrum(p_hat_spectrum), current_si.wavelengths, active)

    return p_hat_lum

# Get the indexes of the previous frame reservoirs that correspond to the current frame pixels
def eval_previous_index(si: mi.SurfaceInteraction3f, previous_sensor: mi.Sensor, current_sensor: mi.Sensor, sampler: mi.Sampler,  film_width: int, film_height: int, pixel_index: mi.UInt32, active: mi.Mask) -> tuple[mi.UInt32, mi.Bool]:

    # Current world-space surface point -> previous sensor
    ds, _ = previous_sensor.sample_direction(si, sampler.next_2d(active), active)

    # Sensor could successfully connect to this point
    valid = active & (ds.pdf > 0)

    # film coordinates
    x = ds.uv.x
    y = ds.uv.y

    # Reprojection must fall inside the image
    valid &= ((x >= 0.0) & (x < film_width) & (y >= 0.0) & (y < film_height))

     # film coordinates -> discrete pixel
    previous_x = mi.UInt32(dr.clamp(x, 0, film_width - 1))
    previous_y = mi.UInt32(dr.clamp(y, 0, film_height - 1))

    # 2D pixel -> collumn-major reservoir index
    previous_index = previous_x + previous_y * film_width

    return previous_index, valid

# samples uniformly from a square neighborhood
def eval_selected_neighbor(sampler: mi.Sampler,  film_width: int, film_height: int, pixel_index: mi.UInt32, radius: int, active: mi.Mask) -> tuple[mi.UInt32, mi.Bool]:
    # 1) Recover integer screen coordinates from the linear pixel index
    current_y = pixel_index // film_width
    current_x = pixel_index - current_y * film_width

    # 2) Randomly select an offset in [-radius, radius]
    rnd = sampler.next_2d(active)

    diameter = 2 * radius + 1

    offset_x = mi.Int32(dr.floor(rnd.x * diameter)) - radius
    offset_y = mi.Int32(dr.floor(rnd.y * diameter)) - radius

    neighbor_x = mi.Int32(current_x) + offset_x
    neighbor_y = mi.Int32(current_y) + offset_y

    # 3) Check whether the selected neighbor exists
    valid_neighbor = (
        active
        & (neighbor_x >= 0)
        & (neighbor_x < film_width)
        & (neighbor_y >= 0)
        & (neighbor_y < film_height)
    )

    # Do not select the current pixel itself
    valid_neighbor &= (offset_x != 0) | (offset_y != 0)

    # 4) Convert back to a row-major linear index
    # Clamp first so that inactive lanes still contain safe indices.
    safe_x = dr.clamp(neighbor_x, 0, film_width - 1)
    safe_y = dr.clamp(neighbor_y, 0, film_height - 1)

    neighbor_index = (mi.UInt32(safe_x) + mi.UInt32(safe_y) * film_width)

    return neighbor_index, valid_neighbor

# Check if we can reuse a candidate from a previous sample space in the current sample space
# Must be applied on the initaial vertice of a path x1
def validate_surface_similarity(current_position: mi.Point3f, current_normal: mi.Normal3f, candidate_position: mi.Point3f, candidate_normal: mi.Normal3f, active: mi.Mask) -> mi.Mask:
    position_error = dr.norm(candidate_position - current_position)
    normal_similarity = dr.dot( dr.normalize(candidate_normal), dr.normalize(current_normal),)

    return (active & (position_error < POSITION_THRESHOLD) & (normal_similarity > NORMAL_THRESHOLD))

