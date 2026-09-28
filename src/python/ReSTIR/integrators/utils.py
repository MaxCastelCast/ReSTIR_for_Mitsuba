import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field

mi.set_variant("llvm_ad_rgb")

POSITION_THRESHOLD = 0.05
NORMAL_THRESHOLD = 0.9


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
class Candidate:
    # Direction sample of this candidate
    direction_sample: mi.DirectionSample3f = field(default_factory=mi.DirectionSample3f)

    # Original pdf of that sample
    pdf: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # The position of the SI associated to the candidate
    surface_position: mi.Point3f = field(
        default_factory=mi.Point3f
    )

    # The surface normal of the SI associated to the candidate
    surface_normal: mi.Normal3f = field(
        default_factory=mi.Normal3f
    )
 

@dataclass
class Reservoir:
    # selected sample
    select: Candidate = field(default_factory = Candidate)

    # sum of all reservoir weights
    w_sum: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # total number of samples
    M: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # contribution weight associated with the reservoir's selected candidate
    W: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    @staticmethod
    def empty(width):
        candidate = Candidate(
            direction_sample=dr.zeros(mi.DirectionSample3f, width),
            pdf=dr.zeros(mi.Float,  width),
            surface_position=dr.zeros(mi.Point3f, width),
            surface_normal=dr.zeros(mi.Normal3f, width)
        )

        return Reservoir(
            select=candidate,
            w_sum=dr.zeros(mi.Float, width),
            M=dr.zeros(mi.Float, width),
            W=dr.zeros(mi.Float, width)
        )

    # Adding a sample to the reservoir
    def add_sample(self, candidate: Candidate, reservoir_weight: mi.Float, rnd1D: mi.Float, active: mi.Mask):

        # Get the mask for currently active Jit lanes with valid weights
        active_candidate = active & (reservoir_weight > 0)

        # Increment the candidate number counter for all valid lanes
        self.M = dr.select(active_candidate, self.M + 1, self.M)

        # Update the total weight of all valid candidates seen so far
        self.w_sum = dr.select(active_candidate, self.w_sum + reservoir_weight, self.w_sum)

        # Replace the candidate only for elements that are active and succeed the random test
        replace = active_candidate & (rnd1D < reservoir_weight/self.w_sum)

        # Replace the selected candidate with the new candidate
        self.select = dr.select(replace, candidate, self.select)

    # Final computation of the reservoir, must be called before merging reservoirs or getting the selected and W arguments
    def finalize(self, p_hat_selected: mi.Float, active: mi.Mask):

        active_r = active & (self.M > 0) & (p_hat_selected > 0)
        safe_M = dr.select(active_r, self.M, 1.0)
        safe_p_hat = dr.select(active_r, p_hat_selected, 1.0)

        self.W = dr.select(active_r, self.w_sum / (safe_M * safe_p_hat) ,0.0)


# Evaluate candidate samples that was from an other sample space in the current sample space
def eval_p_hat_spectrum(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, candidate: Candidate, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, sampler: mi.Sampler, active: mi.Mask) -> mi.Spectrum:
    # Get the direction sample of the candidate
    ds_old = candidate.direction_sample

    # Candidate validity
    active_p = active & (candidate.pdf > 0.)

    # /!\ : The current ds.d is in the previous sample space, and not in the current sample space, so we 
    #       must convert the direction from "ds.p - previous_si.p" to "ds.p - current_si.p"
    delta = ds_old.p - current_si.p
    dist = dr.norm(delta)
    d = delta / dist
    ds = mi.DirectionSample3f(ds_old)

    ds.d = d
    ds.dist = dist

    # Determine the bsdf of that direction (those results are given in local coordinates)
    wo = current_si.to_local(ds.d)
    bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, current_si, wo, active_p)

    # Convert from local BSDF coordinates to world coordinates and apply Mueller matrix transformation
    bsdf_val = current_si.to_world_mueller(bsdf_val, -wo, current_si.wi)

    emitter_val = ds_old.emitter.eval_direction(current_si, ds, active_p)

    # Actual numerator
    p_hat_spectrum = bsdf_val * emitter_val
    safe_pdf = dr.select(active_p, candidate.pdf, 1.0)

    result = dr.select(active_p, p_hat_spectrum / safe_pdf, 0.)

    return dr.select(active_p, result, 1.0)


def eval_p_hat(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, candidate: Candidate, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, sampler: mi.Sampler, active: mi.Mask) -> mi.Float:
    # Evaluate p_hat_spectrum to get the spectrum
    p_hat_spectrum = eval_p_hat_spectrum(scene, current_si, candidate, bsdf, bsdf_ctx, sampler, active)

    # Get a float value with the luminance of the spectrum
    p_hat_lum = mi.luminance(mi.unpolarized_spectrum(p_hat_spectrum), current_si.wavelengths, active)

    return p_hat_lum

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
        s.add_sample(r.select, reservoir_weight, sampler.next_1d(active_r), active_r)

        # Add the nb of samples of this reservoir to our combined reservoir nb
        M_total += dr.select(active_r, r.M, 0.0)

    # Set the total number of samples for the combined reservoir
    s.M = M_total

    active_s = active & (s.M > 0)

    # Compute the new contribution weight for the selected sample of the combined reservoir
    p_hat_select = eval_p_hat(scene, current_si, s.select, bsdf, bsdf_ctx, sampler, active_s)

    active_s &= p_hat_select > 0
    safe_M = dr.select(active_s, s.M, 1.0)
    safe_p_hat = dr.select(active_s, p_hat_select, 1.0)

    s.W = dr.select(active_s, s.w_sum / (safe_M * safe_p_hat), 0.0)

    return s    

# Gathers the reservoirs associated with the indexes
def gather_reservoir(r_set: Reservoir, index: mi.UInt32, active: mi.Mask) -> Reservoir:

    direction_sample_set = dr.gather(dtype=mi.DirectionSample3f, source=r_set.select.direction_sample, index=index, active=active)
    pdf_set = dr.gather(dtype=mi.Float, source=r_set.select.pdf, index=index, active=active)
    surface_position_set = dr.gather(dtype=mi.Point3f, source=r_set.select.surface_position, index=index, active=active)
    surface_normal_set = dr.gather(dtype=mi.Normal3f, source=r_set.select.surface_normal, index=index, active=active)


    candidate_set = Candidate(direction_sample=direction_sample_set, pdf=pdf_set, surface_position=surface_position_set, surface_normal=surface_normal_set)
    w_sum_set = dr.gather(dtype=mi.Float, source=r_set.w_sum, index=index, active=active)
    M_set = dr.gather(dtype=mi.Float, source=r_set.M, index=index, active=active)
    W_set = dr.gather(dtype=mi.Float, source=r_set.W, index=index, active=active)

    return Reservoir(select=candidate_set, w_sum=w_sum_set, W=W_set, M=M_set)

# Store resulting reservoir for next frame
# dr.scatter() modifies its target in place
def scatter_reservoir(previous: Reservoir, current: Reservoir, index: mi.UInt32, active: mi.Mask):

    dr.scatter(previous.select.direction_sample, current.select.direction_sample, index, active=active)
    dr.scatter(previous.select.pdf, current.select.pdf, index, active=active)
    dr.scatter(previous.select.surface_position, current.select.surface_position, index, active=active)
    dr.scatter(previous.select.surface_normal, current.select.surface_normal, index, active=active)

    dr.scatter(previous.w_sum, current.w_sum, index, active=active)
    dr.scatter(previous.M, current.M, index, active=active)
    dr.scatter(previous.W, current.W, index, active=active)

def eval_previous_index(si: mi.SurfaceInteraction3f, previous_sensor: mi.Sensor, current_sensor: mi.Sensor, sampler: mi.Sampler,  film_width: int, film_height: int, pixel_index: mi.UInt32, active: mi.Mask) -> tuple[mi.UInt32, mi.Bool]:

    # Current world-space surface point -> previous sensor
    ds, _ = previous_sensor.sample_direction(si, sampler.next_2d(active), active)

    # Sensor could successfully connect to this point
    valid = active & (ds.pdf > 0)

    # Normalized film coordinates
    x = ds.uv.x
    y = ds.uv.y

    # Reprojection must fall inside the image
    valid &= ((x >= 0.0) & (x < film_width) & (y >= 0.0) & (y < film_height))

     # Normalized film coordinates -> discrete pixel
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

def validate_surface_similarity(current_position: mi.Point3f, current_normal: mi.Normal3f, candidate_position: mi.Point3f, candidate_normal: mi.Normal3f, active: mi.Mask) -> mi.Mask:
    position_error = dr.norm(candidate_position - current_position)
    normal_similarity = dr.dot( dr.normalize(candidate_normal), dr.normalize(current_normal),)

    return (active & (position_error < POSITION_THRESHOLD) & (normal_similarity > NORMAL_THRESHOLD))

