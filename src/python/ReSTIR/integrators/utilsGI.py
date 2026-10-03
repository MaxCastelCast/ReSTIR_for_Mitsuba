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
    # The position of the vertex x1
    x1: mi.Point3f = field(default_factory=mi.Point3f)
    
    # The surface normal at x1
    x1_n: mi.Normal3f = field(default_factory=mi.Normal3f)

    # The position of the vertex x2
    x2: mi.Point3f = field(default_factory=mi.Point3f)

    # The surface normal at x2
    x2_n: mi.Normal3f = field(default_factory=mi.Normal3f)

    # Original pdf of that sample
    pdf: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # unoccluded radiance contribution associated with that light sample
    radiance: mi.Spectrum = field(default_factory=lambda: mi.Spectrum(0.0))

    # random numbers used for the BSDF event at x1
    random_x1: mi.Vector3f = field(default_factory=mi.Vector3f)

    # random numbers used for the BSDF event at x2
    random_x2: mi.Vector3f = field(default_factory=mi.Vector3f)


    @staticmethod
    def make_empty_sample(width):
        return Sample(
            x1=dr.zeros(mi.Point3f, width),
            x1_n=dr.zeros(mi.Normal3f, width),
            x2=dr.zeros(mi.Point3f, width),
            x2_n=dr.zeros(mi.Normal3f, width),
            pdf=dr.zeros(mi.Float, width),
            radiance=dr.zeros(mi.Spectrum, width),
            random_x1=dr.zeros(mi.Vector3f, width),
            random_x2=dr.zeros(mi.Vector3f, width)
        )

@dataclass
class Reservoir:
    # selected sample
    select: Sample = field(default_factory=Sample)

    # sum of all reservoir weights
    w_sum: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # total number of samples
    M: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # contribution weight associated with the reservoir's selected Sample
    W: mi.Float = field(default_factory=lambda: mi.Float(0.0))

    @staticmethod
    def empty(width):
        return Reservoir(
            select=Sample.make_empty_sample(width),
            w_sum=dr.zeros(mi.Float, width),
            M=dr.zeros(mi.Float, width),
            W=dr.zeros(mi.Float, width)
        )

    # Adding a sample to the reservoir
    def add_sample(self, sample: Sample, reservoir_weight: mi.Float, rnd1D: mi.Float, active: mi.Mask):

        # Get the mask for currently active Jit lanes with valid weights
        active_path = active & (reservoir_weight > 0)

        # Increment the Sample number counter for all valid lanes
        self.M = dr.select(active_path, self.M + 1, self.M)

        # Update the total weight of all valid Samples seen so far
        self.w_sum = dr.select(active_path, self.w_sum + reservoir_weight, self.w_sum)

        # Replace the Sample only for elements that are active and succeed the random test
        replace = active_path & (rnd1D < reservoir_weight/self.w_sum)

        # Replace the selected Sample with the new Sample
        self.select = dr.select(replace, sample, self.select)

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
        s.add_sample(r.select, reservoir_weight, sampler.next_1d(active_r), active_r)

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
        x1=dr.gather(mi.Point3f, sample.x1, index, active),
        x1_n=dr.gather(mi.Normal3f, sample.x1_n, index, active),
        x2=dr.gather(mi.Point3f, sample.x2, index, active),
        x2_n=dr.gather(mi.Normal3f, sample.x2_n, index, active),
        pdf=dr.gather(mi.Float, sample.pdf, index, active),
        radiance=dr.gather(mi.Spectrum, sample.radiance, index, active),
        random_x1=dr.gather(mi.Vector3f, sample.random_x1, index, active),
        random_x2=dr.gather(mi.Vector3f, sample.random_x2, index, active)
    )

# Gathers the reservoirs associated with the indexes
def gather_reservoir(r_set: Reservoir, index: mi.UInt32, active: mi.Mask) -> Reservoir:
    sample_set = gather_sample(r_set.select, index, active)
    w_sum_set = dr.gather(dtype=mi.Float, source=r_set.w_sum, index=index, active=active)
    M_set = dr.gather(dtype=mi.Float, source=r_set.M, index=index, active=active)
    W_set = dr.gather(dtype=mi.Float, source=r_set.W, index=index, active=active)

    return Reservoir(select=sample_set, w_sum=w_sum_set, W=W_set, M=M_set)


def scatter_sample(target: Sample, source: Sample, index: mi.UInt32, active: mi.Mask):
    dr.scatter(target.x1, source.x1, index, active=active)
    dr.scatter(target.x1_n, source.x1_n, index, active=active)
    dr.scatter(target.x2, source.x2, index, active=active)
    dr.scatter(target.x2_n, source.x2_n, index, active=active)
    dr.scatter(target.pdf, source.pdf, index, active=active)
    dr.scatter(target.radiance, source.radiance, index, active=active)
    dr.scatter(target.random_x1, source.random_x1, index, active=active)
    dr.scatter(target.random_x2, source.random_x2, index, active=active)


# Store resulting reservoir for next frame
# dr.scatter() modifies its target in place
def scatter_reservoir(previous: Reservoir, current: Reservoir, index: mi.UInt32, active: mi.Mask):

    scatter_sample(previous.select, current.select, index, active)
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
def eval_p_hat_spectrum(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, candidate: Sample, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, sampler: mi.Sampler, active: mi.Mask) -> mi.Spectrum:

    # Check if the candidate is valid for the current sample space
    valid_candidate = active & (candidate.pdf > 0)

    # Compute the bsdf from the current x1 to the candidate x2
    wo = candidate.x2 - current_si.p
    bsdf_val = bsdf.eval(bsdf_ctx, current_si, wo, valid_candidate)
    emitter_val = candidate.radiance

    p_hat_spectrum = bsdf_val * emitter_val
    safe_pdf = dr.select(valid_candidate, candidate.pdf, 1.0)
    result = dr.select(valid_candidate, p_hat_spectrum / safe_pdf, 0.)
    
    return result
    
def eval_p_hat(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, candidate: Sample, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, sampler: mi.Sampler, active: mi.Mask) -> mi.Float:
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


def reconnection_jacobian(source_x1: mi.Point3f, target_x1: mi.Point3f, x2: mi.Point3f, n2: mi.Normal3f, active: mi.Mask) -> mi.Float:

    # Vectors from x2 toward the source and target primary vertices
    delta_source = source_x1 - x2
    delta_target = target_x1 - x2

    # Squared connection lengths
    dist2_source = dr.squared_norm(delta_source)
    dist2_target = dr.squared_norm(delta_target)

    valid = (active & (dist2_source > 0.0) & (dist2_target > 0.0))

    # Safe normalization
    safe_dist_source = dr.sqrt(dr.select(valid, dist2_source, 1.0))

    safe_dist_target = dr.sqrt(dr.select(valid, dist2_target, 1.0))

    dir_source = delta_source / safe_dist_source
    dir_target = delta_target / safe_dist_target

    # Cosines at x2
    cos_source = dr.abs(dr.dot(n2, dir_source))
    cos_target = dr.abs(dr.dot(n2, dir_target))
    valid &= cos_source > 0.0
    valid &= cos_target > 0.0

    safe_cos_source = dr.select(valid, cos_source, 1.0)
    safe_dist2_target = dr.select(valid, dist2_target, 1.0)

    J = (cos_target / safe_cos_source * dist2_source / safe_dist2_target)

    return dr.select(valid, J, 0.0)


def trace_radiance_from_x2(scene: mi.Scene, x2: mi.SurfaceInteraction3f, ray_x2: mi.RayDifferential3f, sampler: mi.Sampler, first_random: mi.Vector3f, active: mi.Mask, max_depth: int = 5) -> mi.Spectrum:

    # Accumulated outgoing radiance from x2
    L = dr.zeros(mi.Spectrum, dr.width(x2.p))

    # Path throughput from x2 to the current vertex
    throughput = mi.Spectrum(1.0)

    # Current path state
    si = x2
    ray = ray_x2
    active_path = active & x2.is_valid()

    bsdf_ctx = mi.BSDFContext()

    for depth in range(max_depth):

        # ------------------------------------------------------
        # 1. Accumulate emission at the current vertex
        # ------------------------------------------------------
        emitter = si.emitter(scene, active_path)
        Le = emitter.eval(si, active_path)

        hit_emitter = active_path & dr.any(Le != 0, axis=0)

        L += dr.select(
            hit_emitter,
            throughput * Le,
            0.0
        )

        # In this simple BSDF-only estimator, stop once a light is hit
        active_path &= ~hit_emitter

        # ------------------------------------------------------
        # 2. Sample the BSDF
        # ------------------------------------------------------
        bsdf = si.bsdf(ray)

        # Replay the stored random numbers for the first event at x2
        if depth == 0:
            sample1 = first_random.x
            sample2 = mi.Point2f(
                first_random.y,
                first_random.z
            )
        else:
            sample1 = sampler.next_1d(active_path)
            sample2 = sampler.next_2d(active_path)

        bs, bsdf_weight = bsdf.sample(
            bsdf_ctx,
            si,
            sample1,
            sample2,
            active_path
        )

        active_path &= bs.pdf > 0.0

        # Mitsuba BSDF sampling weight contains f * cos / pdf
        bsdf_weight = si.to_world_mueller(
            bsdf_weight,
            -bs.wo,
            si.wi
        )

        # Update path throughput
        throughput *= bsdf_weight

        # ------------------------------------------------------
        # 3. Trace one continuation ray
        # ------------------------------------------------------
        wo_world = si.to_world(bs.wo)

        ray = si.spawn_ray(wo_world)

        si = scene.ray_intersect(
            ray,
            ray_flags=mi.RayFlags.All,
            coherent=False,
            active=active_path
        )

        active_path &= si.is_valid()

    return L