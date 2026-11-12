import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field


POSITION_THRESHOLD = 0.05
NORMAL_THRESHOLD = 0.9
MAX_PATH_LENGTH = 2

SHIFT_MAPPING_DISTANCE_THRESHOLD = 1e-3


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float, n_a: float, n_b: float, power: int = 2) -> mi.Float:
    """Compute the MIS weight using the power heuristic."""

    a = n_a * pdf_a
    b = n_b * pdf_b

    # Dr.Jit convert automaticaly the ** operator to dr.power
    a = a ** power
    b = b ** power
    
    w = a / (a + b)

    return dr.select(dr.isfinite(w), w, 0.0)


# The sample is an indexed vertex on a path
@dataclass
class Sample:
    # Outgoing direction of the vertex
    outgoingDirection: mi.Vector3f = field(default_factory=mi.Vector3f)

    # Incoming radiance of the vertex
    incomingRadiance: mi.Spectrum = field(default_factory=lambda: mi.Spectrum(0.0))
    
    # Information about the surface of the current vertex (gives information about the triangle mesh id, and the barycentric coordinates)
    surfaceInformation: mi.PreliminaryIntersection3f = field(default_factory=mi.PreliminaryIntersection3f)

    # Lob information of the next vertices is omitted for now (we assume we can reconnect only with diffuse vertices)

    # The seed that generated that sample
    seed : mi.UInt32 = field(default_factory=lambda: mi.UInt32(0))

    # The reconnection index of the vertex
    k : mi.UInt32 = field(default_factory=lambda: mi.UInt32(0))

    # Precomputed Jacobian term
    J : mi.Float = field(default_factory=lambda: mi.Float(0.0))

    # Pdf value of initial sampling stage
    pdf: mi.Float = field(default_factory=lambda: mi.Float(0.0))
   
    @staticmethod
    def make_empty_sample(width):
        return Sample(
            outgoingDirection=dr.zeros(mi.Vector3f, width),
            incomingRadiance=dr.zeros(mi.Spectrum, width),
            surfaceInformation=dr.zeros(mi.PreliminaryIntersection3f, width),
            seed=dr.zeros(mi.UInt32, width),
            k=dr.zeros(mi.UInt32, width),
            J=dr.zeros(mi.Float, width),
            pdf=dr.zeros(mi.Float, width)
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
            W=dr.zeros(mi.Float, width),
        )

    # Adding a sample to the reservoir
    def add_sample(self, sample: Sample, reservoir_weight: mi.Float, rnd1D: mi.Float, active: mi.Mask) -> mi.Mask:

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

        return replace

    # Final computation of the reservoir, must be called before merging reservoirs or getting the selected and W arguments
    def finalize(self, p_hat_selected: mi.Float, active: mi.Mask, debias_term: mi.Float = None):

        # Use classic M normalization by default
        normalization = (self.M if debias_term is None else debias_term)

        active_r = active & (self.M > 0) & (normalization > 0) & (p_hat_selected > 0)

        safe_norm = dr.select(active_r, normalization, 1.0)
        safe_p_hat = dr.select(active_r, p_hat_selected, 1.0)
        self.W = dr.select(active_r, self.w_sum / (safe_norm * safe_p_hat) ,0.0)

@dataclass
class X1_Geometry_Buffer:
    # Position of the primary vertex of the path
    x1_pos: mi.Point3f = field(default_factory=mi.Point3f)

    # Normal of the primary vertex of the path
    x1_norm: mi.Normal3f = field(default_factory=mi.Normal3f)

    # Primary ray direction of the path
    primary_ray: mi.Ray3f = field(default_factory=mi.Ray3f)

    # Mask to indicate which lanes are valid
    valid: mi.Mask = field(default_factory=mi.Mask)

    @staticmethod
    def empty(width: int):
        return X1_Geometry_Buffer(
            x1_pos=dr.zeros(mi.Point3f, width),
            x1_norm=dr.zeros(mi.Normal3f, width),
            primary_ray=dr.zeros(mi.Ray3f, width),
            valid=dr.zeros(mi.Bool, width)
        )

def gather_x1_geometry_buffer(buffer: X1_Geometry_Buffer, index: mi.UInt32, active: mi.Mask) -> X1_Geometry_Buffer:
    return X1_Geometry_Buffer(
        x1_pos=dr.gather(mi.Point3f, buffer.x1_pos, index, active),
        x1_norm=dr.gather(mi.Normal3f, buffer.x1_norm, index, active),
        primary_ray=dr.gather(mi.Ray3f, buffer.primary_ray, index, active),
        valid=dr.gather(mi.Mask, buffer.valid, index, active)
    )

def scatter_x1_geometry_buffer(target: X1_Geometry_Buffer, source: X1_Geometry_Buffer, index: mi.UInt32, active: mi.Mask):
    dr.scatter(target.x1_pos, source.x1_pos, index, active=active)
    dr.scatter(target.x1_norm, source.x1_norm, index, active=active)
    dr.scatter(target.primary_ray, source.primary_ray, index, active=active)
    dr.scatter(target.valid, source.valid, index, active=active)


def seed_pixel(pixel_index: mi.UInt32, frame_index: mi.UInt32) -> mi.UInt32:
    seed, _ = mi.sample_tea_32(pixel_index, frame_index)
    return seed


def gather_sample(sample: Sample, index: mi.UInt32, active: mi.Mask) -> Sample:
    return Sample(
        outgoingDirection=dr.gather(mi.Vector3f, sample.outgoingDirection, index, active),
        incomingRadiance=dr.gather(mi.Spectrum, sample.incomingRadiance, index, active),
        surfaceInformation=dr.gather(mi.PreliminaryIntersection3f, sample.surfaceInformation, index, active),
        seed=dr.gather(mi.UInt32, sample.seed, index, active),
        k=dr.gather(mi.UInt32, sample.k, index, active),
        J=dr.gather(mi.Float, sample.J, index, active),
        pdf=dr.gather(mi.Float, sample.pdf, index, active)
    )

# Gathers the reservoirs associated with the indexes
def gather_reservoir(r_set: Reservoir, index: mi.UInt32, active: mi.Mask) -> Reservoir:
    sample_set = gather_sample(r_set.select, index, active)
    w_sum_set = dr.gather(dtype=mi.Float, source=r_set.w_sum, index=index, active=active)
    M_set = dr.gather(dtype=mi.Float, source=r_set.M, index=index, active=active)
    W_set = dr.gather(dtype=mi.Float, source=r_set.W, index=index, active=active)

    return Reservoir(select=sample_set, w_sum=w_sum_set, W=W_set, M=M_set)


def scatter_sample(target: Sample, source: Sample, index: mi.UInt32, active: mi.Mask):
    dr.scatter(target.outgoingDirection, source.outgoingDirection, index, active=active)
    dr.scatter(target.incomingRadiance, source.incomingRadiance, index, active=active)
    dr.scatter(target.surfaceInformation, source.surfaceInformation, index, active=active)
    dr.scatter(target.seed, source.seed, index, active=active)
    dr.scatter(target.k, source.k, index, active=active)
    dr.scatter(target.J, source.J, index, active=active)
    dr.scatter(target.pdf, source.pdf, index, active=active)


# Store resulting reservoir for next frame
# dr.scatter() modifies its target in place
def scatter_reservoir(previous: Reservoir, current: Reservoir, index: mi.UInt32, active: mi.Mask):
    scatter_sample(previous.select, current.select, index, active)
    dr.scatter(previous.w_sum, current.w_sum, index, active=active)
    dr.scatter(previous.M, current.M, index, active=active)
    dr.scatter(previous.W, current.W, index, active=active)

    

# Only path tracing contribution for now, no NEE
def trace_postfix_radiance(scene: mi.Scene, xk: mi.SurfaceInteraction3f, ray_xk: mi.RayDifferential3f, sampler: mi.Sampler, active: mi.Mask, max_depth: int = 5) -> tuple[mi.Spectrum, mi.Vector3f]:

    # Intitialize accumulated radiance and throughput
    L = dr.zeros(mi.Spectrum, dr.width(xk.p))
    throughput = mi.Spectrum(1.0)

    # Store the outgoing direction sampled at xk
    outgoing_direction = dr.zeros(mi.Vector3f, dr.width(xk.p))

    si = xk
    ray = ray_xk
    active_path = active & xk.is_valid()
    bsdf_ctx = mi.BSDFContext()

    for depth in range(max_depth):

        # Sample the BSDF at the current vertex
        bsdf = si.bsdf(ray)
        sample1 = sampler.next_float(mi.Float, active_path)
        sample2 = mi.Point2f(sampler.next_float(mi.Float, active_path), sampler.next_float(mi.Float, active_path))

        bs, bsdf_weight = bsdf.sample(bsdf_ctx, si, sample1, sample2, active_path)
        active_bsdf = active_path & (bs.pdf > 0.0)
        bsdf_weight = si.to_world_mueller(bsdf_weight, -bs.wo, si.wi)

        wo_world = si.to_world(bs.wo)

        # Store the direction sampled at xk to return it to the caller
        if depth == 0:
            outgoing_direction = dr.select(active_bsdf, wo_world, outgoing_direction)

        # Update throughput, excluding the BSDF at xk
        if depth > 0:
            throughput *= bsdf_weight

        # Trace the next ray
        ray = si.spawn_ray(wo_world)
        si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=False, active=active_bsdf)
        active_path = active_bsdf & si.is_valid()

        # Evaluate emission at the newly reached vertex
        emitter = si.emitter(scene, active_path)
        Le = emitter.eval(si, active_path)

        L += dr.select(active_path, throughput * Le, 0.0)

    return L, outgoing_direction


def create_path_sampler(path_seed: mi.UInt32, width: int) -> dr.llvm.ad.PCG32 | dr.cuda.ad.PCG32:

    # Generate initial state and sequence for each pixel
    state, sequence = mi.sample_tea_32(path_seed, mi.UInt32(0x12345678))

    # Select the correct Dr.Jit backend
    variant = mi.variant()

    if variant.startswith("llvm"):
        rng = dr.llvm.ad.PCG32(size=width)

    elif variant.startswith("cuda"):
        rng = dr.cuda.ad.PCG32(size=width)

    else:
        raise RuntimeError(f"Unsupported Mitsuba variant for seed generation per pixel: {variant}")

    # Explicitly initialize each lane using its own seed
    rng.seed(initstate=mi.UInt64(state), initseq=mi.UInt64(sequence))

    return rng


def eval_p_hat_spectrum(current_si: mi.SurfaceInteraction3f, reconnection_si: mi.SurfaceInteraction3f, candidate: Sample, bsdf_ctx: mi.BSDFContext, active: mi.Mask) -> mi.Spectrum:

    """
    Evaluate the unnormalized target contribution of a
    ReSTIR PT candidate.

    Assumes:
      - current_si is the vertex immediately before reconnection.
      - reconnection_si is the preserved vertex x_k.
      - reconnection_si.wi corresponds to the CURRENT connection.
      - candidate.incomingRadiance stores the suffix radiance
        excluding the BSDF weight at x_k.
      - candidate.outgoingDirection stores the outgoing world
        direction sampled at x_k.
      - Non-delta BSDFs.
    """

    valid = active & current_si.is_valid() & reconnection_si.is_valid()

    # 1. Evaluate BSDF before reconnection
    # Direction from current vertex to x_k
    delta = reconnection_si.p - current_si.p
    dist2 = dr.squared_norm(delta)
    valid &= dist2 > 0.0
    safe_dist2 = dr.select(valid, dist2, 1.0)
    wo_world = delta * dr.rsqrt(safe_dist2)
    wo_local = current_si.to_local(wo_world)
    bsdf_current = current_si.bsdf()
    bsdf_val_current = bsdf_current.eval(bsdf_ctx, current_si, wo_local, valid)

    # 2. Evaluate BSDF at reconnection vertex
    # Outgoing direction toward preserved suffix
    wo_reconnection_local = reconnection_si.to_local(candidate.outgoingDirection)
    bsdf_reconnection = reconnection_si.bsdf()
    bsdf_val_reconnection, bsdf_pdf_reconnection = bsdf_reconnection.eval_pdf(bsdf_ctx, reconnection_si, wo_reconnection_local, valid)
    valid &= bsdf_pdf_reconnection > 0.0
    safe_pdf = dr.select(valid,bsdf_pdf_reconnection, 1.0)

    # Reconnection BSDF sampling weight
    weight_reconnection = dr.select(valid, bsdf_val_reconnection / safe_pdf,0.0)

    # 3. Evaluate target contribution
    target_spectrum = bsdf_val_current * weight_reconnection * candidate.incomingRadiance

    return dr.select(valid, target_spectrum, mi.Spectrum(0.0))


def eval_p_hat(current_si: mi.SurfaceInteraction3f, reconnection_si: mi.SurfaceInteraction3f, candidate: Sample, bsdf_ctx: mi.BSDFContext, active: mi.Mask) -> mi.Float:
    # Evaluate p_hat_spectrum to get the spectrum
    p_hat_spectrum = eval_p_hat_spectrum(current_si, reconnection_si, candidate, bsdf_ctx, active)

    # Get a float value with the luminance of the spectrum
    p_hat = mi.luminance(mi.unpolarized_spectrum(p_hat_spectrum), current_si.wavelengths, active)

    return dr.select(active & dr.isfinite(p_hat) & (p_hat > 0.0), p_hat, 0.0)


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
    valid_neighbor = (active & (neighbor_x >= 0) & (neighbor_x < film_width) & (neighbor_y >= 0) & (neighbor_y < film_height))

    # Do not select the current pixel itself
    valid_neighbor &= (offset_x != 0) | (offset_y != 0)

    # 4) Convert back to a row-major linear index
    # Clamp first so that inactive lanes still contain safe indices.
    safe_x = dr.clamp(neighbor_x, 0, film_width - 1)
    safe_y = dr.clamp(neighbor_y, 0, film_height - 1)

    neighbor_index = (mi.UInt32(safe_x) + mi.UInt32(safe_y) * film_width)

    return neighbor_index, valid_neighbor


def shift_mapping(scene: mi.Scene, temporal_preliminary_ray: mi.Ray3f, initial_reservoir: Reservoir, current_preliminary_ray: mi.Ray3f, previous_reservoir: Reservoir, previous_index: mi.UInt32, current_sampler: mi.Sampler, active: mi.Mask) -> tuple[ mi.Float, mi.Spectrum, mi.Float, mi.Float, mi.Mask]:
    bsdf_ctx = mi.BSDFContext()

    candidate = previous_reservoir.select

    # 1. Reconstruct the previous and current first vertices
    x1 = scene.ray_intersect(temporal_preliminary_ray, ray_flags=mi.RayFlags.All, coherent=False, active=active)
    y1 = scene.ray_intersect(current_preliminary_ray, ray_flags=mi.RayFlags.All, coherent=False, active=active)

    valid = active & x1.is_valid() & y1.is_valid()& (candidate.k == 2)

    # 2. Reconstruct the preserved reconnection vertex x2
    x2, valid_x2 = reconstruct_reconnection_vertex(scene=scene, current_si=x1, candidate=candidate, active=valid)
    valid &= valid_x2

    # 3. Verify that the shift mapping is feasible
    valid = shift_mapping_criterions(scene=scene, yk_previous=y1, xk_previous=x1, xk=x2, active=valid)

    # 4. Compute the reconnection Jacobian
    J = reconnection_jacobian(source_vertex=x1.p, target_vertex=y1.p, reconnection_vertex=x2.p, reconnection_normal=x2.n, active=valid)
    valid &= dr.isfinite(J) & (J > 0.0)

    # 5. Evaluate the new connection y1 -> x2
    delta = x2.p - y1.p
    dist2 = dr.squared_norm(delta)
    valid &= dist2 > 0.0
    safe_dist2 = dr.select(valid, dist2, 1.0)
    wo_world = delta * dr.rsqrt(safe_dist2)
    wo_local = y1.to_local(wo_world)
    bsdf_y1 = y1.bsdf(current_preliminary_ray)

    bsdf_val_y1, target_bsdf_pdf = bsdf_y1.eval_pdf(bsdf_ctx, y1, wo_local, valid)

    valid &= (
        dr.isfinite(target_bsdf_pdf)
        & (target_bsdf_pdf > 0.0)
    )

    safe_pdf_y1 = dr.select(
        valid,
        target_bsdf_pdf,
        1.0
    )

    # Sampling throughput of the new first connection
    weight_y1 = dr.select(
        valid,
        bsdf_val_y1 / safe_pdf_y1,
        mi.Spectrum(0.0)
    )

    # 6. Update the incoming direction at x2

    x2_shifted = mi.SurfaceInteraction3f(x2)

    delta_back = y1.p - x2.p

    dist2_back = dr.squared_norm(delta_back)

    valid &= dist2_back > 0.0

    safe_dist2_back = dr.select(
        valid,
        dist2_back,
        1.0
    )

    wi_world = delta_back * dr.rsqrt(safe_dist2_back)

    x2_shifted.wi = x2_shifted.to_local(wi_world)

    # 7. Evaluate the preserved suffix at x2

    wo_x2_local = x2_shifted.to_local(
        candidate.outgoingDirection
    )

    bsdf_x2 = x2_shifted.bsdf()

    bsdf_val_x2, bsdf_pdf_x2 = bsdf_x2.eval_pdf(
        bsdf_ctx,
        x2_shifted,
        wo_x2_local,
        valid
    )

    valid &= (
        dr.isfinite(bsdf_pdf_x2)
        & (bsdf_pdf_x2 > 0.0)
    )

    safe_pdf_x2 = dr.select(
        valid,
        bsdf_pdf_x2,
        1.0
    )

    weight_x2 = dr.select(
        valid,
        bsdf_val_x2 / safe_pdf_x2,
        mi.Spectrum(0.0)
    )

    # 8. Calculate the shifted path contribution

    shifted_radiance = (
        weight_y1
        * weight_x2
        * candidate.incomingRadiance
    )

    # 9. Calculate the target function

    target_spectrum = (
        bsdf_val_y1
        * weight_x2
        * candidate.incomingRadiance
    )

    p_hat_temporal = mi.luminance(
        mi.unpolarized_spectrum(target_spectrum),
        y1.wavelengths,
        valid
    )

    valid &= (
        dr.isfinite(p_hat_temporal)
        & (p_hat_temporal > 0.0)
    )

    # 10. Mask invalid outputs

    J = dr.select(valid, J, 0.0)

    shifted_radiance = dr.select(
        valid,
        shifted_radiance,
        mi.Spectrum(0.0)
    )

    p_hat_temporal = dr.select(
        valid,
        p_hat_temporal,
        0.0
    )

    target_bsdf_pdf = dr.select(
        valid,
        target_bsdf_pdf,
        0.0
    )

    return (
        J,
        shifted_radiance,
        p_hat_temporal,
        target_bsdf_pdf,
        valid
    )  



def replay_prefix_until_reconnection(scene: mi.Scene, primary_ray: mi.Ray3f, path_seed: mi.UInt32, k: mi.UInt32, max_k: int, active: mi.Mask) -> tuple[mi.SurfaceInteraction3f, mi.Mask]:
    """
    Replay a path prefix from x1 up to x_(k-1).

    Assumes the stored seed reproduces the prefix currently
    represented by the sample.

    For k=2: returns x1.
    For k=3: returns x2.
    For k=4: returns x3.
    """

    width = dr.width(primary_ray)
    ctx = mi.BSDFContext()

    path_sampler = create_path_sampler(path_seed, width)

    current_ray = mi.Ray3f(primary_ray)
    current_si = scene.ray_intersect(current_ray, ray_flags=mi.RayFlags.All, active=active)

    valid = (
        active
        & current_si.is_valid()
        & (k >= 2)
        & (k <= max_k)
    )

    # Each iteration advances one vertex.
    # To obtain x_(k-1), advance k-2 times.
    for i in range(max_k - 2):

        advance = valid & (k > i + 2)

        bsdf = current_si.bsdf(current_ray)

        sample1 = path_sampler.next_float(
            mi.Float, advance
        )

        sample2 = mi.Point2f(
            path_sampler.next_float(mi.Float, advance),
            path_sampler.next_float(mi.Float, advance)
        )

        bs, _ = bsdf.sample(
            ctx,
            current_si,
            sample1,
            sample2,
            advance
        )

        valid_step = advance & (bs.pdf > 0.0)

        next_ray = current_si.spawn_ray(
            current_si.to_world(bs.wo)
        )

        next_si = scene.ray_intersect(
            next_ray,
            ray_flags=mi.RayFlags.All,
            active=valid_step
        )

        valid &= ~advance | (
            valid_step & next_si.is_valid()
        )

        # Update only lanes that need another vertex.
        current_si = dr.select(
            advance,
            next_si,
            current_si
        )

        current_ray = dr.select(
            advance,
            next_ray,
            current_ray
        )

    return current_si, valid
   

# Here we check distance, roughness and visibility
def shift_mapping_criterions(scene: mi.Scene, yk_previous: mi.SurfaceInteraction3f, xk_previous: mi.SurfaceInteraction3f, xk: mi.SurfaceInteraction3f, active: mi.Mask) -> mi.Mask:
    # Check if the shift mapping is feasible based on the roughness and distance criterion
    active_criterions = active & yk_previous.is_valid() & xk_previous.is_valid() & xk.is_valid()

    # Get the positions, normals and check if they are diffused surfaces:
    yk_previous_pos = yk_previous.p
    xk_previous_pos = xk_previous.p
    xk_pos = xk.p

    # Normals will be needed later on
    # yk_previous_normal = yk_previous.n
    # xk_previous_normal = xk_previous.n
    # xk_normal = xk.n

    threshold2 = SHIFT_MAPPING_DISTANCE_THRESHOLD ** 2

    # Distance criterion : check if the distance between the previous and current vertices is not too short to cause a geometric term explosion
    active_criterions &= (dr.squared_norm(xk_pos - xk_previous_pos) > threshold2) & (dr.squared_norm(xk_pos - yk_previous_pos) > threshold2)

    # Roughness criterion : check if the BSDFs of the previous and current vertices are diffuse
    active_criterions &= mi.has_flag(yk_previous.bsdf().flags(), mi.BSDFFlags.Diffuse) & mi.has_flag(xk_previous.bsdf().flags(), mi.BSDFFlags.Diffuse) & mi.has_flag(xk.bsdf().flags(), mi.BSDFFlags.Diffuse)

    # Check for visibility between yk_previous and xk 
    shadow_ray = yk_previous.spawn_ray_to(xk.p)
    occluded = scene.ray_test(shadow_ray, active=active_criterions)
    active_criterions &= ~occluded
    
    return active_criterions


def compute_temporal_pdf_proxy_at_initial(scene: mi.Scene, temporal_primary_ray: mi.Ray3f, current_primary_ray: mi.Ray3f, initial_reservoir: Reservoir, active: mi.Mask) -> mi.Float:
    """
    Evaluate the temporal sampling strategy's transformed PDF proxy
    at the current pixel's initial reservoir sample.

    For k = 2:

        Initial candidate:
            Y_I = (y1, y2)

        Hypothetical temporal candidate:
            X_T* = (x1, y2)

        Transformed PDF proxy:
            q_T->C(Y_I) = p_T(X_T*) / J_T(X_T*)

    Assumes:
        - Reconnection at k = 2.
        - Regular (non-delta) BSDF sampling.
        - Solid-angle PDFs.
        - Vertex-preserving reconnection mapping.

    Returns:
        The transformed temporal PDF proxy.
        Returns 0 for invalid connections.
    """

    bsdf_ctx = mi.BSDFContext()
    width = dr.width(current_primary_ray)

    # 1. Reconstruct the initial candidate's x2

    # Reconstruct current pixel's first vertex y1
    current_si = scene.ray_intersect(current_primary_ray, active=active)
    valid = active & current_si.is_valid()

    # Replay the initial candidate's BSDF sampling
    # using the stored path seed.
    path_sampler = create_path_sampler(initial_reservoir.select.seed, width)
    sample1 = path_sampler.next_float(mi.Float, valid)
    sample2 = mi.Point2f(path_sampler.next_float(mi.Float, valid), path_sampler.next_float(mi.Float, valid))

    current_bsdf = current_si.bsdf(current_primary_ray)
    bs, _ = current_bsdf.sample(bsdf_ctx, current_si, sample1, sample2, valid)
    valid &= bs.pdf > 0.0

    # Ray from y1 toward the initial sample's x2
    ray_to_x2 = current_si.spawn_ray(current_si.to_world(bs.wo))

    # Reconstruct the initial candidate's x2
    initial_x2 = scene.ray_intersect(ray_to_x2, active=valid)

    valid &= initial_x2.is_valid()

    # 2. Reconstruct the temporal source vertex x1
    temporal_si = scene.ray_intersect(temporal_primary_ray,active=valid)

    valid &= temporal_si.is_valid()

    # 3. Evaluate temporal BSDF PDF toward initial x2
    delta = initial_x2.p - temporal_si.p

    dist2 = dr.squared_norm(delta)
    valid &= dist2 > 0.0

    safe_dist2 = dr.select(valid, dist2, 1.0)

    wo_world = delta * dr.rsqrt(safe_dist2)
    wo_local = temporal_si.to_local(wo_world)

    temporal_bsdf = temporal_si.bsdf(temporal_primary_ray)

    source_pdf = temporal_bsdf.pdf(bsdf_ctx, temporal_si, wo_local, valid)

    valid &= source_pdf > 0.0

    # 4. Check visibility of hypothetical connection
    connection_ray = temporal_si.spawn_ray_to(initial_x2.p)
    occluded = scene.ray_test(connection_ray, active=valid)
    valid &= ~occluded

    # 5. Compute hypothetical reconnection Jacobian
    J = reconnection_jacobian(source_vertex=temporal_si.p, target_vertex=current_si.p, reconnection_vertex=initial_x2.p, reconnection_normal=initial_x2.n, active=valid)
    valid &= (J > 0.0) & dr.isfinite(J)

    # Transform source PDF into current path space
    safe_J = dr.select(valid, J, 1.0)
    proxy = dr.select(valid, source_pdf / safe_J, 0.0)

    return proxy



def reconnection_jacobian(source_vertex: mi.Point3f, target_vertex: mi.Point3f, reconnection_vertex: mi.Point3f, reconnection_normal: mi.Normal3f, active: mi.Mask) -> mi.Float:
    """
    Compute the forward Jacobian determinant of a reconnection shift.

    Source connection:
        x_{k-1} -> x_k

    Shifted connection:
        y_{k-1} -> x_k

    The path suffix starting at x_k is preserved.

    Args:
        source_vertex:
            Position of x_{k-1} in the source path.

        target_vertex:
            Position of y_{k-1} in the shifted path.

        reconnection_vertex:
            Position of x_k = y_k, the first preserved vertex.

        reconnection_normal:
            Surface normal at the reconnection vertex.

        active:
            Mask of active paths.

    Returns:
        Forward Jacobian determinant for the reconnection.
        Returns 0 for geometrically degenerate configurations.
    """

    # Vectors from x2 toward the source and target x1 vertices
    delta_source = source_vertex - reconnection_vertex
    delta_target = target_vertex - reconnection_vertex

    # Squared connection lengths
    dist2_source = dr.squared_norm(delta_source)
    dist2_target = dr.squared_norm(delta_target)

    # Check for none-null distances
    valid = (active & (dist2_source > 0.0) & (dist2_target > 0.0))

    # Safe normalization
    safe_dist_source = dr.sqrt(dr.select(valid, dist2_source, 1.0))
    safe_dist_target = dr.sqrt(dr.select(valid, dist2_target, 1.0))

    dir_source = delta_source / safe_dist_source
    dir_target = delta_target / safe_dist_target

    # Cosines at x2
    cos_source = dr.abs(dr.dot(reconnection_normal, dir_source))
    cos_target = dr.abs(dr.dot(reconnection_normal, dir_target))
    valid &= cos_source > 0.0
    valid &= cos_target > 0.0

    safe_cos_source = dr.select(valid, cos_source, 1.0)
    safe_dist2_target = dr.select(valid, dist2_target, 1.0)

    # Jacobian formula for the reconnection shift form x1 to x2
    J = (cos_target / safe_cos_source * dist2_source / safe_dist2_target)

    return dr.select(valid, J, 0.0)


# Combines an arbitrary number of reservoir and returns the combined reservoirs
# Tuple : [
# Reservoir : current reservoir
# mi.Float : MIS weight
# mi.Float : Target function at shifted candidate
# mi.Float : Forward Jacobian
# mi.Masl : Validity mask
#          ]
def combine_reservoirs(sampler: mi.Sampler, active: mi.Mask, *reservoir_data: tuple[Reservoir, mi.Float, mi.Float, mi.Float, mi.Mask]) -> Reservoir:

    # Initialize output reservoir
    width = dr.width(active)
    s = Reservoir.empty(width)

    M_total = dr.zeros(mi.Float, width)

    # Target function of the selected candidate
    selected_p_hat = dr.zeros(mi.Float, width)

    # Combine incoming reservoirs
    for r, mis, p_hat, J, valid_r in reservoir_data:

        # Only consider valid reservoirs and valid weights
        active_r = (active & valid_r & (r.M > 0) & (r.W > 0.0) & (mis > 0.0) & (p_hat > 0.0) & (J > 0.0) & dr.isfinite(mis) & dr.isfinite(p_hat) & dr.isfinite(J))

        # Generalized RIS resampling weight
        reservoir_weight = mis * p_hat * r.W * J
        active_r &= ((reservoir_weight > 0.0) & dr.isfinite(reservoir_weight))

        reservoir_weight = dr.select(active_r, reservoir_weight, 0.0)

        # Weighted reservoir sampling
        replace = s.add_sample(sample=r.select, reservoir_weight=reservoir_weight, rnd1D=sampler.next_1d(active_r), active=active_r)

        # Remember the target of the selected sample
        selected_p_hat = dr.select(replace, p_hat, selected_p_hat)

        # Preserve the number of represented source proposals
        M_total += dr.select(active_r, r.M, 0.0)

    # The output represents all combined source proposals
    s.M = M_total

    # Generalized RIS contribution weight: Equation (22)
    valid_output = active & (s.w_sum > 0.0) & (selected_p_hat > 0.0) & dr.isfinite(s.w_sum)
    safe_p_hat = dr.select(valid_output, selected_p_hat, 1.0)
    s.W = dr.select(valid_output, s.w_sum / safe_p_hat, 0.0)

    return s


def reconstruct_initial_reconnection(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, current_ray: mi.Ray3f, candidate: Sample, active: mi.Mask) -> tuple[mi.SurfaceInteraction3f, mi.Mask]:

    """
    Reconstruct x2 of a locally generated candidate.

    Only supports k = 2.
    """

    width = dr.width(current_si.p)
    bsdf_ctx = mi.BSDFContext()
    valid = active & current_si.is_valid()

    # Recreate the sampler used for this candidate
    path_sampler = create_path_sampler(candidate.seed, width)

    sample1 = path_sampler.next_float(mi.Float, valid)
    sample2 = mi.Point2f(path_sampler.next_float(mi.Float, valid), path_sampler.next_float(mi.Float, valid))

    # Replay BSDF sampling at x1
    bsdf = current_si.bsdf(current_ray)
    bs, _ = bsdf.sample(bsdf_ctx, current_si, sample1, sample2, valid)

    valid &= bs.pdf > 0.0

    # Reconstruct x2
    ray_to_x2 = current_si.spawn_ray(current_si.to_world(bs.wo))
    reconnection_si = scene.ray_intersect(ray_to_x2, active=valid)
    valid &= reconnection_si.is_valid()

    # Verify that replay reached the stored candidate
    stored_pi = candidate.surfaceInformation
    replayed_pi = scene.ray_intersect_preliminary(ray_to_x2, active=valid)

    valid &= stored_pi.is_valid() & (replayed_pi.prim_index == stored_pi.prim_index) & (dr.norm(replayed_pi.prim_uv - stored_pi.prim_uv) < 1e-5)

    return reconnection_si, valid


def reconstruct_reconnection_vertex(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, candidate: Sample, active: mi.Mask) -> tuple[mi.SurfaceInteraction3f, mi.Mask]:
    pi = candidate.surfaceInformation
    valid = active & current_si.is_valid() & pi.is_valid()

    # Allocate reconstructed positions
    width = dr.width(pi.prim_index)

    xk_pos = dr.zeros(mi.Point3f, width)

    found = dr.zeros(mi.Bool, width)

    # Stored barycentric coordinates
    b1 = pi.prim_uv.x
    b2 = pi.prim_uv.y
    b0 = 1.0 - b1 - b2

    # Process each concrete mesh
    for shape in scene.shapes():

        if not isinstance(shape, mi.Mesh):
            continue

        # Convert concrete mesh to a vectorized shape pointer
        mesh_ptr = mi.ShapePtr(shape)

        # Identify lanes belonging to this mesh
        matching = valid & (pi.shape == mesh_ptr)

        # Fetch triangle vertex indices
        indices = shape.face_indices(pi.prim_index, matching)

        # Fetch triangle vertex positions
        v0 = shape.vertex_position(indices[0], matching)
        v1 = shape.vertex_position(indices[1], matching)
        v2 = shape.vertex_position(indices[2], matching)

        # Reconstruct position using barycentric coordinates
        position = b0 * v0 + b1 * v1 + b2 * v2

        # Update only lanes belonging to this mesh
        xk_pos = dr.select(matching, position, xk_pos)

        found |= matching

    valid &= found

    # Recover barycentric coordinates
    b1 = pi.prim_uv.x
    b2 = pi.prim_uv.y
    b0 = 1.0 - b1 - b2

    # Reconstruct world-space position
    xk_pos = b0 * v0 + b1 * v1 + b2 * v2

    # Direction from current vertex to reconnection vertex
    delta = xk_pos - current_si.p
    dist2 = dr.squared_norm(delta)

    valid &= dist2 > 0.0
    safe_dist2 = dr.select(valid, dist2, 1.0)

    direction = delta * dr.rsqrt(safe_dist2)

    # Construct receiving ray
    ray_to_xk = current_si.spawn_ray(direction)

    # Adapt stored preliminary intersection to receiving ray
    reconstructed_pi = mi.PreliminaryIntersection3f(pi)

    reconstructed_pi.t = dr.dot(xk_pos - ray_to_xk.o, ray_to_xk.d)

    # Reconstruct surface interaction
    xk = reconstructed_pi.compute_surface_interaction(ray=ray_to_xk, ray_flags=mi.RayFlags.All, active=valid)

    valid &= xk.is_valid()

    return xk, valid