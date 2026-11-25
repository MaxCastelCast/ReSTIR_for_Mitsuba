import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field


POSITION_THRESHOLD = 0.05
NORMAL_THRESHOLD = 0.9
MAX_PATH_LENGTH = 6

SHIFT_MAPPING_DISTANCE_THRESHOLD = 1e-3


def mis_weight(*pdfs: mi.Float, selected_index: int = 0, power: int = 2, sample_counts: tuple | None = None) -> mi.Float:
    """
    Compute the generalized MIS power heuristic.

    pdfs:
        Proposal densities of all sampling strategies evaluated at the SAME candidate.

    selected_index:
        Index of the strategy whose MIS weight we want.

    sample_counts:
        Optional number of proposals associated with each strategy.
        Defaults to one proposal per strategy.
    """

    if len(pdfs) == 0:
        raise ValueError("At least one PDF is required.")

    if selected_index < 0 or selected_index >= len(pdfs):
        raise ValueError("Invalid selected_index.")

    if sample_counts is None:
        sample_counts = (1.0,) * len(pdfs)

    if len(sample_counts) != len(pdfs):
        raise ValueError("sample_counts must match the number of PDFs.")

    denominator = mi.Float(0.0)
    numerator = mi.Float(0.0)

    for i, pdf in enumerate(pdfs):
        valid_pdf = dr.isfinite(pdf) & (pdf > 0.0)
        safe_pdf = dr.select(valid_pdf, pdf, 0.0)
        weighted_pdf = sample_counts[i] * safe_pdf
        term = weighted_pdf ** power
        denominator += term

        if i == selected_index:
            numerator = term

    safe_denominator = dr.select(denominator > 0.0, denominator, 1.0)
    weight = numerator / safe_denominator

    return dr.select((denominator > 0.0) & dr.isfinite(weight), weight, 0.0)


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

    # NEE sample generated at the reconnection vertex
    nee_sample: mi.DirectionSample3f = field(default_factory=mi.DirectionSample3f)

    # Whether the NEE sample is valid
    nee_valid: mi.Mask = field(default_factory=lambda: mi.Bool(False))

    # Radiance obtained from NEE
    nee_emitter_weight: mi.Spectrum = field(default_factory=lambda: mi.Spectrum(0.0))
   
    @staticmethod
    def make_empty_sample(width):
        return Sample(
            outgoingDirection=dr.zeros(mi.Vector3f, width),
            incomingRadiance=dr.zeros(mi.Spectrum, width),
            surfaceInformation=dr.zeros(mi.PreliminaryIntersection3f, width),
            seed=dr.zeros(mi.UInt32, width),
            k=dr.zeros(mi.UInt32, width),
            J=dr.zeros(mi.Float, width),
            pdf=dr.zeros(mi.Float, width),
            nee_sample=dr.zeros(mi.DirectionSample3f, width),
            nee_valid=dr.zeros(mi.Bool, width),
            nee_emitter_weight=dr.zeros(mi.Spectrum, width)
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

        # Increment the Sample number counter for all valid lanes
        self.M = dr.select(active, self.M + 1, self.M)

        # Get the mask for currently active Jit lanes with valid weights
        active_path = active & (reservoir_weight > 0) & dr.isfinite(reservoir_weight)

        # Update the total weight of all valid Samples seen so far
        self.w_sum = dr.select(active_path, self.w_sum + reservoir_weight, self.w_sum)

        # Replace the Sample only for elements that are active and succeed the random test
        replace = active_path & (rnd1D < reservoir_weight/self.w_sum)

        # Replace the selected Sample with the new Sample
        self.select = dr.select(replace, sample, self.select)

        return replace

    # Final computation of the reservoir, must be called before merging reservoirs or getting the selected and W arguments
    def finalize(self, p_hat_selected: mi.Float, active: mi.Mask):

        active_r = active & (self.M > 0) & (p_hat_selected > 0)

        safe_M = dr.select(active_r, self.M, 1.0)
        safe_p_hat = dr.select(active_r, p_hat_selected, 1.0)
        self.W = dr.select(active_r, self.w_sum / (safe_M * safe_p_hat) ,0.0)

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
        pdf=dr.gather(mi.Float, sample.pdf, index, active),
        nee_sample=dr.gather(mi.DirectionSample3f, sample.nee_sample, index, active),
        nee_valid=dr.gather(mi.Bool, sample.nee_valid, index, active),
        nee_emitter_weight=dr.gather(mi.Spectrum, sample.nee_emitter_weight, index, active)
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
    dr.scatter(target.nee_sample, source.nee_sample, index, active)
    dr.scatter(target.nee_emitter_weight, source.nee_emitter_weight, index, active)
    dr.scatter(target.nee_valid, source.nee_valid, index, active=active)


# Store resulting reservoir for next frame
# dr.scatter() modifies its target in place
def scatter_reservoir(previous: Reservoir, current: Reservoir, index: mi.UInt32, active: mi.Mask):
    scatter_sample(previous.select, current.select, index, active)
    dr.scatter(previous.w_sum, current.w_sum, index, active=active)
    dr.scatter(previous.M, current.M, index, active=active)
    dr.scatter(previous.W, current.W, index, active=active)

def is_pure_diffuse(bsdf: mi.BSDF) -> mi.Mask:
    flags = bsdf.flags()
    has_diffuse = mi.has_flag(flags, mi.BSDFFlags.Diffuse)
    has_glossy = mi.has_flag(flags, mi.BSDFFlags.Glossy)
    has_delta = mi.has_flag(flags, mi.BSDFFlags.Delta)
    has_null = mi.has_flag(flags, mi.BSDFFlags.Null)
    return has_diffuse & ~has_glossy & ~has_delta & ~has_null

def reconnection_eligible(previous_si: mi.SurfaceInteraction3f, current_si: mi.SurfaceInteraction3f, active: mi.Mask) -> mi.Mask:
    previous_bsdf = previous_si.bsdf()
    current_bsdf = current_si.bsdf()
    previous_diffuse = is_pure_diffuse(previous_bsdf)
    current_diffuse = is_pure_diffuse(current_bsdf)
    distance_squared = dr.squared_norm(current_si.p - previous_si.p)
    valid_distance = distance_squared > SHIFT_MAPPING_DISTANCE_THRESHOLD ** 2
    return active & previous_si.is_valid() & current_si.is_valid() & previous_diffuse & current_diffuse & valid_distance


def trace_prefix_throughput(scene: mi.Scene, path_sampler: dr.llvm.ad.PCG32 | dr.cuda.ad.PCG32, nee_sampler: mi.Sampler, active: mi.Mask, max_depth: int = 5, x1: mi.SurfaceInteraction3f | None = None, ray_x0: mi.RayDifferential3f | None = None) -> tuple:
    """
    Trace a path prefix until the first eligible reconnection pair (x(k-1), xk) is found.

    Returns:
        prefix_throughput: Product of BSDF sampling weights at x1,...,x(k-1).
        xk_pdf: BSDF PDF of the direction sampled at x(k-1) toward xk.
        ray_to_xk: Ray from x(k-1) toward xk.
        prefix_radiance: Accumulated emission and NEE before xk.
        xk_previous: Surface interaction at x(k-1).
        xk_delta: Whether the BSDF event toward xk was delta.
        selected_k: Selected reconnection index.
        found_reconnection: Mask indicating successful reconnection.
    """

    # 1. Initialize the first surface interaction
    if x1 is None:
        x1 = scene.ray_intersect(ray_x0, active=active)

    width = dr.width(ray_x0)
    searching = active & x1.is_valid()
    found_reconnection = dr.zeros(mi.Bool, width)
    selected_k = dr.zeros(mi.UInt32, width)

    # 2. Initialize prefix information
    prefix_throughput = dr.full(mi.Spectrum, 1.0, width)
    prefix_radiance = dr.zeros(mi.Spectrum, width)
    xk_pdf = dr.zeros(mi.Float, width)
    bsdf_ctx = mi.BSDFContext()

    x_current = x1
    current_ray = mi.Ray3f(ray_x0)
    ray_to_xk = mi.Ray3f(ray_x0)
    xk_previous = mi.SurfaceInteraction3f(x1)

    previous_si = mi.SurfaceInteraction3f(x1)
    previous_bsdf_pdf = dr.zeros(mi.Float, width)
    previous_delta = dr.zeros(mi.Bool, width)
    xk_delta = dr.zeros(mi.Bool, width)

    # 3. Trace vertices until an eligible reconnection is found
    for i in range(max_depth - 2):

        # Candidate reconnection index
        candidate_k = i + 2
        trace = searching

        # Retrieve BSDF at the current vertex
        bsdf_xi = x_current.bsdf(current_ray)

        # Accumulate emission at the current prefix vertex
        emitter = x_current.emitter(scene, trace)
        Le = emitter.eval(x_current, trace)

        # Apply MIS to BSDF-sampled emitter hits
        mis_bsdf = dr.ones(mi.Float, width)
        active_mis = trace & (i > 0)
        ds_bsdf = mi.DirectionSample3f(scene, x_current, previous_si)
        emitter_pdf = scene.pdf_emitter_direction(previous_si, ds_bsdf, active=active_mis)
        mis_previous = mis_weight(previous_bsdf_pdf, emitter_pdf, selected_index=0, power=2)
        mis_previous = dr.select(previous_delta, 1.0, mis_previous)
        mis_bsdf = dr.select(active_mis, mis_previous, 1.0)

        emission_contribution = prefix_throughput * Le * mis_bsdf
        prefix_radiance += dr.select(trace, emission_contribution, mi.Spectrum(0.0))

        # Next Event Estimation
        sample_nee = mi.Point2f(nee_sampler.next_1d(trace), nee_sampler.next_1d(trace))
        ds_nee, emitter_weight = scene.sample_emitter_direction(x_current, sample_nee, True, trace)
        active_nee = trace & (ds_nee.pdf > 0.0) & dr.isfinite(ds_nee.pdf)

        wo_nee = x_current.to_local(ds_nee.d)
        bsdf_val_nee, bsdf_pdf_nee = bsdf_xi.eval_pdf(bsdf_ctx, x_current, wo_nee, active_nee)
        bsdf_val_nee = x_current.to_world_mueller(bsdf_val_nee, -wo_nee, x_current.wi)

        mis_nee = mis_weight(ds_nee.pdf, bsdf_pdf_nee, selected_index=0, power=2)
        mis_nee = dr.select(ds_nee.delta, 1.0, mis_nee)

        nee_contribution = prefix_throughput * bsdf_val_nee * emitter_weight * mis_nee
        prefix_radiance += dr.select(active_nee, nee_contribution, mi.Spectrum(0.0))

        # Sample BSDF direction
        sample1 = path_sampler.next_float(mi.Float, trace)
        sample2 = mi.Point2f(path_sampler.next_float(mi.Float, trace), path_sampler.next_float(mi.Float, trace))
        bs, bsdf_weight = bsdf_xi.sample(bsdf_ctx, x_current, sample1, sample2, trace)

        # Check whether sampling succeeded
        valid_sample = trace & (bs.pdf > 0.0) & dr.isfinite(bs.pdf)

        # Convert BSDF weight to world polarization basis
        bsdf_weight = x_current.to_world_mueller(bsdf_weight, -bs.wo, x_current.wi)

        # Accumulate BSDF / PDF throughput
        prefix_throughput = dr.select(trace, dr.select(valid_sample, prefix_throughput * bsdf_weight, mi.Spectrum(0.0)), prefix_throughput)

        # Generate ray toward the next vertex
        wo_world = x_current.to_world(bs.wo)
        next_ray = x_current.spawn_ray(wo_world)
        sampled_delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)

        # Intersect the next vertex
        next_si = scene.ray_intersect(next_ray, ray_flags=mi.RayFlags.All, coherent=False, active=valid_sample)
        valid_next = valid_sample & next_si.is_valid()

        # Test reconnection eligibility
        eligible = reconnection_eligible(x_current, next_si, valid_next)

        # Save the first eligible reconnection
        selected_k = dr.select(eligible, mi.UInt32(candidate_k), selected_k)
        xk_delta = dr.select(eligible, sampled_delta, xk_delta)
        xk_pdf = dr.select(eligible, bs.pdf, xk_pdf)
        ray_to_xk = dr.select(eligible, next_ray, ray_to_xk)
        xk_previous = dr.select(eligible, x_current, xk_previous)

        # Remember which lanes have found a reconnection
        found_reconnection |= eligible

        # Continue searching only for unfinished lanes
        continue_path = valid_next & ~eligible

        # Store BSDF sampling information for MIS at the next prefix vertex
        previous_si = dr.select(continue_path, x_current, previous_si)
        previous_bsdf_pdf = dr.select(continue_path, bs.pdf, previous_bsdf_pdf)
        previous_delta = dr.select(continue_path, sampled_delta, previous_delta)

        # Advance only unfinished paths
        x_current = dr.select(continue_path, next_si, x_current)
        current_ray = dr.select(continue_path, next_ray, current_ray)

        # Update the searching mask
        searching = continue_path

    # 4. Mask invalid final outputs
    prefix_throughput = dr.select(found_reconnection, prefix_throughput, mi.Spectrum(0.0))
    xk_pdf = dr.select(found_reconnection, xk_pdf, 0.0)

    return prefix_throughput, xk_pdf, ray_to_xk, prefix_radiance, xk_previous, xk_delta, selected_k, found_reconnection


def trace_postfix_radiance(scene: mi.Scene, xk: mi.SurfaceInteraction3f, ray_xk: mi.RayDifferential3f, path_sampler: dr.llvm.ad.PCG32 | dr.cuda.ad.PCG32, nee_sampler: mi.Sampler, k: mi.UInt32, active: mi.Mask, max_depth: int = 5) -> tuple[mi.Spectrum, mi.Vector3f]:

    # 1. Initialize accumulated radiance and throughput
    width = dr.width(xk.p)
    L = dr.zeros(mi.Spectrum, width)
    throughput = dr.full(mi.Spectrum, 1.0, width)
    outgoing_direction = dr.zeros(mi.Vector3f, width)

    si = xk
    ray = mi.Ray3f(ray_xk)
    bsdf_ctx = mi.BSDFContext()

    active_path = active & xk.is_valid() & (k >= 2) & (k < max_depth)

    # 2. Trace the suffix starting at xk
    for depth in range(max_depth - 2):

        trace = active_path & (k + depth < max_depth)

        # Retrieve the current BSDF
        bsdf = si.bsdf(ray)

        # 3. NEE at vertices after xk
        if depth > 0:

            ds_nee, emitter_weight = scene.sample_emitter_direction(si, nee_sampler.next_2d(trace), True, trace)
            active_nee = trace & (ds_nee.pdf > 0.0) & dr.isfinite(ds_nee.pdf)

            wo_nee = si.to_local(ds_nee.d)
            bsdf_val_nee, bsdf_pdf_nee = bsdf.eval_pdf(bsdf_ctx, si, wo_nee, active_nee)
            bsdf_val_nee = si.to_world_mueller(bsdf_val_nee, -wo_nee, si.wi)

            mis_nee = mis_weight(ds_nee.pdf, bsdf_pdf_nee, selected_index=0, power=2)
            mis_nee = dr.select(ds_nee.delta, 1.0, mis_nee)

            nee_contribution = throughput * bsdf_val_nee * emitter_weight * mis_nee
            L += dr.select(active_nee, nee_contribution, mi.Spectrum(0.0))

        # 4. BSDF continuation
        sample1 = path_sampler.next_float(mi.Float, trace)
        sample2 = mi.Point2f(path_sampler.next_float(mi.Float, trace), path_sampler.next_float(mi.Float, trace))
        bs, bsdf_weight = bsdf.sample(bsdf_ctx, si, sample1, sample2, trace)

        valid_sample = trace & (bs.pdf > 0.0) & dr.isfinite(bs.pdf)
        bsdf_weight = si.to_world_mueller(bsdf_weight, -bs.wo, si.wi)

        if depth > 0:
            throughput = dr.select(valid_sample, throughput * bsdf_weight, mi.Spectrum(0.0))

        wo_world = si.to_world(bs.wo)

        # 5. Trace next intersection
        next_ray = si.spawn_ray(wo_world)
        next_si = scene.ray_intersect(next_ray, ray_flags=mi.RayFlags.All, coherent=False, active=valid_sample)
        valid_next = valid_sample & next_si.is_valid()

        # 6. Evaluate BSDF-sampled emitter hit
        emitter = next_si.emitter(scene, valid_next)
        Le = emitter.eval(next_si, valid_next)

        ds_bsdf = mi.DirectionSample3f(scene, next_si, si)
        emitter_pdf = scene.pdf_emitter_direction(si, ds_bsdf, active=valid_next)

        mis_bsdf = mis_weight(bs.pdf, emitter_pdf, selected_index=0, power=2)
        mis_bsdf = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta), 1.0, mis_bsdf)

        if depth == 0:
            outgoing_direction = wo_world
        else:
            mis_bsdf = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta), 1.0, mis_bsdf)

        L += dr.select(valid_next, throughput * Le * mis_bsdf, mi.Spectrum(0.0))

        # 7. Continue the suffix
        continue_path = valid_next & (k + depth + 1 < max_depth)

        si = dr.select(continue_path, next_si, si)
        ray = dr.select(continue_path, next_ray, ray)
        active_path = continue_path

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


def eval_p_hat_spectrum(scene: mi.Scene, primary_ray: mi.Ray3f, candidate: Sample, max_k: int, bsdf_ctx: mi.BSDFContext, active: mi.Mask, x1: mi.SurfaceInteraction3f | None = None) -> mi.Spectrum:

     # 1. Replay the prefix and reconstruct the preserved xk
    previous_si, reconnection_si, prefix_throughput, _, valid = replay_path_to_reconnection(scene=scene, primary_ray=primary_ray, path_seed=candidate.seed, k=candidate.k, max_k=max_k, active=active, x1=x1, candidate=candidate, use_stored_xk=True)

    # 2. Evaluate BSDF at y(k-1) toward preserved xk
    delta = reconnection_si.p - previous_si.p
    dist2 = dr.squared_norm(delta)
    valid &= (dist2 > 0.0) & dr.isfinite(dist2)
    safe_dist2 = dr.select(valid, dist2, 1.0)
    wo_world = delta * dr.rsqrt(safe_dist2)
    wo_local = previous_si.to_local(wo_world)

    bsdf_previous = previous_si.bsdf()
    bsdf_val_previous = bsdf_previous.eval(bsdf_ctx, previous_si, wo_local, valid)

    xk_contribution = eval_xk_contribution(reconnection_si, candidate, bsdf_ctx, valid)

    # Compute the complete target contribution
    target_spectrum = prefix_throughput * bsdf_val_previous * xk_contribution

    return dr.select(valid, target_spectrum, mi.Spectrum(0.0))


def eval_p_hat(scene: mi.Scene, primary_ray: mi.Ray3f, candidate: Sample, max_k: int, bsdf_ctx: mi.BSDFContext, active: mi.Mask, x1: mi.SurfaceInteraction3f | None = None) -> mi.Float:

    target_spectrum = eval_p_hat_spectrum(scene=scene, primary_ray=primary_ray, candidate=candidate, max_k=max_k, bsdf_ctx=bsdf_ctx, active=active, x1=x1)
    p_hat = mi.luminance( mi.unpolarized_spectrum(target_spectrum), primary_ray.wavelengths, active)

    return dr.select(active & dr.isfinite(p_hat) & (p_hat > 0.0), p_hat, 0.0)


def eval_xk_contribution(xk: mi.SurfaceInteraction3f, candidate: Sample, bsdf_ctx: mi.BSDFContext, active: mi.Mask) -> mi.Spectrum:

    # 1. Retrieve the BSDF at the reconnection vertex
    bsdf_xk = xk.bsdf()

    # 2. Evaluate the BSDF-sampled suffix contribution
    wo_local = xk.to_local(candidate.outgoingDirection)
    bsdf_val, bsdf_pdf = bsdf_xk.eval_pdf(bsdf_ctx, xk, wo_local, active)
    bsdf_val = xk.to_world_mueller(bsdf_val, -wo_local, xk.wi)
    valid_bsdf = active & (bsdf_pdf > 0.0) & dr.isfinite(bsdf_pdf)
    safe_pdf = dr.select(valid_bsdf, bsdf_pdf, 1.0)
    bsdf_weight = dr.select(valid_bsdf, bsdf_val / safe_pdf, mi.Spectrum(0.0))
    suffix_contribution = bsdf_weight * candidate.incomingRadiance

    # 3. Retrieve the stored NEE sample
    ds_nee = candidate.nee_sample
    active_nee = active & candidate.nee_valid
    active_nee &= (ds_nee.pdf > 0.0) & dr.isfinite(ds_nee.pdf)

    # 4. Evaluate the BSDF toward the sampled emitter
    wo_nee = xk.to_local(ds_nee.d)
    bsdf_val_nee, bsdf_pdf_nee = bsdf_xk.eval_pdf(bsdf_ctx, xk, wo_nee, active_nee)
    bsdf_val_nee = xk.to_world_mueller(bsdf_val_nee, -wo_nee, xk.wi)

    # 5. Compute the NEE MIS weight
    mis_nee = mis_weight(ds_nee.pdf, bsdf_pdf_nee, selected_index=0, power=2)
    mis_nee = dr.select(ds_nee.delta, 1.0, mis_nee)

    # 6. Evaluate the NEE contribution
    nee_contribution = bsdf_val_nee * candidate.nee_emitter_weight * mis_nee
    nee_contribution = dr.select(active_nee, nee_contribution, mi.Spectrum(0.0))

    # 7. Combine both estimators
    xk_contribution = suffix_contribution + nee_contribution

    return dr.select(active, xk_contribution, mi.Spectrum(0.0))


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

def shift_mapping(scene: mi.Scene, temporal_preliminary_ray: mi.Ray3f, temporal_reservoir: Reservoir, current_preliminary_ray: mi.Ray3f, max_depth: int , active: mi.Mask) -> tuple[mi.Float, mi.Float, mi.Float, mi.Mask]:
    bsdf_ctx = mi.BSDFContext()
    candidate = temporal_reservoir.select
    width = dr.width(current_preliminary_ray)

    # 1. Construct the needed vertices for our shift mappign
    # TO OPTIMIZE : can use geometric buffer of x1 for the path x
    xk_previous, xk, _, _, valid_replay_xk  = replay_path_to_reconnection(scene=scene, primary_ray=temporal_preliminary_ray, path_seed=candidate.seed, k=candidate.k, max_k=MAX_PATH_LENGTH, active=active, x1 = None, candidate= candidate, use_stored_xk=True)
    # TODO : is the path "y" using the right seed for replay ? 
    yk_previous, _, yk_prefix_throughput, prefix_path_pdf_y, valid_replay_yk = replay_path_to_reconnection(scene=scene, primary_ray=current_preliminary_ray, path_seed=candidate.seed, k=candidate.k, max_k=MAX_PATH_LENGTH, active=active, x1 = None, candidate= candidate, use_stored_xk=True)
    valid = active & valid_replay_xk & valid_replay_yk & xk_previous.is_valid() & xk.is_valid() & yk_previous.is_valid()

    # 2. Verify that the shift mapping is feasible
    valid = shift_mapping_criterions(scene=scene, yk_previous=yk_previous, xk_previous=xk_previous, xk=xk, active=valid)

    # 3. Compute the reconnection Jacobian
    J = reconnection_jacobian(source_vertex=xk_previous.p, target_vertex=yk_previous.p, reconnection_vertex=xk.p, reconnection_normal=xk.n, active=valid)
    valid &= dr.isfinite(J) & (J > 0.0)


    # 4. Evaluate the new connection yk_previous -> xk
    delta = xk.p - yk_previous.p
    dist2 = dr.squared_norm(delta)
    valid &= dist2 > 0.0
    safe_dist2 = dr.select(valid, dist2, 1.0)
    wo_world = delta * dr.rsqrt(safe_dist2)
    wo_local = yk_previous.to_local(wo_world)
    bsdf_yk = yk_previous.bsdf()
    bsdf_val_yk_previous, target_bsdf_pdf = bsdf_yk.eval_pdf(bsdf_ctx, yk_previous, wo_local, valid)
    valid &= dr.isfinite(target_bsdf_pdf) & (target_bsdf_pdf > 0.0)

    # 5. Update the incoming direction at xk
    xk_shifted = mi.SurfaceInteraction3f(xk)
    delta_back = yk_previous.p - xk.p
    dist2_back = dr.squared_norm(delta_back)
    valid &= dist2_back > 0.0
    safe_dist2_back = dr.select(valid, dist2_back, 1.0)
    wi_world = delta_back * dr.rsqrt(safe_dist2_back)
    xk_shifted.wi = xk_shifted.to_local(wi_world)

    # Evaluate the preserved suffix and NEE using the shifted incoming direction
    xk_contribution = eval_xk_contribution(xk_shifted, candidate, bsdf_ctx, valid)

    # 7. Calculate the target function
    target_reconnection_pdf = target_bsdf_pdf
    target_spectrum = yk_prefix_throughput * bsdf_val_yk_previous * xk_contribution
    p_hat_temporal = mi.luminance(mi.unpolarized_spectrum(target_spectrum), yk_previous.wavelengths, valid)
    valid &= dr.isfinite(p_hat_temporal) & (p_hat_temporal > 0.0)

    # 8. Mask invalid outputs
    J = dr.select(valid, J, 0.0)
    p_hat_temporal = dr.select(valid, p_hat_temporal, 0.0)
    target_reconnection_pdf = dr.select(valid, target_reconnection_pdf, 0.0)

    return J, p_hat_temporal, target_reconnection_pdf, valid


def replay_prefix_until_reconnection(scene: mi.Scene, primary_ray: mi.Ray3f, path_seed: mi.UInt32, k: mi.UInt32, max_k: int, active: mi.Mask) -> tuple[mi.SurfaceInteraction3f, mi.Mask]:
    """
    Replay a path prefix from x1 up to x_(k-1).

    Assumes the stored seed reproduces the prefix currently represented by the sample.

    For k=2: returns x1.
    For k=3: returns x2.
    For k=4: returns x3.
    """

    width = dr.width(primary_ray)
    ctx = mi.BSDFContext()

    path_sampler = create_path_sampler(path_seed, width)

    current_ray = mi.Ray3f(primary_ray)
    current_si = scene.ray_intersect(current_ray, ray_flags=mi.RayFlags.All, coherent=False ,active=active)

    valid = active & current_si.is_valid() & (k >= 2) & (k <= max_k)

    # Each iteration advances one vertex.
    # To obtain x_(k-1), advance k-2 times.
    for i in range(max_k - 2):

        advance = valid & (k > i + 2)

        bsdf = current_si.bsdf(current_ray)

        sample1 = path_sampler.next_float(mi.Float, advance)
        sample2 = mi.Point2f(path_sampler.next_float(mi.Float, advance), path_sampler.next_float(mi.Float, advance))

        bs, _ = bsdf.sample(ctx, current_si, sample1, sample2, advance)

        valid_step = advance & (bs.pdf > 0.0)

        next_ray = current_si.spawn_ray(current_si.to_world(bs.wo))
        next_si = scene.ray_intersect(next_ray, ray_flags=mi.RayFlags.All, coherent=False, active=valid_step)

        valid &= ~advance | (valid_step & next_si.is_valid())

        # Update only lanes that need another vertex.
        current_si = dr.select(advance, next_si, current_si)
        current_ray = dr.select(advance, next_ray, current_ray)

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


def compute_shifted_pdf_proxy(scene: mi.Scene, source_primary_ray: mi.Ray3f, target_primary_ray: mi.Ray3f, candidate: Sample, max_k: int, active: mi.Mask) -> tuple[mi.Float, mi.Float, mi.Mask]:

    bsdf_ctx = mi.BSDFContext()

    # 1. Validate candidate
    valid = active & (candidate.k >= 2) & (candidate.k <= max_k) & candidate.surfaceInformation.is_valid()

    # 2. Replay source prefix and reconstruct preserved xk
    source_previous, source_xk, _, _, valid_source = replay_path_to_reconnection(scene=scene, primary_ray=source_primary_ray, path_seed=candidate.seed, k=candidate.k, max_k=max_k, active=valid, candidate=candidate, use_stored_xk=True)
    valid &= valid_source

    # 3. Replay target prefix and reconstruct the same preserved xk
    target_previous, target_xk, _, target_prefix_pdf, valid_target = replay_path_to_reconnection(scene=scene, primary_ray=target_primary_ray, path_seed=candidate.seed, k=candidate.k, max_k=max_k, active=valid, candidate=candidate, use_stored_xk=True)
    valid &= valid_target

    # 4. Validate source and target reconnection compatibility
    valid = shift_mapping_criterions(scene=scene, yk_previous=target_previous, xk_previous=source_previous, xk=target_xk, active=valid)

    # 5. Evaluate source BSDF PDF toward preserved xk
    delta = source_xk.p - source_previous.p
    dist2 = dr.squared_norm(delta)
    valid &= (dist2 > 0.0) & dr.isfinite(dist2)
    safe_dist2 = dr.select(valid, dist2, 1.0)
    wo_world = delta * dr.rsqrt(safe_dist2)
    wo_local = source_previous.to_local(wo_world)
    source_bsdf = source_previous.bsdf()
    source_pdf = source_bsdf.pdf(bsdf_ctx, source_previous, wo_local, valid)
    valid &= (source_pdf > 0.0) & dr.isfinite(source_pdf)


    # 6. Check visibility of the hypothetical source connection
    source_connection_ray = source_previous.spawn_ray_to(source_xk.p)
    source_occluded = scene.ray_test(source_connection_ray, active=valid)
    valid &= ~source_occluded

    # 7. Compute forward Jacobian from source to target
    J = reconnection_jacobian(source_vertex=source_previous.p, target_vertex=target_previous.p, reconnection_vertex=target_xk.p, reconnection_normal=target_xk.n, active=valid)
    valid &= (J > 0.0) & dr.isfinite(J)

    # 8. Transform source connection PDF into target domain
    valid &= (target_prefix_pdf > 0.0) & dr.isfinite(target_prefix_pdf)
    safe_J = dr.select(valid, J, 1.0)
    proxy_pdf = dr.select(valid, source_pdf / safe_J, 0.0)

    # 9. Mask invalid outputs
    valid &= dr.isfinite(proxy_pdf)
    J = dr.select(valid, J, 0.0)
    proxy_pdf = dr.select(valid, proxy_pdf, 0.0)

    return proxy_pdf, J, valid

def replay_path_to_reconnection(scene: mi.Scene, primary_ray: mi.Ray3f, path_seed: mi.UInt32, k: mi.UInt32, max_k: int, active: mi.Mask, x1: mi.SurfaceInteraction3f | None = None, candidate: Sample | None = None, use_stored_xk: bool = False) -> tuple[mi.SurfaceInteraction3f, mi.SurfaceInteraction3f, mi.Spectrum, mi.Float, mi.Mask]:
    
    if max_k < 2:
        raise ValueError("max_k must be >= 2.")

    if use_stored_xk and candidate is None:
        raise ValueError("candidate is required when use_stored_xk=True.")

    width = dr.width(primary_ray)
    bsdf_ctx = mi.BSDFContext()
    path_sampler = create_path_sampler(path_seed, width)
    prefix_throughput = dr.full(mi.Spectrum, 1.0, width)
    prefix_pdf = dr.full(mi.Float, 1.0, width)

    # 1. Initialize x1
    current_ray = mi.Ray3f(primary_ray)

    if x1 is None:
        x1 = scene.ray_intersect(current_ray, ray_flags=mi.RayFlags.All, coherent=False, active=active)

    current_si = x1
    valid = active & current_si.is_valid() & (k >= 2) & (k <= max_k)

    if use_stored_xk:
        valid &= candidate.surfaceInformation.is_valid()

    # 2. Initialize output vertices
    previous_si = dr.zeros(mi.SurfaceInteraction3f, width)
    reconnection_si = dr.zeros(mi.SurfaceInteraction3f, width)
    reached_reconnection = dr.zeros(mi.Bool, width)

    # 3. Replay the path prefix
    for i in range(max_k - 1):

        # Stop before xk when reconstructing a stored candidate
        if use_stored_xk:
            trace = valid & (k > i + 2)
        else:
            trace = valid & (k >= i + 2)

        # Sample the current vertex BSDF
        bsdf = current_si.bsdf(current_ray)
        sample1 = path_sampler.next_float(mi.Float, trace)
        sample2 = mi.Point2f(path_sampler.next_float(mi.Float, trace), path_sampler.next_float(mi.Float, trace))
        bs, bsdf_weight= bsdf.sample(bsdf_ctx, current_si, sample1, sample2, trace)
        bsdf_weight = current_si.to_world_mueller(bsdf_weight, -bs.wo, current_si.wi)

        # Trace the next ray
        valid_sample = trace & (bs.pdf > 0.0) & dr.isfinite(bs.pdf)
        next_ray = current_si.spawn_ray(current_si.to_world(bs.wo))
        next_si = scene.ray_intersect(next_ray, ray_flags=mi.RayFlags.All, coherent=False, active=valid_sample)
        valid_step = valid_sample & next_si.is_valid()

        # Determine whether we reached the requested reconnection vertex
        continue_path = valid_step & (k > i + 2)

        # Accumulate prefix throughput
        prefix_throughput = dr.select(continue_path, prefix_throughput * bsdf_weight, prefix_throughput)
        prefix_pdf = dr.select(continue_path, prefix_pdf * bs.pdf, prefix_pdf)

        if not use_stored_xk:
            reconnect = valid_step & (k == i + 2)
            previous_si = dr.select(reconnect, current_si, previous_si)
            reconnection_si = dr.select(reconnect, next_si, reconnection_si)
            reached_reconnection |= reconnect

        # Update validity
        valid &= ~trace | valid_step

        # Advance only the lanes that still need more vertices
        current_si = dr.select(continue_path, next_si, current_si)
        current_ray = dr.select(continue_path, next_ray, current_ray)

    # 4. Recover the preserved reconnection vertex if requested
    if use_stored_xk:
        previous_si = current_si
        reconnection_si, valid_xk = reconstruct_reconnection_vertex(scene=scene, current_si=previous_si, candidate=candidate, active=valid)
        valid &= valid_xk
    else:
        valid &= reached_reconnection

    # 5. Validate output
    valid &= previous_si.is_valid() & reconnection_si.is_valid()

    return previous_si, reconnection_si, prefix_throughput, prefix_pdf, valid


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

    # This compensates for finalize() dividing by M_total.
    valid_output = active & (M_total > 0.0) & (selected_p_hat > 0.0)
    s.w_sum = dr.select(valid_output, s.w_sum * M_total, 0.0)

    # All 1/M normalization happens in finalize()
    s.finalize(p_hat_selected=selected_p_hat, active=valid_output)

    return s


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