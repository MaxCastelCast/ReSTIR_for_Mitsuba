import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field

mi.set_variant("llvm_ad_rgb")
dr.set_flag(dr.JitFlag.Debug, True)


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float, n_a: float, n_b: float, power: int = 2) -> mi.Float:
    """Compute the MIS weight using the power heuristic."""

    a = n_a * pdf_a
    b = n_b * pdf_b

    # Dr.Jit convert automaticaly the ** operator to dr.power
    a = a ** power
    b = b ** power
    
    w = a / (a + b)

    return dr.select(dr.isfinite(w), w, 0.0)

from dataclasses import dataclass, field

@dataclass
class Candidate:
    # Direction sample of this candidate
    direction_sample: mi.DirectionSample3f = field(default_factory=mi.DirectionSample3f)

    # Original pdf of that sample
    pdf: mi.Float = field(default_factory=lambda: mi.Float(0.0))
 

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
            pdf=dr.zeros(mi.Float,  width)
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
    ds = candidate.direction_sample

    # Candidate validity
    active_p = active & (candidate.pdf > 0.)

    # /!\ : The current ds.d is in the previous sample space, and not in the current sample space, so we 
    #       must convert the direction from "ds.p - previous_si.p" to "ds.p - current_si.p"
    d = dr.normalize(ds.p - current_si.p)

    # Determine the bsdf of that direction (those results are given in local coordinates)
    wo = current_si.to_local(d)
    bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, current_si, wo, active_p)

    # Convert from local BSDF coordinates to world coordinates and apply Mueller matrix transformation
    bsdf_val = current_si.to_world_mueller(bsdf_val, -wo, current_si.wi)

    emitter_val = ds.emitter.eval_direction(current_si, ds, active_p)

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
def combine_reservoirs(scene: mi.Scene, current_si: mi.SurfaceInteraction3f, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, sampler: mi.Sampler, active: mi.Mask, *reservoirs : Reservoir) -> Reservoir:
    # Initialize the return reservoire
    width = dr.width(current_si.p)
    s = Reservoir.empty(width)
    M_total = dr.zeros(mi.Float, width)

    # Combine the reservoirs by adding their selected candidates and the associate reservoir weight
    for r in reservoirs:

        # Reservoir exists only where it contains samples
        active_r = active & (r.M > 0)

        # We need to evaluate p_hat for the current pixel/surface_interaction so that the candidates can be used for the current context 
        p_hat = eval_p_hat(scene, current_si, r.select, bsdf, bsdf_ctx, sampler, active_r)

        # Compute their reservoir weight which is r_w = r.w_sum * (p_hat_current / p_hat_previous) : determine how this weight is important in this context compared to its original one
        reservoir_weight = p_hat * r.W * r.M

        # Add this candidate in the new reservoir
        s.add_sample(r.select, reservoir_weight, sampler.next_1d(active_r), active_r)

        # Add the nb of samples of this reservoir to our combined reservoir nb
        M_total += r.M

    # Set the total number of samples for the combined reservoir
    s.M = M_total

    # Compute the new contribution weight for the selected sample of the combined reservoir
    p_hat_select = eval_p_hat(scene, current_si, s.select, bsdf, bsdf_ctx, sampler, active_r)
    active_s = active_r & (s.M > 0) & (p_hat_select > 0)
    s.W = dr.select(active_s, s.w_sum/(s.M * dr.select(active_s, p_hat_select, 1.0)), 0.0)

    return s    

# Gathers the reservoirs associated with the indexes
def gather_reservoir(r_set: Reservoir, index: mi.UInt32) -> Reservoir:

    direction_sample_set = dr.gather(mi.DirectionSample3f, r_set.select.direction_sample, index)
    pdf_set = dr.gather(mi.Float, r_set.select.pdf, index)

    candidate_set = Candidate(direction_sample=direction_sample_set, pdf=pdf_set)
    w_sum_set = dr.gather(mi.Float, r_set.w_sum, index)
    M_set = dr.gather(mi.Float, r_set.M, index)
    W_set = dr.gather(mi.Float, r_set.W, index)

    return Reservoir(select=candidate_set, w_sum=w_sum_set, W=W_set, M=M_set)

# Store resulting reservoir for next frame
# dr.scatter() modifies its target in place
def scatter_reservoir(previous: Reservoir, current: Reservoir, index: mi.UInt32):

    dr.scatter(previous.select.direction_sample, current.select.direction_sample, index)
    dr.scatter(previous.select.pdf, current.select.pdf, index)

    dr.scatter(previous.w_sum, current.w_sum, index)
    dr.scatter(previous.M, current.M, index)
    dr.scatter(previous.W, current.W, index)

