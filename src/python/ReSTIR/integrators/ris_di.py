import mitsuba as mi
import drjit as dr
from dataclasses import dataclass

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


@dataclass
class Candidate:
    # direction sample of this candidate
    direction_sample: mi.DirectionSample3f

    # radiance value of the selected sample
    L: mi.Spectrum

    # validity of the candidate
    valid: mi.Mask
 

@dataclass
class Reservoir:
    # selected sample
    select: Candidate

    # sum of all reservoir weights
    w_sum: mi.Float

    # total number of samples
    M: mi.Float

    # contribution weight associated with the reservoir's selected candidate
    W: mi.Float 

    # Adding a sample to the reservoir
    def add_sample(self, candidate: Candidate, reservoir_weight: mi.Float, rnd1D: mi.Float, active: mi.Mask):
        # Get the mask for currently active Jit lanes with valid weights
        active_candidate = active & candidate.valid & (reservoir_weight > 0)

        # Increment the valid candidate number counter for all valid lanes
        self.M = dr.select(active_candidate, self.M + 1, self.M)

        # Update the total weight of all valid candidates seen so far
        self.w_sum = dr.select(active_candidate, self.w_sum + reservoir_weight, self.w_sum)

        # Replace the candidate only for elements that are active and succeed the random test
        replace = active_candidate & (rnd1D < reservoir_weight/self.w_sum)

        # Replace the selected candidate with the new candidate
        self.select = dr.select(replace, candidate, self.select)

class DirectRISIntegrator(mi.SamplingIntegrator):

    def __init__(self, props):
        super().__init__(props)

        # Cannot specify shading_samples together with
        # emitter_samples and/or bsdf_samples
        # if (props.has_property("shading_samples") and (props.has_property("emitter_samples") or props.has_property("bsdf_samples"))):
        #     raise ValueError("Cannot specify both 'shading_samples' and ('emitter_samples' and/or 'bsdf_samples').")

        # Number of shading samples -- shorthand for setting both
        # emitter_samples and bsdf_samples
        shading_samples = props.get("shading_samples", 1)

        # Number of samples using emitter sampling
        self.emitter_samples = props.get(
            "emitter_samples",
            shading_samples
        )

        # Number of samples using BSDF sampling
        self.bsdf_samples = props.get(
            "bsdf_samples",
            shading_samples
        )

        # At least one sampling strategy must be enabled
        if self.emitter_samples + self.bsdf_samples == 0:
            raise ValueError(
                "Must have at least 1 BSDF or emitter sample!"
            )
        

    def sample(self, scene : mi.Scene, sampler : mi.Sampler, ray : mi.RayDifferential3f, medium : mi.Medium =None, active : mi.Mask =True) -> tuple[mi.Spectrum, mi.Mask, list]:
        """
        Main rendering routine for one vectorized set of rays.

        Returns:
            radiance, valid, aovs
        """

        # Get the current intersection with the scene
        si = scene.ray_intersect(ray, ray_flags = mi.RayFlags.All, coherent = True, active = active)

        # We can perform correct computations only on rays that intersected with the scene
        valid_ray = active & si.is_valid()
        active &= si.is_valid()

        # Returned pixel color (we accumulate on it)
        result = mi.Spectrum(0.)

        # Add directly visible emission
        emitter = si.emitter(scene, active)
        result += emitter.eval(si, active)

        # Instantiate the reservoir and the bsdf (needed in both MIS cases)
        laneNb = dr.width(si.p)
        empty_ds = dr.zeros(mi.DirectionSample3f, laneNb)
        
        r = Reservoir(Candidate(empty_ds, mi.Spectrum(0.0), mi.Mask(False)), mi.Float(0.0), mi.Float(0.0), mi.Float(0.0))
        bsdf = si.bsdf(ray)
        bsdf_ctx = mi.BSDFContext()

        # Add the Emitter Samples inside the reservoir
        for i in range(self.emitter_samples):
            # Sample an emitter point in the scene, we set visibility check to false because our target function is p_hat = BRDF(x) * L_e(x) * G(x)
            # ds : direction sample
            # emitter_val : the pixel value at the emitter divided by the emitter sampling pdf : L_e(y) / pdf_emitter(y)
            ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(active), False, active)

            # We remove from the valid computations the ones with zero probability of sampling this emitter point
            active_e = active & (ds.pdf > 0.)

            ## For the MIS weights, we need also need the probability of sampling that same point with BSDF sampling 
            wo = si.to_local(ds.d) # BSDF computations in Mitsuba are done in local coordinates

            # Evaluate the BSDF at the intersection point for the sampled direction
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_e)

            # Convert back to world coordinates
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            # Compute the MIS weight for the area sampling case, BSDF sampling cannot sample that direction from a delta surface (because it is impossible that it will match the perfect reflection direction, so p_bsdf(x) = 0)
            # We have the BSDF sampling pdf to 0 for delta surfaces and therefore mis_area = p_emitter^2 / (p_emitter^2 + 0^2) = 1
            emitter_pdf = ds.pdf
            mis_area = dr.select(ds.delta, 1., mis_weight(emitter_pdf, bsdf_pdf, self.emitter_samples, self.bsdf_samples))

            # Compute the evaluation of the target function "p_hat" at this candidate point, which is p_hat = BRDF(x) * L_e(x) * G(x)
            # emitter_val contains the emitter contribution divided by
            # the emitter-direction sampling PDF ds.pdf, so the candidate weight w_xi is already included in emitter_val
            candidate_weight_spectrum = bsdf_val * emitter_val
            p_hat_and_wxi = mi.luminance(mi.unpolarized_spectrum(candidate_weight_spectrum), si.wavelengths, active_e); 

            # Construct the candidate structure for the reservoir sampling step
            candidate = Candidate(ds, candidate_weight_spectrum, active_e) 

            # Compute the reservoir weight : w_i = mis_area * p_hat * w_xi (we must divide by the number of samples for the emitter case)
            reservoir_weight = mis_area * p_hat_and_wxi / self.emitter_samples

            # Add the candidate to the reservoir
            # The added random number is needed to detemine if this candidate succesfully replaces the currently selected candidate or not
            r.add_sample(candidate= candidate, reservoir_weight= reservoir_weight, rnd1D= sampler.next_1d(active_e), active= active_e)

        # Add the BSDF Samples inside the reservoir
        for i in range(self.bsdf_samples):
            # Sample a bsdf point in the scene from the intersection point si
            # bs : bsdf sample obtained (not directly a direction sample!)
            # bs_val : the bsdf evaluation divided by the probability of sampling that value : bs_val = BSDF(x) = p_bsdf(x)
            # /!\ : Those results are returned in local coordinates
            bs, bsdf_val = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)

            # Convert from local BSDF coordinates to world coordinates and apply Mueller matrix transformation
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi)

            # We remove from the valid computations the Jit lines that have a zero probability of sampling this direction
            active_b = active & (bs.pdf != 0.) 


            # Trace the ray in the sampled direction and intersect against the scene
            wo_world = si.to_world(bs.wo)
            si_new = scene.ray_intersect(si.spawn_ray(wo_world), active_b)

            # Check if the ray hit an emitter, and if so, keep them as acive in the Jit SIMD lane
            emitter = si_new.emitter(scene, active_b)
            emitter_v = emitter.eval(si_new, active_b)
            active_b &= dr.any(emitter_v != 0, axis=0)

            
            # To compute the multiple importance sampling weight, we need to compute the probability of sampling this same direction using emitter sampling
            # Since we had an intersection with an emitter, we evaluate the emitter contribution at this point (to avoid repetitive computations)
            emitter_val = emitter.eval(si_new, active_b)

            # Create a direction sample from the original intersection point to the bsdf intersection point (that is on an emitter)
            ds = mi.DirectionSample3f(scene, si_new, si)

            # Determine probability of having sampled that same direction using Emitter sampling. 
            emitter_pdf = scene.pdf_emitter_direction(si, ds, active_b)

            # We want to set the emitter contribution pdf to 0 for delta BSDFs, since they cannot be sampled by the emitter sampling technique
            delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            emitter_pdf = dr.select(delta, 0., emitter_pdf)

            # Compute the MIS weight for the BSDF sampling case
            bsdf_pdf = bs.pdf
            mis_bsdf = mis_weight(bsdf_pdf, emitter_pdf, self.bsdf_samples, self.emitter_samples)

            # Compute the evaluation of the target function at this candidate point, which is p_hat = BRDF(x) * L_e(x) * G(x)
            # Remark : bsdf_val is already divided by bsdf_pdf, so the weight of the candidate sample w_xi is already included in bsdf_val
            candidate_weight_spectrum = bsdf_val * emitter_val

            # Since for the reservoir weight we need a float value, we compute the luminance of the candidate weight spectrum to get a single float value
            p_hat_and_wxi = mi.luminance(mi.unpolarized_spectrum(candidate_weight_spectrum), si.wavelengths, active_b)

            # Construct the candidate for the reservoir
            candidate = Candidate(ds, candidate_weight_spectrum, active_b)

            # Compute the reservoir weight : w_i = mis_area * p_hat * w_xi (we must divide by the number of samples for the bsdf case)
            reservoir_weight = mis_bsdf * p_hat_and_wxi / self.bsdf_samples

            # Add the candidate to the reservoir
            # The added random number is needed to detemine if this candidate succesfully replaces the currently selected candidate or not
            r.add_sample(candidate=candidate, reservoir_weight=reservoir_weight, rnd1D= sampler.next_1d(active_b), active= active_b)


        # Get the selected candidate 
        selected_candidate = r.select

        # To get the integral estimator, we need to compute I = f(X) * W_X
        p_hat_X = selected_candidate.L

        # Add the visibility check to the selected candidate to determine f(X) = p_hat(X) * V(X)
        f_X = mi.Spectrum(0.)

        # Trace a ray from the original intersection point to the selected candidate point and check for visibility
        shadow_ray = si.spawn_ray_to(selected_candidate.direction_sample.p)

        # Check if we have an occlusion
        occluded = scene.ray_test(shadow_ray, selected_candidate.valid)
        visible = selected_candidate.valid & ~occluded

        # Set the value to 0 if we have an occlusion
        f_X = dr.select(visible, selected_candidate.L, 0.)

        # Get the evaluation of the selected candidiate : f(X) = p_hat(X) * Visibility(X) 
        selected_importance = mi.luminance(mi.unpolarized_spectrum(selected_candidate.L), si.wavelengths, selected_candidate.valid)

        # Valid candidates must have striclty positive importance, otherwise they are considered invalid
        active_r = selected_candidate.valid & (selected_importance > 0.)

        # Compute the contribution weight W_x = w_sum / p_hat(X)
        W_x = dr.select(active_r, r.w_sum / dr.select(active_r, selected_importance, 1.), 0.)

        # the result is the product of the selected candidate's evaluation of the actual function f(x) and its contribution weight W_x
        result += dr.select(active_r, f_X * W_x, 0.)


        return result, valid_ray, []


    def to_string(self):
        return "DirectRISIntegrator[]"


# Register the integrator as a Mitsuba plugin
mi.register_integrator(
    "direct_ris",
    lambda props: DirectRISIntegrator(props)
)