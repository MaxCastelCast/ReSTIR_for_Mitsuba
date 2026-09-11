#include <mitsuba/render/integrator.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-direct:

Resampled Importance Sampling Direct Illumination Integrator (:monosp:`direct_ris`)
-------------------------------------------------

.. pluginparameters::

 * - shading_samples
   - |int|
   - This convenience parameter can be used to set both :code:`emitter_samples` and
     :code:`bsdf_samples` at the same time.

 * - emitter_samples
   - |int|
   - Optional more fine-grained parameter: specifies the number of samples that should be generated
     using the direct illumination strategies implemented by the scene's emitters.
     (Default: set to the value of :monosp:`shading_samples`)

 * - bsdf_samples
   - |int|
   - Optional more fine-grained parameter: specifies the number of samples that should be generated
     using the BSDF sampling strategies implemented by the scene's surfaces.
     (Default: set to the value of :monosp:`shading_samples`)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters.
     (Default: no, i.e. |false|)

 */


template <typename Float, typename Spectrum>
class DirectRISIntegrator : public SamplingIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    DirectRISIntegrator(const Properties &props) : Base(props) {
        if (props.has_property("shading_samples")
            && (props.has_property("emitter_samples") ||
                props.has_property("bsdf_samples"))) {
            Throw("Cannot specify both 'shading_samples' and"
                  " ('emitter_samples' and/or 'bsdf_samples').");
        }

        /// Number of shading samples -- this parameter is a shorthand notation
        /// to set both 'emitter_samples' and 'bsdf_samples' at the same time
        size_t shading_samples = props.get<size_t>("shading_samples", 1);

        /// Number of samples to take using the emitter sampling technique
        m_emitter_samples = props.get<size_t>("emitter_samples", shading_samples);

        /// Number of samples to take using the BSDF sampling technique
        m_bsdf_samples = props.get<size_t>("bsdf_samples", shading_samples);

        if (m_emitter_samples + m_bsdf_samples == 0)
            Throw("Must have at least 1 BSDF or emitter sample!");

    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &ray, const Medium * /* medium */, Float * /* aovs */, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // ----------------------- Candidate structure -----------------------
        // The Candidate structure needed for reservoir sampling
        struct Candidate {
            // The sampled direction and associated information
            DirectionSample3f direction_sample;
            // Inverse proposal density 1 / q(x) associated with this candidate
            Float w = 0.f;
            // the radiance value of that candidate
            Spectrum L = 0.f;
            // Validity mask of this candidate for Dr.Jit
            Mask valid = false;

            DRJIT_STRUCT(Candidate, direction_sample, w, L, valid)
        };

        // ----------------------- Reservoir structure -----------------------
        // The Reservoir structure needed for reservoir sampling
        struct Reservoir {
            // Currently selected candidate
            Candidate selected;
            // Total weight of all valid candidates seen so far
            Float w_sum = 0.f;


            // Add a candidate to the reservoir using its unnormalized RIS weight.
            // The candidate replaces the current reservoir sample with probability
            // selection_weight / w_sum.
            //  candidate: the new candidate to be added
            //  selection_weight: unnormalized RIS/reservoir weight of this candidate
            //  rnd1D: a random number in [0,1) used to decide whether to replace the current selected candidate with the new candidate
            //  active: the validity mask of this candidate for Dr.Jit from the integrator's perspective

            void add_sample(const Candidate &candidate, Float selection_weight, Float rnd1D, Mask active) {

                // Apply the reservoir sampling algorithm only on active candidates
                Mask active_candidate = active && candidate.valid && selection_weight > 0.f;

                // Update the total weight of all valid candidates seen so far
                dr::masked(w_sum, active_candidate) += selection_weight;

                // We replace the candidate only for elements that are active and succeed the random test
                Mask replace = active_candidate && (rnd1D < selection_weight/w_sum);

                // Replace the selected candidate with the new candidate
                selected = dr::select(replace, candidate, selected);
            }
        };

        // ----------------------- Intersection Point -----------------------
        // Intersection point
        SurfaceInteraction3f si = scene->ray_intersect(ray, +RayFlags::All, /* coherent = */ true, active);

        // Resulting radiance value (accumulated over all samples)
        Spectrum result(0.f);

        // ----------------------- Visible emitters -----------------------

        if (m_hide_emitters) {
            // Skip all area emitters along this ray
            Mask skip_emitters = si.is_valid() && (si.shape->emitter() != nullptr) && active;

            if (dr::any_or<true>(skip_emitters)) {
                Ray3f ray_skip = si.spawn_ray(ray.d);
                PreliminaryIntersection3f pi = Base::skip_area_emitters(scene, ray_skip, true, skip_emitters);
                SurfaceInteraction3f si_after_skip = pi.compute_surface_interaction(ray, +RayFlags::All, skip_emitters);
                dr::masked(si, skip_emitters) = si_after_skip;
            }
        } else {
            EmitterPtr emitter_vis = si.emitter(scene, active);
            if (dr::any_or<true>(emitter_vis != nullptr))
                result += emitter_vis->eval(si, active);
        }

        Mask valid_ray = active && si.is_valid();

        active &= si.is_valid();
        if (dr::none_or<false>(active))
            return { result, valid_ray };

        // ----------------------- Select RIS candidates : Emitter sampling -----------------------

        Reservoir r;

        BSDFContext ctx;
        BSDFPtr bsdf = si.bsdf(ray);
        auto flags = bsdf->flags();
        Mask sample_emitter = active && has_flag(flags, BSDFFlags::Smooth);

        for(int i = 0; i < m_emitter_samples; ++i){
            // Initialize the Mask for active Jit lines, the direction sampled and the
            // emitter_val which is the emitter contribution divided by
            // the emitter-direction sampling PDF ds.pdf
            Mask active_e = sample_emitter;
            DirectionSample3f ds;
            Spectrum emitter_val;

            // Sample an emitter point in the scene, we set visibility check to false because our target function is p_hat = BRDF(x) * L_e(x) * G(x)
            std::tie(ds, emitter_val) = scene->sample_emitter_direction(si, sampler->next_2d(sample_emitter), false, sample_emitter);

            // We remove from the valid computations the Jit lines that have a zero probability of sampling this emitter point
            active_e &= ds.pdf != 0.f;

            // If all the Jit lines are invalid, we skip this iteration
            if (dr::none_or<false>(active_e))
                continue;
            
            // To compute the multiple importance sampling weight, we need to compute the probability of sampling this same direction using BSDF sampling
            Vector3f wo = si.to_local(ds.d);
            Spectrum bsdf_val;
            Float bsdf_pdf;
            std::tie(bsdf_val, bsdf_pdf) = bsdf->eval_pdf(ctx, si, wo, active_e);

            // Convert from local BSDF coordinates to world coordinates and apply Mueller matrix transformation
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

            // Compute the MIS weight for the area sampling case
            Float emitter_pdf = ds.pdf;
            Float mis_area = mis_weight(emitter_pdf, bsdf_pdf, m_emitter_samples, m_bsdf_samples, 1);

            // Compute the weight of that candidate sample 
            Float w_xi = 1 / ds.pdf;

            // Compute the evaluation of the target function at this candidate point, which is p_hat = BRDF(x) * L_e(x) * G(x)
            // emitter_val contains the emitter contribution divided by
            // the emitter-direction sampling PDF ds.pdf, so the candidate weight w_xi is already included in emitter_val
            Spectrum candidate_weight_spectrum = bsdf_val * emitter_val;
            Float p_hat_and_wxi = mitsuba::luminance(unpolarized_spectrum(candidate_weight_spectrum), si.wavelengths, active_e);    
        

            // Construct the candidate structure for the reservoir sampling step
            Candidate candidate;
            candidate.direction_sample = ds;
            candidate.w = w_xi;
            candidate.L = candidate_weight_spectrum;
            candidate.valid = active_e;

            // Construct the reservoir weight (we must divide by the number of samples for the emitter case)
            Float reservoir_weight = mis_area * p_hat_and_wxi / m_emitter_samples;

            // Add the candidate to the reservoir
            r.add_sample(candidate, reservoir_weight, sampler->next_1d(active_e), active_e);
        }

        // ------------------------ Select RIS candidates : BSDF sampling -------------------------

        for(int i = 0; i < m_bsdf_samples; ++i){
            // Sample a direction using the BSDF sampling technique
            Mask active_b = active;
            BSDFSample3f bs;
            Spectrum bsdf_val;
            // The first random number chooses between reflection an refraction, the pair of random numbers is used to sample the direction
            std::tie(bs, bsdf_val) = bsdf->sample(ctx, si, sampler->next_1d(active_b), sampler->next_2d(active_b), active_b);

            // Convert from local BSDF coordinates to world coordinates and apply Mueller matrix transformation
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            // We remove from the valid computations the Jit lines that have a zero probability of sampling this direction
            active_b = active_b && bs.pdf != 0.f && dr::any(unpolarized_spectrum(bsdf_val) != 0.f);

            // If all the Jit lines are invalid, we skip this iteration
            if (dr::none_or<false>(active_b))
                continue;

            // Trace the ray in the sampled direction and intersect against the scene
            Vector3f wo_world = si.to_world(bs.wo);
            SurfaceInteraction3f si_bsdf = scene->ray_intersect(si.spawn_ray(wo_world), active_b);

            // Check if the ray hit an emitter, and if so, keep them as acive in the Jit SIMD lane
            EmitterPtr emitter = si_bsdf.emitter(scene, active_b);
            active_b &= (emitter != nullptr);

            if (dr::any_or<true>(active_b)) {

                // To compute the multiple importance sampling weight, we need to compute the probability of sampling this same direction using emitter sampling
                // Since we had an intersection with an emitter, we evaluate the emitter contribution at this point (to avoid repetitive computations)
                Spectrum emitter_val = emitter->eval(si_bsdf, active_b);

                // Create a direction sample from the original intersection point to the bsdf intersection point (that is on an emitter)
                DirectionSample3f ds(scene, si_bsdf, si);

                // Determine probability of having sampled that same direction using Emitter sampling. 
                Float emitter_pdf = scene->pdf_emitter_direction(si, ds, active_b);

                // We want to set the emitter contribution pdf to 0 for delta BSDFs, since they cannot be sampled by the emitter sampling technique
                Mask delta = has_flag(bs.sampled_type, BSDFFlags::Delta);
                emitter_pdf = dr::select(delta, 0.f, emitter_pdf);

                // Compute the MIS weight for the BSDF sampling case
                Float bsdf_pdf = bs.pdf;
                Float mis_bsdf = mis_weight(bsdf_pdf, emitter_pdf, m_bsdf_samples, m_emitter_samples, 1);

                // Compute the weight of that candidate sample
                Float w_xi = 1 / bs.pdf;

                // Compute the evaluation of the target function at this candidate point, which is p_hat = BRDF(x) * L_e(x) * G(x)
                // Remark : bsdf_val is already divided by bsdf_pdf, so the weight of the candidate sample w_xi is already included in bsdf_val
                Spectrum candidate_weight_spectrum = bsdf_val * emitter_val;

                // Since for the reservoir weight we need a float value, we compute the luminance of the candidate weight spectrum to get a single float value
                Float p_hat_and_wxi = mitsuba::luminance( unpolarized_spectrum(candidate_weight_spectrum), si.wavelengths, active_b);

                // Construct the candidate structure for the reservoir sampling step
                Candidate candidate;
                candidate.direction_sample = ds;
                candidate.w = w_xi;
                candidate.L = candidate_weight_spectrum;
                candidate.valid = active_b;

                // Construct the reservoir weight (we must divide by the number of samples for the BSDF case)
                Float reservoir_weight = mis_bsdf * p_hat_and_wxi / m_bsdf_samples;

                // Add the candidate to the reservoir
                r.add_sample(candidate, reservoir_weight, sampler->next_1d(active_b), active_b);
            }
            
        }

        // ------------------------ Get the selected candidate -------------------------

        // Get the selected candidate from the reservoir
        Candidate selected_candidate = r.selected;

        // To get the estimator, we need to compute I = f(X) * W_X
        Spectrum f_x = selected_candidate.L;

        // Add the visibility check to the selected candidate to determine f(X) = p_hat(X) * V(X)
        if (dr::any_or<true>(selected_candidate.valid)) {
            // Trace a ray from the original intersection point to the selected candidate point and check for visibility
            Ray3f shadow_ray = si.spawn_ray_to(selected_candidate.direction_sample.p);
            SurfaceInteraction3f shadow_pi = scene->ray_intersect(shadow_ray, +RayFlags::All, true, selected_candidate.valid);
            Mask visible = selected_candidate.valid && !shadow_pi.is_valid();
            f_x = dr::select(visible, selected_candidate.L, 0.f);
        }

        // Get the evaluation of the selected candidiate : f(X) = p_hat(X) * Visibility(X) 
        Float selected_importance = mitsuba::luminance(unpolarized_spectrum(selected_candidate.L), si.wavelengths, selected_candidate.valid);
        
        // Valid candidates must have striclty positive importance, otherwise they are considered invalid
        Mask active_r = selected_candidate.valid && selected_importance > 0.f;

        // Compute the contribution weight W_x = w_sum / q(x) where q(x) is the proposal distribution of the selected candidate
        Float W_x = dr::select(active_r, r.w_sum / dr::select(active_r, selected_importance, 1.f), 0.f);

        // the result should is the product of the selected candidate's evaluation of the actual funciton f(x) (which is the p_hat function with the visibility check) and its contribution weight W_x
        result += dr::select(active_r, f_x * W_x, 0.f);

        return { result, valid_ray };
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DirectIntegrator[" << std::endl
            << "  emitter_samples = " << m_emitter_samples << "," << std::endl
            << "  bsdf_samples = " << m_bsdf_samples << std::endl
            << "]";
        return oss.str();
    }

    Float mis_weight(Float pdf_a, Float pdf_b, ScalarFloat n_a, ScalarFloat n_b, int pow) const {         
        Float a = n_a * pdf_a;
        Float b = n_b * pdf_b;

        // a = std::pow(a, pow);
        // b = std::pow(b, pow);

        Float w = a / (a + b);
        return dr::select(dr::isfinite(w), w, 0.f);
    }

    MI_DECLARE_CLASS(DirectRISIntegrator)
private:
    size_t m_emitter_samples;
    size_t m_bsdf_samples;
    ScalarFloat m_frac_bsdf, m_frac_lum;
    ScalarFloat m_weight_bsdf, m_weight_lum;

    MI_TRAVERSE_CB(Base, m_frac_bsdf, m_frac_lum, m_weight_bsdf, m_weight_lum)
};

MI_EXPORT_PLUGIN(DirectRISIntegrator)
NAMESPACE_END(mitsuba)
