
import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field
from typing import Tuple, Union
from utils import *

dr.set_flag(dr.JitFlag.Debug, True)

class TemporalReuseIntegrator(mi.ad.integrators.common.ADIntegrator):

    # Grid containing the reservoirs of the previous frame
    previousGrid: Reservoir = None
    #temporal_reuse_grid: Reservoir = None
    #spacial_reuse_grid: Reservoir = None
    previous_camera: mi.Sensor = None

    # Previous camera position

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

        # No previous frame exists initially
        self.previousGrid = None

    def store_previous_camera(self, sensor: mi.Sensor, width: int, height: int) -> None:
        """
        Store an independent snapshot of the current camera so that it can be
        used for temporal reprojection during the next frame.

        The sensor passed by the viewer is modified in-place when the camera
        moves, so simply doing

            self.previous_camera = sensor

        would not preserve the previous frame's camera.
        """

        # Create an independent perspective sensor.
        previous_camera = mi.load_dict({
            "type": "perspective",
            "film": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
            }
        })

        # Parameters of the current and previous cameras
        current_params = mi.traverse(sensor)
        previous_params = mi.traverse(previous_camera)

        print("CURRENT SENSOR PARAMS:")
        for key in current_params.keys():
            print("  ", key)

        print("SNAPSHOT SENSOR PARAMS:")
        for key in previous_params.keys():
            print("  ", key)

        # Copy every camera parameter that exists in both sensors.
        for key in previous_params.keys():
            if key in current_params:
                previous_params[key] = current_params[key]

        print("current to_world:",
            sensor.world_transform().matrix)

        print("previous to_world BEFORE:",
            previous_camera.world_transform().matrix)

        previous_params.update()

        print("previous to_world AFTER:",
            previous_camera.world_transform().matrix)

        # Store the independent snapshot for the next frame.
        self.previous_camera = previous_camera

    def sample_rays(
            self,
            scene: mi.Scene,
            sensor: mi.Sensor,
            sampler: mi.Sampler,
        ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
            """
            Sample a 2D grid of primary rays for a given sensor
    
            Returns a tuple containing
    
            - the set of sampled rays
            - a ray weight (usually 1 if the sensor's response function is sampled
              perfectly)
            - the continuous 2D image-space positions associated with each ray
            """
    
            film = sensor.film()
            film_size = film.crop_size()
            rfilter = film.rfilter()
            border_size = rfilter.border_size()
    
            if film.sample_border():
                film_size += 2 * border_size
    
            spp = sampler.sample_count()
    
            # Compute discrete sample position
            idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)
    
            # Try to avoid a division by an unknown constant if we can help it
            log_spp = dr.log2i(spp)
            if 1 << log_spp == spp:
                idx >>= dr.opaque(mi.UInt32, log_spp)
            else:
                idx //= dr.opaque(mi.UInt32, spp)
    
            # Compute the position on the image plane
            pos = mi.Vector2i()
            pos.y = idx // film_size[0]
            pos.x = dr.fma(mi.UInt32(mi.Int32(-film_size[0])), pos.y, idx)
    
            if film.sample_border():
                pos -= border_size
    
            pos += mi.Vector2i(film.crop_offset())
    
            # Cast to floating point and add random offset
            pos_f = mi.Vector2f(pos) + sampler.next_2d()
    
            # Re-scale the position to [0, 1]^2
            scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
            offset = -mi.ScalarVector2f(film.crop_offset()) * scale
            pos_adjusted = dr.fma(pos_f, scale, offset)
    
            aperture_sample = mi.Vector2f(0.0)
            if sensor.needs_aperture_sample():
                aperture_sample = sampler.next_2d()
    
            time = sensor.shutter_open()
            if sensor.shutter_open_time() > 0:
                time += sampler.next_1d() * sensor.shutter_open_time()
    
            wavelength_sample = 0
            if mi.is_spectral:
                wavelength_sample = sampler.next_1d()
    
            with dr.resume_grad():
                ray, weight = sensor.sample_ray_differential(
                    time=time,
                    sample1=wavelength_sample,
                    sample2=pos_adjusted,
                    sample3=aperture_sample
                )
    
            # With box filter, ignore random offset to prevent numerical instabilities
            splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

            pixel_index = pos.x + pos.y * film_size[0]
    
            return ray, weight, splatting_pos, pixel_index

    # Main rendering loop
    def render(self: mi.SamplingIntegrator,
                   scene: mi.Scene,
                   sensor: Union[int, mi.Sensor] = 0,
                   seed: mi.UInt32 = 0,
                   spp: int = 0,
                   develop: bool = True,
                   evaluate: bool = True) -> mi.TensorXf:
            if not develop:
                raise Exception("develop=True must be specified when "
                                "invoking AD integrators")
    
            if isinstance(sensor, int):
                sensor = scene.sensors()[sensor]
    
            film = sensor.film()
    
            # Disable derivatives in all of the following
            with dr.suspend_grad():
                # Prepare the film and sample generator for rendering
                sampler, spp = self.prepare(
                    sensor=sensor,
                    seed=seed,
                    spp=spp,
                    aovs=self.aov_names()
                )

                # Initialize the previousGrid with the right size
                film_size = film.crop_size()
                width  = int(film_size[0])
                height = int(film_size[1])
                pixel_count = width * height

                # Allocate temporal storage only once
                if self.previousGrid is None:
                    self.previousGrid = Reservoir.empty(pixel_count)
    
                # Generate a set of rays starting at the sensor + their discrete pixel indices
                ray, weight, splatting_pos, pixel_index = self.sample_rays(scene, sensor, sampler)
                    
                # Launch the Monte Carlo sampling process in primal mode
                L, valid, current_reservoir, aovs = self.sample(
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    sensor=sensor,
                    film_width=width,
                    film_height=height,
                    active=mi.Bool(True)
                )

                # apply temporal reuse : TODO (separate the classic sampling from the spacial/temporal sampling)

                # Store resulting reservoir for next frame
                scatter_reservoir(self.previousGrid, current_reservoir, pixel_index, valid)

                # Store previous camera (done in viewer.py)
                #self.store_previous_camera(sensor, width, height)
    
                # Prepare an ImageBlock as specified by the film
                block = film.create_block()
    
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)
    
                # Accumulate into the image block
                TemporalReuseIntegrator._splat_to_block(
                    block, film, splatting_pos,
                    value=L * weight,
                    weight=1.0,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )
    
                # Explicitly delete any remaining unused variables
                del sampler, ray, weight, splatting_pos, L, valid
    
                # Perform the weight division and return an image tensor
                film.put_block(block)
    
                return film.develop()
        

    def sample(self, scene : mi.Scene, sensor: mi.Sensor, sampler : mi.Sampler, ray : mi.RayDifferential3f, medium : mi.Medium = None, film_width: int = 0, film_height: int = 0,  active : mi.Mask =True) -> tuple[mi.Spectrum, mi.Mask, Reservoir, list]:
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
        current_reservoir = Reservoir.empty(dr.width(ray))
        bsdf = si.bsdf(ray)
        bsdf_ctx = mi.BSDFContext()

        # Add the Emitter Samples inside the reservoir
        for i in range(self.emitter_samples):
            # Sample an emitter point in the scene, we set visibility check to false because our target function is p_hat = BRDF(x) * L_e(x) * G(x)
            # ds : direction sample
            # emitter_weight : the pixel value at the emitter divided by the emitter sampling pdf : L_e(y) / pdf_emitter(y)
            ds, emitter_weight = scene.sample_emitter_direction(si, sampler.next_2d(active), False, active)

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
            # emitter_weight contains the emitter contribution divided by
            # the emitter-direction sampling PDF ds.pdf, so the candidate weight w_xi is already included in emitter_weight
            candidate_weight_spectrum = bsdf_val * emitter_weight
            p_hat_and_wxi = mi.luminance(mi.unpolarized_spectrum(candidate_weight_spectrum), si.wavelengths, active_e); 

            # Construct the candidate structure for the reservoir sampling step
            candidate = Candidate(ds, ds.pdf) 


            # Compute the reservoir weight : w_i = mis_area * p_hat * w_xi (we must divide by the number of samples for the emitter case)
            reservoir_weight = mis_area * p_hat_and_wxi / self.emitter_samples

            # Add the candidate to the reservoir
            # The added random number is needed to detemine if this candidate succesfully replaces the currently selected candidate or not
            current_reservoir.add_sample(candidate= candidate, reservoir_weight= reservoir_weight, rnd1D= sampler.next_1d(active_e), active= active_e)

        # Add the BSDF Samples inside the reservoir
        for i in range(self.bsdf_samples):
            # Sample a bsdf point in the scene from the intersection point si
            # bs : bsdf sample obtained (not directly a direction sample!)
            # bsdf_weight : the bsdf evaluation divided by the probability of sampling that value : bs_val = BSDF(x) = p_bsdf(x)
            # /!\ : Those results are returned in local coordinates
            bs, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)

            # Convert from local BSDF coordinates to world coordinates and apply Mueller matrix transformation
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bs.wo, si.wi)

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
            # Remark : bsdf_weight is already divided by bsdf_pdf, so the weight of the candidate sample w_xi is already included in bsdf_val
            candidate_weight_spectrum = bsdf_weight * emitter_val

            # Since for the reservoir weight we need a float value, we compute the luminance of the candidate weight spectrum to get a single float value
            p_hat_and_wxi = mi.luminance(mi.unpolarized_spectrum(candidate_weight_spectrum), si.wavelengths, active_b)

            # Construct the candidate for the reservoir
            candidate = Candidate(ds, bs.pdf)

            # Compute the reservoir weight : w_i = mis_area * p_hat * w_xi (we must divide by the number of samples for the bsdf case)
            reservoir_weight = mis_bsdf * p_hat_and_wxi / self.bsdf_samples

            # Add the candidate to the reservoir
            # The added random number is needed to detemine if this candidate succesfully replaces the currently selected candidate or not
            current_reservoir.add_sample(candidate=candidate, reservoir_weight=reservoir_weight, rnd1D= sampler.next_1d(active_b), active= active_b)

        # Evaluate p_hat_(current si)
        p_hat_current = eval_p_hat(scene, si, current_reservoir.select, bsdf, bsdf_ctx, sampler, active)

        # Evaluate the contribution weight W of the current reservoir
        current_reservoir.finalize(p_hat_current, active)

        ## Temportal Reuse ##
        if self.previous_camera is not None:

            # 1) Determine which temporal sample corresponds to our current pixel
            previous_index, valid_reprojection = eval_previous_index(si, self.previous_camera, sampler, film_width, film_height, active)
            post_cam_move_previous_reservoir = gather_reservoir(self.previousGrid, previous_index, valid_reprojection)

            # 2) Merge this reservoir with the current reservoir
            current_reservoir = combine_reservoirs(scene, si, bsdf, bsdf_ctx, sampler, active, current_reservoir, post_cam_move_previous_reservoir)


        print("TESSSST")

        #######################

        ## Get the selected candidate ##
        selected_candidate = current_reservoir.select

        # To get the integral estimator, we need to compute I = f(X) * W_X
        p_hat_X_spectrum = eval_p_hat_spectrum(scene, si, selected_candidate, bsdf, bsdf_ctx, sampler, active)
        p_hat = eval_p_hat(scene, si, selected_candidate, bsdf, bsdf_ctx, sampler, active)

        # Add the visibility check to the selected candidate to determine f(X) = p_hat(X) * V(X)
        f_X = mi.Spectrum(0.)
        
        # Trace a ray from the original intersection point to the selected candidate point and check for visibility
        shadow_ray = si.spawn_ray_to(selected_candidate.direction_sample.p)

        # Check if we have an occlusion
        occluded = scene.ray_test(shadow_ray, active)
        visible = ~occluded

        # Get the evaluation of the selected candidiate : f(X) = p_hat(X) * Visibility(X) 
        f_X = dr.select(visible, p_hat_X_spectrum, 0.)

        # Valid candidates must have striclty positive importance, otherwise they are considered invalid
        active_r = p_hat > 0.

        # the result is the product of the selected candidate's evaluation of the actual function f(x) and its contribution weight W_x
        result += dr.select(active_r, f_X * current_reservoir.W, 0.)
        

        return result, valid_ray, current_reservoir, []


    def to_string(self):
        return "TemporalReuseIntegrator[]"


# Register the integrator as a Mitsuba plugin
mi.register_integrator(
    "temp_reuse",
    lambda props: TemporalReuseIntegrator(props)
)