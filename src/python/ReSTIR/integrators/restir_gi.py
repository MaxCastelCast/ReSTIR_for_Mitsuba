
import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field
from typing import Tuple, Union
from utilsGI import *

POSITION_THRESHOLD = 0.05
NORMAL_THRESHOLD = 0.9

SPATIAL_NEIGHBOR_COUNT = 3
SPATIAL_NEIGHBOR_RADIUS = 10

class Restir_Gi_Integrator(mi.ad.integrators.common.ADIntegrator):

    # Grid containing the reservoirs of the previous frame
    previousGrid: Reservoir = None
    currentGrid: Reservoir = None
    previous_camera: mi.Sensor = None

    # Number of ReSTIR GI proposals per primary sample
    initial_candidates: int = 1

    def __init__(self, props):
        super().__init__(props)

        self.initial_candidates = props.get("initial_candidates", 1)
        

    def prepare_frame(self, sensor: mi.Sensor) -> None:
        if self.previous_camera is not None:
            return

        if not isinstance(sensor, mi.ProjectiveCamera):
            raise TypeError(
                f"ReSTIR DI requires a ProjectiveCamera, got {type(sensor)}"
            )

        film = sensor.film()
        film_size = film.crop_size()

        # Convert the JIT transform to a scalar CPU transform
        to_world = mi.ScalarTransform4f(
            dr.slice(sensor.world_transform().matrix, 0)
        )

        self.previous_camera = mi.load_dict({
            "type": "perspective",
            "to_world": to_world,
            "film": {
                "type": "hdrfilm",
                "width": int(film_size[0]),
                "height": int(film_size[1]),
            }
        })


    def prepare_next_frame(self, sensor: mi.Sensor) -> None:
        if self.previous_camera is None:
            return

        # Extract current camera transform as a scalar transform
        to_world = mi.ScalarTransform4f(
            dr.slice(sensor.world_transform().matrix, 0)
        )

        # Update previous_camera
        params = mi.traverse(self.previous_camera)
        params["to_world"] = mi.Transform4f(to_world)
        params.update()

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
                film_width  = int(film_size[0])
                film_height = int(film_size[1])
                pixel_count = film_width * film_height

                # Allocate temporal storage only once
                if self.previousGrid is None:
                    self.previousGrid = Reservoir.empty(pixel_count)
                    self.currentGrid  = Reservoir.empty(pixel_count)

                # Generate a set of rays starting at the sensor + their discrete pixel indices
                ray, weight, splatting_pos, pixel_index = self.sample_rays(scene, sensor, sampler)


                # PHASE 1
                initial_reservoir, si, valid_ray, direct_emission = self.sample_initial_reservoir(scene, sampler, ray, mi.Bool(True))

                sampler.schedule_state()
                dr.eval(initial_reservoir.w_sum, 
                        initial_reservoir.M, 
                        initial_reservoir.W, 
                        initial_reservoir.select.pdf, 
                        initial_reservoir.select.x1,
                        initial_reservoir.select.x1_n,
                        initial_reservoir.select.x2,
                        initial_reservoir.select.x2_n,
                        si.p, si.n, si.sh_frame.n, si.wi, si.uv, 
                        valid_ray, direct_emission
                    )

                temporal_reservoir = self.sample_temporal_reuse(scene, sensor, sampler, ray, si, initial_reservoir, valid_ray, film_width, film_height, pixel_index)

                sampler.schedule_state()
                dr.eval(
                    temporal_reservoir.w_sum,
                    temporal_reservoir.M,
                    temporal_reservoir.W,
                    temporal_reservoir.select.pdf,
                    temporal_reservoir.select.x1,
                    temporal_reservoir.select.x1_n,
                    temporal_reservoir.select.x2,
                    temporal_reservoir.select.x2_n,
                    temporal_reservoir.select.radiance,
                    temporal_reservoir.select.random_x1,
                    temporal_reservoir.select.random_x2,
                )

                spacial_reservoir = self.sample_spacial_reuse(scene, sampler, ray, si, temporal_reservoir, valid_ray, film_width, film_height, pixel_index)

                sampler.schedule_state()
                dr.eval(
                    spacial_reservoir.w_sum,
                    spacial_reservoir.M,
                    spacial_reservoir.W,
                    spacial_reservoir.select.pdf,
                    spacial_reservoir.select.x1,
                    spacial_reservoir.select.x1_n,
                    spacial_reservoir.select.x2,
                    spacial_reservoir.select.x2_n,
                    spacial_reservoir.select.radiance,
                    spacial_reservoir.select.random_x1,
                    spacial_reservoir.select.random_x2
                )


                L = self.evaluate_reservoir(scene, sampler, ray, si, spacial_reservoir, valid_ray, direct_emission)


                valid = valid_ray
                aovs = []
                

                # Store resulting reservoir for next frame
                scatter_reservoir(self.currentGrid, temporal_reservoir, pixel_index, valid_ray)

                sampler.schedule_state()
                dr.eval(
                    self.currentGrid.w_sum,
                    self.currentGrid.M,
                    self.currentGrid.W,
                    self.currentGrid.select.pdf,
                    self.currentGrid.select.x1,
                    self.currentGrid.select.x1_n,
                    self.currentGrid.select.x2,
                    self.currentGrid.select.x2_n,
                    self.currentGrid.select.radiance,
                    self.currentGrid.select.random_x1,
                    self.currentGrid.select.random_x2
                )

                self.previousGrid, self.currentGrid = (self.currentGrid, self.previousGrid)
                #self.prepare_next_frame(sensor)

    
                # Prepare an ImageBlock as specified by the film
                block = film.create_block()
    
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)
    
                # Accumulate into the image block
                Restir_Gi_Integrator._splat_to_block(
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
        


    def sample_initial_reservoir(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, active: mi.Mask = True) -> tuple[Reservoir, mi.SurfaceInteraction3f, mi.Mask, mi.Spectrum]:
        x1 = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=True, active=active)
        valid_ray = active & x1.is_valid()
        active_x1 = valid_ray

        emitter_x1 = x1.emitter(scene, active_x1)
        direct_emission = emitter_x1.eval(x1, active_x1)

        current_reservoir = Reservoir.empty(dr.width(ray))

        bsdf_ctx = mi.BSDFContext()
        bsdf_x1 = x1.bsdf(ray)

        ## NEE x1 Case ##
        # Sample a point/direction on a light
        ds_nee, emitter_weight = scene.sample_emitter_direction(x1, sampler.next_2d(active_x1), True, active_x1)

        active_nee = (active_x1 & (ds_nee.pdf > 0.0))

        # Direction toward sampled light in local BSDF frame
        wo_nee = x1.to_local(ds_nee.d)

        # Evaluate BSDF and its PDF for this same direction
        bsdf_val_nee, bsdf_pdf_nee = bsdf_x1.eval_pdf(bsdf_ctx, x1, wo_nee, active_nee)
        bsdf_val_nee = x1.to_world_mueller(bsdf_val_nee, -wo_nee, x1.wi)

        # If the light sample is delta, BSDF sampling cannot produce
        # the same point/direction in the usual continuous sense.
        mis_nee = dr.select(ds_nee.delta, 1.0, mis_weight(ds_nee.pdf, bsdf_pdf_nee, 1.0, 1.0))

        nee_contribution_x1 = (bsdf_val_nee * emitter_weight * mis_nee)
        direct_nee = dr.select(active_nee, nee_contribution_x1, 0.0)
        

        # Generate initial ReSTIR GI candidates
        for _ in range(self.initial_candidates):

            rD1, rD2, rD3 = sampler.next_1d(active_x1), sampler.next_1d(active_x1), sampler.next_1d(active_x1)

            random_x1 = mi.Vector3f(rD1, rD2, rD3)

            bs_1, bsdf_weight_1 = bsdf_x1.sample(bsdf_ctx, x1, rD1, mi.Point2f(rD2, rD3), active_x1)

            active_x2 = (active_x1 & (bs_1.pdf > 0.0))

            # Convert BSDF sampling weight to world-space Mueller representation
            bsdf_weight_1 = x1.to_world_mueller(bsdf_weight_1, -bs_1.wo, x1.wi)

            wo_world_1 = x1.to_world(bs_1.wo)
            ray_x2 = x1.spawn_ray(wo_world_1)

            x2 = scene.ray_intersect(ray_x2, ray_flags=mi.RayFlags.All, coherent=True, active=active_x2)
            active_x2 &= x2.is_valid()

            rD4, rD5, rD6 = sampler.next_1d(active_x2), sampler.next_1d(active_x2), sampler.next_1d(active_x2)
            random_x2 = mi.Vector3f(rD4, rD5, rD6)

            ## BSDF x1 Case ##
            Lo_x2 = trace_radiance_from_x2(scene=scene, x2=x2, ray_x2=ray_x2, sampler=sampler, first_random=random_x2, active=active_x2)

        
            # Construct a candidate sample for the reservoir
            candidate = Sample(x1=x1.p, x1_n=x1.n, x1_si = x1, x2=x2.p, x2_n=x2.n, pdf=bs_1.pdf, radiance=Lo_x2, random_x1=random_x1, random_x2=random_x2)

            # bsdf contribution at x1
            candidate_weight_spectrum = bsdf_weight_1 * Lo_x2
            reservoir_weight = mi.luminance(mi.unpolarized_spectrum(candidate_weight_spectrum), x1.wavelengths, active_x2)

            current_reservoir.add_sample(sample=candidate, reservoir_weight=reservoir_weight, rnd1D=sampler.next_1d(active_x2), active=active_x2)

        p_hat_current = eval_p_hat(scene, x1, current_reservoir.select, bsdf_x1, bsdf_ctx, sampler, active_x1)
        current_reservoir.finalize(p_hat_current, active_x1)

        direct_lighting = direct_emission + direct_nee

        return (current_reservoir, x1, valid_ray, direct_lighting)


    def sample_temporal_reuse(self, scene: mi.Scene, sensor: mi.Sensor, sampler: mi.Sampler, ray: mi.RayDifferential3f, si: mi.SurfaceInteraction3f, initial_reservoir: Reservoir, valid_ray: mi.Mask, film_width: int, film_height: int, pixel_index: mi.UInt32) -> Reservoir:
        active = valid_ray

        # No temporal history yet
        if self.previous_camera is None:
            return initial_reservoir

        bsdf = si.bsdf(ray)
        bsdf_ctx = mi.BSDFContext()

        # 1) Determine which temporal sample corresponds to our current pixel
        previous_index, valid_reprojection = eval_previous_index(si, self.previous_camera, sensor, sampler, film_width, film_height, pixel_index, active)

        # Gather previous-frame reservoir
        previous_reservoir = gather_reservoir(self.previousGrid, previous_index, valid_reprojection)

        # /!\ We have to first check if we can use temporal reuse with geometry similarity tests 
        # Use geometric check to verify if the temporal sample is useful for our current geometry
        x1_surface_pos = previous_reservoir.select.x1
        x1_surface_normal = previous_reservoir.select.x1_n
        valid_history = validate_surface_similarity(si.p, si.n, x1_surface_pos, x1_surface_normal, valid_reprojection)

        # 2) Merge this reservoir with the current reservoir
        temporal_reservoir = combine_reservoirs(scene, si, bsdf, bsdf_ctx, sampler, False, active , (initial_reservoir, active), (previous_reservoir, valid_history))

        return temporal_reservoir

    def sample_spacial_reuse(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, si: mi.SurfaceInteraction3f, temporal_reservoir: Reservoir, valid_ray: mi.Mask, film_width: int, film_height: int, pixel_index: mi.UInt32) -> Reservoir:
        active = valid_ray
        bsdf = si.bsdf(ray)
        bsdf_ctx = mi.BSDFContext()

        # Create a list of reservoir to be combined later on at the end
        reservoirs = [(temporal_reservoir, active)]
        
        # Concatenate the reservoir samples to be combined at the end
        for _ in range(SPATIAL_NEIGHBOR_COUNT):
            # 1) Determine which spacial samples correspond t our current pixels
            neighbor_index, valid_neighbors = eval_selected_neighbor(sampler, film_width, film_height, pixel_index, SPATIAL_NEIGHBOR_RADIUS, active)

            # Gather the neighbors
            neighbor_reservoir = gather_reservoir(temporal_reservoir, neighbor_index, valid_neighbors)

            # /!\ We have to first check if we can use spacial reuse with geometry similarity tests 
            # Use geometric check to verify if the temporal sample is useful for our current geometry
            valid_neighborhood = validate_surface_similarity(si.p, si.n, neighbor_reservoir.select.x1, neighbor_reservoir.select.x1_n, valid_neighbors)

            # Append the reservoir to our list of reservoirs
            reservoirs.append((neighbor_reservoir, valid_neighborhood))


        # 2) Merge all the neighbor reservoirs with the current reservoir
        spacial_reservoir = combine_reservoirs(scene, si, bsdf, bsdf_ctx, sampler, True, active, *reservoirs)

        return spacial_reservoir



    def evaluate_reservoir(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, si: mi.SurfaceInteraction3f, reservoir: Reservoir, valid_ray: mi.Mask, direct_emission: mi.Spectrum) -> mi.Spectrum:
        active = valid_ray

        result = mi.Spectrum(direct_emission)

        bsdf = si.bsdf(ray)
        bsdf_ctx = mi.BSDFContext()

        ## Get the selected candidate ##
        selected_candidate = reservoir.select

        # To get the integral estimator, we need to compute I = f(X) * W_X
        p_hat_X_spectrum = eval_p_hat_spectrum(scene, si, selected_candidate, bsdf, bsdf_ctx, sampler, active)
        p_hat = eval_p_hat(scene, si, selected_candidate, bsdf, bsdf_ctx, sampler, active)

        
        # Trace a ray from the original intersection point to the selected candidate point and check for visibility
        shadow_ray = si.spawn_ray_to(selected_candidate.x2)

        # Check if we have an occlusion
        occluded = scene.ray_test(shadow_ray, active)
        visible = ~occluded

        # Get the evaluation of the selected candidiate : f(X) = p_hat(X) * Visibility(X) 
        f_X = dr.select(visible, p_hat_X_spectrum, 0.)

        # Valid candidates must have striclty positive importance, otherwise they are considered invalid
        active_r = active & (p_hat > 0.)

        result += dr.select(active_r, f_X * reservoir.W, 0.)

        return result



    def to_string(self):
        return "TemporalReuseIntegrator[]"


# Register the integrator as a Mitsuba plugin
mi.register_integrator(
    "restir_gi",
    lambda props: Restir_Gi_Integrator(props)
)