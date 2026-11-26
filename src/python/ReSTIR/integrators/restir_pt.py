
import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field
from typing import Tuple, Union
from utilsPT import *

POSITION_THRESHOLD = 0.05
NORMAL_THRESHOLD = 0.9

SPATIAL_NEIGHBOR_COUNT = 1
SPATIAL_NEIGHBOR_RADIUS = 5
TEMPORAL_M_CAP = 20

IS_TEMPORAL_ACTIVE = True

# TODO : Make Spacial Reuse Compile correctly
IS_SPACIAL_ACTIVE = False

# PATH_LENGTH : The total number of edges (counting the one from the camera to the first vertex x1)
PATH_LENGTH = 6

class Restir_PT_Integrator(mi.ad.integrators.common.ADIntegrator):

    # Grid containing the reservoirs of the previous frame
    previousGrid: Reservoir = None
    currentGrid: Reservoir = None
    previous_camera: mi.Sensor = None

    previous_x1_buffer: X1_Geometry_Buffer = None
    current_x1_buffer: X1_Geometry_Buffer = None


    # Number of ReSTIR PT proposals per primary sample
    initial_path_count: int = 1

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
                    self.previous_x1_buffer = X1_Geometry_Buffer.empty(pixel_count)
                    self.current_x1_buffer = X1_Geometry_Buffer.empty(pixel_count)

                # Generate a set of rays starting at the sensor + their discrete pixel indices
                ray, weight, splatting_pos, pixel_index = self.sample_rays(scene, sensor, sampler)


                # PHASE 1
                initial_reservoir, si, valid_ray, short_path_radiance = self.sample_initial_reservoir(scene, sampler, mi.UInt32(seed), pixel_index, ray, mi.Bool(True))

                # Fill the primary surface information for temporal reuse for next frame
                current_geometry = X1_Geometry_Buffer(x1_pos=si.p, x1_norm=si.n, primary_ray=mi.Ray3f(ray), valid=valid_ray)
                scatter_x1_geometry_buffer(self.current_x1_buffer, current_geometry, pixel_index, active=mi.Bool(True))

                sampler.schedule_state()
                dr.eval(initial_reservoir, si, valid_ray, short_path_radiance, ray, self.current_x1_buffer)

                temporal_reservoir = self.sample_temporal_reuse(scene, sensor, sampler, ray, si, initial_reservoir, valid_ray, film_width, film_height, pixel_index)

                sampler.schedule_state()
                dr.eval(temporal_reservoir)

                spacial_reservoir = self.sample_spatial_reuse(scene=scene, sampler=sampler, ray=ray, si=si, initial_reservoir=temporal_reservoir, valid_ray=valid_ray, film_width=film_width, film_height=film_height, pixel_index=pixel_index)

                sampler.schedule_state()
                dr.eval(spacial_reservoir)

                # Evaluate the selected reservoir
                L = self.evaluate_reservoir(scene=scene, sampler=sampler, ray=ray, si=si, reservoir=spacial_reservoir, valid_ray=valid_ray, direct_emission=short_path_radiance)

                valid = valid_ray
                aovs = []
                

                # Store resulting reservoir for next frame
                scatter_reservoir(self.currentGrid, temporal_reservoir, pixel_index, valid_ray)

                sampler.schedule_state()
                dr.eval(self.currentGrid, self.current_x1_buffer)

                self.previousGrid, self.currentGrid = (self.currentGrid, self.previousGrid)
                self.previous_x1_buffer, self.current_x1_buffer = (self.current_x1_buffer, self.previous_x1_buffer)

    
                # Prepare an ImageBlock as specified by the film
                block = film.create_block()
    
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)
    
                # Accumulate into the image block
                Restir_PT_Integrator._splat_to_block(
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
        


    # ReSTIR PT initial reservoir sampling, without NEE.
    # PATH_LENGTH: Total number of edges, including the camera -> x1 edge.
    def sample_initial_reservoir(self, scene: mi.Scene, sampler: mi.Sampler, frame_id: mi.UInt32, pixel_index: mi.UInt32, ray: mi.RayDifferential3f, active: mi.Mask = True) -> tuple[Reservoir, mi.SurfaceInteraction3f, mi.Mask, mi.Spectrum]:

        # Initialize reservoir
        width = dr.width(ray)
        current_reservoir = Reservoir.empty(width)
        selected_p_hat = dr.zeros(mi.Float, width)
        bsdf_ctx = mi.BSDFContext()

        # Initialize short-path radiance accumulator
        short_path_radiance = dr.zeros(mi.Spectrum, width)

        # Seed the sampler for the current pixel
        pixel_seed = seed_pixel(pixel_index, frame_id)

        # Compute primary intersection x1
        x1 = scene.ray_intersect(ray, active=active)
        valid_ray = active & x1.is_valid()

        # Generate initial ReSTIR PT candidates
        for path_index in range(self.initial_path_count):

            # Create a deterministic sampler for the current path
            path_seed, _ = mi.sample_tea_32(pixel_seed, mi.UInt32(path_index))
            path_sampler = create_path_sampler(path_seed, width)

            # Trace prefix from x1 to x(k-1), then sample the ray toward xk
            prefix_throughput, incoming_xk_pdf, ray_to_xk, prefix_radiance, xk_previous, xk_previous_delta, selected_k, valid_prefix = trace_prefix_throughput(scene=scene, path_sampler=path_sampler, nee_sampler=sampler, active=valid_ray, max_depth=PATH_LENGTH, x1=x1, ray_x0=ray)

            # Accumulate prefix emission (average across initial paths)
            short_path_radiance += prefix_radiance / self.initial_path_count

            # Intersect the reconnection vertex xk
            preliminary_xk = scene.ray_intersect_preliminary(ray=ray_to_xk, coherent=False, active=valid_prefix)
            xk = preliminary_xk.compute_surface_interaction(ray=ray_to_xk, ray_flags=mi.RayFlags.All, active=valid_prefix)
            valid_xk = valid_prefix & xk.is_valid()

            # Accumulate emission at xk using MIS
            emitter_xk = xk.emitter(scene, valid_xk)
            Le_xk = emitter_xk.eval(xk, valid_xk)

            # Construct the direction sample corresponding to x1 -> xk
            ds_bsdf = mi.DirectionSample3f(scene, xk, xk_previous)

            # Evaluate the probability that NEE would sample this emitter
            emitter_pdf = scene.pdf_emitter_direction(xk_previous, ds_bsdf, active=valid_xk)

            # Compute the BSDF sampling MIS weight
            mis_bsdf = mis_weight(incoming_xk_pdf, emitter_pdf, selected_index=0, power=2)
            mis_bsdf = dr.select(xk_previous_delta, 1.0, mis_bsdf)
            
            # Accumulate the weighted emitter contribution
            emission_contribution = prefix_throughput * Le_xk * mis_bsdf
            short_path_radiance += dr.select(valid_xk, emission_contribution / self.initial_path_count, mi.Spectrum(0.0))

            # Trace postfix starting at xk
            incoming_radiance, outgoing_direction = trace_postfix_radiance(scene=scene, xk=xk, ray_xk=ray_to_xk, path_sampler=path_sampler, nee_sampler= sampler, k= selected_k,  active=valid_xk, max_depth=PATH_LENGTH)

            # Evaluate BSDF at xk for the sampled outgoing direction    
            bsdf_xk = xk.bsdf(ray_to_xk)
            wo_local = xk.to_local(outgoing_direction)
            bsdf_val, bsdf_pdf = bsdf_xk.eval_pdf(bsdf_ctx, xk, wo_local, valid_xk)
            bsdf_val = xk.to_world_mueller(bsdf_val, -wo_local, xk.wi)

            # Compute BSDF sampling weight at xk
            safe_pdf = dr.select(bsdf_pdf > 0.0, bsdf_pdf, 1.0)
            bsdf_weight = dr.select(bsdf_pdf > 0.0, bsdf_val / safe_pdf, mi.Spectrum(0.0))

            # Evaluate the NEE at xk 
            xk_nee, emitter_weight = scene.sample_emitter_direction(xk, sampler.next_2d(valid_xk), True, valid_xk)
            active_nee = valid_xk & (xk_nee.pdf > 0.0) & dr.isfinite(xk_nee.pdf)

            # Construct ReSTIR PT candidate
            candidate = Sample(outgoingDirection=outgoing_direction, incomingRadiance=incoming_radiance, surfaceInformation=preliminary_xk, seed=path_seed, k=mi.UInt32(selected_k), J=mi.Float(1.0), pdf=incoming_xk_pdf, nee_sample=xk_nee, nee_valid=active_nee, nee_emitter_weight=emitter_weight)

            # Compute full importance-sampled candidate contribution
            xk_contribution = eval_xk_contribution(xk, candidate, bsdf_ctx, valid_xk)
            candidate_contribution = prefix_throughput * xk_contribution
            
            # Compute reservoir importance weight
            reservoir_weight = mi.luminance(mi.unpolarized_spectrum(candidate_contribution), xk.wavelengths, valid_xk)
            reservoir_weight = dr.select(valid_xk, reservoir_weight, 0.0)

            # Add candidate to the reservoir (all valid primary rays count toward M)
            replace = current_reservoir.add_sample(sample=candidate, reservoir_weight=reservoir_weight, rnd1D=sampler.next_1d(valid_ray), active=valid_ray)

            # Recover target function by removing the xk-1 -> xk proposal PDF
            p_hat_candidate = reservoir_weight * incoming_xk_pdf

            # Save target importance of the selected candidate
            selected_p_hat = dr.select(replace, p_hat_candidate, selected_p_hat)

        # Finalize reservoir
        current_reservoir.finalize(p_hat_selected=selected_p_hat, active=valid_ray)

        return current_reservoir, x1, valid_ray, short_path_radiance 

    def sample_temporal_reuse(self, scene: mi.Scene, sensor: mi.Sensor, sampler: mi.Sampler, ray: mi.RayDifferential3f, si: mi.SurfaceInteraction3f, initial_reservoir: Reservoir, valid_ray: mi.Mask, film_width: int, film_height: int, pixel_index: mi.UInt32) -> Reservoir:
        active = valid_ray

        # No temporal history yet
        if self.previous_camera is None and IS_TEMPORAL_ACTIVE:
            return initial_reservoir

        # 1) Determine which temporal sample corresponds to our current pixel
        previous_index, valid_reprojection = eval_previous_index(si, self.previous_camera, sensor, sampler, film_width, film_height, pixel_index, active)

        # Gather previous-frame reservoir
        previous_reservoir = gather_reservoir(self.previousGrid, previous_index, valid_reprojection)

        # Apply M-cap to avoid historical sample dominance
        apply_M_cap(previous_reservoir=previous_reservoir, m_cap=TEMPORAL_M_CAP)
        
        # Get previous X1 geometry to know from were our paths start from
        previous_x1_geometry = gather_x1_geometry_buffer(self.previous_x1_buffer, previous_index, valid_reprojection)
        active_history = valid_reprojection & previous_x1_geometry.valid & (previous_reservoir.W > 0)

        # Try to perform a shift mapping between the previous and current samples for temporal reuse
        # If the shift mapping is successful, we can use the previous sample for temporal reuse
        # If the shift mapping is not successful, we cannot use the previous sample for temporal reuse and we keep the current sample as is
        temporal_primary_ray = previous_x1_geometry.primary_ray
        current_primary_ray = mi.Ray3f(ray)
        J, p_hat_temporal, target_bsdf_pdf, valid_shift_mapping = shift_mapping(scene=scene, temporal_preliminary_ray=temporal_primary_ray, temporal_reservoir=previous_reservoir, current_preliminary_ray=current_primary_ray, max_depth=PATH_LENGTH, active=active_history)


        # Current strategy evaluated at the freshly generated initial candidate
        initial_sample_pdf = dr.select(active & (initial_reservoir.W > 0.0), initial_reservoir.select.pdf, 0.0)

        # Temporal strategy evaluated at the initial candidate
        proxy_temporal_sample_pdf, _, valid_temporal_proxy = compute_shifted_pdf_proxy(scene=scene, source_primary_ray=temporal_primary_ray, target_primary_ray=current_primary_ray, candidate=initial_reservoir.select, max_k=PATH_LENGTH, active=active_history)

        proxy_temporal_sample_pdf = dr.select(valid_temporal_proxy, proxy_temporal_sample_pdf, 0.0)

        # Current strategy evaluated at the shifted temporal candidate
        proxy_initial_sample_pdf = dr.select(valid_shift_mapping, target_bsdf_pdf, 0.0)

        # Temporal strategy evaluated at its selected candidate, expressed in the current domain
        temporal_sample_pdf, _, valid_temporal_pdf = compute_shifted_pdf_proxy(scene=scene, source_primary_ray=temporal_primary_ray, target_primary_ray=current_primary_ray, candidate=previous_reservoir.select, max_k=PATH_LENGTH, active=active_history & valid_shift_mapping)

        temporal_sample_pdf = dr.select(valid_temporal_pdf, temporal_sample_pdf, 0.0)

        # Compute the MIS weights
        mis_initial = mis_weight(initial_sample_pdf, proxy_temporal_sample_pdf, selected_index=0, power=1)
        mis_temporal = mis_weight(temporal_sample_pdf, proxy_initial_sample_pdf, selected_index=0, power=1)

        # Add the new jacobian for the shift mapping in the temporal sample
        previous_reservoir.select.J = J

        # Compute the target radiance function of both the initial function and the target function
        initial_p_hat = eval_p_hat(scene=scene, primary_ray=mi.Ray3f(ray), candidate=initial_reservoir.select, max_k=PATH_LENGTH, bsdf_ctx=mi.BSDFContext(), active=active & (initial_reservoir.W > 0.0), x1=si)
        valid_initial = initial_p_hat > 0.0
        temporal_p_hat = p_hat_temporal

        # Create the reservoir n-uples for the reservoir combination phase
        initial_reservoir_tuple = (initial_reservoir, mis_initial, initial_p_hat, 1, valid_initial)
        temporal_reservoir_tuple = (previous_reservoir, mis_temporal, temporal_p_hat, J, valid_shift_mapping)

        # Merge this reservoir with the current reservoir
        temporal_reservoir = combine_reservoirs(sampler, active, initial_reservoir_tuple, temporal_reservoir_tuple)

        return temporal_reservoir


    def sample_spatial_reuse(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, si: mi.SurfaceInteraction3f, initial_reservoir: Reservoir, valid_ray: mi.Mask, film_width: int, film_height: int, pixel_index: mi.UInt32) -> Reservoir:


        if not IS_SPACIAL_ACTIVE:
            return initial_reservoir

        active = valid_ray
        bsdf_ctx = mi.BSDFContext()
        current_primary_ray = mi.Ray3f(ray)

        # 1. Evaluate the current candidate
        current_p_hat = eval_p_hat(scene=scene, primary_ray=current_primary_ray, candidate=initial_reservoir.select, max_k=PATH_LENGTH, bsdf_ctx=bsdf_ctx, active=active & (initial_reservoir.W > 0.0), x1=si)
        valid_current = (active & (initial_reservoir.W > 0.0) & (current_p_hat > 0.0) & dr.isfinite(current_p_hat))

        # Each entry contains:
        # reservoir, primary_ray, p_hat, J, strategy_enabled, candidate_valid
        strategy_data = [(initial_reservoir, current_primary_ray, current_p_hat, mi.Float(1.0), active, valid_current)]

        # 2. Gather neighboring strategies
        for _ in range(SPATIAL_NEIGHBOR_COUNT):

            neighbor_index, valid_neighbor = eval_selected_neighbor(sampler, film_width, film_height, pixel_index, SPATIAL_NEIGHBOR_RADIUS, active)
            neighbor_reservoir = gather_reservoir(initial_reservoir, neighbor_index, valid_neighbor)
            neighbor_geometry = gather_x1_geometry_buffer(self.current_x1_buffer, neighbor_index, valid_neighbor)
            neighbor_primary_ray = neighbor_geometry.primary_ray

            # Primary-surface compatibility
            position_difference = dr.norm(si.p - neighbor_geometry.x1_pos)
            normal_similarity = dr.dot(si.n, neighbor_geometry.x1_norm)

            strategy_enabled = valid_neighbor & neighbor_geometry.valid
            strategy_enabled &= (position_difference < POSITION_THRESHOLD) & (normal_similarity > NORMAL_THRESHOLD)

            # A strategy can participate even if its selected sample cannot be shifted
            valid_candidate_source = strategy_enabled & (neighbor_reservoir.W > 0.0)

            # Shift the selected candidate
            J, p_hat_spatial, target_bsdf_pdf, valid_shift = shift_mapping(scene=scene, temporal_preliminary_ray=neighbor_primary_ray, temporal_reservoir=neighbor_reservoir, current_preliminary_ray=current_primary_ray, max_depth=PATH_LENGTH, active=valid_candidate_source)

            valid_shift &= (J > 0.0) & dr.isfinite(J)
            valid_shift &= (p_hat_spatial > 0.0) & dr.isfinite(p_hat_spatial)

            strategy_data.append((neighbor_reservoir, neighbor_primary_ray, p_hat_spatial, J, strategy_enabled, valid_shift))

        # 3. Compute MIS weights for all candidates
        combined_data = []

        for candidate_index, (candidate_reservoir, candidate_primary_ray, candidate_p_hat, candidate_J, candidate_strategy_enabled, candidate_valid) in enumerate(strategy_data):

            candidate_pdfs = []

            for strategy_index, (strategy_reservoir, strategy_primary_ray, strategy_p_hat, strategy_J, strategy_enabled, strategy_candidate_valid) in enumerate(strategy_data):

                # Evaluate the strategy at THIS candidate, independently of its own selected sample
                active_proxy = active & strategy_enabled & candidate_valid

                proxy_pdf, proxy_J, valid_proxy = compute_shifted_pdf_proxy(scene=scene, source_primary_ray=strategy_primary_ray, target_primary_ray=current_primary_ray, candidate=candidate_reservoir.select, max_k=PATH_LENGTH, active=active_proxy)

                proxy_pdf = dr.select(valid_proxy & dr.isfinite(proxy_pdf) & (proxy_pdf > 0.0), proxy_pdf, 0.0)

                candidate_pdfs.append(proxy_pdf)

            # Balance heuristic
            mis_candidate = mis_weight(*candidate_pdfs, selected_index=candidate_index, power=1)

            valid_candidate = candidate_valid & (candidate_pdfs[candidate_index] > 0.0)
            valid_candidate &= (mis_candidate > 0.0) & dr.isfinite(mis_candidate)

            combined_data.append((candidate_reservoir, mis_candidate, candidate_p_hat, candidate_J, valid_candidate))

        # 4. Combine the reservoirs
        spatial_reservoir = combine_reservoirs(sampler, active, *combined_data)

        return spatial_reservoir

    def evaluate_reservoir(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, si: mi.SurfaceInteraction3f, reservoir: Reservoir, valid_ray: mi.Mask, direct_emission: mi.Spectrum) -> mi.Spectrum:

        # 1. Initialize result
        result = mi.Spectrum(direct_emission)
        active = valid_ray & (reservoir.W > 0.0)
        bsdf_ctx = mi.BSDFContext()

        # 2. Retrieve selected candidate
        selected_candidate = reservoir.select
        primary_ray = mi.Ray3f(ray)

        # 3. Replay prefix and reconstruct preserved reconnection vertex xk
        previous_si, xk, _, _, valid_reconnection = replay_path_to_reconnection(scene=scene, primary_ray=primary_ray, path_seed=selected_candidate.seed, k=selected_candidate.k, max_k=PATH_LENGTH, active=active, x1=si, candidate=selected_candidate, use_stored_xk=True)

        # 4. Check visibility between y(k-1) and xk
        shadow_ray = previous_si.spawn_ray_to(xk.p)
        occluded = scene.ray_test(shadow_ray, active=valid_reconnection)
        valid_reconnection &= ~occluded

        # 5. Evaluate complete target spectrum (including prefix throughput)
        target_spectrum = eval_p_hat_spectrum(scene=scene, primary_ray=primary_ray, candidate=selected_candidate, max_k=PATH_LENGTH, bsdf_ctx=bsdf_ctx, active=valid_reconnection, x1=si)

        # 6. Apply reservoir contribution weight
        result += dr.select(valid_reconnection, target_spectrum * reservoir.W, mi.Spectrum(0.0))

        return result



    def to_string(self):
        return "ReSTIR_PT_Integrator[]"


# Register the integrator as a Mitsuba plugin
mi.register_integrator(
    "restir_pt",
    lambda props: Restir_PT_Integrator(props)
)