
import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field
from typing import Tuple, Union
from utilsPT import *

POSITION_THRESHOLD = 0.05
NORMAL_THRESHOLD = 0.9

SPATIAL_NEIGHBOR_COUNT = 3
SPATIAL_NEIGHBOR_RADIUS = 10

# PATH_LENGTH : The total number of edges (counting the one from the camera to the first vertex x1)
PATH_LENGTH = 6

class Restir_PT_Integrator(mi.ad.integrators.common.ADIntegrator):

    # Grid containing the reservoirs of the previous frame
    previousGrid: Reservoir = None
    currentGrid: Reservoir = None
    previous_camera: mi.Sensor = None

    previous_x1_buffer: X1_Geometry_Buffer = None
    current_x1_buffer: X1_Geometry_Buffer = None


    # Number of ReSTIR GI proposals per primary sample
    initial_path_count: int = 2

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
                initial_reservoir, si, valid_ray = self.sample_initial_reservoir(scene, sampler, mi.UInt32(seed), pixel_index, ray, mi.Bool(True))

                # Fill the primary surface information for temporal reuse for next frame
                current_geometry = X1_Geometry_Buffer(x1_pos=si.p, x1_norm=si.n, primary_ray=mi.Ray3f(ray), valid=valid_ray)
                scatter_x1_geometry_buffer(self.current_x1_buffer, current_geometry, pixel_index, active=mi.Bool(True))

                sampler.schedule_state()
                dr.eval(
                    initial_reservoir.w_sum,
                    initial_reservoir.M,
                    initial_reservoir.W,
                    initial_reservoir.select.pdf,
                    initial_reservoir.select.J,
                    initial_reservoir.select.seed,
                    initial_reservoir.select.incomingRadiance,
                    initial_reservoir.select.outgoingDirection,
                    si.p, si.n, si.sh_frame.n, si.wi, si.uv, 
                    valid_ray,
                    self.current_x1_buffer.x1_pos,
                    self.current_x1_buffer.x1_norm
                )

                temporal_reservoir = self.sample_temporal_reuse(scene, sensor, sampler, ray, si, initial_reservoir, valid_ray, film_width, film_height, pixel_index)

                sampler.schedule_state()
                dr.eval(
                    temporal_reservoir.w_sum,
                    temporal_reservoir.M,
                    temporal_reservoir.W,
                    temporal_reservoir.select.pdf,
                    temporal_reservoir.select.J,
                    temporal_reservoir.select.seed,
                    temporal_reservoir.select.incomingRadiance,
                    temporal_reservoir.select.outgoingDirection,
                )

                # spacial_reservoir = self.sample_spacial_reuse(scene, sampler, ray, si, temporal_reservoir, valid_ray, film_width, film_height, pixel_index)

                # sampler.schedule_state()
                # dr.eval(
                #     spacial_reservoir.w_sum,
                #     spacial_reservoir.M,
                #     spacial_reservoir.W,
                #     spacial_reservoir.select.pdf,
                #     spacial_reservoir.select.x1,
                #     spacial_reservoir.select.x1_n,
                #     spacial_reservoir.select.x2,
                #     spacial_reservoir.select.x2_n,
                #     spacial_reservoir.select.radiance,
                # )


                #L = self.evaluate_reservoir(scene, sampler, ray, si, spacial_reservoir, valid_ray, direct_emission)

                # Evaluate the selected reservoir
                L = self.evaluate_reservoir(scene=scene, sampler=sampler, ray=ray, si=si, reservoir=temporal_reservoir, valid_ray=valid_ray, direct_emission=mi.Spectrum(0.0))

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
                    self.currentGrid.select.J,
                    self.currentGrid.select.seed,
                    self.currentGrid.select.incomingRadiance,
                    self.currentGrid.select.outgoingDirection,
                )

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
        


    # ReSTIR PT intitial reservoir sampling, do simple path tracing to generate a set of initial candidates and store them in the reservoir (without NEE for now)
    # PATH_LENGTH : The total number of edges (counting the one from the camera to the first vertex x1)
    def sample_initial_reservoir(self, scene: mi.Scene, reservoir_sampler: mi.Sampler, frame_id: mi.UInt32, pixel_index: mi.UInt32, ray: mi.RayDifferential3f, active: mi.Mask = True) -> tuple[Reservoir, mi.SurfaceInteraction3f, mi.Mask]:

        # Initialize the reservoir
        width = dr.width(ray)
        current_reservoir = Reservoir.empty(width)
        bsdf_ctx = mi.BSDFContext()

        # The importance of each sample 
        selected_p_hat = dr.zeros(mi.Float, width)

        # Seed the sampler for the current pixel 
        pixel_seed = seed_pixel(pixel_index, frame_id)

        # Compute primary intersection
        x1 = scene.ray_intersect(ray, active=active)
        valid_ray = active & x1.is_valid()

        # Store the pdf from x1 toward x2 : the probability to sample x2
        reconnection_pdf = 0

        # Generate initial ReSTIR PT candidates
        # For now, we only choose paths with reconnection vertices = 2
        for path_index in range(self.initial_path_count):

            # Intitialize accumulated radiance and throughput
            throughput = mi.Spectrum(1.0)
            L = dr.zeros(mi.Spectrum, width)
            current_ray = ray
            active_path = valid_ray

            # Seed the sampler for the current path 
            path_seed, _ = mi.sample_tea_32(pixel_seed, mi.UInt32(path_index))
            path_sampler = create_path_sampler(path_seed, width)

            for i in range(PATH_LENGTH):

                # Trace the ray and compute the surface interaction
                preliminary_x = scene.ray_intersect_preliminary(ray=current_ray, coherent=(i == 0), active=active_path)
                x = preliminary_x.compute_surface_interaction(ray=current_ray, ray_flags=mi.RayFlags.All, active=active_path)

                active_path &= x.is_valid()
                if i == 0 : valid_ray = active_path

                # Evaluate emitted radiance at current vertex
                emitter = x.emitter(scene, active_path)
                Le = emitter.eval(x, active_path)
                L += dr.select(active_path, throughput * Le, 0.0)

                # Get the BSDF at the current vertex
                bsdf_x = x.bsdf(current_ray)

                # Construct a candidate sample for the reservoir (for now we only consider paths with reconnection vertices = 2)
                if i == 1:
                    # Compute postfix
                    incoming_radiance, outgoing_direction  = trace_postfix_radiance(scene, x, current_ray, path_sampler, active_path, PATH_LENGTH - i - 1)

                    # Compute the radiance contribution of the current candidate sample
                    wo_local = x.to_local(outgoing_direction)
                    bsdf_val, bsdf_pdf = bsdf_x.eval_pdf(bsdf_ctx, x, wo_local, active_path)
                    bsdf_val = x.to_world_mueller(bsdf_val, -wo_local, x.wi)
                    safe_pdf = dr.select(bsdf_pdf > 0.0, bsdf_pdf, 1.0)
                    bsdf_weight = dr.select(bsdf_pdf > 0.0, bsdf_val / safe_pdf, 0.0)
                    candidate_contribution = throughput * bsdf_weight * incoming_radiance

                    # Construct a candidate sample for the reservoir
                    candidate = Sample(outgoingDirection=outgoing_direction, incomingRadiance=incoming_radiance, surfaceInformation=preliminary_x, seed=path_seed, k=mi.UInt32(i+1), J=mi.Float(1.0), pdf=reconnection_pdf)
                    reservoir_weight = mi.luminance(mi.unpolarized_spectrum(candidate_contribution), x.wavelengths, active_path)

                    # Add candidate to reservoir
                    replace = current_reservoir.add_sample(sample=candidate, reservoir_weight=reservoir_weight, rnd1D=reservoir_sampler.next_1d(active_path), active=active_path)

                    # Recover the target function by removing
                    # the division by the x1 -> x2 sampling PDF
                    p_hat_candidate = reservoir_weight * bs_k.pdf

                    # Remember p_hat of the selected candidate
                    selected_p_hat = dr.select(replace, p_hat_candidate, selected_p_hat)

                    # The postfix helper has already traced the remaining path
                    break


                # Evaluate the BSDF for the next sample 
                rD1, rD2, rD3 =  path_sampler.next_float(mi.Float, active_path),  path_sampler.next_float(mi.Float, active_path), path_sampler.next_float(mi.Float, active_path)
                bs_k, bsdf_weight_k = bsdf_x.sample(bsdf_ctx, x, rD1, mi.Point2f(rD2, rD3), active_path)

                # For k=2, this is the PDF of x1 -> x2
                if i == 0:
                    reconnection_pdf = bs_k.pdf

                active_path_next = (active_path & (bs_k.pdf > 0.0))
                bsdf_weight_k = x.to_world_mueller(bsdf_weight_k, -bs_k.wo, x.wi)

                # Update throughput
                throughput *= bsdf_weight_k

                # Outgoing direction in world space
                wo_world_k = x.to_world(bs_k.wo)

                # Go to the next ray
                active_path = active_path_next
                current_ray = x.spawn_ray(wo_world_k)

        # Finalize the reservoir using the importance
        # of the selected candidate
        current_reservoir.finalize(p_hat_selected=selected_p_hat, active=valid_ray)

        return (current_reservoir, x1, valid_ray)

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
        
        # Get previous X1 geometry to know from were our paths start from
        previous_x1_geometry = gather_x1_geometry_buffer(self.previous_x1_buffer, previous_index, valid_reprojection)
        active_history = valid_reprojection & previous_x1_geometry.valid & (previous_reservoir.W > 0)

        # Try to perform a shift mapping between the previous and current samples for temporal reuse
        # If the shift mapping is successful, we can use the previous sample for temporal reuse
        # If the shift mapping is not successful, we cannot use the previous sample for temporal reuse and we keep the current sample as is
        temporal_primary_ray = previous_x1_geometry.primary_ray
        current_primary_ray = mi.Ray3f(ray)
        J, shifted_radiance, p_hat_temporal, target_bsdf_pdf, valid_shift_mapping = shift_mapping(scene, temporal_primary_ray, initial_reservoir, current_primary_ray, previous_reservoir,previous_index, sampler,  active_history)

    
        # Current strategy evaluated at initial candidate
        initial_sample_pdf = initial_reservoir.select.pdf

        # Temporal strategy evaluated at initial candidate
        proxy_temporal_sample_pdf = compute_temporal_pdf_proxy_at_initial(scene=scene, temporal_primary_ray=temporal_primary_ray, current_primary_ray=current_primary_ray, initial_reservoir=initial_reservoir, active=active_history)
        
        # Current strategy evaluated at shifted temporal candidate
        proxy_initial_sample_pdf = target_bsdf_pdf

        # Temporal strategy evaluated at shifted temporal candidate
        valid_J = valid_shift_mapping & (J > 0.0) & dr.isfinite(J)
        safe_J = dr.select(valid_J, J, 1.0)
        temporal_sample_pdf = dr.select(valid_J,previous_reservoir.select.pdf / safe_J, 0.0)

        # Compute the MIS weights of the initial sample and the temporal sample
        mis_initial = mis_weight(initial_sample_pdf, proxy_temporal_sample_pdf, 1, 1, 1)
        mis_temporal = mis_weight(temporal_sample_pdf, proxy_initial_sample_pdf, 1, 1, 1)

        # Add the new jacobian for the shift mapping in the temporal sample
        previous_reservoir.select.J = J

        # Compute the target radiance function of both the initial function and the target function
        initial_x2, valid_initial = reconstruct_initial_reconnection(scene=scene, current_si=si, current_ray=current_primary_ray, candidate=initial_reservoir.select, active=active)
        initial_p_hat = eval_p_hat(current_si=si, reconnection_si=initial_x2, candidate=initial_reservoir.select, bsdf_ctx=bsdf_ctx, active=valid_initial)
        temporal_p_hat = p_hat_temporal

        # Create the reservoir n-uples for the reservoir combination phase
        initial_reservoir_tuple = (initial_reservoir, mis_initial, initial_p_hat, 1, active)
        temporal_reservoir_tuple = (previous_reservoir, mis_temporal, temporal_p_hat, J, valid_shift_mapping)

        # Merge this reservoir with the current reservoir
        temporal_reservoir = combine_reservoirs(sampler, active, initial_reservoir_tuple, temporal_reservoir_tuple)

        return temporal_reservoir

    # def sample_spacial_reuse(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, si: mi.SurfaceInteraction3f, temporal_reservoir: Reservoir, valid_ray: mi.Mask, film_width: int, film_height: int, pixel_index: mi.UInt32) -> Reservoir:
    #     active = valid_ray
    #     bsdf = si.bsdf(ray)
    #     bsdf_ctx = mi.BSDFContext()

    #     # Create a list of reservoir to be combined later on at the end
    #     reservoirs = [(temporal_reservoir, active)]
        
    #     # Concatenate the reservoir samples to be combined at the end
    #     for _ in range(SPATIAL_NEIGHBOR_COUNT):
    #         # 1) Determine which spacial samples correspond t our current pixels
    #         neighbor_index, valid_neighbors = eval_selected_neighbor(sampler, film_width, film_height, pixel_index, SPATIAL_NEIGHBOR_RADIUS, active)

    #         # Gather the neighbors
    #         neighbor_reservoir = gather_reservoir(temporal_reservoir, neighbor_index, valid_neighbors)

    #         # /!\ We have to first check if we can use spacial reuse with geometry similarity tests 
    #         # Use geometric check to verify if the temporal sample is useful for our current geometry
    #         #valid_neighborhood = validate_surface_similarity(si.p, si.n, neighbor_reservoir.select.x1, neighbor_reservoir.select.x1_n, valid_neighbors)

    #         # Append the reservoir to our list of reservoirs
    #         #reservoirs.append((neighbor_reservoir, valid_neighborhood))


    #     # 2) Merge all the neighbor reservoirs with the current reservoir
    #     spacial_reservoir = combine_reservoirs(scene, si, bsdf, bsdf_ctx, sampler, True, active, *reservoirs)

    #     return spacial_reservoir



    def evaluate_reservoir(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, si: mi.SurfaceInteraction3f, reservoir: Reservoir, valid_ray: mi.Mask, direct_emission: mi.Spectrum) -> mi.Spectrum:

        # 1. Initialize result
        result = mi.Spectrum(direct_emission)
        active = valid_ray & (reservoir.W > 0.0)
        bsdf_ctx = mi.BSDFContext()

        # 2. Retrieve selected candidate
        selected_candidate = reservoir.select

        # 3. Reconstruct preserved reconnection vertex
        x2, valid_reconnection = reconstruct_reconnection_vertex(scene=scene, current_si=si, candidate=selected_candidate, active=active)

        # 4. Check visibility from current x1 to x2
        shadow_ray = si.spawn_ray_to(x2.p)
        occluded = scene.ray_test(shadow_ray,active=valid_reconnection)
        valid_reconnection &= ~occluded

        # 5. Evaluate selected candidate's target
        target_spectrum = eval_p_hat_spectrum(current_si=si, reconnection_si=x2, candidate=selected_candidate, bsdf_ctx=bsdf_ctx, active=valid_reconnection)

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