const cutoff_mask_input_file = "data/reference_sim.jld2"

function cutoff_mask_config(grid; cutoff_mask = 1, kwargs...)
    return OfflineFilterConfig(original_data_filename = cutoff_mask_input_file,
                               var_names_to_filter = ("b",),
                               velocity_names = ("u", "w"),
                               grid = grid,
                               N = 2,
                               freq_c = 1e-4,
                               map_to_mean = true,
                               compute_mean_velocities = true,
                               cutoff_mask = cutoff_mask;
                               kwargs...)
end

function cutoff_mask_model(config, input_data)
    filtered_vars = create_filtered_vars(config)
    model = OceananigansLagrangianFilter.LagrangianFilter(
        config.grid;
        tracers = filtered_vars,
        auxiliary_fields = create_original_vars(config),
        forcing = create_forcing(filtered_vars, config),
        advection = config.advection,
    )
    initialise_filtered_vars_from_data(model, input_data, config)
    update_input_data!(model, input_data)
    return model
end

@testset "Offline cutoff mask" begin
    b_fts = FieldTimeSeries(cutoff_mask_input_file, "b")
    u_fts = FieldTimeSeries(cutoff_mask_input_file, "u")
    w_fts = FieldTimeSeries(cutoff_mask_input_file, "w")
    input_data = (var_data = (b_fts,), velocity_data = (u_fts, w_fts))
    grid = b_fts.grid

    @testset "scalar masks retain the legacy path" begin
        legacy = cutoff_mask_config(grid)
        unit_mask = cutoff_mask_config(grid; cutoff_mask = 1.0)
        half_mask = cutoff_mask_config(grid; cutoff_mask = 0.5)
        half_frequency = OfflineFilterConfig(
            original_data_filename = cutoff_mask_input_file,
            var_names_to_filter = ("b",),
            velocity_names = ("u", "w"),
            grid = grid,
            N = 2,
            freq_c = 0.5e-4,
            map_to_mean = true,
            compute_mean_velocities = true,
        )

        @test isnothing(legacy.cutoff_mask)
        @test isnothing(unit_mask.cutoff_mask)
        @test legacy.filter_params == unit_mask.filter_params
        @test isnothing(half_mask.cutoff_mask)
        @test half_mask.filter_params == half_frequency.filter_params
        @test_throws ErrorException cutoff_mask_config(grid; cutoff_mask = 0)
        @test_throws ErrorException cutoff_mask_config(grid; cutoff_mask = Inf)
    end

    @testset "field validation" begin
        valid_mask = CenterField(grid)
        set!(valid_mask, 1)
        @test cutoff_mask_config(grid; cutoff_mask = valid_mask).cutoff_mask === valid_mask

        invalid_values = CenterField(grid)
        set!(invalid_values, 0)
        @test_throws ErrorException cutoff_mask_config(grid; cutoff_mask = invalid_values)

        face_mask = XFaceField(grid)
        set!(face_mask, 1)
        @test_throws ErrorException cutoff_mask_config(grid; cutoff_mask = face_mask)

        @test_throws ErrorException cutoff_mask_config(
            grid;
            cutoff_mask = valid_mask,
            compute_Eulerian_filter = true,
        )

        relaxation_mask(x, z, p) = 1
        @test cutoff_mask_config(
            grid;
            cutoff_mask = valid_mask,
            boundary_relaxation = true,
            relax_timescale = 1hour,
            mask_func = relaxation_mask,
            mask_params = (;),
        ).boundary_relaxation

        other_grid = RectilinearGrid(CPU(), size = (4, 4), x = (0, 1), z = (-1, 0),
                                     topology = (Periodic, Flat, Bounded))
        wrong_grid_mask = CenterField(other_grid)
        set!(wrong_grid_mask, 1)
        @test_throws ErrorException cutoff_mask_config(grid; cutoff_mask = wrong_grid_mask)
    end

    @testset "a constant field is equivalent to a scalar cutoff" begin
        mask_value = 0.5
        mask = CenterField(grid)
        set!(mask, mask_value)

        scalar_config = cutoff_mask_config(grid; cutoff_mask = mask_value)
        field_config = cutoff_mask_config(grid; cutoff_mask = mask)
        scalar_model = cutoff_mask_model(scalar_config, input_data)
        field_model = cutoff_mask_model(field_config, input_data)

        for name in keys(scalar_model.tracers)
            # Spatial-mask states use reference coefficients and are M times
            # the scalar-cutoff states; their physical outputs agree.
            @test interior(getproperty(field_model.tracers, name)) ==
                  mask_value .* interior(getproperty(scalar_model.tracers, name))
        end

        scalar_simulation = Simulation(scalar_model, Δt = 60.0, stop_iteration = 1)
        field_simulation = Simulation(field_model, Δt = 60.0, stop_iteration = 1)
        run!(scalar_simulation)
        run!(field_simulation)

        for name in keys(scalar_model.tracers)
            @test interior(getproperty(field_model.tracers, name)) ==
                  mask_value .* interior(getproperty(scalar_model.tracers, name))
        end

        scalar_outputs = create_output_fields(scalar_model, scalar_config)
        field_outputs = create_output_fields(field_model, field_config)
        @test keys(field_outputs) == keys(scalar_outputs)
        for name in keys(scalar_outputs)
            @test interior(field_outputs[name]) == interior(scalar_outputs[name])
        end
    end

    @testset "boundary relaxation targets use the local cutoff" begin
        # Exercise both the single-exponential map and the cosine/sine pair.
        relaxation_mask(x, z, p) = p.strength
        for N in (1, 2)
            mask = CenterField(grid)
            set!(mask, 0.5)
            common = (original_data_filename = cutoff_mask_input_file,
                      var_names_to_filter = ("b",), velocity_names = ("u", "w"),
                      grid = grid, N = N, freq_c = 1.0,
                      boundary_relaxation = true, relax_timescale = 10.0,
                      mask_func = relaxation_mask, mask_params = (; strength = 0.4))
            spatial = OfflineFilterConfig(; common..., cutoff_mask = mask)
            scalar = OfflineFilterConfig(; common..., cutoff_mask = 0.5)
            spatial_forcing = create_forcing(create_filtered_vars(spatial), spatial)
            scalar_forcing = create_forcing(create_filtered_vars(scalar), scalar)

            for component in (N == 1 ? (:C1,) : (:C1, :S1))
                key = Symbol("xi_u_", component)
                masked = spatial_forcing[key][3]
                uniform = scalar_forcing[key][3]
                @test masked.field_dependencies == (:u, key, OceananigansLagrangianFilter.Utils.CUTOFF_MASK_FIELD)
                @test uniform.field_dependencies == (:u, key)

                velocity, map_state, M = 2.3, 0.125, 0.5
                # The masked map state is M times the scalar-cutoff state.
                target = uniform.parameters[1] * velocity
                @test masked.func(0.1, -0.2, 0.0, velocity, M * target, M, masked.parameters) ≈ 0 atol=1e-14
                @test masked.func(0.1, -0.2, 0.0, velocity, M * map_state, M, masked.parameters) ≈
                      M * uniform.func(0.1, -0.2, 0.0, velocity, map_state, uniform.parameters) rtol=1e-12
                @test masked.func(0.1, -0.2, 0.0, velocity, M * target + 0.125, M, masked.parameters) ≈
                      -0.125 * 0.4 / 10 rtol=1e-12
            end
        end
    end

    @testset "spatial cutoff and boundary relaxation run together" begin
        mask = Field{Center, Nothing, Nothing}(grid)
        set!(mask, x -> 1 + 0.2sin(2π * x / 10_000))
        relaxation_mask(x, z, p) = p.strength
        config = cutoff_mask_config(grid; cutoff_mask = mask,
                                    boundary_relaxation = true,
                                    relax_timescale = 3600.0,
                                    mask_func = relaxation_mask,
                                    mask_params = (; strength = 0.4))
        model = cutoff_mask_model(config, input_data)
        run!(Simulation(model; Δt = 60.0, stop_iteration = 1))
        @test all(isfinite, interior(create_output_fields(model, config)["b_Lagrangian_filtered"]))
    end

    @testset "reduced spatial mask" begin
        mask = Field{Center, Nothing, Center}(grid)
        set!(mask, (x, z) -> 1 + 1e-5 * x)
        config = cutoff_mask_config(grid; cutoff_mask = mask)
        model = cutoff_mask_model(config, input_data)
        simulation = Simulation(model, Δt = 60.0, stop_iteration = 1)
        run!(simulation)

        for output in values(create_output_fields(model, config))
            @test all(isfinite, interior(output))
        end
    end

    @testset "constant tracer crossing mask gradients" begin
        # A normalized one-sided filter returns 1/2 for f=1. This remains
        # exactly true under advection, even when the mask changes along a
        # trajectory. Use the saved grid but prescribe all input values here.
        b_fts.data .= 1
        u_fts.data .= 1000
        w_fts.data .= 0
        mask = Field{Center, Nothing, Nothing}(grid)
        set!(mask, x -> 1 + 0.5sin(2π * x / 10_000))
        config = cutoff_mask_config(grid; cutoff_mask = mask)
        model = cutoff_mask_model(config, input_data)
        run!(Simulation(model; Δt = 0.02, stop_time = 1.0))
        output = create_output_fields(model, config)["b_Lagrangian_filtered"]
        @test maximum(abs, interior(output) .- 0.5) < 1e-12
    end
end
