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
        @test_throws ErrorException cutoff_mask_config(
            grid;
            cutoff_mask = valid_mask,
            boundary_relaxation = true,
            relax_timescale = 1hour,
            mask_func = relaxation_mask,
            mask_params = (;),
        )

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
            @test interior(getproperty(field_model.tracers, name)) ==
                  interior(getproperty(scalar_model.tracers, name))
        end

        scalar_simulation = Simulation(scalar_model, Δt = 60.0, stop_iteration = 1)
        field_simulation = Simulation(field_model, Δt = 60.0, stop_iteration = 1)
        run!(scalar_simulation)
        run!(field_simulation)

        for name in keys(scalar_model.tracers)
            @test interior(getproperty(field_model.tracers, name)) ==
                  interior(getproperty(scalar_model.tracers, name))
        end

        scalar_outputs = create_output_fields(scalar_model, scalar_config)
        field_outputs = create_output_fields(field_model, field_config)
        @test keys(field_outputs) == keys(scalar_outputs)
        for name in keys(scalar_outputs)
            @test interior(field_outputs[name]) == interior(scalar_outputs[name])
        end
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
end
