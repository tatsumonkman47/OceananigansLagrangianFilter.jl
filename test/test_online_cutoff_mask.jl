@testset "Online spatial cutoff" begin
    grid = RectilinearGrid(CPU(), size = (16, 4), x = (0, 2π), z = (-1, 0),
                           topology = (Periodic, Flat, Bounded))
    mask = CenterField(grid)
    set!(mask, (x, z) -> 1 + 0.5cos(x))
    common = (grid = grid, var_names_to_filter = ("signal",),
              velocity_names = ("u",), N = 1, freq_c = 1.0)
    config = OnlineFilterConfig(; common..., cutoff_mask = mask)

    @test config.cutoff_mask === mask
    @test keys(cutoff_mask_auxiliary_fields(config)) ==
          (OceananigansLagrangianFilter.Utils.CUTOFF_MASK_FIELD,)
    @test isnothing(OnlineFilterConfig(; common...).cutoff_mask)
    @test OnlineFilterConfig(; common..., cutoff_mask = 0.5).filter_params ==
          OnlineFilterConfig(; common..., freq_c = 0.5).filter_params
    @test_throws ErrorException OnlineFilterConfig(; common..., cutoff_mask = 0)
    @test_throws ErrorException compute_Eulerian_filter!(config)
    @test_throws ErrorException compute_time_shift!(config)

    second_mask = CenterField(grid)
    set!(second_mask, 1.2)
    first_config = OnlineFilterConfig(; common..., cutoff_mask = mask, label = "_first")
    second_config = OnlineFilterConfig(; common..., cutoff_mask = second_mask, label = "_second")
    mask_fields = merge(cutoff_mask_auxiliary_fields(first_config),
                        cutoff_mask_auxiliary_fields(second_config))
    @test length(keys(mask_fields)) == 2
    for labeled_config in (first_config, second_config)
        mask_key = only(keys(cutoff_mask_auxiliary_fields(labeled_config)))
        forcing = create_forcing(create_filtered_vars(labeled_config), labeled_config)
        signal_key = Symbol("signal", labeled_config.label, "_C1")
        @test forcing[signal_key][1].field_dependencies[end] == mask_key
    end

    filtered_vars = create_filtered_vars(config)
    model = NonhydrostaticModel(grid;
        tracers = (filtered_vars..., :signal),
        forcing = create_forcing(filtered_vars, config),
        auxiliary_fields = cutoff_mask_auxiliary_fields(config))
    set!(model, signal = 1)
    initialise_filtered_vars_from_model(model, config)
    set!(model.tracers.signal_C1, 0)
    run!(Simulation(model; Δt = 0.01, stop_time = 0.1))

    result = create_output_fields(model, config)["signal_Lagrangian_filtered"]
    expected = 1 .- exp.(-interior(mask) .* 0.1)
    @test maximum(abs, interior(result) .- expected) < 2e-5

    pair_config = OnlineFilterConfig(; merge(common, (; N = 2))..., cutoff_mask = mask,
                                     map_to_mean = false, compute_mean_velocities = false)
    pair_vars = create_filtered_vars(pair_config)
    pair_model = NonhydrostaticModel(grid;
        tracers = (pair_vars..., :signal),
        forcing = create_forcing(pair_vars, pair_config),
        auxiliary_fields = cutoff_mask_auxiliary_fields(pair_config))
    set!(pair_model, signal = 1)
    initialise_filtered_vars_from_model(pair_model, pair_config)
    set!(pair_model.tracers.signal_C1, 0)
    set!(pair_model.tracers.signal_S1, 0)
    run!(Simulation(pair_model; Δt = 0.01, stop_time = 0.1))

    params = pair_config.filter_params
    a, b, c, d = params.a1, params.b1, params.c1, params.d1
    q = interior(mask) .* 0.1
    decay = exp.(-c .* q)
    cosine_integral = (c .+ decay .* (-c .* cos.(d .* q) .+ d .* sin.(d .* q))) ./ (c^2 + d^2)
    sine_integral = (d .- decay .* (c .* sin.(d .* q) .+ d .* cos.(d .* q))) ./ (c^2 + d^2)
    pair_expected = a .* cosine_integral .+ b .* sine_integral
    pair_result = create_output_fields(pair_model, pair_config)["signal_Lagrangian_filtered"]
    @test maximum(abs, interior(pair_result) .- pair_expected) < 2e-5

    map_config = OnlineFilterConfig(; common..., cutoff_mask = mask,
                                    map_to_mean = true, compute_mean_velocities = true)
    map_vars = create_filtered_vars(map_config)
    map_model = NonhydrostaticModel(grid;
        tracers = (map_vars..., :signal),
        forcing = create_forcing(map_vars, map_config),
        auxiliary_fields = cutoff_mask_auxiliary_fields(map_config))
    set!(map_model, signal = 1, u = 2)
    initialise_filtered_vars_from_model(map_model, map_config)
    mean_u = create_output_fields(map_model, map_config)["u_Lagrangian_filtered"]
    @test maximum(abs, interior(mean_u) .- 2) < 1e-12
end
