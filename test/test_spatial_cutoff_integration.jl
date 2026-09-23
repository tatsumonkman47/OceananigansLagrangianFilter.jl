using Oceananigans.OutputWriters: write_output!
using Oceananigans.Units: Time

# Manufactured input: x(t+s) = x(t) + U s, M(x) = 1 + ε cos(x).
# Save analytic snapshots, so no source-model discretization contaminates the test.
function write_spatial_cutoff_input(filename, grid, U; stop_time = 96.0)
    model = OceananigansLagrangianFilter.LagrangianFilter(grid;
                tracers = (:constant, :material, :signal))
    set!(model.velocities.u, U)
    set!(model.velocities.w, 0)
    set!(model.tracers.constant, 1)
    outputs = (; model.tracers..., u = model.velocities.u, w = model.velocities.w)
    writer = JLD2Writer(model, outputs; filename, schedule = TimeInterval(0.1),
                       array_type = Array{Float64}, overwrite_existing = true)
    for (n, t) in enumerate(0:0.1:stop_time)
        set!(model.tracers.material, (x, z) -> 1 + 0.2cos(x - U * t))
        set!(model.tracers.signal, cos(2t))
        model.clock.time = t
        model.clock.iteration = n - 1
        write_output!(writer, model)
    end
    return nothing
end

# Independent reference: integrate the normalized kernel along the exact
# trajectory, without the package's coefficient or tendency constructors.
# These are the closed-form N=1 and N=2 kernels at reference cutoff 1.
function spatial_cutoff_integral(x, t, U, N; ε = 0.5, h = 0.005, horizon = 80.0)
    signal = displacement = mass = 0.0
    nsteps = round(Int, 2horizon / h)
    for n in 0:nsteps
        s = -horizon + n * h
        M = 1 + ε * cos(x + U * s)
        τ = U == 0 ? (1 + ε * cos(x)) * s :
                     s + ε / U * (sin(x + U * s) - sin(x))
        r = abs(τ)
        G = N == 1 ? exp(-r) / 2 :
                     exp(-r / sqrt(2)) * (cos(r / sqrt(2)) + sin(r / sqrt(2))) / (2sqrt(2))
        # Composite Simpson quadrature, including the Jacobian dτ = M ds.
        weight = (n == 0 || n == nsteps) ? 1 : isodd(n) ? 4 : 2
        K = weight * M * G * h / 3
        mass += K
        signal += K * cos(2(t + s))
        displacement += K * U * s
    end
    return (; mass, signal, displacement)
end

@testset "Spatial cutoff: analytical offline validation" begin
    mktempdir() do directory
        # The x-z topology also exercises the package's existing mean-position
        # regridder. Four vertical rows suffice: all fields are invariant in z.
        grid = RectilinearGrid(CPU(); size = (48, 4), x = (0, 2π), z = (-1, 0),
                               topology = (Periodic, Flat, Bounded))
        mask = Field{Center, Nothing, Nothing}(grid)
        set!(mask, x -> 1 + 0.5cos(x))
        xs = collect(xnodes(CenterField(grid)))

        for U in (0.0, 1.0)
            input = joinpath(directory, "input_$(U).jld2")
            write_spatial_cutoff_input(input, grid, U)

            for N in (U == 0 ? (1, 2, 4) : (1, 2))
                @testset "U=$U, N=$N" begin
                    # Order 4 only needs the frequency-response check. Omitting
                    # unused fields and maps keeps compilation of this case small.
                    maps = N != 4
                    variables = maps ? ("constant", "material", "signal") : ("signal",)
                    velocities = maps ? ("u", "w") : ("u",)
                    output = joinpath(directory, "filtered_$(U)_$(N).jld2")
                    config = OfflineFilterConfig(original_data_filename = input,
                                output_filename = output,
                                forward_output_filename = joinpath(directory, "forward.jld2"),
                                backward_output_filename = joinpath(directory, "backward.jld2"),
                                var_names_to_filter = variables,
                                velocity_names = velocities, grid = grid, backend = InMemory(),
                                N = N, freq_c = 1.0, cutoff_mask = mask,
                                Δt = 0.08, T_out = 4.0, advection = Centered(order = 4),
                                map_to_mean = maps,
                                compute_mean_velocities = maps,
                                compute_Eulerian_filter = false, output_netcdf = false,
                                delete_intermediate_files = true)
                    run_offline_Lagrangian_filter(config)

                    signal = FieldTimeSeries(output, "signal_Lagrangian_filtered")
                    if maps
                        constant = FieldTimeSeries(output, "constant_Lagrangian_filtered")
                        material = FieldTimeSeries(output, "material_Lagrangian_filtered")
                        displacement = FieldTimeSeries(output, "xi_u")
                        velocity = FieldTimeSeries(output, "u_Lagrangian_filtered")
                        remapped_constant = FieldTimeSeries(output, "constant_Lagrangian_filtered_at_mean")
                    end

                    # Exclude the finite-window startup and turnaround transients.
                    # Input interpolation contributes O((ω Δt_input)^2) error;
                    # the moving case additionally has spatial advection error.
                    for t in (40.0, 48.0, 56.0)
                        if maps
                            @test maximum(abs, interior(constant[Time(t)]) .- 1) < 2e-6
                            @test maximum(abs, interior(remapped_constant[Time(t)]) .- 1) < 2e-6
                            expected_material = 1 .+ 0.2cos.(xs .- U * t)
                            @test maximum(abs, interior(material[Time(t)])[:, 1, 2] .- expected_material) < 2e-3
                        end

                        if U == 0
                            # Exact physical-frequency response for stationary particles.
                            gain = 1 ./ (1 .+ (2 ./ (1 .+ 0.5cos.(xs))).^(2N))
                            @test maximum(abs, interior(signal[Time(t)])[:, 1, 2] .- gain .* cos(2t)) < 2e-3
                            if maps
                                @test maximum(abs, interior(displacement[Time(t)])) < 1e-12
                            end
                        else
                            references = [spatial_cutoff_integral(x, t, U, N) for x in xs]
                            @test maximum(abs, [r.mass for r in references] .- 1) < 1e-8
                            @test maximum(abs, interior(signal[Time(t)])[:, 1, 2] .- [r.signal for r in references]) < 3e-3
                            @test maximum(abs, interior(displacement[Time(t)])[:, 1, 2] .- [r.displacement for r in references]) < 3e-3
                        end
                    end

                    if U != 0
                        # Mean velocity is D(x+ξ)/Dt, not the scalar-filtered U.
                        δ = 1e-3
                        expected_velocity = [U * (1 +
                            (spatial_cutoff_integral(x + δ, 48.0, U, N).displacement -
                             spatial_cutoff_integral(x - δ, 48.0, U, N).displacement) / (2δ)) for x in xs]
                        @test maximum(abs, interior(velocity[Time(48.0)])[:, 1, 2] .- expected_velocity) < 3e-3
                    end
                end
            end
        end
    end
end
