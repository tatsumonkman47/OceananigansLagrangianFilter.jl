using Oceananigans.OutputWriters: write_output!
using Oceananigans.Units: Time

# Prescribe snapshots for particles moving at unit speed through M(x) = 1 + 0.5cos(x).
function write_spatial_cutoff_input(filename, grid)
    model = OceananigansLagrangianFilter.LagrangianFilter(grid; tracers = (:constant, :signal))
    set!(model.velocities.u, 1.0)
    set!(model.velocities.w, 0.0)
    set!(model.tracers.constant, 1.0)
    outputs = (; model.tracers..., u = model.velocities.u, w = model.velocities.w)
    writer = JLD2Writer(model, outputs; filename, schedule = TimeInterval(0.1),
                       array_type = Array{Float64}, overwrite_existing = true)
    for (n, t) in enumerate(0:0.1:64)
        set!(model.tracers.signal, cos(2t))
        model.clock.time = t
        model.clock.iteration = n - 1
        write_output!(writer, model)
    end
end

# Independent N=2 reference: integrate M(x+s) G(τ(t+s)-τ(t)) along a trajectory.
function spatial_cutoff_reference(x, t; h = 0.01, horizon = 32.0)
    signal = mass = 0.0
    nsteps = round(Int, 2horizon / h)
    for n in 0:nsteps
        s = -horizon + n * h
        M = 1 + 0.5cos(x + s)
        τ = s + 0.5(sin(x + s) - sin(x))
        r = abs(τ) / sqrt(2)
        G = exp(-r) * (cos(r) + sin(r)) / (2sqrt(2))
        weight = (n == 0 || n == nsteps) ? 1 : isodd(n) ? 4 : 2
        K = weight * M * G * h / 3
        mass += K
        signal += K * cos(2(t + s))
    end
    return (; mass, signal)
end

@testset "Spatial cutoff follows the normalized trajectory kernel" begin
    mktempdir() do directory
        grid = RectilinearGrid(CPU(); size = (48, 4), x = (0, 2π), z = (-1, 0),
                               topology = (Periodic, Flat, Bounded))
        mask = Field{Center, Nothing, Nothing}(grid)
        set!(mask, x -> 1 + 0.5cos(x))
        input = joinpath(directory, "input.jld2")
        output = joinpath(directory, "filtered.jld2")
        write_spatial_cutoff_input(input, grid)

        config = OfflineFilterConfig(original_data_filename = input,
                    output_filename = output,
                    forward_output_filename = joinpath(directory, "forward.jld2"),
                    backward_output_filename = joinpath(directory, "backward.jld2"),
                    var_names_to_filter = ("constant", "signal"),
                    velocity_names = ("u", "w"), grid = grid, backend = InMemory(),
                    N = 2, freq_c = 1.0, cutoff_mask = mask,
                    Δt = 0.08, T_out = 4.0, advection = Centered(order = 4),
                    map_to_mean = false, compute_mean_velocities = false,
                    output_netcdf = false, delete_intermediate_files = true)
        run_offline_Lagrangian_filter(config)

        t = 32.0 # Exclude startup and turnaround transients.
        constant = FieldTimeSeries(output, "constant_Lagrangian_filtered")[Time(t)]
        signal = FieldTimeSeries(output, "signal_Lagrangian_filtered")[Time(t)]
        xs = collect(xnodes(CenterField(grid)))
        reference = [spatial_cutoff_reference(x, t) for x in xs]
        @test maximum(abs, interior(constant) .- 1) < 2e-6
        @test maximum(abs, [r.mass for r in reference] .- 1) < 1e-6
        @test maximum(abs, interior(signal)[:, 1, 2] .- [r.signal for r in reference]) < 3e-3
    end
end
