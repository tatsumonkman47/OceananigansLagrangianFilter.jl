```@meta
EditURL = "../../../examples/offline_filter_spatial_cutoff.jl"
```


# Offline filtering with a spatially varying cutoff

This manufactured example has a known particle trajectory, so we can check the
PDE filter against direct integration along that trajectory. A uniform velocity
carries particles across a stationary cutoff mask M(x) = 1 + 0.5 cos(x).
The reference cutoff is 1, giving local cutoffs between 0.5 and 1.5.

The spatial filter uses a clock dτ = M(x(t)) dt along each trajectory. It is a
Butterworth-squared filter in τ. Its kernel in physical time includes the
Jacobian M(x(s)), which ensures that constants are preserved. When M is constant
along a trajectory, this reduces to the usual filter with cutoff M × freq_c.
See the offline filtering equations for the precise definition and limitations.

## Save analytic input data

````julia
using OceananigansLagrangianFilter
using Oceananigans.Units: Time
using Oceananigans.OutputWriters: write_output!
using Printf

filename_stem = "spatial_cutoff_advection"
grid = RectilinearGrid(CPU(); size = (48, 4), x = (0, 2π), z = (-1, 0),
                       topology = (Periodic, Flat, Bounded))

U = 1.0
ε = 0.5
ω = 2.0
````

Use the ordinary Oceananigans model and output writer to store prescribed
snapshots. We do not step this model: both velocity and tracer histories are
known exactly. The z direction is invariant and permits the existing x-z
mean-position regridder to be exercised.

````julia
model = NonhydrostaticModel(grid; tracers = (:constant, :material, :signal))
set!(model.velocities.u, U)
set!(model.tracers.constant, 1)
outputs = (; model.tracers..., u = model.velocities.u, w = model.velocities.w)
writer = JLD2Writer(model, outputs;
                    filename = filename_stem * ".jld2",
                    schedule = TimeInterval(0.1),
                    array_type = Array{Float64}, overwrite_existing = true)
````

The material tracer is constant following each particle. The signal tracer
oscillates in time and is attenuated by the filter.

````julia
for (n, t) in enumerate(0:0.1:96)
    set!(model.tracers.material, (x, z) -> 1 + 0.2cos(x - U * t))
    set!(model.tracers.signal, cos(ω * t))
    model.clock.time = t
    model.clock.iteration = n - 1
    write_output!(writer, model)
end
````

## Perform offline filtering

````julia
cutoff_mask = Field{Center, Nothing, Nothing}(grid)
set!(cutoff_mask, x -> 1 + ε * cos(x))

filter_config = OfflineFilterConfig(
    original_data_filename = filename_stem * ".jld2",
    output_filename = filename_stem * "_filtered.jld2",
    forward_output_filename = filename_stem * "_forward.jld2",
    backward_output_filename = filename_stem * "_backward.jld2",
    var_names_to_filter = ("constant", "material", "signal"),
    velocity_names = ("u", "w"), grid = grid, backend = InMemory(),
    N = 2, freq_c = 1.0, cutoff_mask = cutoff_mask,
    Δt = 0.08, T_out = 4.0, advection = Centered(order = 4),
    map_to_mean = true, compute_mean_velocities = true,
    compute_Eulerian_filter = false, output_netcdf = false,
    delete_intermediate_files = true)

run_offline_Lagrangian_filter(filter_config)
````

## Compare with direct trajectory integration

A particle at x at time t was at x + U s at time t + s (using unwrapped x for
displacements). The elapsed filter time is
τ(t+s) - τ(t) = s + ε/U [sin(x+Us) - sin(x)].
For N=2 and reference cutoff 1, the kernel is
G(r) = exp(-|r|/√2) [cos(|r|/√2) + sin(|r|/√2)] / (2√2).
We integrate M(x+Us) G(τ(t+s)-τ(t)) times the signal or displacement.

````julia
function trajectory_reference(x, t; U = 1.0, ε = 0.5, ω = 2.0)
    signal = displacement = mass = 0.0
    h = 0.005
    for n in 0:32000
        s = -80 + n * h
        τ = s + ε / U * (sin(x + U * s) - sin(x))
        r = abs(τ) / sqrt(2)
        G = exp(-r) * (cos(r) + sin(r)) / (2sqrt(2))
        weight = (n == 0 || n == 32000) ? 1 : isodd(n) ? 4 : 2
        K = weight * (1 + ε * cos(x + U * s)) * G * h / 3
        mass += K
        signal += K * cos(ω * (t + s))
        displacement += K * U * s
    end
    return (; mass, signal, displacement)
end
````

Compare halfway through the stored interval, away from endpoint transients.

````julia
t = 48.0
xs = collect(xnodes(CenterField(grid)))
reference = [trajectory_reference(x, t; U, ε, ω) for x in xs]
read_profile(name) = interior(FieldTimeSeries(filter_config.output_filename, name)[Time(t)])[:, 1, 2]
constant = read_profile("constant_Lagrangian_filtered")
material = read_profile("material_Lagrangian_filtered")
signal = read_profile("signal_Lagrangian_filtered")
displacement = read_profile("xi_u")

@printf("Maximum constant-tracer error:       %.3e\n", maximum(abs, constant .- 1))
@printf("Maximum material-tracer error:       %.3e\n", maximum(abs, material .- (1 .+ 0.2cos.(xs .- U*t))))
@printf("Maximum signal quadrature error:     %.3e\n", maximum(abs, signal .- [r.signal for r in reference]))
@printf("Maximum displacement error:          %.3e\n", maximum(abs, displacement .- [r.displacement for r in reference]))
````

The CPU example run with Julia 1.10.11 gives:

````text
Maximum constant-tracer error:       0.000e+00
Maximum material-tracer error:       1.689e-04
Maximum signal quadrature error:     8.234e-04
Maximum displacement error:          1.675e-05
````

A compact profile illustrates agreement across the cutoff range. These are
values at instantaneous positions; `_at_mean` fields in the JLD2 file are the
additional interpolation to the computed mean positions.

````julia
@printf("\n       x    cutoff    PDE signal    reference\n")
for i in 1:8:length(xs)
    @printf("%8.3f  %8.3f  %12.6f  %12.6f\n", xs[i], 1 + ε*cos(xs[i]), signal[i], reference[i].signal)
end
````

The input and final JLD2 files are retained for inspection. For example,
FieldTimeSeries(filter_config.output_filename, "signal_Lagrangian_filtered_at_mean")
loads the remapped signal. Reduce the input sampling interval, Δt, and grid
spacing to check convergence; increasing the time window reduces endpoint error.

Spatial masks also support boundary relaxation, using a separate relaxation
mask to select the boundary region. The optional Eulerian post-processing
filter does not yet support spatial masks. This example validates the adaptive
filter, not the physical suitability of a cutoff for a particular ocean flow.
