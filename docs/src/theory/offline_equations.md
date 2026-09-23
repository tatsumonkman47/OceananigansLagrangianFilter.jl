# Offline Lagrangian filtering equations

This page describes the Lagrangian filtering equations for the 'offline' configuration of *OceananigansLagrangianFilter.jl*. The offline scheme runs a forward pass very similar to the [online configuration](@ref "Online Lagrangian filtering equations"), before running a backward pass through the offline data and combinging the backward and forward outputs. 

We compute the Lagrangian mean of some scalar ``f`` as
```math
\begin{equation}\label{fstardef}
    f^*(\vb*{\varphi}(\vb*{a},t),t) = \int_{-\infty}^\infty G(t-s)f(\vb*{\varphi}(\vb*{a},s),s) \, \mathrm{d} s\,,
\end{equation}
```
and optionally compute
```math
\begin{equation}\label{Xidefonline}
\vb*{\Xi}(\vb*{\varphi}(\vb*{a},t),t) = \int_{-\infty}^\infty \alpha e^{-\alpha(t-s)}\vb*{\varphi}(\vb*{a},s)\,\mathrm{d} s\,,
\end{equation}
```
so that the generalised Lagrangian mean ``\bar{f}^{\mathrm{L}}`` (see definition in [Lagrangian averaging](@ref "Lagrangian averaging")) can be recovered by a post-processing interpolation step using
```math
\begin{equation}
\bar{f}^{\mathrm{L}}(\vb*{\Xi}(\vb*{x},t),t) = f^*(\vb*{x},t)\,.
\end{equation}
```
We consider even filter kernels composed of sums of exponentials of the absolute value of ``t``:
```math
\begin{equation}
    G(t) = \sum_{n=1}^{N/2} e^{-c_n |t|} \left( a_n \cos(d_n |t|) + b_n \sin(d_n |t|) \right)\,.
\end{equation}
```
We define a set of ``N`` weight functions, for even ``N``. For ``k = 1,...,N/2`` we have
```math
\begin{align}
    G_{Ck}(t) &=e^{-c_k|t|}\cos d_k |t|\,, \\
    G_{Sk}(t) &=e^{-c_k|t|}\sin d_k |t|\,, \\
\end{align}
```
For ``t>0``, we have
```math
\begin{align} \label{forward_G_derivs}
    G_{Ck}'(t) &= - c_kG_{Ck}(t) - d_k G_{Sk}(t) \\
    G_{Sk}'(t) &= - c_kG_{Sk}(t) + d_k G_{Ck}(t) \\
\end{align}
```
and for ``t<0``
```math
\begin{align} \label{backward_G_derivs}
    G_{Ck}'(t) &= -G_{Ck}'(-t) = c_kG_{Ck}(t) + d_k G_{Sk}(t) \\
    G_{Sk}'(t) &= -G_{Sk}'(-t) = c_kG_{Sk}(t) - d_k G_{Ck}(t) \\
\end{align}
```
We then define a corresponding set of ``N`` filtered scalars
```math
\begin{align}\label{forwardgdef}
    g_{Ck}(\vb*{\varphi}(\vb*{a},t),t) &= \int_{-\infty}^t G_{Ck}(t-s)f(\vb*{\varphi}(\vb*{a},s),s)\, \mathrm{d} s\\
    g_{Sk}(\vb*{\varphi}(\vb*{a},t),t) &= \int_{-\infty}^t G_{Sk}(t-s)f(\vb*{\varphi}(\vb*{a},s),s)\, \mathrm{d} s\\
\end{align}
```
so that 
```math
\begin{equation}
    f_1^*(\vb*{x},t) = \sum_{n=1}^{N/2} a_n g_{Cn}(\vb*{x},t) + b_n g_{Sn}(\vb*{x},t)\,.
\end{equation}
```
We derive PDEs for the ``g_{Ck}`` and ``g_{Sk}`` by taking the time derivative of \eqref{forwardgdef}:
```math
\begin{align}
\frac{\partial g_{Ck}}{\partial t} + \vb*{u} \cdot \nabla g_{Ck} &= f - c_k g_{Ck} - d_k g_{Sk} \\
\frac{\partial g_{Sk}}{\partial t} + \vb*{u} \cdot \nabla g_{Sk} &=  - c_k g_{Sk} + d_k g_{Ck}
\end{align}
```
This system of equations can be solved with initial conditions ``g_{Ck}(\vb*{x},0) = g_{Sk}(\vb*{x},0)=0``.

If we want to find ``\bar{f}^{\mathrm{L}}``, we define map functions with which to interpolate ``f^*`` to ``\bar{f}^{\mathrm{L}}`` after the simulation. We define 
```math
\begin{align}
    \vb*{\Xi}_{Ck}(\vb*{\varphi}(\vb*{a},t),t) &= \int_{-\infty}^t G_{Ck}(t-s)\vb*{\varphi}(\vb*{a},s) \mathrm{d} s\\
    \vb*{\Xi}_{Sk}(\vb*{\varphi}(\vb*{a},t),t) &= \int_{-\infty}^t G_{Sk}(t-s)\vb*{\varphi}(\vb*{a},s) \mathrm{d} s\,,
\end{align}
```
so that
```math
\begin{equation}
\vb*{\Xi}(\vb*{x},t) = \sum_{n=1}^{N/2} a_n \vb*{\Xi}_{Ck}(\vb*{x},t) + b_n \vb*{\Xi}_{Sk}(\vb*{x},t)
\end{equation}
```
and
```math
\begin{align}
\frac{\partial \vb*{\Xi}_{Ck}}{\partial t} + \vb*{u} \cdot \nabla \vb*{\Xi}_{Ck} &= \vb*{x} - c_k \vb*{\Xi}_{Ck} - d_k \vb*{\Xi}_{Sk} \\
\frac{\partial \vb*{\Xi}_{Sk}}{\partial t} + \vb*{u} \cdot \nabla \vb*{\Xi}_{Sk} &=  - c_k \vb*{\Xi}_{Sk} + d_k \vb*{\Xi}_{Ck}\,.
\end{align}
```
To define perturbation equations, we set:
```math
\begin{align}
    \vb*{\Xi}_{Ck} &= \vb*{\xi}_{Ck} + \frac{c_k}{c_k^2 + d_k^2}\vb*{x} \\
    \vb*{\Xi}_{Sk} &= \vb*{\xi}_{Sk} + \frac{d_k}{c_k^2 + d_k^2}\vb*{x}\,,
\end{align}
```
where the coefficients of ``\vb*{x}`` are needed because each of the filters ``G_{Ck}`` and ``G_{Sk}`` are not individually normalised over the interval ``[-\infty,0]``.
The perturbation map equations are then given by
```math
\begin{align}
\frac{\partial \vb*{\xi}_{Ck}}{\partial t} + \vb*{u} \cdot \nabla \vb*{\xi}_{Ck} &= -\frac{c_k}{c_k^2 + d_k^2}\vb*{u} - c_k \vb*{\xi}_{Ck} - d_k \vb*{\xi}_{Sk} \\
\frac{\partial \vb*{\xi}_{Sk}}{\partial t} + \vb*{u} \cdot \nabla \vb*{\xi}_{Sk} &= -\frac{d_k}{c_k^2 + d_k^2}\vb*{u} - c_k \vb*{\xi}_{Sk} + d_k \vb*{\xi}_{Ck}\,.
\end{align}
```
Backward-pass equations of the same form are solved by time-reversing the velocity and field data, and changing the sign of the velocity. The final filtered field is then reconstructed by summing the forwards and backwards pass outputs at each time. 

## Spatially varying cutoff equations

The spatial-cutoff option extends the constant-coefficient filter above by choosing
a strictly positive, stationary mask ``M(\boldsymbol{x})`` and defining a filter
clock along each particle trajectory ``\boldsymbol{X}(t)``:

```math
\tau(t)-\tau(s)=\int_s^t M(\boldsymbol{X}(r))\,\mathrm{d}r.
```

The adaptive mean at the instantaneous particle position is defined by

```math
f^*(\boldsymbol{X}(t),t)=\int_{-\infty}^{\infty}
 G\bigl(\tau(t)-\tau(s)\bigr) f(\boldsymbol{X}(s),s)
 M(\boldsymbol{X}(s))\,\mathrm{d}s.
```

Here ``G`` is the **reference** kernel, using the unscaled coefficients from
`freq_c`. Changing variables from ``s`` to ``\tau(s)`` shows that the kernel
integrates to one whenever the reference kernel does. Consequently a constant,
or a materially conserved tracer, is preserved in the infinite-window,
continuum limit. The finite computation retains the usual endpoint and
discretization errors.

For constant ``M=m`` along a trajectory, the kernel becomes ``mG(m(t-s))``,
with physical-frequency response ``1/[1+(\omega/(m\,\mathrm{freq_c}))^{2N}]``.
For varying ``M`` this is a Butterworth-squared filter in ``\tau``, and has no
single time-invariant transfer function in physical time. In particular it is
different from freezing ``M(\boldsymbol{X}(t))`` throughout the averaging window.
This clock-based definition is an extension of the constant-cutoff formulation;
it is not an assertion that arbitrary coefficient substitution preserves the
original physical-time convolution.

Writing ``D_t=\partial_t+\boldsymbol{u}\cdot\nabla``, the forward tracer equations
are

```math
D_t g_{Ck}=M(f-c_k g_{Ck}-d_k g_{Sk}),\qquad
D_t g_{Sk}=M(-c_k g_{Sk}+d_k g_{Ck}).
```

Reconstruct with the reference coefficients ``\sum_k(a_k g_{Ck}+b_k g_{Sk})``.
Both the source and decay terms must carry ``M``. Scaling only decay and then
multiplying the output by the local mask generally fails to preserve constants
when particles cross gradients.

For the position equations, define
``\kappa_{Ck}=c_k/(c_k^2+d_k^2)`` and
``\kappa_{Sk}=d_k/(c_k^2+d_k^2)`` using reference coefficients. Subtracting
``\kappa_{Ck}\boldsymbol{x}`` and ``\kappa_{Sk}\boldsymbol{x}`` from the
corresponding filtered position variables gives

```math
\begin{aligned}
D_t\boldsymbol{\xi}_{Ck}
 &= -M(c_k\boldsymbol{\xi}_{Ck}+d_k\boldsymbol{\xi}_{Sk})
    -\kappa_{Ck}\boldsymbol{u},\\
D_t\boldsymbol{\xi}_{Sk}
 &= M(-c_k\boldsymbol{\xi}_{Sk}+d_k\boldsymbol{\xi}_{Ck})
    -\kappa_{Sk}\boldsymbol{u}.
\end{aligned}
```

The displacement reconstruction also uses reference ``a_k,b_k``. The existing
forward/backward sum and interpolation to mean position still apply. The
physical-time derivative of the mean map uses one factor of ``M`` multiplying
the reference mean-velocity reconstruction, followed by the usual subtraction
of the backward contribution. With a varying clock, this map-derived mean
velocity need not equal the adaptive scalar filter applied to velocity.

The locally equilibrated initialization is ``g_{Ck}=\kappa_{Ck}f`` and
``g_{Sk}=\kappa_{Sk}f``. The corresponding displacement initialization is

```math
\boldsymbol{\xi}_{Ck}
=\frac{d_k^2-c_k^2}{M(c_k^2+d_k^2)^2}\boldsymbol{u},\qquad
\boldsymbol{\xi}_{Sk}
=\frac{-2c_kd_k}{M(c_k^2+d_k^2)^2}\boldsymbol{u}.
```

These initialize local equilibria, not the unknown prehistory of a particle
crossing gradients. Exclude endpoint transients when comparing with the
infinite-window definition. The single-exponential case follows by setting
``d_1=0`` and omitting the sine variables.
