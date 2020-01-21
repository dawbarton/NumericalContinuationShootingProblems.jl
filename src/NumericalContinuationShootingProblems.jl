"""
    NumericalContinuationShootingProblems

This module implements a basic single shooting method for use with ordinary
differential equations (ODEs). The solvers are provided by OrdinaryDiffEq.jl.

This assumes that the ODE takes a `Vector` or `SVector` input and produces a
corresponding output. For small ODEs (i.e., fewer than around 20 dimensions)
`SVector` inputs are often significantly faster.

See [`add_shootingproblem!`](@ref) for details.
"""
module NumericalContinuationShootingProblems

using DocStringExtensions
using NumericalContinuation: add_var!, add_func!, add_pars!
using OrdinaryDiffEq: solve, remake, ODEProblem, Tsit5

export add_shootingproblem!, periodic

struct ShootingProblem{T, O, S, B}
    abstol::T
    reltol::T
    odeprob::O
    solver::S
    bc::B
end

function (shoot::ShootingProblem)(res, u, p, tspan)
    sol = solve(remake(shoot.odeprob, u0=u, p=p, tspan=(tspan[1], tspan[2])), 
        shoot.solver, save_everystep=false, save_start=false, abstol=shoot.abstol, reltol=shoot.reltol)
    shoot.bc(res, u, sol[end], p, tspan)
end

"""
    $SIGNATURES

Periodic boundary conditions for numerical shooting.
"""
periodic(res, u0, u1, p, tspan) = (res .= u1 .- u0)

"""
$SIGNATURES

Construct an ODE shooting problem (boundary value problem) of the form 

```math
   u' = f(u, p, t),
```
for `t∈[t₀, t₁]` where `u` is the state and `p` are the parameters, and add it
to the problem structure. The function `f` is assumed to be an appropriate
input to `ODEProblem` of the OrdinaryDiffEq.jl package.

# Parameters

* `prob` : the underlying continuation problem.
* `name::String` : the name of the algebraic zero problem.
* `f` : the function to use for the zero problem. It takes either three
  arguments (`u`, `p`, and `t`) or four arguments for an in-place version
  (`dudt`, `u`, `p`, and `t`).
* `u0` : the initial state value (typically either a Vector or an SVector).
* `p0` : the initial parameter values.
* `tspan` : the integration time span as a vector or tuple.
* `bc` : (optional) the boundary conditions to use, specified as a function
  with five arguments (`res`, `u0`, `u1`, `p`, `tspan`) where `res` is the
  residual. If not specified, periodic boundary conditions will be used.
* `pnames` : (keyword, optional) the names of the parameters (default:
  auto-generated names).
* `solver` : (keyword, optional) the IVP solver to use (default: Tsit5).
* `reltol` : (keyword, optional) the absolute tolerance for the IVP solver to
  use (default: 1e-6).
* `abstol` : (keyword, optional) the absolute tolerance for the IVP solver to
  use (default: 1e-6).

# Example

The Hopf bifurcation normal form
```
hopf = (out, u, p, t) -> (out[1] = p[1]*u[1] - u[2] + p[2]*u[1]*(u[1]^2 + u[2]^2); 
                          out[2] = u[1] + p[1]*u[2] + p[2]*u[2]*(u[1]^2 + u[2]^2))
prob = ProblemStructure()
add_shootingproblem!(prob, "hopf", hopf, [1.0, 0.0], [1.0, -1.0], [0, 2π], periodic)
```
"""
function add_shootingproblem!(prob, name::String, f, u0, p0, tspan, bc=periodic; pnames=nothing, solver=Tsit5(), reltol=1e-6, abstol=1e-8)
    _tspan = length(tspan) == 1 ? [zero(tspan[1]), tspan[1]] : [tspan[1], tspan[2]]
    odeprob = ODEProblem(f, u0, (_tspan[1], _tspan[2]), p0)
    shooting = ShootingProblem(reltol, abstol, odeprob, solver, bc)
    # Check for parameter names
    _pnames = pnames !== nothing ? [string(pname) for pname in pnames] : ["$(name).p$i" for i in 1:length(p0)]
    if length(_pnames) != length(p0)
        throw(ArgumentError("Length of parameter vector does not match number of parameter names"))
    end
    # Create the necessary continuation variables and add the function
    uidx = add_var!(prob, "$(name).u", length(u0), u0=u0)
    pidx = add_var!(prob, "$(name).p", length(p0), u0=p0)
    tidx = add_var!(prob, "$(name).tspan", 2, u0=_tspan)
    fidx = add_func!(prob, name, length(u0), shooting, [uidx, pidx, tidx])
    add_pars!(prob, _pnames, pidx, active=false)
    add_pars!(prob, ("$(name).t0", "$(name).t1"), tidx, active=false)
    return prob
end

end # module
