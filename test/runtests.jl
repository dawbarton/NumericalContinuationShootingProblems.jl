using NumericalContinuation
using NumericalContinuationShootingProblems
using StaticArrays
using Test

NC = NumericalContinuation

@testset "Shooting problems" begin
    hopf = (out, u, p, t) -> (out[1] = p[1]*u[1] - u[2] + p[2]*u[1]*(u[1]^2 + u[2]^2); 
                              out[2] = u[1] + p[1]*u[2] + p[2]*u[2]*(u[1]^2 + u[2]^2))
    prob = NC.ProblemStructure()
    @test_throws ArgumentError add_shootingproblem!(prob, "hopf", hopf, [1.0, 0.0], [1.0, -1.0], [0, 2π], abstol=1e-8, pnames=["α", "β", "γ"])
    add_shootingproblem!(prob, "hopf", hopf, [1.0, 0.0], [1.0, -1.0], [0, 2π], abstol=1e-8)
    NC.initialize!(prob)
    out = ones(6)
    u0 = NC.get_u0(Float64, NC.get_vars(prob))
    d0 = NC.get_data(NC.get_data(prob))
    get_funcs(prob)[:embedded](out, u0, data=d0, prob=prob)
    @test isapprox(out, zero(out), atol=1e-5)

    hopf2 = (u, p, t) -> SVector(p[1]*u[1] - u[2] + p[2]*u[1]*(u[1]^2 + u[2]^2),
                                 u[1] + p[1]*u[2] + p[2]*u[2]*(u[1]^2 + u[2]^2))
    prob = NC.ProblemStructure()
    add_shootingproblem!(prob, "hopf", hopf2, SVector(1.0, 0.0), SVector(1.0, -1.0), [0, 2π], abstol=1e-8)
    NC.initialize!(prob)
    out = ones(6)
    u0 = NC.get_u0(Float64, NC.get_vars(prob))
    d0 = NC.get_data(NC.get_data(prob))
    get_funcs(prob)[:embedded](out, u0, data=d0, prob=prob)
    @test isapprox(out, zero(out), atol=1e-5)
end
