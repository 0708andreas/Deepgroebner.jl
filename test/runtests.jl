using Deepgroebner
using Test

@testset "Deepgroebner.jl" begin
    f = [term(1, (1, 2, 3)), term(3, (3, 2, 1))]
    t = term(2, (1, 2, 2))
    s = term(4, (2, 2, 1))
    @test LT(f) == term(3, (3, 2, 1))
    @test ! Deepgroebner.div(t, LT(f))
    @test   Deepgroebner.div(s, LT(f))
end
