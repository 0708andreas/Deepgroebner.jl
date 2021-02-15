using Deepgroebner
using Test

@testset "Deepgroebner.jl" begin
    f = [term(1, (1, 2, 3)), term(3, (3, 2, 1))]
    g = [term(4, (1, 2, 3)), term(1, (1, 2, 0))]
    h = [term(2, (1, 2, 3)), term(3, (3, 2, 1))]
    t = term(2, (1, 2, 2))
    s = term(4, (2, 2, 1))
    @test LT(f) == term(3, (3, 2, 1))
    @test ! Deepgroebner.div(t, LT(f))
    @test   Deepgroebner.div(s, LT(f))
    @test minus(f, g) == [term(-3, (1, 2, 3)), term(-1, (1, 2, 0)), term(3, (3, 2, 1))]
    @test mdiv([term(1, (1, 2, 3)), term(1, (0, 1, 1))], [[term(1, (1, 0, 0))]]) == [term(1.0, (0, 1, 1))]
end
