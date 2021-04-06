using Random
using StableRNGs
using ReinforcementLearning

include("groebner.jl")

export GroebnerEnvParams,
    GroebnerEnv,
    rand_env,
    buchberger_test,
    eval_model

struct GroebnerEnvParams
    nvars::Int
    maxdeg::Int
    npols::Int
end

struct MatrixSpace{T}
    size::Tuple{Int, Int}
end

Base.in(x, s::MatrixSpace{T}) where T = (x isa Array{T,2}) && size(x) == s.size
Base.rand(s::MatrixSpace{T}) where T = rand(T, size)

Base.show(io::IO, params::GroebnerEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(GroebnerEnvParams)], ",")
)

mutable struct GroebnerEnv{N, R<:AbstractRNG} <: AbstractEnv
    params::GroebnerEnvParams
    G::Array{Array{term{N}, 1}, 1}
    P::Vector{NTuple{2, Array{term{N}, 1}}}
    reward::Int
    done::Bool
    t::Int
    rng::R
end

function rand_env(p::GroebnerEnvParams)
    G = [[term(1., ntuple(x -> rand(1:p.maxdeg), p.nvars)),
              term(1., ntuple(x -> rand(1:p.maxdeg), p.nvars))]
             for _ in 1:p.npols]
    P = [(G[i], G[j])
             for i in 1:length(G)
             for j in i:length(G)]
    return GroebnerEnv(p, G, P, 0, false, 0, Random.GLOBAL_RNG)
end



p(env::GroebnerEnv) = length(env.P)

RLBase.action_space(env::GroebnerEnv) = env.done ? ([1]) : (1:p(env))
RLBase.state_space(env::GroebnerEnv) = MatrixSpace{Int}((p(env),
                                                    4*env.params.nvars))
RLBase.reward(env::GroebnerEnv) = env.reward
RLBase.is_terminated(env::GroebnerEnv) = env.done
RLBase.state(env::GroebnerEnv) = !env.done ?
    hcat([vcat(vcat(collect.(getproperty.(f,:a))...),
               vcat(collect.(getproperty.(g,:a))...)
               )
          for (f, g) in env.P]...) : Array{Int}(undef, 0, 0)


function RLBase.reset!(env::GroebnerEnv{N, R}) where {N, R}
    env.done = false
    env.reward = 0
    # env.t = 0
    env.G = [[term(1., ntuple(x -> rand(1:env.params.maxdeg), env.params.nvars)),
              term(1., ntuple(x -> rand(1:env.params.maxdeg), env.params.nvars))]
             for _ in 1:env.params.npols]
    env.P = [(env.G[i], env.G[j])
             for i in 1:length(env.G)
             for j in i:length(env.G)]
    env.t = 0
end

function buchberger_test(env::GroebnerEnv, model)
    i = 0
    reward = 0
    while length(env.P) > 0
        env(model(state(env)))
        i = i+1
        reward = reward + env.reward
    end
    return i, reward
end

function eval_model(env::GroebnerEnv, model)
    iters = 0
    reward = 0
    for i in 1:100
        RLBase.reset!(env)
        (i, r) = buchberger_test(env, model)
        @assert is_groebner_basis(env.G)
        iters = iters + i
        reward = reward + r
    end
    return Base.:/(iters, 100), Base.:/(reward, 100)
end


function (env::GroebnerEnv{N, R})(a) where {N, R}
    @assert a in 1:p(env)
    (f, g) = env.P[a]
    deleteat!(env.P, a)
    r, reward = mdiv_count(S(f, g), env.G)
    if length(r) == 1
        append!(r, [term{N}(0.0, ntuple(x->0, N))])
    end
    
    reward = -1*(1 + reward) # +1 from S-poly construction
    if r != []
        env.P = update!(env.P, env.G, r)
        env.G = push!(env.G, r)
    end
    if length(env.P) == 0
        env.done = true
    end
    env.reward = reward
    env.t = env.t + 1
    if env.t > 10_000
        env.done = true
        env.reward = -1_000_000
        println("Stopped after 10_000 selections")
    end
end




# struct CartPoleEnvParams{T}
#     gravity::T
#     masscart::T
#     masspole::T
#     totalmass::T
#     halflength::T
#     polemasslength::T
#     forcemag::T
#     dt::T
#     thetathreshold::T
#     xthreshold::T
#     max_steps::Int
# end

# Base.show(io::IO, params::CartPoleEnvParams) = print(
#     io,
#     join(["$p=$(getfield(params, p))" for p in fieldnames(CartPoleEnvParams)], ","),
# )

# mutable struct CartPoleEnv{T,R<:AbstractRNG} <: AbstractEnv
#     params::CartPoleEnvParams{T}
#     state::Array{T,1}
#     action::Int
#     done::Bool
#     t::Int
#     rng::R
# end

# """
#     CartPoleEnv(;kwargs...)
# # Keyword arguments
# - `T = Float64`
# - `gravity = T(9.8)`
# - `masscart = T(1.0)`
# - `masspole = T(0.1)`
# - `halflength = T(0.5)`
# - `forcemag = T(10.0)`
# - `max_steps = 200`
# - 'dt = 0.02'
# - `rng = Random.GLOBAL_RNG`
# """
# function CartPoleEnv(;
#     T = Float64,
#     gravity = 9.8,
#     masscart = 1.0,
#     masspole = 0.1,
#     halflength = 0.5,
#     forcemag = 10.0,
#     max_steps = 200,
#     dt = 0.02,
#     rng = Random.GLOBAL_RNG,
# )
#     params = CartPoleEnvParams{T}(
#         gravity,
#         masscart,
#         masspole,
#         masscart + masspole,
#         halflength,
#         masspole * halflength,
#         forcemag,
#         dt,
#         Base.:/(2 * 12 * π , 360),
#         2.4,
#         max_steps,
#     )
#     high = cp = CartPoleEnv(params, zeros(T, 4), 2, false, 0, rng)
#     reset!(cp)
#     cp
# end

# CartPoleEnv{T}(; kwargs...) where {T} = CartPoleEnv(; T = T, kwargs...)

# function RLBase.reset!(env::CartPoleEnv{T}) where {T<:Number}
#     env.state[:] = T(0.1) * rand(env.rng, T, 4) .- T(0.05)
#     env.t = 0
#     env.action = 2
#     env.done = false
#     nothing
# end

# RLBase.action_space(env::CartPoleEnv) = Base.OneTo(2)

# RLBase.state_space(env::CartPoleEnv{T}) where {T} = Space(
#     ClosedInterval{T}[
#         (-2*env.params.xthreshold)..(2*env.params.xthreshold),
#         -1e38..1e38,
#         (-2*env.params.thetathreshold)..(2*env.params.thetathreshold),
#         -1e38..1e38,
#     ],
# )

# RLBase.reward(env::CartPoleEnv{T}) where {T} = env.done ? zero(T) : one(T)
# RLBase.is_terminated(env::CartPoleEnv) = env.done
# RLBase.state(env::CartPoleEnv) = env.state

# function (env::CartPoleEnv)(a)
#     @assert a in (1, 2)
#     env.action = a
#     env.t += 1
#     force = a == 2 ? env.params.forcemag : -env.params.forcemag
#     x, xdot, theta, thetadot = env.state
#     costheta = cos(theta)
#     sintheta = sin(theta)
#     tmp = (force + env.params.polemasslength * thetadot^2 * sintheta) / env.params.totalmass
#     thetaacc =
#         (env.params.gravity * sintheta - costheta * tmp) / (
#             env.params.halflength *
#             (4 / 3 - env.params.masspole * costheta^2 / env.params.totalmass)
#         )
#     xacc = tmp - env.params.polemasslength * thetaacc * costheta / env.params.totalmass
#     env.state[1] += env.params.dt * xdot
#     env.state[2] += env.params.dt * xacc
#     env.state[3] += env.params.dt * thetadot
#     env.state[4] += env.params.dt * thetaacc
#     env.done =
#         abs(env.state[1]) > env.params.xthreshold ||
#         abs(env.state[3]) > env.params.thetathreshold ||
#         env.t > env.params.max_steps
#     nothing
# end

# Random.seed!(env::CartPoleEnv, seed) = Random.seed!(env.rng, seed)
