using Flux
using ReinforcementLearning
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Zygote
using Zygote: @adjoint
using Dates
using TensorBoardLogger
using StableRNGs
using Logging
using Setfield
using SliceMap
using StatsBase
using ElasticArrays

include("GroebnerEnv.jl")


# From https://github.com/FluxML/Zygote.jl/issues/317 in order to solve
# MethodError: no method matching iterate(::Nothing)
fixnothing(x) = x
@adjoint fixnothing(x) = fixnothing(x), function(y)
    if y === nothing || y isa AbstractArray{Nothing}
        return (zero(x),)
    else
        return (y,)
    end
end

export Replicate,
    experiment,
    pg_experiment,
    strat_rand,
    strat_real_first,
    strat_normal,
    strat_degree

struct Batch{A}
    b::A
end

struct Replicate{F, T}
    reducer::F
    model::T
end

# @functor Replicate
Flux.trainable(m::Replicate) = Flux.trainable(m.model)

# (m::Replicate)(x::AbstractArray) = m.reducer(slicemap(m.model, x; dims=1))
(m::Replicate)(x::AbstractArray) = m.reducer(m.model(x))
(m::Replicate)(x::Batch{<:AbstractArray}) = m.(x.b)
# (m::Replicate)(x::AbstractArray) = m.reducer([m.model(c) for c in eachcol(x)])
(m::Replicate)(xs::Vararg{<:AbstractArray}) = m.(xs)
(m::Replicate)(xs::Tuple) = m(xs...)

Base.getindex(m::Replicate, i::Integer) = model
Base.getindex(m::Replicate, i::AbstractVector) = Replicate(m.reducer, m.model)

function Base.show(io::IO, m::Replicate)
    print(io, "Replicate(", m.reducer, ", ")
    print(io, m.model)
    print(io, ")")
end



mutable struct MyDQNLearner{Q,F,R} <: AbstractLearner
    approximator::Q
    loss_func::F
    γ::Float32
    sampler::BatchSampler
    min_replay_history::Int
    rng::R
    # for debugging
    loss::Float32
end

Flux.functor(x::MyDQNLearner) = (Q = x.approximator,), y -> begin
    x = @set x.approximator = y.Q
    x
end

(learner::MyDQNLearner)(env) =
    env |>
    state |>
    x -> send_to_device(device(learner), x) |> learner.approximator |> send_to_host
# (learner::MyDQNLearner)(env) =
#     learner.approximator(state(env))

# (learner::MyDQNLearner)(env) =
#     env |>
#     state |>
#     learner.approximator

function MyDQNLearner(;
    approximator::Q,
    loss_func::F = huber_loss,
    γ = 0.99f0,
    batch_size = 32,
    min_replay_history = 32,
    rng = Random.GLOBAL_RNG,
) where {Q,F}
    MyDQNLearner{Q,F,typeof(rng)}(
        approximator,
        loss_func,
        γ,
        BatchSampler{SARTS}(batch_size),
        min_replay_history,
        rng,
        0.0,
    )
end

function RLBase.update!(learner::MyDQNLearner, traj::AbstractTrajectory)
    Q = learner.approximator
    γ = learner.γ
    loss_func = learner.loss_func

    s = traj[:state]
    a = traj[:action]
    r = traj[:reward]
    t = traj[:terminal]

    send_to_device(device(Q), s)

    if length(r) == 0 || length(s) == 0 || length(t) == 0 || length(a) == 0
        return
    end



    # s_ = s[2:end]
    # s = s[1:end-1]
    # a = a[1:end-1]
    # println("a = ", a)

    # if ! t[end]
    #     return
    # end
    # println(r[end])
    # println(s[1])
    gs = gradient(params(Q)) do
        q = [Q(s[i])[a[i]] for i in 1:(length(s)-1)]
        # qs = [Q(s[i]) for i in 1:(length(s))]
        # q = [qs[i][a[i]] for i in 1:(length(s)-1)]
        # q_ = vcat([r[i] + maximum(Q(s[i+1])) for i in 1:(length(s) - 1)], [r[end]])
        # q_ = [t[i] ? r[i] : r[i] + maximum(Q(s[i+1])) for i in 1:(length(s)-1)]
        #
        xyz = 2+2
        qs = [(sort(Q(s[i])))[end] for i in 1:(length(s))]
        q_ = [r[i] + (1 - t[i]) * qs[i+1] for i in 1:(length(s)-1)]

        # Zygote.ignore() do
        #     println(q)
        #     println(Q(s[1]))
        #     println(q_)
        # end
        

        # loss = Flux.huber_loss(q, q_)
        loss = sum([(fixnothing(q[i] - q_[i]))^2 for i in 1:length(q)])

        Zygote.ignore() do
            learner.loss = loss
            if any(t)
                # println(findall(t))
            end

        end
        
        loss
    end
    # dump(gs)
    # println("Got to update!(Q, gs)")
    # println(gs.grads)
    RLBase.update!(Q, gs)

end




function experiment(
    seed = 123,
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_CartPole_$(t)")
    end
    log_dir = joinpath(save_dir, "tb_log")
    lg = TBLogger(log_dir, min_level = Logging.Info)
    rng = StableRNG(seed)

    params = GroebnerEnvParams(3, 4, 3)
    env = GroebnerEnv{3, StableRNG}(params,
                                    Array{Array{term{3},1},1}[],
                                    Array{NTuple{2, Array{term{3}, 1}}}[],
                                    0,
                                    false,
                                    0,
                                    rng)
    RLBase.reset!(env)
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = MyDQNLearner(
                approximator = NeuralNetworkApproximator(
                    # model = Replicate(x -> softmax(x; dims=2)[:], Chain(
                    model = Replicate(x -> dropdims(x; dims=1), Chain(
                        Dense(4*3, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, 1, x -> x; initW = glorot_uniform(rng)),
                    )) |> gpu ,
                    # model = Replicate(x -> [y[1] for y in x], Chain(
                    #     Dense(4*3, 10, relu, initW = glorot_uniform(rng)),
                    #     Dense(10, 1, relu, initW = glorot_uniform(rng))
                    # )) |> gpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                # batch_size = 1,
                min_replay_history = 100,
                loss_func = Flux.huber_loss,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularVectorSARTTrajectory(
            capacity = 1000,
            # state = Array{Int, 2} => (),
            state = Array{Int, 2},
            # next_state = Array{Int, 2}
        ),
    )

    stop_condition = StopAfterEpisode(1000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        # DoEveryNStep() do t, agent, env
        #     with_logger(lg) do
        #         @info "training" loss = agent.policy.learner.loss
        #     end
        # end,
        # DoEveryNEpisode() do t, agent, env
        #     with_logger(lg) do
        #         @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
        #             0
        #     end
        # end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.
    You can view the runtime logs with `tensorboard --logdir $log_dir`.
    Some useful statistics are stored in the `hook` field of this experiment.
    """

    Experiment(agent, env, stop_condition, hook, description)
end





function experiment(
    params :: GroebnerEnvParams,
    episodes :: Int,
    seed = 123,
    save_dir = nothing,
)
    n = params.nvars
    d = params.maxdeg
    s = params.npols

    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_CartPole_$(t)")
    end
    log_dir = joinpath(save_dir, "tb_log")
    lg = TBLogger(log_dir, min_level = Logging.Info)
    rng = StableRNG(seed)

    env = GroebnerEnv{n, StableRNG}(params,
                                    Array{Array{term{n},1},1}[],
                                    Array{NTuple{2, Array{term{n}, 1}}}[],
                                    0,
                                    false,
                                    0,
                                    rng)
    RLBase.reset!(env)
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = MyDQNLearner(
                approximator = NeuralNetworkApproximator(
                    # model = Replicate(x -> softmax(x; dims=2)[:], Chain(
                    model = Replicate(x -> dropdims(x; dims=1), Chain(
                        Dense(4*n, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, 1, x -> x; initW = glorot_uniform(rng)),
                    )) |> gpu ,
                    # model = Replicate(x -> [y[1] for y in x], Chain(
                    #     Dense(4*3, 10, relu, initW = glorot_uniform(rng)),
                    #     Dense(10, 1, relu, initW = glorot_uniform(rng))
                    # )) |> gpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                # batch_size = 1,
                min_replay_history = 100,
                loss_func = Flux.huber_loss,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularVectorSARTTrajectory(
            capacity = 1000,
            # state = Array{Int, 2} => (),
            state = Array{Int, 2},
            # next_state = Array{Int, 2}
        ),
    )

    stop_condition = StopAfterEpisode(1000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        # DoEveryNStep() do t, agent, env
        #     with_logger(lg) do
        #         @info "training" loss = agent.policy.learner.loss
        #     end
        # end,
        # DoEveryNEpisode() do t, agent, env
        #     with_logger(lg) do
        #         @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
        #             0
        #     end
        # end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.
    You can view the runtime logs with `tensorboard --logdir $log_dir`.
    Some useful statistics are stored in the `hook` field of this experiment.
    """

    Experiment(agent, env, stop_condition, hook, description)
end

















# function ElasticArrayTrajectory(; kwargs...)
#     Trajectory(map(kwargs.data) do x
#         ElasticArray{eltype(first(x))}(undef, last(x)..., 0)
#     end)
# end

# const SARTV = (:state, :action, :reward, :terminal, :value)
# const ElasticSARTVTrajectory = Trajectory{
#     <:NamedTuple{SARTV,<:Tuple{<:ElasticArray,<:ElasticArray,<:ElasticArray,<:ElasticArray,<:ElasticArray}},
# }

# function ElasticSARTVTrajectory(;
#     state = Int => (),
#     action = Int => (),
#     reward = Float32 => (),
#     terminal = Bool => (),
#     value = Int => ()
# )
#     ElasticArrayTrajectory(;
#         state = state,
#         action = action,
#         reward = reward,
#         terminal = terminal,
#         value = value
#     )
# end

# function Base.length(t::ElasticSARTVTrajectory)
#     x = t[:terminal]
#     size(x, ndims(x))
# end


Base.@kwdef mutable struct VPGPolicy{
    A<:NeuralNetworkApproximator,
    B<:Union{NeuralNetworkApproximator,Nothing},
    R<:AbstractRNG,
} <: AbstractPolicy
    approximator::A
    baseline::B = nothing
    γ::Float32 = 0.99f0 # discount factor
    α_θ = 1.0f0 # step size of policy
    α_w = 1.0f0 # step size of baseline
    batch_size::Int = 1024
    rng::R = Random.GLOBAL_RNG
    loss::Float32 = 0.0f0
    baseline_loss::Float32 = 0.0f0
end

"""
About continuous action space, see
* [Diagonal Gaussian Policies](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
* [Clipped Action Policy Gradient](https://arxiv.org/pdf/1802.07564.pdf)
"""

function (π::VPGPolicy)(env::AbstractEnv)
    to_dev(x) = send_to_device(device(π.approximator), x)

    logits = env |> state |> to_dev |> π.approximator

    # dist = logits |> softmax |> π.dist
    dist = softmax(logits; dims=2)
    w = Weights(dropdims(dist; dims=1))
    # action = π.action_space[rand(π.rng, dist)]
    # action = rand(π.rng, dist)
    action = sample(π.rng, 1:length(w), w)
    action
end

function (π::VPGPolicy)(env::MultiThreadEnv)
    error("not implemented")
    # TODO: can PG support multi env? PG only get updated at the end of an episode.
end

function RLBase.update!(
    trajectory::ElasticSARTTrajectory,
    policy::VPGPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(trajectory[:state], state(env))
    push!(trajectory[:action], action)

    # push!(trajectory[:value], buchberger_test(copy(env), policy)[2])
    # println(is_groebner_basis(env.G))
end

function RLBase.update!(
    t::ElasticSARTTrajectory,
    ::VPGPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end

RLBase.update!(::VPGPolicy, ::ElasticSARTTrajectory, ::AbstractEnv, ::PreActStage) = nothing

function RLBase.update!(
    π::VPGPolicy,
    traj::ElasticSARTTrajectory,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    local model = π.approximator
    # to_dev(x) = send_to_device(device(model), x)
    to_dev(x) = x

    states = traj[:state]
    actions = traj[:action] |> Array # need to convert ElasticArray to Array, or code will fail on gpu. `log_prob[CartesianIndex.(A, 1:length(A))`
    gains = traj[:reward] |> x -> discount_rewards(x, π.γ)
    # values = traj[:value] |> Array

    for idx in Iterators.partition(shuffle(1:length(traj[:terminal])), π.batch_size)
        S = select_last_dim(states, idx) |> to_dev
        A = actions[idx]
        G = gains[idx] |> x -> Flux.unsqueeze(x, 1) |> to_dev
        # gains is a 1 colomn array, but the ouput of flux model is 1 row, n_batch columns array. so unsqueeze it.


        if π.baseline isa NeuralNetworkApproximator
            gs = gradient(Flux.params(π.baseline)) do
                δ = G .- [maximum(π.baseline(s)) for s in S]
                loss = mean(δ .^ 2) * π.α_w # mse
                Zygote.ignore() do
                    π.baseline_loss = loss
                end
                loss
            end
            RLBase.update!(π.baseline, gs)
        elseif π.baseline isa Nothing
            # Normalization. See
            # (http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/hw2_final.pdf)
            # (https://web.stanford.edu/class/cs234/assignment3/solution.pdf)
            # normalise should not be used with baseline. or the loss of the policy will be too small.
            δ = G |> x -> Flux.normalise(x, dims = 2)
        end

        # push!(values, 0)
        # advantages = generalized_advantage_estimation(gains, values, 0.99, 0.97; terminal = traj[:terminal])

        gs = gradient(Flux.params(model)) do
            log_prob = [logsoftmax(model(s); dims=2) for s in S]
            # log_probₐ = log_prob[CartesianIndex.(A, 1:length(A))]
            log_probₐ = [log_prob[i][A[i]] for i in 1:length(A)]
            loss = -mean(log_probₐ .* δ) * π.α_θ
            # loss = -sum(log_probₐ .* δ)
            # loss = -1 * (sum(G) * sum(log_probₐ)) # sum(G) og ikke over delta, for så misser vi den absolutte størrelse af G da delta er normaliseret
            # loss = -sum(log_probₐ .* G)
            # loss = -sum(G)
            # loss = -mean(log_probₐ .* G)
            Zygote.ignore() do
                π.loss = loss
                # println(G)
                # println(log_probₐ .* δ)
            end
            loss
        end
        RLBase.update!(model, gs)
        # println(π.loss)
        # println(gs.grads)
        # Flux.Optimise.update!(model.optimizer, Flux.params(model), gs)
    end
end


global_losses = Float32[]

function pg_experiment(  params :: GroebnerEnvParams,
    episodes :: Int,
    gamma = 0.99f0,
    save_dir = nothing,
    seed = 123)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_VPG_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)

    n = params.nvars
    d = params.maxdeg
    s = params.npols

    # env = GroebnerEnv{n, StableRNG}(params,
    #                                 Array{Array{term{n},1},1}[],
    #                                 Array{NTuple{2, Array{term{n}, 1}}}[],
    #                                 0,
    #                                 false,
    #                                 0,
    #                                 rng)

    env = rand_env(params)

    agent = Agent(
        policy = VPGPolicy(
            approximator = NeuralNetworkApproximator(
                # model = Replicate(x -> dropdims(x, dims=1), Chain(
                # # model = Chain(
                #     Dense(n*4, 64, relu; initW = glorot_uniform(rng)),
                #     Dense(64, 1, x -> x; initW = glorot_uniform(rng)),
                # )),
                model = Chain(
                    Dense(n*4, 64, relu; initW = glorot_uniform(rng)),
                    Dense(64, 64, relu; initW = glorot_uniform(rng)),
                    Dense(64, 1; initW = glorot_uniform(rng))
                ),
                optimizer = ADAM(),

            ) |> cpu,
            baseline = NeuralNetworkApproximator(
                # model = Replicate(x -> dropdims(x, dims=1), Chain(
                model = Chain(
                    Dense(n*4, 64, relu; initW = glorot_uniform(rng)),
                    Dense(64, 64, relu; initW = glorot_uniform(rng)),
                    Dense(64, 1; initW = glorot_uniform(rng)),
                ),
                optimizer = ADAM(),
            ) |> cpu,
            γ = gamma,
            rng = rng,
        ),
        trajectory = ElasticSARTTrajectory(state = Vector{Array{Int, 2}} => ()),
                                           # reward = Int => ()),
    )
    # VPG is updated after each episode
    stop_condition = StopAfterEpisode(episodes)

    total_reward_per_episode = TotalRewardPerEpisode()
    # total_loss_per_episode = TotalLoosPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        # DoEveryNEpisode() do t, agent, env
        #     push!(global_losses, agent.policy.loss)
        # end
        # DoEveryNEpisode() do t, agent, env
        #     with_logger(lg) do
        #         @info(
        #             "training",
        #             loss = agent.policy.loss,
        #             baseline_loss = agent.policy.baseline_loss,
        #             reward = total_reward_per_episode.rewards[end],
        #         )
        #     end
        # end,
    )

    description = "# Make Gröbner bases with Policy Gradients"

    Experiment(agent, env, stop_condition, hook, description)
end


function strat_rand(env::GroebnerEnv)
    return rand(1:length(env.P))
end

function strat_first(env::GroebnerEnv)
    function comp((f, g), (h, i))
        p1 = (findfirst(Ref(f) .== env.G), findfirst(Ref(g) .== env.G))
        p2 = (findfirst(Ref(h) .== env.G), findfirst(Ref(i) .== env.G))
        p1 < p2
    end
    return sortperm(env.P; lt=comp)[1]
end


function strat_real_first(env::GroebnerEnv)
    return 1
end

function strat_degree(env::GroebnerEnv)
    function comp((f, g), (h, i))
        lcm1 = lcm(LT(f), LT(g))
        lcm2 = lcm(LT(h), LT(i))
        return sum(lcm1.a) < sum(lcm2.a)
    end
    return sortperm(env.P, lt=comp)[1]
end

function strat_truedeg(env::GroebnerEnv)
    function comp((f, g), (h, i))
        
        return sum(sum.(getproperty.(S(f, g), :a))) < sum(sum.(getproperty.(S(h, i), :a)))
    end
    return sortperm(env.P, lt=comp)[1]
end

function strat_norm_or_queue(env::GroebnerEnv)
    strat = rand([strat_real_first, strat_degree])
    return strat(env)
end



function strat_normal(env::GroebnerEnv)
    function comp(p1, p2)
        f, g = p1
        h, i = p2
        lcm1 = lcm(LT(f), LT(g))
        lcm2 = lcm(LT(h), LT(i))
        return lcm1.a != lcm2.a && ! gt(lcm1.a, lcm2.a)
    end
    
    P_ = sortperm(env.P, lt = comp )[1]
    return P_
end


function eval_strats()
    params = [
        (2, 3, 10),
        (3, 3, 10),
        (4, 3, 10),
        (6, 3, 10),
        (7, 3, 10),
        (8, 3, 10),

        (3, 10, 4),
        (3, 10, 10)
    ]

    strats = [
        strat_rand,
        # strat_first,
        strat_real_first,
        strat_degree,
        strat_normal
    ]

    results = []

    for param in params
        for strat in strats
            env = rand_env(GroebnerEnvParams(param...))
            res = eval_model(env, strat; iters=500)
            push!(results, (res, param, strat))
        end
    end
    return results
end







function easy_experiment(
    episodes :: Int,
    gamma = 1.0f0,
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_VPG_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)

    env = EasyEnv([[1.0]], 0, false, 0)
    RLBase.reset!(env)

    agent = Agent(
        policy = VPGPolicy(
            approximator = NeuralNetworkApproximator(
                model = Replicate(x -> dropdims(x, dims=1), Chain(
                # model = Chain(
                    Dense(20, 64, relu; initW = glorot_uniform(rng)),
                    Dense(64, 1, x -> x; initW = glorot_uniform(rng)),
                )),
                optimizer = ADAM(),
            ) |> cpu,
            baseline = NeuralNetworkApproximator(
                model = Replicate(x -> dropdims(x, dims=1), Chain(
                    Dense(20, 64, relu; initW = glorot_uniform(rng)),
                    Dense(64, 1, x -> x; initW = glorot_uniform(rng)),
                )),
                optimizer = ADAM(),
            ) |> cpu,
            γ = gamma,
            rng = rng,
        ),
        trajectory = ElasticSARTTrajectory(state = Vector{Array{Float32, 2}} => (),
                                           reward = Float32 => ()),
    )
    # VPG is updated after each episode
    stop_condition = StopAfterEpisode(episodes)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNEpisode() do t, agent, env
            push!(global_losses, agent.policy.loss)
        end
        # DoEveryNEpisode() do t, agent, env
        #     with_logger(lg) do
        #         @info(
        #             "training",
        #             loss = agent.policy.loss,
        #             baseline_loss = agent.policy.baseline_loss,
        #             reward = total_reward_per_episode.rewards[end],
        #         )
        #     end
        # end,
    )

    description = "# Play CartPole with VPG"

    Experiment(agent, env, stop_condition, hook, description)
end
