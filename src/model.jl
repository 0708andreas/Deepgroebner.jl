using Flux
using ReinforcementLearning
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Zygote
using Dates
using TensorBoardLogger
using StableRNGs
using Logging
using Setfield
using SliceMap

include("GroebnerEnv.jl")

export Replicate,
    experiment

struct Replicate{F, T}
    reducer::F
    model::T
end

# @functor Replicate
Flux.trainable(m::Replicate) = Flux.trainable(m.model)

# (m::Replicate)(x::AbstractArray) = m.reducer(mapslices(m.model, x; dims=1))
(m::Replicate)(x::AbstractArray) = m.reducer(slicemap(m.model, x; dims=1))
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

    if length(r) == 0
        return
    end
    

    # s_ = s[2:end]
    # s = s[1:end-1]
    # a = a[1:end-1]
    # println("a = ", a)

    if length(s) == 0
        return
    end
    if ! t[end]
        return
    end
    # println(r[end])
    # println(s[1])
    gs = gradient(params(Q)) do
        q = [Q(s[i])[a[i]] for i in 1:(length(s))]
        q_ = vcat([maximum(Q(s[i+1])) for i in 1:(length(s) - 1)], [r[end]])

        # Zygote.ignore() do
        #     println(q)
        #     println(Q(s[1]))
        #     println(q_)
        # end
        

        # loss = Flux.huber_loss(q, q_)
        loss = sum([abs(q[i] - q_[i]) for i in 1:length(q)])

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

    params = GroebnerEnvParams(3, 4, 2)
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
                    model = Replicate(x -> softmax(x; dims=2)[:], Chain(
                        Dense(4*3, 10, relu; initW = glorot_uniform(rng)),
                        Dense(10, 1, relu; initW = glorot_uniform(rng)),
                    )) |> cpu,
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

    stop_condition = StopAfterStep(1_000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                    0
            end
        end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.
    You can view the runtime logs with `tensorboard --logdir $log_dir`.
    Some useful statistics are stored in the `hook` field of this experiment.
    """

    Experiment(agent, env, stop_condition, hook, description)
end



function experiment2(
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

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na; initW = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
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
            state = Vector{Float32}
        ),
    )

    stop_condition = StopAfterStep(10_000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                    0
            end
        end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.
    You can view the runtime logs with `tensorboard --logdir $log_dir`.
    Some useful statistics are stored in the `hook` field of this experiment.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
