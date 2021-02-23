using ReinforcementLearning



mutable struct TTTEnv
    state::Array{Int, 2} # 0=Empty, 1=Tic, 2=Tac
    done::Bool
end

function RLBase.reset!(env::TTTEnv)
    env.state = fill(0, (3, 3))
    env.done = false
end
RLBase.state(env::TTTEnv) = env.state

function is_win(env::TTTEnv, player)
    b = env.state .== 1
    @inbounds begin
        b[1, 1, p] & b[1, 2, p] & b[1, 3, p] ||
            b[2, 1, p] & b[2, 2, p] & b[2, 3, p] ||
            b[3, 1, p] & b[3, 2, p] & b[3, 3, p] ||
            b[1, 1, p] & b[2, 1, p] & b[3, 1, p] ||
            b[1, 2, p] & b[2, 2, p] & b[3, 2, p] ||
            b[1, 3, p] & b[2, 3, p] & b[3, 3, p] ||
            b[1, 1, p] & b[2, 2, p] & b[3, 3, p] ||
            b[1, 3, p] & b[2, 2, p] & b[3, 1, p]
    end
end


mutable struct TTTLearner <: AbstractLearner
    model::Dict{Array{Int, 2}, Array{Int, 2}}
end

(learner::TTTLearner)(env) =
    let weights = learner[RLBase.state(env)]
        findfirst(maximum(weights) .== weights)
    end

RLBase.update!()
